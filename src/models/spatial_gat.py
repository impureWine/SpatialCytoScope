"""Pure PyTorch sparse spatial attention, multitask losses and training CLI."""

import argparse
import random
from pathlib import Path
from time import perf_counter
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import Tensor, nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from src.config_schema import PROJECT_ROOT, detect_device, load_config
from src.models.baseline import BaselineSpatialModel


class SparseGATLayer(nn.Module):
    """Multihead destination aggregation with spatially weighted attention."""

    def __init__(self, in_features: int, out_features: int, heads: int = 1) -> None:
        """Initialize concatenated attention heads.

        Args:
            in_features: Input feature width.
            out_features: Output width per head.
            heads: Number of concatenated independent heads.

        Raises:
            ValueError: If a dimension is not positive.
        """
        super().__init__()
        if min(in_features, out_features, heads) < 1:
            raise ValueError("All attention dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.projection = nn.Linear(in_features, heads * out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(heads, out_features))
        self.bias = nn.Parameter(torch.zeros(heads * out_features))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        """Aggregate j into i for each edge (i, j), concatenating the heads.

        Args:
            h: Finite N by in_features tensor.
            edge_index: Long 2 by E tensor; row zero is receiving node i.
            edge_weight: Nonnegative E-vector on the same device as h.

        Returns:
            N by (heads times out_features) transformed features. Empty incoming
            groups produce zero before the learned bias. Callers add self loops.

        Raises:
            ValueError: If tensor shapes, dtypes, devices or edge values are invalid.
        """
        if (h.ndim != 2 or h.shape[1] != self.in_features or edge_index.ndim != 2
                or edge_index.shape[0] != 2 or edge_index.dtype != torch.long
                or edge_weight.shape != (edge_index.shape[1],)
                or h.device != edge_index.device or h.device != edge_weight.device):
            raise ValueError("Invalid feature, edge or weight tensor contract")
        if (not torch.isfinite(h).all() or not torch.isfinite(edge_weight).all()
                or torch.any(edge_weight < 0)
                or (edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= h.shape[0]))):
            raise ValueError("Features and weights must be finite and indices in range")
        n = h.shape[0]
        features = self.projection(h).reshape(n, self.heads, self.out_features)
        receiver, neighbor = edge_index
        src = (features * self.a_src).sum(dim=-1)
        dst = (features * self.a_dst).sum(dim=-1)
        scores = F.leaky_relu(src[receiver] + dst[neighbor], negative_slope=0.2) * edge_weight[:, None]
        groups = receiver[:, None].expand(-1, self.heads)
        maximum = torch.full((n, self.heads), -torch.inf, device=h.device, dtype=h.dtype)
        maximum.scatter_reduce_(0, groups, scores.detach(), reduce="amax", include_self=True)
        exponential = torch.exp(scores - maximum[receiver])
        denominator = torch.zeros((n, self.heads), device=h.device, dtype=h.dtype)
        denominator.index_add_(0, receiver, exponential)
        attention = exponential / denominator[receiver].clamp_min(torch.finfo(h.dtype).tiny)
        if h.device.type == "mps":
            # Metal has no sparse matrix multiplication backend.
            aggregated = torch.zeros_like(features)
            aggregated.index_add_(0, receiver, attention[..., None] * features[neighbor])
        else:
            aggregated = torch.stack([
                torch.sparse.mm(torch.sparse_coo_tensor(edge_index, attention[:, head],
                                (n, n), device=h.device, check_invariants=False).coalesce(), features[:, head])
                for head in range(self.heads)
            ], dim=1)
        return aggregated.reshape(n, self.heads * self.out_features) + self.bias


class SpatialGATAE(nn.Module):
    """Spatial attention autoencoder with domain and cell-mixture supervision."""

    def __init__(self, n_genes: int, n_domains: int = 5, hidden_dim: int = 128,
                 latent_dim: int = 30, heads: int = 4) -> None:
        """Construct the two-layer encoder and three task heads.

        Args:
            n_genes: Number of aligned expression features.
            n_domains: Number of training domain classes.
            hidden_dim: Per-head first-layer feature width.
            latent_dim: Spatial embedding width, fixed at 30 by the slot contract.
            heads: Number of first-layer concatenated heads.

        Raises:
            ValueError: If the embedding width violates the AnnData contract.
        """
        super().__init__()
        if latent_dim != 30 or n_domains < 2:
            raise ValueError("Require latent_dim=30 and at least two domains")
        self.model_args = {"n_genes": n_genes, "n_domains": n_domains, "hidden_dim": hidden_dim,
                           "latent_dim": latent_dim, "heads": heads}
        self.encoder1 = SparseGATLayer(n_genes, hidden_dim, heads)
        self.encoder2 = SparseGATLayer(hidden_dim * heads, latent_dim, 1)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_genes))
        self.domain_head = nn.Linear(latent_dim, n_domains)
        self.cell_head = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 6), nn.Softmax(dim=-1))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> dict[str, Tensor]:
        """Calculate embeddings and all three task outputs.

        Args:
            x: N by G log-normalized expression matrix.
            edge_index: Directed sparse graph edge indices, including self loops.
            edge_weight: Corresponding spatial weights, with unit self-loop weights.

        Returns:
            Dictionary containing z, reconstruction, domain_logits and cell_proportions.
        """
        z = self.encoder2(F.elu(self.encoder1(x, edge_index, edge_weight)), edge_index, edge_weight)
        return {"z": z, "reconstruction": self.decoder(z), "domain_logits": self.domain_head(z),
                "cell_proportions": self.cell_head(z)}


def multitask_loss(outputs: dict[str, Tensor], x: Tensor, domain_true: Tensor,
                   cell_true: Tensor, edge_index: Tensor, edge_weight: Tensor,
                   lambda_task: float = 1.0, lambda_smooth: float = 0.3) -> dict[str, Tensor]:
    """Evaluate reconstruction, CE, forward KL(predicted || truth), and smoothness.

    Args:
        outputs: Network forward outputs.
        x: Normalized expression reconstruction target.
        domain_true: Long vector of contiguous training domain indices.
        cell_true: Nonnegative N by 6 reference proportions summing to one.
        edge_index: Original spatial edges without self loops.
        edge_weight: Original Gaussian spatial weights.
        lambda_task: Shared weight for CE and KL supervision.
        lambda_smooth: Weight of mean weighted squared embedding distance.

    Returns:
        Scalar tensors total, reconstruction, domain, cell_kl and smoothness.

    Raises:
        ValueError: If supervision or loss weights are invalid.
    """
    if (not np.isfinite([lambda_task, lambda_smooth]).all() or min(lambda_task, lambda_smooth) < 0
            or cell_true.shape != outputs["cell_proportions"].shape
            or not torch.isfinite(cell_true).all() or torch.any(cell_true < 0)
            or not torch.allclose(cell_true.sum(-1), torch.ones_like(cell_true[:, 0]), atol=1e-5)):
        raise ValueError("Require valid simplex cell targets and nonnegative finite loss weights")
    reconstruction = F.mse_loss(outputs["reconstruction"], x)
    domain = F.cross_entropy(outputs["domain_logits"], domain_true)
    epsilon = torch.finfo(x.dtype).eps
    pred = outputs["cell_proportions"].clamp_min(epsilon)
    pred = pred / pred.sum(-1, keepdim=True)
    truth = cell_true.clamp_min(epsilon)
    truth = truth / truth.sum(-1, keepdim=True)
    kl = (pred * (pred.log() - truth.log())).sum(-1).mean()
    # Average over the original directed spatial edges, excluding added self loops.
    z = outputs["z"]
    if edge_weight.numel():
        receiver, neighbor = edge_index
        squared_distance = (z[receiver] - z[neighbor]).square().sum(-1)
        smoothness = (edge_weight * squared_distance).mean()
    else:
        smoothness = z.sum() * 0
    total = reconstruction + lambda_task * (domain + kl) + lambda_smooth * smoothness
    return {"total": total, "reconstruction": reconstruction, "domain": domain,
            "cell_kl": kl, "smoothness": smoothness}


def graph_tensors(adata: ad.AnnData, device: torch.device, self_loops: bool = True) -> tuple[Tensor, Tensor]:
    """Convert Gaussian spatial weights into device tensors.

    Args:
        adata: Object with spatial_weights in obsp.
        device: Destination tensor device.
        self_loops: Add unit diagonal for attention; disable for smoothing.

    Returns:
        Long edge index and float32 edge weights.

    Raises:
        ValueError: If spatial weights are missing, negative or nonfinite.
    """
    if "spatial_weights" not in adata.obsp:
        raise ValueError("Run spatial graph construction before modeling")
    weights = sparse.csr_matrix(adata.obsp["spatial_weights"], dtype=np.float32).copy()
    weights.setdiag(0)
    weights.eliminate_zeros()
    if not np.isfinite(weights.data).all() or np.any(weights.data < 0):
        raise ValueError("Spatial weights must be finite and nonnegative")
    if self_loops:
        weights = weights + sparse.eye(adata.n_obs, dtype=np.float32, format="csr")
    coo = weights.tocoo()
    return (torch.as_tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device),
            torch.as_tensor(coo.data, dtype=torch.float32, device=device))


def expression_tensor(adata: ad.AnnData, gene_names: Sequence[str], device: torch.device) -> Tensor:
    """Align inference genes exactly to training order before densifying.

    Args:
        adata: Log-normalized expression data.
        gene_names: Ordered training feature vocabulary.
        device: Destination tensor device.

    Returns:
        Finite N by G float32 tensor.

    Raises:
        ValueError: If genes are duplicated, missing or expression is nonfinite.
    """
    if (not adata.var_names.is_unique or len(set(gene_names)) != len(gene_names)
            or not set(gene_names) <= set(adata.var_names)):
        raise ValueError("Inference requires all unique training genes; rerun preprocessing with a shared gene vocabulary")
    matrix = adata[:, list(gene_names)].X
    array = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(array).all():
        raise ValueError("Expression must be finite")
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def predict_slice(model: SpatialGATAE, adata: ad.AnnData, gene_names: Sequence[str],
                  domain_classes: Sequence[int | str], device: torch.device) -> ad.AnnData:
    """Write GAT predictions without reading inference labels or cell truth.

    Args:
        model: Fitted spatial model.
        adata: Slice with normalized expression and spatial weights.
        gene_names: Training gene order.
        domain_classes: Training class vocabulary in logit order.
        device: Device hosting the model.

    Returns:
        Input with spatial embedding and both GAT prediction slots populated.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(expression_tensor(adata, gene_names, device), *graph_tensors(adata, device))
    adata.obsm["X_spatial_gat"] = outputs["z"].cpu().numpy().astype(np.float32)
    indices = outputs["domain_logits"].argmax(-1).cpu().numpy()
    adata.obs["domain_pred_gat"] = pd.Categorical(np.asarray(domain_classes)[indices])
    proportions = outputs["cell_proportions"].cpu().numpy().astype(np.float32)
    adata.obsm["cell_proportions_pred_gat"] = proportions
    adata.obsm["cell_proportions_pred"] = proportions.copy()
    return adata


def main() -> None:
    """Train only on the training slice, save artifacts and infer both slices.

    Raises:
        ValueError: If training supervision or shared inference features are missing.
        FloatingPointError: If optimization produces a nonfinite loss.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    started = perf_counter()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(1)
    device = torch.device(detect_device())
    train_path = cfg.paths.processed_dir / "slice_train.h5ad"
    test_path = cfg.paths.processed_dir / "slice_test.h5ad"
    train, test = ad.read_h5ad(train_path), ad.read_h5ad(test_path)
    genes = train.var_names.tolist()
    # Validate both feature vocabularies before training or overwriting any file.
    x = expression_tensor(train, genes, device)
    expression_tensor(test, genes, device)
    baseline = BaselineSpatialModel()
    with threadpool_limits(limits=1):
        baseline.fit(train)
    classes = baseline.classifier.classes_.tolist()
    class_to_index = {label: index for index, label in enumerate(classes)}
    domain_true = torch.tensor([class_to_index[label] for label in train.obs["domain_true"]], dtype=torch.long, device=device)
    cell_true = torch.as_tensor(np.asarray(train.obsm["cell_proportions_true"]), dtype=torch.float32, device=device)
    edges, weights = graph_tensors(train, device)
    smooth_edges, smooth_weights = graph_tensors(train, device, self_loops=False)
    model = SpatialGATAE(len(genes), len(classes), cfg.model.hidden_dim, cfg.model.latent_dim, cfg.model.heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(cfg.model.epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = multitask_loss(model(x, edges, weights), x, domain_true, cell_true,
                                smooth_edges, smooth_weights, cfg.model.lambda_task, cfg.model.lambda_smooth)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}")
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
        optimizer.step()
        record = {name: float(value.detach().cpu()) for name, value in losses.items()}
        history.append(record)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{cfg.model.epochs}: loss={record['total']:.5f}", flush=True)
    output_dir = cfg.paths.processed_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline.save(output_dir / "baseline.joblib")
    # CPU tensors plus primitive metadata allow portable weights_only=True loading.
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_args": model.model_args, "gene_names": genes, "domain_classes": classes,
                "cell_type_names": [str(name) for name in train.uns["cell_type_names"]], "seed": cfg.seed,
                "history": history, "config": cfg.model.model_dump()}, output_dir / "spatial_gat.pt")
    for data, path in ((train, train_path), (test, test_path)):
        baseline.predict(data)
        predict_slice(model, data, genes, classes, device)
        data.uns["model_training"] = {"seed": cfg.seed, "epochs": cfg.model.epochs,
                                       "training_slice": "slice_train", "device": str(device),
                                       "cell_proportions_pred_source": "spatial_gat"}
        temporary = path.with_name(f".{path.stem}.modeling.h5ad")
        data.write_h5ad(temporary, compression="gzip")
        temporary.replace(path)
        print(f"Updated {path}: embedding={data.obsm['X_spatial_gat'].shape}", flush=True)
    print(f"Device={device}; elapsed={perf_counter() - started:.2f}s; models={output_dir}")


if __name__ == "__main__":
    main()
