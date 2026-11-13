"""Quality control, expression manifolds, spatial graphs and permutation Moran I."""

import argparse
from pathlib import Path
from time import perf_counter

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from src.config_schema import AppConfig as Config
from src.config_schema import PROJECT_ROOT, load_config

MARKERS = ("CD3D", "CD19", "CD68", "EPCAM", "COL1A1", "MKI67", "NKG7", "FCER1A")


def run_qc_and_manifold(adata: ad.AnnData, cfg: Config) -> ad.AnnData:
    """Filter raw counts and calculate a reproducible expression manifold.

    Args:
        adata: Raw input; an existing counts layer takes precedence over X.
        cfg: Validated application configuration.

    Returns:
        Independent AnnData with float32 log-normalized X, retained raw counts,
        30-dimensional PCA, two-dimensional UMAP and categorical leiden_pca.

    Raises:
        ValueError: If counts are invalid or QC leaves insufficient dimensions.
    """
    result = adata.copy()
    result.X = result.layers.get("counts", result.X).copy().astype(np.float32)
    values = result.X.data if sparse.issparse(result.X) else np.asarray(result.X)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Counts must be finite and nonnegative")
    if result.n_obs == 0 or result.n_vars == 0:
        raise ValueError("QC requires a nonempty expression matrix")
    result.var["mt"] = result.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(result, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    keep = ((result.obs["total_counts"] >= cfg.qc.min_counts)
            & (result.obs["n_genes_by_counts"] >= cfg.qc.min_genes)
            & (result.obs["pct_counts_mt"] <= cfg.qc.max_mt_pct))
    result = result[keep].copy()
    if min(result.shape) <= 30:
        raise ValueError(f"QC retained {result.shape}; 30-component PCA requires >30 spots and genes")
    result.layers["counts"] = result.X.copy()
    sc.pp.normalize_total(result, target_sum=1e4)
    sc.pp.log1p(result)
    result.X = result.X.astype(np.float32)
    sc.pp.highly_variable_genes(result, n_top_genes=min(cfg.qc.n_top_genes, result.n_vars),
                                flavor="seurat", subset=False)
    selected = result.var["highly_variable"].to_numpy() | result.var_names.isin(MARKERS)
    result.var["highly_variable"] = selected
    result = result[:, selected].copy()
    if result.n_vars <= 30:
        raise ValueError("HVG selection plus markers must retain at least 31 genes for PCA")
    with threadpool_limits(limits=1):
        sc.pp.pca(result, n_comps=30, random_state=cfg.seed, svd_solver="arpack")
        sc.pp.neighbors(result, n_neighbors=min(15, result.n_obs - 1), n_pcs=30,
                        use_rep="X_pca", random_state=cfg.seed, transformer="sklearn")
        sc.tl.umap(result, random_state=cfg.seed, maxiter=100)
        sc.tl.leiden(result, key_added="leiden_pca", resolution=0.6,
                     random_state=cfg.seed, flavor="igraph", directed=False, n_iterations=2)
    result.obsm["X_pca"] = result.obsm["X_pca"].astype(np.float32)
    result.obsm["X_umap"] = result.obsm["X_umap"].astype(np.float32)
    result.uns["data_state"] = "log1p_normalized_hvg"
    result.uns["preprocessing"] = {"seed": cfg.seed, "input_spots": adata.n_obs,
                                    "retained_spots": result.n_obs, "target_sum": 1e4,
                                    "marker_genes_retained": [g for g in MARKERS if g in result.var_names]}
    return result


def build_spatial_graph(adata: ad.AnnData, n_neighbors: int = 6, sigma: float = 50.0) -> ad.AnnData:
    """Build a symmetric union-kNN graph and self-loop normalized operator.

    Args:
        adata: Input object with N by 2 spatial pixel coordinates; modified in place.
        n_neighbors: Positive number of neighbors, capped at N minus one.
        sigma: Positive Gaussian bandwidth in the same units as coordinates.

    Returns:
        Input with CSR binary connectivities, Gaussian weights and normalized adjacency.

    Raises:
        ValueError: If parameters or coordinates are invalid or fewer than two spots exist.
    """
    if (isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int)
            or n_neighbors < 1 or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError("n_neighbors must be a positive integer and sigma finite and positive")
    coords = np.asarray(adata.obsm.get("spatial", []), dtype=np.float64)
    n = adata.n_obs
    if n < 2 or coords.shape != (n, 2) or not np.isfinite(coords).all():
        raise ValueError("Require at least two spots and finite spatial coordinates of shape (N, 2)")
    k = min(n_neighbors, n - 1)
    _, indices = cKDTree(coords).query(coords, k=k + 1, workers=1)
    # Remove by index, rather than position, so coincident spots remain neighbors.
    rows = np.repeat(np.arange(n), k)
    cols = np.concatenate([candidates[candidates != i][:k] for i, candidates in enumerate(indices)])
    adjacency = sparse.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    edges = adjacency.tocoo()
    distance = np.linalg.norm(coords[edges.row] - coords[edges.col], axis=1)
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-0.5 * np.square(distance / sigma))
    weighted = sparse.csr_matrix((weights, (edges.row, edges.col)), shape=(n, n))
    with_loops = weighted + sparse.eye(n, format="csr")
    inv_sqrt = sparse.diags(1 / np.sqrt(np.asarray(with_loops.sum(axis=1)).ravel()))
    adata.obsp["spatial_connectivities"] = adjacency
    adata.obsp["spatial_weights"] = weighted
    adata.obsp["spatial_norm_adj"] = (inv_sqrt @ with_loops @ inv_sqrt).tocsr()
    adata.uns["spatial_graph"] = {"n_neighbors": k, "sigma": sigma, "symmetrization": "union"}
    return adata


def compute_morans_i(adata: ad.AnnData, top_k: int = 50, n_perms: int = 100) -> pd.DataFrame:
    """Calculate vectorized Moran I with upper-tail permutations and global BH FDR.

    Args:
        adata: Log-normalized expression and a symmetric spatial_weights matrix.
        top_k: Maximum number of genes to return, sorted by descending Moran I.
        n_perms: Number of shared row permutations; each preserves gene covariance.

    Returns:
        Top genes with gene, morans_i, p_value and fdr columns. The same records
        are stored as dictionaries in uns['svg_results']. Constant genes receive
        I=0, p=1 and are excluded from the top list. FDR covers all tested genes.

    Raises:
        ValueError: If dimensions, expression, weights or parameters are invalid.
    """
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (top_k, n_perms)):
        raise ValueError("top_k and n_perms must be positive integers")
    n = adata.n_obs
    if n < 3 or adata.n_vars == 0 or "spatial_weights" not in adata.obsp:
        raise ValueError("Require >=3 spots, genes and spatial_weights")
    weights = sparse.csr_matrix(adata.obsp["spatial_weights"], dtype=np.float64)
    asymmetry = weights - weights.T
    if (not np.isfinite(weights.data).all() or np.any(weights.data < 0)
            or np.any(weights.diagonal() != 0)
            or np.max(np.abs(asymmetry.data), initial=0) > 1e-10):
        raise ValueError("Weights must be finite, nonnegative, symmetric and have zero diagonal")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Spatial weights must have a positive sum")
    expression = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    z = np.asarray(expression, dtype=np.float64).copy()
    if not np.isfinite(z).all():
        raise ValueError("Expression must be finite")
    z -= z.mean(axis=0)
    scale = z.std(axis=0)
    valid = scale > np.finfo(np.float64).eps
    z /= np.where(valid, scale, 1)
    denominator = np.einsum("ij,ij->j", z, z)
    denominator = np.where(valid, denominator, 1)
    observed = (n / total) * np.einsum("ij,ij->j", z, weights @ z) / denominator
    observed[~valid] = 0
    seed = int(adata.uns.get("preprocessing", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    exceedances = np.zeros(adata.n_vars, dtype=np.int64)
    for _ in range(n_perms):
        permuted = z[rng.permutation(n)]
        statistic = (n / total) * np.einsum("ij,ij->j", permuted, weights @ permuted) / denominator
        exceedances += statistic >= observed - 1e-12
    p_values = (exceedances + 1.0) / (n_perms + 1.0)
    p_values[~valid] = 1
    order = np.argsort(p_values, kind="stable")
    adjusted = p_values[order] * adata.n_vars / np.arange(1, adata.n_vars + 1)
    adjusted = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    fdr = np.empty_like(adjusted)
    fdr[order] = adjusted
    results = pd.DataFrame({"gene": adata.var_names.to_numpy(), "morans_i": observed,
                            "p_value": p_values, "fdr": fdr})
    top = results.loc[valid].sort_values("morans_i", ascending=False, kind="stable").head(top_k).reset_index(drop=True)
    adata.uns["svg_results"] = top.to_dict(orient="records")
    adata.uns["moran_test"] = {"seed": seed, "n_perms": n_perms, "tail": "greater",
                               "fdr_method": "Benjamini-Hochberg", "n_genes_tested": adata.n_vars,
                               "n_constant_genes": int((~valid).sum()), "weight_key": "spatial_weights"}
    return top


def main() -> None:
    """Process both raw slices and save interoperable H5AD files.

    H5AD cannot encode lists of dictionaries. Only during writing, svg_results
    becomes a DataFrame with identical named columns; the in-memory list is
    restored afterward. A table read from H5AD supports to_dict('records').

    Raises:
        FileNotFoundError: If a raw slice has not been generated yet.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    sources = [cfg.paths.raw_dir / f"slice_{split}.h5ad" for split in ("train", "test")]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Missing {source}; run python -m src.data_loader --mode synthetic")
    started = perf_counter()
    for source in sources:
        result = run_qc_and_manifold(ad.read_h5ad(source), cfg)
        build_spatial_graph(result, cfg.spatial.n_neighbors, cfg.spatial.sigma)
        table = compute_morans_i(result)
        records = result.uns["svg_results"]
        result.uns["svg_results"] = table
        target = cfg.paths.processed_dir / source.name
        try:
            result.write_h5ad(target, compression="gzip")
        finally:
            result.uns["svg_results"] = records
        print(f"Saved {target}: {result.shape}, PCA={result.obsm['X_pca'].shape}, "
              f"UMAP={result.obsm['X_umap'].shape}, SVGs={len(table)}")
    print(f"Preprocessing completed in {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
