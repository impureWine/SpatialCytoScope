"""FastAPI and shared read-only analytics for the spatial pathology workbench."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator, Literal

import anndata as ad
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse
from scipy.stats import mannwhitneyu
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SPATIAL_DATA_DIR", str(ROOT / "data/processed")))
CELL_TYPES = ("T_cell", "B_cell", "Macrophage", "NK_cell", "Dendritic", "Tumor_Epi")
MODEL_KEYS = {"baseline": ("domain_pred_baseline", "cell_proportions_pred_baseline"),
              "spatial_gat": ("domain_pred_gat", "cell_proportions_pred_gat")}
SLICE_NAMES = ("slice_train", "slice_test")


class ROIRequest(BaseModel):
    """Strict ROI payload using zero-based AnnData row positions."""

    model_config = ConfigDict(extra="forbid")
    spot_indices: list[Annotated[int, Field(strict=True, ge=0)]] = Field(min_length=1, max_length=100000)
    model_type: Literal["spatial_gat", "baseline"] = "spatial_gat"


def load_slices(data_dir: Path = DATA_DIR) -> dict[str, ad.AnnData]:
    """Load both modeled slices once.

    Args:
        data_dir: Directory containing the two processed H5AD files.

    Returns:
        Mapping from canonical slice names to independent read-only inputs.

    Raises:
        FileNotFoundError: If a slice has not been generated.
    """
    return {name: ad.read_h5ad(data_dir / f"{name}.h5ad") for name in SLICE_NAMES}


def svg_records(adata: ad.AnnData) -> list[dict[str, str | float]]:
    """Decode either in-memory or H5AD SVG representations.

    Args:
        adata: Slice containing svg_results.

    Returns:
        JSON-compatible records with finite Moran statistics.
    """
    values = adata.uns.get("svg_results", [])
    records = values.to_dict("records") if isinstance(values, pd.DataFrame) else list(values)
    return [{"gene": str(row["gene"]), "morans_i": float(row["morans_i"]),
             "p_value": float(row["p_value"]), "fdr": float(row["fdr"])} for row in records]


def model_metrics(adata: ad.AnnData, model_type: str) -> dict[str, float | None]:
    """Calculate measured domain agreement and spatial fragmentation.

    Args:
        adata: Slice with model predictions and optional ground truth.
        model_type: Key identifying baseline or spatial_gat.

    Returns:
        ARI, NMI, PAS (fraction with more than half neighbors disagreeing),
        edge boundary rate and cell-proportion RMSE, or null when unavailable.
    """
    domain_key, cell_key = MODEL_KEYS[model_type]
    metrics: dict[str, float | None] = dict.fromkeys(["ari", "nmi", "pas", "boundary_rate", "cell_rmse"])
    if domain_key not in adata.obs:
        return metrics
    labels = adata.obs[domain_key].astype(str).to_numpy()
    if "domain_true" in adata.obs and not adata.obs["domain_true"].isna().any():
        truth = adata.obs["domain_true"].astype(str).to_numpy()
        metrics["ari"] = float(adjusted_rand_score(truth, labels))
        metrics["nmi"] = float(normalized_mutual_info_score(truth, labels))
    if "spatial_connectivities" in adata.obsp:
        adjacency = sparse.csr_matrix(adata.obsp["spatial_connectivities"]).copy()
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
        adjacency.data[:] = 1
        edges = adjacency.tocoo()
        disagree = labels[edges.row] != labels[edges.col]
        degree = np.asarray(adjacency.sum(1)).ravel()
        differing = np.bincount(edges.row, weights=disagree.astype(float), minlength=adata.n_obs)
        connected = degree > 0
        if connected.any():
            metrics["pas"] = float(np.mean(differing[connected] > degree[connected] / 2))
            metrics["boundary_rate"] = float(disagree.mean())
    if cell_key in adata.obsm and "cell_proportions_true" in adata.obsm:
        metrics["cell_rmse"] = float(np.sqrt(np.mean((adata.obsm[cell_key] - adata.obsm["cell_proportions_true"]) ** 2)))
    return metrics


def slice_summary(adata: ad.AnnData, slice_name: str) -> dict[str, object]:
    """Build the same measured summary for local and HTTP modes.

    Args:
        adata: Analyzed slice.
        slice_name: Canonical slice identifier.

    Returns:
        Spot count, genes, SVGs, model metrics and baseline-relative changes.
    """
    metrics = {name: model_metrics(adata, name) for name in MODEL_KEYS}
    base, gat = metrics["baseline"], metrics["spatial_gat"]
    ari_change = None
    pas_reduction = None
    if base["ari"] is not None and gat["ari"] is not None and abs(base["ari"]) > 1e-12:
        ari_change = 100 * (gat["ari"] - base["ari"]) / abs(base["ari"])
    if base["pas"] is not None and gat["pas"] is not None and base["pas"] > 0:
        pas_reduction = 100 * (base["pas"] - gat["pas"]) / base["pas"]
    return {"slice_name": slice_name, "n_spots": adata.n_obs, "n_genes": adata.n_vars,
            "gene_names": adata.var_names.tolist(), "top_svgs": svg_records(adata),
            "cell_type_names": list(CELL_TYPES), "models": metrics,
            "ari_improvement_pct": ari_change, "pas_reduction_pct": pas_reduction,
            "pas_definition": "fraction of connected spots with >50% differently labeled spatial neighbors"}


def analyze_roi(adata: ad.AnnData, payload: ROIRequest) -> dict[str, object]:
    """Compare unique ROI rows to all remaining spots without mutating data.

    Args:
        adata: Slice with log1p-normalized X and model cell proportions.
        payload: Validated indices and chosen prediction model.

    Returns:
        Top eight differential genes and mean six-cell proportions. Log2FC uses
        mean expm1(X) abundances plus 1e-9, not means of log expression. U tests
        are two-sided, asymptotic and tie-corrected; BH covers all retained genes.

    Raises:
        ValueError: If indices are out of bounds, background is empty or predictions missing.
    """
    indices = np.unique(payload.spot_indices)
    if indices[-1] >= adata.n_obs:
        raise ValueError("spot_indices contains an out-of-range position")
    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[indices] = True
    if mask.all():
        raise ValueError("ROI must leave at least one background Spot")
    cell_key = MODEL_KEYS[payload.model_type][1]
    if cell_key not in adata.obsm:
        raise ValueError("Selected model predictions are unavailable; run Step 3")
    matrix = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    roi, background = matrix[mask].astype(np.float64), matrix[~mask].astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Expression contains nonfinite values")
    test = mannwhitneyu(roi, background, axis=0, alternative="two-sided", method="asymptotic")
    p_values = np.nan_to_num(test.pvalue, nan=1.0)
    logfc = np.log2((np.expm1(roi).mean(0) + 1e-9) / (np.expm1(background).mean(0) + 1e-9))
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order] * adata.n_vars / np.arange(1, adata.n_vars + 1)
    fdr = np.empty_like(p_values)
    fdr[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    table = pd.DataFrame({"gene": adata.var_names, "log2fc": logfc, "p_value": p_values, "fdr": fdr})
    table["magnitude"] = np.abs(logfc)
    top = table.sort_values(["p_value", "magnitude", "gene"], ascending=[True, False, True]).head(8)
    proportions = np.asarray(adata.obsm[cell_key])[mask].mean(0)
    return {"n_roi": int(mask.sum()), "n_background": int((~mask).sum()),
            "model_type": payload.model_type, "spot_indices": indices.tolist(),
            "differential_genes": top.drop(columns="magnitude").to_dict("records"),
            "cell_proportions": {name: float(value) for name, value in zip(CELL_TYPES, proportions)},
            "test": "two-sided asymptotic Mann-Whitney U; BH over all retained genes"}


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load slice snapshots on startup and release them on shutdown.

    Args:
        application: FastAPI service instance.

    Yields:
        Control to the running service after loading both slices.
    """
    application.state.slices = load_slices()
    yield
    application.state.slices.clear()


app = FastAPI(title="SpatialCytoScope API", version="0.4.0", lifespan=lifespan)


def get_slice(request: Request, slice_name: str) -> ad.AnnData:
    """Resolve a canonical slice name from the startup snapshot.

    Args:
        request: Incoming request containing application state.
        slice_name: slice_train or slice_test.

    Returns:
        Loaded AnnData snapshot.

    Raises:
        HTTPException: For an unknown slice.
    """
    if slice_name not in request.app.state.slices:
        raise HTTPException(status_code=404, detail="Unknown slice; choose slice_train or slice_test")
    return request.app.state.slices[slice_name]


@app.get("/api/v1/health")
def health(request: Request) -> dict[str, object]:
    """Report startup readiness.

    Args:
        request: Active API request.

    Returns:
        Service status and loaded slice names.
    """
    return {"status": "ok", "loaded_slices": sorted(request.app.state.slices)}


@app.get("/api/v1/slice/{slice_name}/summary")
def summary(slice_name: str, request: Request) -> dict[str, object]:
    """Return measured slice summary.

    Args:
        slice_name: Canonical slice identifier.
        request: Active API request.

    Returns:
        JSON-compatible metadata and model metrics.
    """
    return slice_summary(get_slice(request, slice_name), slice_name)


@app.post("/api/v1/slice/{slice_name}/roi_analysis")
def roi_analysis(slice_name: str, payload: ROIRequest, request: Request) -> dict[str, object]:
    """Calculate ROI statistics in the synchronous worker pool.

    Args:
        slice_name: Canonical slice identifier.
        payload: Validated ROI selection.
        request: Active API request.

    Returns:
        Differential genes and mean predicted cell fractions.

    Raises:
        HTTPException: If the selection is invalid.
    """
    data = get_slice(request, slice_name)
    try:
        return analyze_roi(data, payload)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
