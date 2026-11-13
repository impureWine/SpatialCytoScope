"""Numerical and data-contract regression tests for the complete modeling pipeline."""

from pathlib import Path
from typing import Iterator

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from src.data_loader import generate_synthetic_visium
from src.evaluation import chaos_score, deconvolution_metrics, integration_metrics, mean_variance_align, percentage_abnormal_spots
from src.models.spatial_gat import SpatialGATAE, graph_tensors
from src.preprocessing import compute_morans_i


@pytest.fixture(autouse=True)
def check_sparse_invariants() -> Iterator[None]:
    """Enable native PyTorch sparse invariant checks throughout each unit test.

    Yields:
        Control while native sparse operators validate their structural contracts.
    """
    with torch.sparse.check_sparse_tensor_invariants():
        yield


@pytest.fixture
def synthetic() -> ad.AnnData:
    """Provide a small fully structured offline slice.

    Returns:
        Reproducible 96-Spot, 80-gene raw AnnData.
    """
    return generate_synthetic_visium(n_spots=96, n_genes=80, seed=42)


def test_synthetic_contract_and_roundtrip(synthetic: ad.AnnData, tmp_path: Path) -> None:
    """Check independent counts, six-cell truth, markers and persisted RGB geometry.

    Args:
        synthetic: Small synthetic fixture.
        tmp_path: Isolated pytest temporary directory.
    """
    assert synthetic.shape == (96, 80)
    assert synthetic.X.dtype == np.float32
    assert (synthetic.X != synthetic.layers["counts"]).nnz == 0
    assert synthetic.X is not synthetic.layers["counts"]
    assert synthetic.obsm["spatial"].shape == (96, 2)
    assert set(synthetic.obs.domain_true) == set(range(5))
    assert synthetic.obsm["cell_proportions_true"].shape == (96, 6)
    assert np.allclose(synthetic.obsm["cell_proportions_true"].sum(1), 1)
    assert {"CD3D", "CD19", "CD68", "EPCAM", "COL1A1", "MKI67"} <= set(synthetic.var_names)
    image = synthetic.uns["spatial_meta"]["hires_image"]
    assert image.shape == (500, 500, 3) and image.dtype == np.uint8
    target = tmp_path / "fixture.h5ad"
    synthetic.write_h5ad(target)
    restored = ad.read_h5ad(target)
    assert np.array_equal(restored.obsm["spatial"], synthetic.obsm["spatial"])
    assert np.array_equal(restored.uns["spatial_meta"]["hires_image"], image)


def test_synthetic_reproducibility_and_shift(synthetic: ad.AnnData) -> None:
    """Ensure seeded reproducibility and a real independent batch perturbation.

    Args:
        synthetic: Seeded reference slice.
    """
    repeat = generate_synthetic_visium(n_spots=96, n_genes=80, seed=42)
    shifted = generate_synthetic_visium(n_spots=96, n_genes=80, batch_id="Batch_2", seed=43)
    assert (synthetic.X != repeat.X).nnz == 0
    assert (synthetic.X != shifted.X).nnz > 0
    assert not np.array_equal(synthetic.obsm["cell_proportions_true"], shifted.obsm["cell_proportions_true"])


def test_moran_ring_bounds_and_permutation() -> None:
    """Verify exact Moran values and [-1,1] bounds on a regular undirected ring."""
    n = 12
    angle = np.arange(n) * 2 * np.pi / n
    x = np.column_stack((2 + np.cos(angle), 2 + np.sin(angle), np.ones(n))).astype(np.float32)
    data = ad.AnnData(x, var=pd.DataFrame(index=["cos", "sin", "constant"]))
    rows = np.repeat(np.arange(n), 2)
    cols = np.column_stack(((np.arange(n) - 1) % n, (np.arange(n) + 1) % n)).ravel()
    weights = sparse.csr_matrix((np.ones(2 * n), (rows, cols)), shape=(n, n))
    data.obsp["spatial_weights"] = weights
    result = compute_morans_i(data, top_k=3, n_perms=19)
    z = x[:, :2].astype(float) - x[:, :2].mean(0)
    expected = n / weights.sum() * np.diag(z.T @ weights @ z) / np.diag(z.T @ z)
    assert len(result) == 2
    assert result.morans_i.between(-1 - 1e-7, 1 + 1e-7).all()
    assert np.allclose(result.morans_i, expected, atol=1e-6)
    assert result.p_value.between(1 / 20, 1).all()
    assert np.all(result.fdr >= result.p_value)
    pd.testing.assert_frame_equal(result, compute_morans_i(data, top_k=3, n_perms=19))


def test_gat_forward_simplex_and_backward() -> None:
    """Check actual multihead encoder dimensions, simplex outputs and gradients."""
    torch.manual_seed(42)
    torch.set_num_threads(1)
    data = ad.AnnData(np.ones((9, 32), dtype=np.float32))
    weights = sparse.eye(9, format="csr") + sparse.diags([np.ones(8), np.ones(8)], [-1, 1], shape=(9, 9))
    data.obsp["spatial_weights"] = weights
    edge, weight = graph_tensors(data, torch.device("cpu"))
    model = SpatialGATAE(32, n_domains=5)
    x = torch.randn(9, 32)
    outputs = model(x, edge, weight)
    assert outputs["z"].shape == (9, 30)
    assert outputs["reconstruction"].shape == (9, 32)
    assert outputs["domain_logits"].shape == (9, 5)
    assert outputs["cell_proportions"].shape == (9, 6)
    assert torch.all(outputs["cell_proportions"] >= 0)
    assert torch.allclose(outputs["cell_proportions"].sum(1), torch.ones(9), atol=1e-6)
    loss = (outputs["reconstruction"] - x).square().mean() + outputs["domain_logits"].square().mean() + outputs["cell_proportions"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("minority_count,expected", [(0, 0.0), (1, 1 / 7), (2, 2 / 7), (3, 3 / 7), (4, 3 / 7)])
def test_pas_exact_threshold(minority_count: int, expected: float) -> None:
    """Check >=4 disagreements using a seven-Spot complete six-neighbor graph.

    Args:
        minority_count: Number assigned to the alternate label.
        expected: Hand-derived abnormal fraction.
    """
    coordinates = np.column_stack((np.arange(7), np.zeros(7))).astype(float)
    labels = [1] * minority_count + [0] * (7 - minority_count)
    assert percentage_abnormal_spots(coordinates, labels) == pytest.approx(expected)


def test_pas_irregular_geometry_and_small_input() -> None:
    """Compare spatial kNN selection to an independent full-distance reference."""
    rng = np.random.default_rng(3)
    xy = rng.normal(size=(25, 2))
    labels = rng.integers(0, 3, 25)
    distances = cdist(xy, xy)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1)[:, :6]
    expected = np.mean((labels[nearest] != labels[:, None]).sum(1) >= 4)
    assert percentage_abnormal_spots(xy, labels) == expected
    with pytest.raises(ValueError):
        percentage_abnormal_spots(xy[:6], labels[:6])


def test_chaos_nearest_distance_sd() -> None:
    """Check population SD and omission of a singleton cluster by hand."""
    xy = np.array([[0, 0], [1, 0], [4, 0], [10, 0]], dtype=float)
    assert chaos_score(xy, [0, 0, 0, 1]) == pytest.approx(np.std([1, 1, 3]))


def test_deconvolution_perfect_and_constant_targets() -> None:
    """Verify perfect predictions and explicit undefined constant correlations."""
    truth = np.random.default_rng(42).dirichlet(np.ones(6), size=20)
    metrics, _ = deconvolution_metrics(truth, truth)
    assert metrics["RMSE"] == 0
    assert metrics["Mean_Pearson_r"] == pytest.approx(1)
    assert metrics["Cosine_similarity"] == pytest.approx(1)
    constant = np.full((10, 6), 1 / 6)
    metrics, _ = deconvolution_metrics(constant, constant)
    assert np.isnan(metrics["Mean_Pearson_r"]) and metrics["Valid_Pearson_types"] == 0


def test_batch_moment_alignment_and_mixing() -> None:
    """Check moment correction, finite ASW and iLISI bounds on two batches."""
    first = np.random.default_rng(42).normal(size=(35, 4))
    features = np.vstack((first, first * 2 + 8))
    labels = np.array(["A"] * 35 + ["B"] * 35)
    corrected = mean_variance_align(features, labels)
    assert np.allclose(corrected[:35].mean(0), corrected[35:].mean(0))
    assert np.allclose(corrected[:35].std(0), corrected[35:].std(0))
    assert np.array_equal(features[:35], first)
    with threadpool_limits(limits=1):
        before = integration_metrics(features, labels, perplexity=10)
        after = integration_metrics(corrected, labels, perplexity=10)
    assert -1 <= after["Batch_ASW"] <= 1
    assert 1 <= after["iLISI"] <= 2 + 1e-10
    assert after["Batch_ASW_mixing"] > before["Batch_ASW_mixing"]
