"""Offline synthetic and online Visium raw-data ingestion.

Raw ingestion deliberately keeps X as float32 counts. Normalization and learned
embeddings belong to preprocessing; no estimated quantity is presented as truth.
"""

import argparse
from pathlib import Path
from time import perf_counter

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.ndimage import gaussian_filter

from src.config_schema import PROJECT_ROOT, detect_device, load_config

CELL_TYPES = ("T_cell", "B_cell", "Macrophage", "NK_cell", "Dendritic", "Tumor_Epi")
DOMAINS = ("Tumor_Core", "Invasive_Margin", "Immune_TLS", "Stroma", "Normal_Tissue")
MT_GENES = (
    "MT-CO1", "MT-CO2", "MT-CO3", "MT-ND1", "MT-ND2", "MT-ND3", "MT-ND4",
    "MT-ND4L", "MT-ND5", "MT-ND6", "MT-ATP6", "MT-ATP8", "MT-CYB", "MT-RNR1",
    "MT-RNR2", "MT-TF", "MT-TV", "MT-TL1", "MT-TI", "MT-TQ",
)
MARKERS = ("CD3D", "CD19", "CD68", "EPCAM", "COL1A1", "MKI67", "NKG7", "FCER1A")


def _domains(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.int64]:
    """Assign geometric regions shared by spots and histology pixels.

    Args:
        x: Horizontal pixel coordinates.
        y: Vertical pixel coordinates.

    Returns:
        Domain IDs from zero to four.
    """
    radius = np.sqrt(((x - 210) / 1.05) ** 2 + ((y - 250) / 0.92) ** 2)
    labels = np.full(x.shape, 4, dtype=np.int64)
    labels[(x < 315) & (y > 120)] = 3
    labels[radius < 175] = 1
    labels[radius < 108] = 0
    labels[((x - 370) / 65) ** 2 + ((y - 155) / 80) ** 2 < 1] = 2
    return labels


def _histology(rng: np.random.Generator) -> NDArray[np.uint8]:
    """Render domain-aware eosin texture and elliptical hematoxylin nuclei.

    Args:
        rng: Independent seeded image random number generator.

    Returns:
        A 500 by 500 RGB uint8 image.
    """
    yy, xx = np.mgrid[:500, :500].astype(np.float64)
    regions = _domains(xx, yy)
    palette = np.array([[234, 184, 204], [240, 193, 209], [225, 196, 220],
                        [245, 201, 211], [247, 217, 224]], dtype=np.float64)
    texture = gaussian_filter(rng.normal(size=(500, 500)), 2.0)
    texture /= max(float(texture.std()), 1e-6)
    fibers = np.sin(xx * 0.12 + yy * 0.04 + texture) * (regions == 3)
    rgb = palette[regions] + (texture * 2.8 + fibers * 4)[..., None]
    nuclei = np.zeros((500, 500), dtype=np.float64)
    density = np.array([0.035, 0.025, 0.070, 0.012, 0.018])
    centers = rng.random((500, 500)) < density[regions]
    for cy, cx in np.argwhere(centers):
        rx, ry = rng.uniform(1.0, 2.4, size=2)
        y0, y1 = max(0, cy - 4), min(500, cy + 5)
        x0, x1 = max(0, cx - 4), min(500, cx + 5)
        patch = np.exp(-(((xx[y0:y1, x0:x1] - cx) / rx) ** 2
                         + ((yy[y0:y1, x0:x1] - cy) / ry) ** 2) * 1.6)
        nuclei[y0:y1, x0:x1] = np.maximum(nuclei[y0:y1, x0:x1], patch)
    opacity = nuclei[..., None] * 0.82
    rgb = rgb * (1 - opacity) + np.array([79, 57, 125]) * opacity
    return np.clip(rgb, 0, 255).astype(np.uint8)


def generate_synthetic_visium(
    n_spots: int = 1200, n_genes: int = 800, n_domains: int = 5,
    batch_id: str = "Batch_1", seed: int = 42,
) -> ad.AnnData:
    """Generate reproducible structured raw counts with known cell mixtures.

    Args:
        n_spots: Number of hexagonal lattice locations, at least 50.
        n_genes: Number of genes, at least 28 to include required markers.
        n_domains: Exactly five, matching the fixed tissue ontology.
        batch_id: Batch_1 for training or Batch_2 for shifted evaluation.
        seed: Nonnegative NumPy random seed.

    Returns:
        Raw AnnData with counts, geometry, truth labels and synthetic histology.

    Raises:
        ValueError: If dimensions, batch, seed or ontology are invalid.
    """
    if (not all(isinstance(v, int) and not isinstance(v, bool)
                for v in (n_spots, n_genes, n_domains, seed))
            or n_spots < 50 or n_genes < 28 or n_domains != 5 or seed < 0
            or batch_id not in {"Batch_1", "Batch_2"}):
        raise ValueError("Require n_spots>=50, n_genes>=28, n_domains=5, seed>=0 and a known batch")
    sample_seed, basis_seed, image_seed = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(sample_seed)
    # The biological basis is fixed across seeds/batches; held-out spots remain independent.
    basis_rng = np.random.default_rng(314159)
    image_rng = np.random.default_rng(image_seed)
    n_cols = int(np.ceil(np.sqrt(n_spots * np.sqrt(3) / 2)))
    row, col = np.divmod(np.arange(n_spots), n_cols)
    xy = np.column_stack((col + (row % 2) * 0.5, row * np.sqrt(3) / 2))
    xy = 12 + 476 * xy / xy.max(axis=0)
    labels = _domains(xy[:, 0], xy[:, 1])
    alpha = np.array([[1, .4, 1.5, .4, .4, 24], [7, 2, 8, 3, 2, 12],
                      [16, 18, 3, 4, 5, 1], [2, 1, 5, 1, 2, 3], [2, 2, 2, 2, 2, 14]])
    proportions = np.vstack([rng.dirichlet(alpha[d]) for d in labels]).astype(np.float32)
    basis = basis_rng.gamma(1.8, 1.1, size=(6, n_genes))
    for cell, genes in enumerate(np.array_split(np.arange(28, n_genes), 6)):
        basis[cell, genes] *= 6
    basis[:, :20] *= 0.24
    for gene, cell in [(20, 0), (21, 1), (22, 2), (23, 5), (26, 3), (27, 4)]:
        basis[:, gene] = 0.15
        basis[cell, gene] = 32
    mean = proportions @ basis
    mean[:, 24] += np.where(labels == 3, 28, 1)
    mean[:, 25] += np.where(labels == 0, 22, 2)
    mean *= rng.lognormal(0, .22, size=(n_spots, 1))
    if batch_id == "Batch_2":
        shift_rng = np.random.default_rng(basis_seed)
        mean *= shift_rng.lognormal(.18, .3, size=(1, n_genes))
        mean += rng.gamma(1.2, .7, size=mean.shape)
    counts = rng.negative_binomial(12.0, 12.0 / (12.0 + mean)).astype(np.float32)
    genes = list(MT_GENES) + list(MARKERS) + [f"GENE_{i:04d}" for i in range(28, n_genes)]
    obs = pd.DataFrame(index=[f"{batch_id}_{i:05d}" for i in range(n_spots)])
    obs["batch"] = pd.Categorical([batch_id] * n_spots)
    obs["domain_true"] = pd.Categorical(labels, categories=range(5))
    obs["domain_name"] = pd.Categorical([DOMAINS[d] for d in labels], categories=DOMAINS)
    obs["array_row"], obs["array_col"] = row, col
    result = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs,
                        var=pd.DataFrame({"mt": [g.startswith("MT-") for g in genes]}, index=genes))
    result.layers["counts"] = result.X.copy()
    result.obsm["spatial"] = xy.astype(np.float32)
    result.obsm["cell_proportions_true"] = proportions
    result.uns["cell_type_names"] = list(CELL_TYPES)
    result.uns["domain_names"] = list(DOMAINS)
    result.uns["spatial_meta"] = {"hires_image": _histology(image_rng), "scalefactor": 1.0}
    result.uns["data_state"] = "raw_counts_not_normalized"
    result.uns["simulation"] = {"seed": seed, "batch_id": batch_id, "n_domains": 5,
                                "biological_basis_seed": 314159}
    return result


def load_real_visium(sample_id: str, save_dir: Path) -> ad.AnnData:
    """Download a Scanpy Visium sample and derive explicit proxy labels.

    Args:
        sample_id: Sample identifier accepted by scanpy.datasets.visium_sge.
        save_dir: Directory for download caching and the resulting H5AD file.

    Returns:
        Raw Visium AnnData with marker-derived reference labels, never truth.

    Raises:
        ValueError: If sample ID, image, spatial coordinates or markers are absent.
    """
    import scanpy as sc

    if not sample_id or Path(sample_id).name != sample_id or "/" in sample_id or "\\" in sample_id:
        raise ValueError("sample_id must be a nonempty sample name without path separators")
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    result = sc.datasets.visium_sge(sample_id=sample_id, path=save_dir / sample_id, include_hires_tiff=False)
    result.var_names_make_unique()
    result.X = result.X.astype(np.float32)
    result.layers["counts"] = result.X.copy()
    result.var["mt"] = result.var_names.str.upper().str.startswith("MT-")
    if "spatial" not in result.obsm or "spatial" not in result.uns:
        raise ValueError("Visium sample has no spatial metadata")
    library = next(iter(result.uns["spatial"].values()))
    if "hires" not in library.get("images", {}):
        raise ValueError("Visium sample has no hires image")
    image = np.asarray(library["images"]["hires"])[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image * (255 if image.max() <= 1 else 1), 0, 255).astype(np.uint8)
    result.uns["spatial_meta"] = {"hires_image": image,
                                  "scalefactor": float(library["scalefactors"]["tissue_hires_scalef"])}
    panels = (("CD3D", "CD3E", "TRAC"), ("CD19", "MS4A1", "CD79A"),
              ("CD68", "LST1", "C1QA"), ("NKG7", "GNLY", "KLRD1"),
              ("FCER1A", "CD1C", "CLEC10A"), ("EPCAM", "KRT8", "KRT18"))
    normalized = result.copy()
    sc.pp.normalize_total(normalized, target_sum=1e4)
    sc.pp.log1p(normalized)
    scores = np.zeros((result.n_obs, 6), dtype=np.float32)
    coverage: dict[str, list[str]] = {}
    for i, panel in enumerate(panels):
        present = [gene for gene in panel if gene in normalized.var_names]
        coverage[CELL_TYPES[i]] = present
        if present:
            scores[:, i] = np.asarray(normalized[:, present].X.mean(axis=1)).ravel()
    if not any(coverage.values()):
        raise ValueError("No reference marker genes found in sample")
    reference = np.array(CELL_TYPES)[scores.argmax(axis=1)].astype(object)
    reference[scores.max(axis=1) <= 0] = "Unassigned"
    result.obs["cell_type_reference"] = pd.Categorical(reference)
    result.obsm["cell_type_reference_scores"] = scores
    result.uns["reference_labels"] = {"method": "highest mean log1p library-normalized marker expression",
                                      "is_ground_truth": False, "marker_coverage": coverage}
    result.uns["cell_type_names"] = list(CELL_TYPES)
    result.uns["data_state"] = "raw_counts_not_normalized"
    result.write_h5ad(save_dir / f"{sample_id}.h5ad", compression="gzip")
    return result


def main() -> None:
    """Run the selected ingestion mode and persist H5AD files.

    Raises:
        SystemExit: If real mode has no sample identifier.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/config.yaml")
    parser.add_argument("--sample-id")
    args = parser.parse_args()
    if args.mode == "real" and not args.sample_id:
        parser.error("--sample-id is required for real mode")
    config = load_config(args.config)
    started = perf_counter()
    config.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "synthetic":
        for split, batch, seed in (("train", "Batch_1", config.seed),
                                   ("test", "Batch_2", config.seed + 1)):
            result = generate_synthetic_visium(batch_id=batch, seed=seed)
            target = config.paths.raw_dir / f"slice_{split}.h5ad"
            result.write_h5ad(target, compression="gzip")
            print(f"Saved {target}: {result.n_obs} spots x {result.n_vars} genes")
    else:
        result = load_real_visium(args.sample_id, config.paths.raw_dir)
        print(f"Saved {args.sample_id}: {result.n_obs} spots x {result.n_vars} genes")
    print(f"Device: {detect_device()}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
