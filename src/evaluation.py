"""Frozen cross-slice evaluation, spatial diagnostics and unsupervised batch audit."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from src.config_schema import PROJECT_ROOT, load_config
from src.models.baseline import BaselineSpatialModel

MODEL_KEYS = {"Baseline": ("domain_pred_baseline", "cell_proportions_pred_baseline"),
              "Spatial-GAT-AE": ("domain_pred_gat", "cell_proportions_pred_gat")}
CELL_TYPES = ("T_cell", "B_cell", "Macrophage", "NK_cell", "Dendritic", "Tumor_Epi")
COLORS = ["#95BCE5", "#F39DA0", "#A3C9A8", "#C4B5FD", "#FCD34D", "#94A3B8", "#F9A8D4", "#6EE7B7"]


def percentage_abnormal_spots(coordinates: NDArray[np.floating], labels: Sequence[object],
                              n_neighbors: int = 6) -> float:
    """Measure spots with at least four disagreeing neighbors among exactly six.

    Args:
        coordinates: N by 2 finite pixel coordinates.
        labels: N supervised predicted domain labels.
        n_neighbors: Must equal six for this fixed PAS definition.

    Returns:
        Abnormal-spot fraction between zero and one; multiply by 100 for percent.

    Raises:
        ValueError: If fewer than seven spots or invalid labels/coordinates exist.
    """
    xy = np.asarray(coordinates, dtype=np.float64)
    groups = np.asarray(labels)
    if (n_neighbors != 6 or xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 7
            or groups.shape != (len(xy),) or not np.isfinite(xy).all() or pd.isna(groups).any()):
        raise ValueError("PAS requires >=7 finite 2D spots, complete labels and exactly six neighbors")
    _, candidates = cKDTree(xy).query(xy, k=7, workers=1)
    neighbors = np.vstack([row[row != i][:6] for i, row in enumerate(candidates)])
    disagreements = (groups[neighbors] != groups[:, None]).sum(axis=1)
    return float(np.mean(disagreements >= 4))


def chaos_score(coordinates: NDArray[np.floating], labels: Sequence[object]) -> float:
    """Calculate the requested within-cluster nearest-distance SD variant of CHAOS.

    Args:
        coordinates: N by 2 finite pixel coordinates.
        labels: N complete cluster labels.

    Returns:
        Unweighted mean population SD of within-cluster nearest-neighbor distances,
        in pixels. Singleton clusters are omitted; NaN if no eligible cluster exists.

    Raises:
        ValueError: If coordinate or label dimensions are invalid.
    """
    xy = np.asarray(coordinates, dtype=np.float64)
    groups = np.asarray(labels)
    if (xy.ndim != 2 or xy.shape[1] != 2 or groups.shape != (len(xy),)
            or not np.isfinite(xy).all() or pd.isna(groups).any()):
        raise ValueError("CHAOS requires finite 2D coordinates and complete labels")
    deviations: list[float] = []
    for label in pd.unique(groups):
        points = xy[groups == label]
        if len(points) >= 2:
            distances, _ = cKDTree(points).query(points, k=2, workers=1)
            deviations.append(float(np.std(distances[:, 1], ddof=0)))
    return float(np.mean(deviations)) if deviations else float("nan")


def deconvolution_metrics(truth: NDArray[np.floating], prediction: NDArray[np.floating]) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Evaluate RMSE, per-cell-type Pearson correlation and mean spot cosine.

    Args:
        truth: N by 6 true simplex proportions.
        prediction: N by 6 predicted simplex proportions.

    Returns:
        Overall metrics and six per-cell-type metric records. Undefined Pearson
        correlations are excluded from the mean and their number is reported.

    Raises:
        ValueError: If shapes, values or simplex constraints are invalid.
    """
    actual, pred = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if actual.ndim != 2 or actual.shape[1] != 6 or actual.shape != pred.shape or len(actual) < 2:
        raise ValueError("Require matching N by 6 proportions for at least two spots")
    for matrix in (actual, pred):
        if not np.isfinite(matrix).all() or (matrix < 0).any() or not np.allclose(matrix.sum(1), 1, atol=1e-5):
            raise ValueError("Cell proportions must be finite nonnegative simplex rows")
    records: list[dict[str, object]] = []
    correlations: list[float] = []
    for column, cell in enumerate(CELL_TYPES):
        a, b = actual[:, column] - actual[:, column].mean(), pred[:, column] - pred[:, column].mean()
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        correlation = float(a @ b / denominator) if denominator > 1e-14 else float("nan")
        if np.isfinite(correlation):
            correlations.append(correlation)
        records.append({"Cell_type": cell, "RMSE": float(np.sqrt(np.mean((actual[:, column] - pred[:, column]) ** 2))),
                        "Pearson_r": correlation})
    cosine = np.sum(actual * pred, axis=1) / (np.linalg.norm(actual, axis=1) * np.linalg.norm(pred, axis=1))
    return {"RMSE": float(np.sqrt(np.mean((actual - pred) ** 2))),
            "Mean_Pearson_r": float(np.mean(correlations)) if correlations else float("nan"),
            "Cosine_similarity": float(np.clip(cosine, -1, 1).mean()),
            "Valid_Pearson_types": float(len(correlations))}, records


def evaluate_slice(adata: ad.AnnData, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate both stored frozen predictions against a slice's ground truth.

    Args:
        adata: Analyzed, modeled slice with synthetic truth annotations.
        split: Train or Test display name.

    Returns:
        Overall metrics, per-cell metrics and observed off-diagonal confusion counts.

    Raises:
        ValueError: If truth or predictions are missing or labels are incomplete.
    """
    if "domain_true" not in adata.obs or "cell_proportions_true" not in adata.obsm:
        raise ValueError("Supervised evaluation requires domain and cell-mixture truth")
    rows: list[dict[str, object]] = []
    cells: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    truth = adata.obs["domain_true"].astype(str).to_numpy()
    for name, (domain_key, cell_key) in MODEL_KEYS.items():
        if domain_key not in adata.obs or cell_key not in adata.obsm or adata.obs[domain_key].isna().any():
            raise ValueError(f"Missing complete {name} predictions; run Step 3")
        pred = adata.obs[domain_key].astype(str).to_numpy()
        row: dict[str, object] = {"Slice": split, "Model": name, "N_spots": adata.n_obs,
                                 "ARI": adjusted_rand_score(truth, pred), "NMI": normalized_mutual_info_score(truth, pred),
                                 "Macro_F1": f1_score(truth, pred, average="macro", zero_division=0),
                                 "PAS": percentage_abnormal_spots(adata.obsm["spatial"], pred),
                                 "CHAOS_px": chaos_score(adata.obsm["spatial"], pred)}
        metrics, detail = deconvolution_metrics(adata.obsm["cell_proportions_true"], adata.obsm[cell_key])
        row.update(metrics)
        rows.append(row)
        cells.extend({"Slice": split, "Model": name, **entry} for entry in detail)
        wrong = pd.DataFrame({"True_domain": truth, "Predicted_domain": pred})
        counts = wrong.loc[truth != pred].value_counts().reset_index(name="N_errors")
        errors.extend({"Slice": split, "Model": name, **entry} for entry in counts.to_dict("records"))
    return pd.DataFrame(rows), pd.DataFrame(cells), pd.DataFrame(errors)


def mean_variance_align(features: NDArray[np.floating], batches: Sequence[object]) -> NDArray[np.float64]:
    """Align each batch's feature moments to the first observed reference batch.

    Args:
        features: Finite N by G expression features in a shared gene order.
        batches: N batch identifiers, with training batch occurring first.

    Returns:
        Copy aligned using per-batch population means and SDs. This is an
        unsupervised transductive diagnostic, not ComBat or frozen zero-shot inference.

    Raises:
        ValueError: If feature shapes, batches or reference variances are invalid.
    """
    x = np.asarray(features, dtype=np.float64)
    labels = np.asarray(batches)
    if x.ndim != 2 or labels.shape != (len(x),) or not np.isfinite(x).all() or pd.isna(labels).any():
        raise ValueError("Require finite features and one batch label per row")
    unique = pd.unique(labels)
    if len(unique) < 2 or any(np.sum(labels == label) < 2 for label in unique):
        raise ValueError("Require at least two observations in each of two or more batches")
    reference = x[labels == unique[0]]
    mean, deviation = reference.mean(0), reference.std(0)
    aligned = x.copy()
    for label in unique[1:]:
        mask = labels == label
        group = x[mask]
        aligned[mask] = (group - group.mean(0)) / np.maximum(group.std(0), 1e-8) * deviation + mean
    return aligned


def integration_metrics(features: NDArray[np.floating], batches: Sequence[object],
                        perplexity: float = 30.0) -> dict[str, float]:
    """Measure batch silhouettes and Gaussian-perplexity local inverse Simpson diversity.

    Args:
        features: N by D common-basis features, without separately fitted PCA axes.
        batches: At least two batch categories.
        perplexity: Target effective neighborhood size, capped below N minus one.

    Returns:
        Signed Batch_ASW, mean one-minus-absolute silhouette mixing score, and mean
        iLISI. With two batches iLISI ranges from one to two; higher means mixing.

    Raises:
        ValueError: If features, batches or perplexity are invalid.
    """
    x, labels = np.asarray(features, dtype=np.float64), np.asarray(batches)
    if (x.ndim != 2 or len(x) < 4 or labels.shape != (len(x),) or not np.isfinite(x).all()
            or pd.isna(labels).any() or not 1 < len(pd.unique(labels)) < len(x)
            or not np.isfinite(perplexity) or perplexity <= 1):
        raise ValueError("Require finite features, valid batch labels and perplexity >1")
    silhouettes = silhouette_samples(x, labels, metric="euclidean")
    effective = min(perplexity, len(x) - 1.01)
    k = min(len(x) - 1, int(np.ceil(3 * effective)))
    distances, indices = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=1).fit(x).kneighbors(x)
    neighbors = np.vstack([row[row != i][:k] for i, row in enumerate(indices)])
    squared = np.vstack([row[idx != i][:k] ** 2 for i, (row, idx) in enumerate(zip(distances, indices))])
    squared -= squared.min(axis=1, keepdims=True)
    low, high, beta = np.zeros(len(x)), np.full(len(x), np.inf), np.ones(len(x))
    for _ in range(50):
        weights = np.exp(-squared * beta[:, None])
        sums = weights.sum(1)
        entropy = np.log(sums) + beta * (weights * squared).sum(1) / sums
        broad = entropy > np.log(effective)
        low = np.where(broad, beta, low)
        high = np.where(broad, high, beta)
        beta = np.where(broad, np.where(np.isfinite(high), (beta + high) / 2, beta * 2), (beta + low) / 2)
    weights = np.exp(-squared * beta[:, None])
    weights /= weights.sum(1, keepdims=True)
    mass = np.column_stack([(weights * (labels[neighbors] == label)).sum(1) for label in pd.unique(labels)])
    return {"Batch_ASW": float(silhouettes.mean()), "Batch_ASW_mixing": float((1 - np.abs(silhouettes)).mean()),
            "iLISI": float(np.mean(1 / np.square(mass).sum(1)))}


def batch_audit(train: ad.AnnData, test: ad.AnnData, seed: int) -> tuple[pd.DataFrame, list[NDArray[np.float64]], NDArray[np.str_]]:
    """Evaluate expression moment alignment in one common PCA coordinate system.

    Args:
        train: Training slice.
        test: Independent test slice.
        seed: PCA random state.

    Returns:
        Batch metrics, before/after projections and batch identities. Domain ASW
        and within-domain batch silhouettes are retained as biology controls.

    Raises:
        ValueError: If fewer than 31 common unique genes exist.
    """
    genes = train.var_names.intersection(test.var_names, sort=False)
    if len(genes) < 31 or not train.var_names.is_unique or not test.var_names.is_unique:
        raise ValueError("Batch audit needs at least 31 shared unique genes")
    arrays: list[NDArray[np.float64]] = []
    for data in (train, test):
        matrix = data[:, genes].X
        arrays.append(np.asarray(matrix.toarray() if sparse.issparse(matrix) else matrix, dtype=np.float64))
    expression = np.vstack(arrays)
    batches = np.asarray(["Batch_1"] * train.n_obs + ["Batch_2"] * test.n_obs)
    domains = np.concatenate([data.obs["domain_true"].astype(str).to_numpy() for data in (train, test)])
    corrected = mean_variance_align(expression, batches)
    pca = PCA(n_components=30, random_state=seed, svd_solver="full").fit(expression)
    projections = [pca.transform(value) for value in (expression, corrected)]
    records: list[dict[str, object]] = []
    for method, features in zip(("Uncorrected", "Mean-variance aligned"), projections):
        metrics: dict[str, object] = integration_metrics(features, batches)
        conditional: list[float] = []
        weights: list[int] = []
        for label in pd.unique(domains):
            mask = domains == label
            if len(pd.unique(batches[mask])) > 1 and mask.sum() > 3:
                conditional.append(float(silhouette_samples(features[mask], batches[mask]).mean()))
                weights.append(int(mask.sum()))
        metrics.update({"Method": method, "Domain_ASW": float(silhouette_samples(features, domains).mean()),
                        "Within_domain_Batch_ASW": float(np.average(conditional, weights=weights)),
                        "N_common_genes": len(genes)})
        records.append(metrics)
    return pd.DataFrame(records), projections, batches


def performance_retention(metrics: pd.DataFrame) -> pd.DataFrame:
    """Quantify frozen test performance relative to the training slice.

    Args:
        metrics: Overall Train/Test metric table for both frozen models.

    Returns:
        Test/train percentages for higher-is-better metrics, and train/test for RMSE.
        Ratios with nonpositive denominators or nonpositive training scores are NaN.
        Values above 100 are retained without capping.
    """
    records: list[dict[str, object]] = []
    for model in MODEL_KEYS:
        group = metrics[metrics.Model == model].set_index("Slice")
        for metric in ("ARI", "NMI", "Macro_F1", "RMSE", "Mean_Pearson_r", "Cosine_similarity"):
            train, test = float(group.loc["Train", metric]), float(group.loc["Test", metric])
            numerator, denominator = (train, test) if metric == "RMSE" else (test, train)
            ratio = 100 * numerator / denominator if train > 0 and denominator > 0 and np.isfinite([train, test]).all() else float("nan")
            records.append({"Model": model, "Metric": metric, "Train": train, "Test": test, "Retention_pct": ratio})
    return pd.DataFrame(records)


def require_matplotlib_panel_alignment(figure: Figure, destination: Path) -> None:
    """Record and check the rendered single-panel or equal-width paired layout.

    Args:
        figure: Final Matplotlib figure.
        destination: Output stem for the layout manifest.

    Raises:
        ValueError: If paired panels differ in height or width, or layout is unsupported.
    """
    figure.canvas.draw()
    width, height = figure.get_size_inches() * 72
    panels = []
    for index, axis in enumerate(figure.axes):
        rect = axis.get_position()
        panels.append({"id": chr(97 + index), "bbox_pt": [rect.x0 * width, rect.y0 * height, rect.x1 * width, rect.y1 * height]})
    if len(panels) not in (1, 2):
        raise ValueError("This export contract supports one panel or two comparable horizontal panels")
    if len(panels) == 2:
        a, b = [np.asarray(panel["bbox_pt"]) for panel in panels]
        if max(abs((a[2] - a[0]) - (b[2] - b[0])), abs(a[1] - b[1]), abs(a[3] - b[3])) > 1.5:
            raise ValueError("Comparable panel plot rectangles are not aligned within 1.5 pt")
    layout = {"schema_version": 1, "backend": "matplotlib", "figure": {"width_pt": width, "height_pt": height}, "panels": panels}
    destination.with_suffix(".layout.json").write_text(json.dumps(layout, indent=2), encoding="utf-8")


def save_figure(figure: Figure, destination: Path) -> None:
    """Export a fixed-size 320 dpi PNG and editable PDF/SVG after geometry checks.

    Args:
        figure: Final Matplotlib figure.
        destination: Output stem without extension.
    """
    require_matplotlib_panel_alignment(figure, destination)
    figure.savefig(destination.with_suffix(".png"), dpi=320, facecolor="white")
    figure.savefig(destination.with_suffix(".pdf"), facecolor="white")
    figure.savefig(destination.with_suffix(".svg"), facecolor="white")
    svg_path = destination.with_suffix(".svg")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8", newline="\n")
    plt.close(figure)


def make_figures(metrics: pd.DataFrame, batch: pd.DataFrame, projections: list[NDArray[np.float64]],
                 batches: NDArray[np.str_], directory: Path) -> None:
    """Render three complementary evidence figures without invented uncertainty.

    Args:
        metrics: Actual frozen model scores.
        batch: Actual before/after integration metrics.
        projections: Common-basis uncorrected and aligned 30D projections.
        batches: Batch identity vector.
        directory: Output directory for the three figures and source tables.
    """
    sns.set_theme(style="white", palette=COLORS)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False, "text.color": "#0F172A"})
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    figure.subplots_adjust(left=.12, right=.97, bottom=.19, top=.75)
    positions = np.arange(3)
    for index, row in metrics.reset_index(drop=True).iterrows():
        values = [row[key] for key in ("ARI", "NMI", "Macro_F1")]
        bars = axis.bar(positions + (index - 1.5) * .18, values, width=.17,
                        color=COLORS[index], edgecolor="#475569", linewidth=.35,
                        label=f"{row.Model} / {row.Slice}")
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=6)
    axis.set(xticks=positions, xticklabels=["ARI", "NMI", "Macro-F1"], ylim=(min(-.05, float(metrics.ARI.min()) - .08), 1.12),
             ylabel="Classification score", title="Stored frozen pipeline predictions")
    figure.legend(*axis.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.52, .98), ncol=2, frameon=False)
    save_figure(figure, directory / "classification_comparison")
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    figure.subplots_adjust(left=.14, right=.70, bottom=.20, top=.86)
    points = metrics.copy()
    points["PAS_percent"] = points.PAS * 100
    sns.scatterplot(data=points, x="PAS_percent", y="CHAOS_px", hue="Model", style="Slice", s=85,
                    palette={"Baseline": COLORS[0], "Spatial-GAT-AE": COLORS[1]}, ax=axis)
    axis.set(xlabel="Abnormal spots, PAS (%)", ylabel="Within-domain nearest-distance SD (px)", title="Spatial fragmentation and distance variability")
    axis.set_xscale("symlog", linthresh=1)
    axis.set_yscale("symlog", linthresh=.03)
    axis.set_xlim(0, max(1, float(points.PAS_percent.max()) * 1.3))
    axis.set_ylim(0, max(.05, float(points.CHAOS_px.max()) * 1.8))
    axis.set_xlabel("PAS (%, symmetric-log scale; linear below 1%)")
    axis.set_ylabel("Nearest-distance SD (px, symmetric-log scale)")
    axis.legend(bbox_to_anchor=(1.03, 1), loc="upper left", frameon=False)
    save_figure(figure, directory / "spatial_continuity")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharex=True, sharey=True)
    figure.subplots_adjust(left=.11, right=.97, bottom=.25, top=.76, wspace=.28)
    joined = np.vstack([value[:, :2] for value in projections])
    low, high = joined.min(0), joined.max(0)
    margin = np.maximum((high - low) * .06, .1)
    for index, (axis, values) in enumerate(zip(axes, projections)):
        for group, color in zip(pd.unique(batches), COLORS[:2]):
            mask = batches == group
            axis.scatter(values[mask, 0], values[mask, 1], s=3, alpha=.38, c=color, label=group, rasterized=True)
        row = batch.iloc[index]
        axis.set_title(f"{row.Method}\nASW {row.Batch_ASW:.3f}; iLISI {row.iLISI:.3f}", fontsize=8, pad=10)
        axis.set(xlabel="Common PC1", xlim=(low[0] - margin[0], high[0] + margin[0]), ylim=(low[1] - margin[1], high[1] + margin[1]))
    axes[0].set_ylabel("Common PC2")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .01), ncol=2, markerscale=2.5, frameon=False)
    save_figure(figure, directory / "batch_alignment")


def markdown_table(table: pd.DataFrame) -> str:
    """Render Markdown without adding a tabulate runtime dependency.

    Args:
        table: Scalar-valued table.

    Returns:
        Markdown table with four-decimal numeric precision and NA for undefined values.
    """
    lines = ["| " + " | ".join(map(str, table.columns)) + " |", "| " + " | ".join(["---"] * len(table.columns)) + " |"]
    for row in table.itertuples(index=False, name=None):
        values = ["NA" if isinstance(value, (float, np.floating)) and not np.isfinite(value)
                  else f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value).replace("|", "/") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    """Evaluate frozen outputs, export source tables, figures and a grounded report.

    Raises:
        ValueError: If supervision, feature alignment or provenance is incomplete.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    started = perf_counter()
    directory = PROJECT_ROOT / "docs/figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths = [cfg.paths.processed_dir / f"slice_{name}.h5ad" for name in ("train", "test")]
    train, test = [ad.read_h5ad(path) for path in paths]
    if set(train.obs_names) & set(test.obs_names):
        raise ValueError("Train and test Spot identifiers overlap")
    for data in (train, test):
        if data.uns.get("model_training", {}).get("training_slice") != "slice_train":
            raise ValueError("Frozen prediction provenance must identify slice_train as training source")
    if ("simulation" in train.uns and "simulation" in test.uns
            and train.uns["simulation"]["seed"] == test.uns["simulation"]["seed"]):
        raise ValueError("Independent synthetic test slices require different sampling seeds")
    results = [evaluate_slice(data, split) for data, split in ((train, "Train"), (test, "Test"))]
    metrics, cells, errors = [pd.concat([entry[index] for entry in results], ignore_index=True) for index in range(3)]
    with threadpool_limits(limits=1):
        batch, projections, batches = batch_audit(train, test, cfg.seed)
    pd.DataFrame({"Spot_ID": list(train.obs_names) + list(test.obs_names), "Batch": batches,
                  "PC1_uncorrected": projections[0][:, 0], "PC2_uncorrected": projections[0][:, 1],
                  "PC1_aligned": projections[1][:, 0], "PC2_aligned": projections[1][:, 1]}).to_csv(directory / "batch_projection.csv", index=False)
    retention = performance_retention(metrics)
    baseline = BaselineSpatialModel.load(cfg.paths.processed_dir / "models/baseline.joblib")
    # Diagnose independently fitted PCA axes using the frozen training PCs; never refit the model.
    genes = train.var_names.tolist()
    reference = train.X.toarray() if sparse.issparse(train.X) else np.asarray(train.X)
    aligned = test[:, genes].copy()
    expression = aligned.X.toarray() if sparse.issparse(aligned.X) else np.asarray(aligned.X)
    aligned.obsm["X_pca"] = ((expression - reference.mean(0)) @ np.asarray(train.varm["PCs"])).astype(np.float32)
    baseline.predict(aligned)
    aligned_metrics = evaluate_slice(aligned, "Test_shared_train_PCA")[0]
    aligned_metrics = aligned_metrics[aligned_metrics.Model == "Baseline"].reset_index(drop=True)
    for name, table in (("model_metrics", metrics), ("cell_metrics", cells), ("confusion_errors", errors),
                        ("batch_metrics", batch), ("zero_shot_retention", retention), ("shared_pca_baseline", aligned_metrics)):
        table.to_csv(directory / f"{name}.csv", index=False)
    make_figures(metrics, batch, projections, batches, directory)
    test_rows = metrics[metrics.Slice == "Test"].set_index("Model")
    diagnostics = []
    for model in MODEL_KEYS:
        row = test_rows.loc[model]
        diagnostics.append(f"- {model}：独立测试 ARI={row.ARI:.4f}，Macro-F1={row.Macro_F1:.4f}，RMSE={row.RMSE:.4f}；PAS={row.PAS * 100:.2f}%。")
    fingerprints = "\n".join(f"- `{path.name}` SHA256：`{hashlib.sha256(path.read_bytes()).hexdigest()}`" for path in paths)
    report = ["# SpatialCytoScope 科学评估报告", "## 数据与评估设计",
              f"随机种子 {cfg.seed}。训练切片 {train.n_obs} Spots；测试切片 {test.n_obs} Spots。两个合成切片来自不同随机种子；Batch_2 额外含乘性偏移与背景噪声。",
              "两切片共享固定六边形网格与几何结构域模板。独立性体现在计数、细胞比例采样和批次扰动；本实验没有验证新组织几何或真实患者切片的泛化。",
              "本报告读取 Step 3 冻结预测，不训练或微调任何模型。训练分数为拟合内分数；测试分数来自独立切片的零样本预测。只有一对合成切片，不将 Spot 当作独立生物学重复，不报告跨切片置信区间或显著性。",
              "独立性核验通过：训练与测试 Spot ID 不重叠、合成采样种子不同，两个切片的模型来源均记录为 slice_train。元数据核验提供可复查记录，并非对训练过程无泄漏的形式化证明。",
              fingerprints, "## 分类、解卷积与空间连续性（原管线冻结输出）", markdown_table(metrics),
              "Baseline 原管线的测试 PCA 与训练 PCA 分别拟合，坐标系不同。这张主表保留真实已存储输出；跨模型迁移优劣应同时查看下文的共享训练 PCA 对照。",
              "Macro-F1 直接比较监督类别 ID，不事后重排标签。RMSE 汇总所有 Spot×6 类细胞；Pearson r 先在各细胞类型内跨 Spot 计算，再对有效类型等权平均；常量列相关性记为 NA 并排除。Cosine 为各 Spot 的六维向量余弦均值。",
              "PAS 使用原始像素坐标的恰好 6 个近邻：至少 4 个标签不同即异常。表中为 0–1 比例。该定义与 Step 4 UI 使用对称 union-kNN 全部邻居的 PAS 不同，不能直接混用。",
              "CHAOS_px 按本项目指定定义，取每个簇内最近邻距离的总体标准差，再对至少 2 个 Spot 的簇等权平均；单例簇不计。它是最近邻距离变异指标，不是标准 CHAOS 的平均最近邻距离定义。均匀但分散的簇也可得到低值，应结合 PAS 与分型指标解读。",
              "## 分细胞类型解卷积", markdown_table(cells), "## 跨批次特征分布", markdown_table(batch),
              "先按共享基因顺序合并 log1p 标准化表达，拟合一个共同 30 维 PCA；校正前后均投影到这套固定基底。没有直接拼接两个独立 PCA 坐标系。",
              "校正采用均值方差对齐：以 Batch_1 为参考，将 Batch_2 各基因的均值和总体标准差对齐。该方法不是 ComBat。估计 Batch_2 矩使用其全部无标签特征，因此属于转导式批次诊断，不是冻结模型的零样本校正收益。",
              "Batch_ASW 为批次标签的平均有符号 silhouette；Batch_ASW_mixing 为 mean(1−|silhouette|)，越高表示混合更充分。iLISI 使用欧氏近邻、Gaussian 熵匹配（perplexity=30，最多 90 个邻居，不含自身）、批次概率的逆 Simpson 指数均值，两批次范围 [1,2]。非官方整合基准软件输出，未做额外归一化。",
              "Domain_ASW 检查生物结构域分离度；Within_domain_Batch_ASW 是各真实结构域内批次 silhouette 按 Spot 数加权的均值。混合指标改善并不自动代表生物信号保持。",
              "## 零样本性能保持率", markdown_table(retention),
              "ARI/NMI/Macro-F1/Pearson/Cosine 使用 Test÷Train×100%；RMSE 使用 Train÷Test×100%，使越高均表示更好的保持。非正训练分数或无效分母记 NA；不截断超过 100% 的比率。这里只量化冻结模型，不把均值方差对齐后的诊断特征用于重新训练。",
              "## 误差分析", *diagnostics,
              f"测试切片 Spatial-GAT 相对原 Baseline：ARI 差值 {test_rows.loc['Spatial-GAT-AE', 'ARI'] - test_rows.loc['Baseline', 'ARI']:+.4f}，Macro-F1 差值 {test_rows.loc['Spatial-GAT-AE', 'Macro_F1'] - test_rows.loc['Baseline', 'Macro_F1']:+.4f}，RMSE 差值 {test_rows.loc['Spatial-GAT-AE', 'RMSE'] - test_rows.loc['Baseline', 'RMSE']:+.4f}（RMSE 越低越好）。空间连续性与监督任务精度应分别判断。",
              "### Baseline 的 PCA 坐标系诊断", markdown_table(aligned_metrics),
              f"原始测试 Baseline ARI={test_rows.loc['Baseline', 'ARI']:.4f}；测试表达改投影到冻结训练 PCA 后，ARI={aligned_metrics.iloc[0].ARI:.4f}。该附加分析不修改原 H5AD，不重新拟合 RandomForest/Ridge，也不使用测试标签进行映射或调参；它区分特征坐标漂移与模型自身迁移误差。主表保留原管线的实际输出。",
              "### 最多见的跨域误判", markdown_table(errors.sort_values("N_errors", ascending=False).groupby(["Slice", "Model"], sort=False).head(3)),
              "真实测试标签仅用于最终打分和误差分组。结构边界 Spot 的细胞混合、组织区域重叠及 Batch_2 特征偏移共同影响跨切片预测；当前设计没有独立改变每个因素，不能单独归因。",
              "## 图表与复现", "所有图由实际 CSV 表和共同 PCA 特征生成；无模拟误差条。PNG 为 320 dpi，PDF 保留可编辑文字。",
              "![分类对比](docs/figures/classification_comparison.png)", "![空间连续性](docs/figures/spatial_continuity.png)",
              "![批次分布](docs/figures/batch_alignment.png)",
              "项目根目录执行 `python -m src.evaluation`，再执行 `python -m pytest -q tests/test_pipeline.py`。源表位于 docs/figures/*.csv。"]
    (PROJECT_ROOT / "EVALUATION_REPORT.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print(batch.to_string(index=False))
    print(f"Report: {PROJECT_ROOT / 'EVALUATION_REPORT.md'}; elapsed={perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
