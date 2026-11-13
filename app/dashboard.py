"""Standalone Streamlit spatial workbench with optional FastAPI analytics."""

import base64
import html
import io
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import anndata as ad
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from scipy import sparse

from app.api import CELL_TYPES, DATA_DIR, MODEL_KEYS, ROIRequest, analyze_roi, slice_summary, svg_records

PALETTE = ["#95BCE5", "#F39DA0", "#A3C9A8", "#C4B5FD", "#FCD34D", "#94A3B8", "#F9A8D4", "#6EE7B7"]
CONTINUOUS = [[0, "#F1F5F9"], [0.5, "#95BCE5"], [1, "#F39DA0"]]


@st.cache_resource(show_spinner="正在读取切片…")
def local_slice(path: str, modified_ns: int) -> ad.AnnData:
    """Cache a local slice until its modification timestamp changes.

    Args:
        path: Absolute H5AD filename.
        modified_ns: File modification timestamp used as a cache key.

    Returns:
        Shared read-only AnnData snapshot.
    """
    return ad.read_h5ad(path)


@st.cache_data(show_spinner=False)
def local_summary(name: str, modified_ns: int, _data: ad.AnnData) -> dict[str, Any]:
    """Cache measured summary using the current slice revision.

    Args:
        name: Slice identifier.
        modified_ns: File revision cache key.
        _data: Read-only slice, excluded from expensive hashing.

    Returns:
        Shared analytic summary.
    """
    return slice_summary(_data, name)


@st.cache_data(show_spinner=False, max_entries=64)
def local_roi(name: str, modified_ns: int, indices: tuple[int, ...], model: str,
              _data: ad.AnnData) -> dict[str, Any]:
    """Cache region statistics across visual-only slider changes.

    Args:
        name: Slice name used as cache key.
        modified_ns: Slice revision.
        indices: Unique selected row positions.
        model: Prediction model key.
        _data: Read-only expression and predictions.

    Returns:
        Differential expression and cell composition summary.
    """
    return analyze_roi(_data, ROIRequest(spot_indices=list(indices), model_type=model))


def api_json(base_url: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call an optional API from the Streamlit server.

    Args:
        base_url: HTTP service address.
        endpoint: API path.
        payload: JSON POST body, or None for a GET.

    Returns:
        Decoded JSON response.

    Raises:
        URLError: If the API cannot be reached within five seconds.
        HTTPError: If the server rejects the request.
    """
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(base_url.rstrip("/") + endpoint, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.load(response)


@st.cache_data(show_spinner=False, max_entries=4)
def image_source(image_bytes: bytes, shape: tuple[int, ...]) -> str:
    """Encode the actual tissue image as an inline PNG.

    Args:
        image_bytes: Raw uint8 RGB pixel bytes.
        shape: Height by width by three pixel dimensions.

    Returns:
        Base64 data URL for Plotly layout.images.
    """
    buffer = io.BytesIO()
    Image.fromarray(np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def chart_style(figure: go.Figure, height: int = 240) -> go.Figure:
    """Apply consistent quiet chart typography and surfaces.

    Args:
        figure: Plotly figure to style.
        height: Chart height in CSS pixels.

    Returns:
        Styled figure.
    """
    figure.update_layout(template="plotly_white", height=height, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
                         font={"family": "Segoe UI, Microsoft YaHei, sans-serif", "size": 12, "color": "#0F172A"},
                         margin={"l": 12, "r": 12, "t": 12, "b": 20},
                         legend={"orientation": "h", "y": -0.07, "x": 0, "font": {"size": 11}})
    return figure


def domain_title(label: str, data: ad.AnnData) -> str:
    """Resolve numeric training class IDs to tissue names when available.

    Args:
        label: Predicted domain label.
        data: Slice with the optional domain_names vocabulary.

    Returns:
        Human-readable domain label.
    """
    names = list(data.uns.get("domain_names", []))
    if label.isdigit() and int(label) < len(names):
        return f"{label} · {names[int(label)]}"
    return label


def spatial_figure(data: ad.AnnData, labels: np.ndarray, values: np.ndarray | None,
                   he_opacity: float, spot_opacity: float, radius: int) -> go.Figure:
    """Create registered histology and selectable WebGL spot layers.

    Args:
        data: Slice holding image and full-resolution spatial coordinates.
        labels: Model domain labels for hover text and categorical coloring.
        values: Optional continuous expression or cell abundance values.
        he_opacity: Background image opacity.
        spot_opacity: Scatter opacity.
        radius: Spot radius in CSS pixels; Plotly receives twice this size.

    Returns:
        Equal-aspect spatial figure with pixel-aligned reversed Y axis.
    """
    meta = data.uns["spatial_meta"]
    image = np.asarray(meta["hires_image"], dtype=np.uint8)
    coords = np.asarray(data.obsm["spatial"], dtype=float) * float(meta["scalefactor"])
    figure = go.Figure()
    if values is None:
        for index, label in enumerate(sorted(set(labels), key=str)):
            selected = np.flatnonzero(labels == label)
            figure.add_trace(go.Scattergl(x=coords[selected, 0], y=coords[selected, 1], mode="markers", name=domain_title(str(label), data),
                             customdata=np.column_stack((selected, labels[selected])),
                             marker={"color": PALETTE[index % len(PALETTE)], "size": radius * 2, "opacity": spot_opacity},
                             hovertemplate="Spot %{customdata[0]} · %{customdata[1]}<extra></extra>"))
    else:
        figure.add_trace(go.Scattergl(x=coords[:, 0], y=coords[:, 1], mode="markers", name="Spots",
                         customdata=np.column_stack((np.arange(data.n_obs), labels)),
                         marker={"color": values, "colorscale": CONTINUOUS, "size": radius * 2,
                                 "opacity": spot_opacity, "colorbar": {"thickness": 9, "len": 0.7}},
                         hovertemplate="Spot %{customdata[0]} · %{customdata[1]}<br>丰度 %{marker.color:.3f}<extra></extra>"))
    height, width = image.shape[:2]
    figure.add_layout_image(source=image_source(image.tobytes(), image.shape), xref="x", yref="y", x=0, y=0,
                            sizex=width, sizey=height, xanchor="left", yanchor="top", sizing="stretch", opacity=he_opacity, layer="below")
    chart_style(figure, 420)
    xmax = max(float(width), float(coords[:, 0].max()) + 8)
    ymax = max(float(height), float(coords[:, 1].max()) + 8)
    figure.update_xaxes(range=[0, xmax], showgrid=False, zeroline=False, visible=False, constrain="domain")
    figure.update_yaxes(range=[ymax, 0], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1)
    figure.update_layout(dragmode="lasso", clickmode="event+select", uirevision="spatial",
                         selectionrevision="spatial", margin={"l": 0, "r": 0, "t": 0, "b": 25})
    return figure


def selected_indices(event: Any, n_spots: int) -> list[int]:
    """Read global spot positions rather than trace-local point indices.

    Args:
        event: Streamlit Plotly selection event.
        n_spots: Upper bound for valid row positions.

    Returns:
        Sorted unique positions selected across every categorical trace.
    """
    positions: set[int] = set()
    for point in event.get("selection", {}).get("points", []):
        custom = point.get("customdata")
        if custom is not None:
            try:
                value = int(custom[0])
                if 0 <= value < n_spots:
                    positions.add(value)
            except (TypeError, ValueError, IndexError):
                continue
    return sorted(positions)


def number(value: float | None, percent: bool = False) -> str:
    """Format missing metrics without inventing a zero.

    Args:
        value: Optional measured metric.
        percent: Whether the value is already in percentage units.

    Returns:
        Compact displayed number or an em dash.
    """
    return "—" if value is None else (f"{value:+.1f}%" if percent else f"{value:.3f}")


def main() -> None:
    """Render slice controls, dual-layer pathology and responsive ROI analysis."""
    st.set_page_config(page_title="SpatialCytoScope · 空间微区工作台", layout="wide", initial_sidebar_state="auto")
    st.markdown(f"<style>{(ROOT / 'app/assets/style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## SpatialCytoScope")
        st.caption("空间转录组 · 组织微区解析")
        slice_label = st.selectbox("切片", ["Train · Batch 1", "Test · Batch 2"])
        name = "slice_train" if slice_label.startswith("Train") else "slice_test"
        engine = st.selectbox("算法引擎", ["Spatial-GAT-AE", "Baseline"])
        model = "spatial_gat" if engine == "Spatial-GAT-AE" else "baseline"
        layer = st.selectbox("图层着色", ["结构域分型", "单基因表达热图", "免疫细胞浸润丰度"])
        path = DATA_DIR / f"{name}.h5ad"
        if not path.is_file():
            st.error("切片尚未就绪，请先完成 Step 1–3 数据管线。")
            st.stop()
        stamp = path.stat().st_mtime_ns
        data = local_slice(str(path), stamp)
        domain_key, cell_key = MODEL_KEYS[model]
        if domain_key not in data.obs or cell_key not in data.obsm:
            st.error("缺少模型预测，请先运行 Step 3。")
            st.stop()
        values = None
        if layer == "单基因表达热图":
            genes = data.var_names.tolist()
            gene = st.selectbox("基因", genes, index=genes.index("CD3D") if "CD3D" in genes else 0)
            matrix = data[:, gene].X
            values = matrix.toarray().ravel() if sparse.issparse(matrix) else np.asarray(matrix).ravel()
        elif layer == "免疫细胞浸润丰度":
            cell = st.selectbox("细胞类型", CELL_TYPES)
            values = np.asarray(data.obsm[cell_key])[:, CELL_TYPES.index(cell)]
        st.markdown("### 双层穿透")
        he = st.slider("H&E 病理底图透明度", 0.0, 1.0, 0.70, 0.05)
        opacity = st.slider("Spot 散点不透明度", 0.1, 1.0, 0.85, 0.05)
        radius = st.slider("Spot 散点像素半径", 4, 16, 5)
        st.markdown("### 数据连接")
        mode = st.radio("统计模式", ["本地独立模式", "FastAPI 模式"], label_visibility="collapsed")
        api_url = st.text_input("API 地址", os.environ.get("SPATIAL_API_URL", "http://127.0.0.1:8000")) if mode == "FastAPI 模式" else ""
        st.caption("模拟 H&E / 合成切片" if "simulation" in data.uns else "组织病理切片")
    summary = local_summary(name, stamp, data)
    if mode == "FastAPI 模式":
        try:
            summary = api_json(api_url, f"/api/v1/slice/{name}/summary")
            if summary["n_spots"] != data.n_obs or summary["gene_names"] != data.var_names.tolist():
                raise ValueError("API 切片与本地可视化数据不一致")
        except (URLError, HTTPError, ValueError, KeyError) as error:
            st.warning(f"API 暂不可用，已切回本地统计：{error}")
            mode = "本地独立模式"
            summary = local_summary(name, stamp, data)
    st.markdown('<div class="lab-nav"><h1><span class="lab-mark"></span>SpatialCytoScope</h1>'
                f'<p>{html.escape(slice_label)} · {html.escape(engine)}<br>空间组织结构与微区浸润工作台</p></div>', unsafe_allow_html=True)
    st.caption("在组织切片上圈选微区，联合查看组织分型、免疫浸润与差异表达。")
    gat = summary["models"]["spatial_gat"]
    kpis = [("当前切片 Spot", f"{summary['n_spots']:,}", f"{summary['n_genes']:,} 个保留基因"),
            ("空间高变基因", str(len(summary["top_svgs"])), "Moran’s I 排名记录数"),
            ("Spatial-GAT 结构域 ARI", number(gat["ari"]), f"相比 Baseline {number(summary['ari_improvement_pct'], True)}"),
            ("空间噪点率 PAS 降幅", number(summary["pas_reduction_pct"], True), "相比 Baseline · 负值表示升高")]
    with st.container(key="kpi_row"):
        kpi_columns = st.columns(4)
    for column, (label, value, note) in zip(kpi_columns, kpis):
        column.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
                        f'<div class="kpi-note"><span class="delta-badge">{note}</span></div></div>', unsafe_allow_html=True)
    # Remaining content width minus the shared gap is allocated in the exact ratio 58:42.
    with st.container(key="workspace"):
        left, right = st.columns([0.58, 0.42], gap="medium")
    labels = data.obs[domain_key].astype(str).to_numpy()
    key = f"spatial_{name}_{model}_{layer}"
    with left:
        with st.container(border=False, key="tissue_panel"):
            st.subheader("组织切片 · 双层穿透")
            st.caption("套索或框选 Spot；清空选择后可按结构域分析。")
            event = st.plotly_chart(spatial_figure(data, labels, values, he, opacity, radius),
                                    key=key, on_select="rerun", selection_mode=("lasso", "box"),
                                    config={"displaylogo": False, "scrollZoom": True}, width="stretch")
        a, b = st.columns(2, gap="small")
        with a, st.container(border=False, key="umap_panel"):
            st.subheader("UMAP 表征")
            coords = np.asarray(data.obsm["X_umap"])
            figure = go.Figure()
            for i, label in enumerate(sorted(set(labels))):
                mask = labels == label
                figure.add_trace(go.Scattergl(x=coords[mask, 0], y=coords[mask, 1], mode="markers", name=label,
                                 marker={"size": 4, "color": PALETTE[i % len(PALETTE)], "opacity": 0.8}, showlegend=False))
            chart_style(figure, 210)
            figure.update_xaxes(visible=False)
            figure.update_yaxes(visible=False)
            st.plotly_chart(figure, key="umap", width="stretch", config={"displayModeBar": False})
        with b, st.container(border=False, key="svg_panel"):
            st.subheader("Moran’s I · Top 10")
            records = svg_records(data)[:10][::-1]
            figure = chart_style(go.Figure(go.Bar(x=[r["morans_i"] for r in records], y=[r["gene"] for r in records],
                                                 orientation="h", marker_color=PALETTE[0])), 210)
            figure.update_layout(margin={"l": 0, "r": 5, "t": 5, "b": 20})
            figure.update_xaxes(title=None, showgrid=False)
            st.plotly_chart(figure, key="svg", width="stretch", config={"displayModeBar": False})
    with right:
        with st.container(border=False, key="roi_panel"):
            st.subheader("微区深度解析")
            selected = selected_indices(event, data.n_obs)
            if not selected:
                domains = sorted(set(labels), key=str)
                choice = st.selectbox("选择结构域作为 ROI", domains, format_func=lambda value: domain_title(value, data), key=f"roi_{name}_{model}")
                selected = np.flatnonzero(labels == choice).tolist()
            if not selected or len(selected) >= data.n_obs:
                st.info("请选择部分 Spot，至少保留一个背景 Spot 以计算差异表达。")
                return
            st.caption(f"ROI {len(selected):,} Spots / 背景 {data.n_obs - len(selected):,} Spots")
            try:
                if mode == "FastAPI 模式":
                    result = api_json(api_url, f"/api/v1/slice/{name}/roi_analysis", {"spot_indices": selected, "model_type": model})
                else:
                    result = local_roi(name, stamp, tuple(selected), model, data)
            except (URLError, HTTPError, ValueError) as error:
                st.warning(f"API 微区统计不可用，使用本地计算：{error}")
                result = local_roi(name, stamp, tuple(selected), model, data)
            st.markdown("### 免疫细胞浸润")
            fractions = [result["cell_proportions"][name] for name in CELL_TYPES]
            radar = chart_style(go.Figure(go.Scatterpolar(r=fractions + fractions[:1], theta=list(CELL_TYPES) + [CELL_TYPES[0]],
                                      fill="toself", fillcolor="rgba(149,188,229,0.25)", line={"color": PALETTE[0], "width": 2}, name="ROI 平均比例")), 235)
            radar.update_layout(polar={"radialaxis": {"range": [0, max(0.3, max(fractions) * 1.15)], "tickformat": ".0%", "nticks": 4},
                                       "angularaxis": {"tickfont": {"size": 11}}}, showlegend=False,
                                margin={"l": 52, "r": 52, "t": 20, "b": 20})
            st.plotly_chart(radar, key="radar", width="stretch", config={"displayModeBar": False})
            st.markdown("### ROI 差异基因")
            candidates = [row["gene"] for row in result["differential_genes"]]
            show_genes = st.multiselect("查看表达分布", candidates, default=candidates[:4], key=f"violin_{name}_{model}")
            if show_genes:
                matrix = data[:, show_genes].X
                expression = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
                mask = np.zeros(data.n_obs, dtype=bool)
                mask[selected] = True
                violin = go.Figure()
                for name_group, rows, color, side in (("ROI", mask, PALETTE[1], "negative"), ("背景", ~mask, PALETTE[0], "positive")):
                    violin.add_trace(go.Violin(x=np.tile(show_genes, int(rows.sum())), y=expression[rows].ravel(), name=name_group,
                                     side=side, line_color=color, fillcolor=color, opacity=0.7, points=False, meanline_visible=True,
                                     scalemode="width", spanmode="hard"))
                chart_style(violin, 220)
                violin.update_layout(violinmode="overlay", yaxis_title="log1p 表达")
                st.plotly_chart(violin, key="violin", width="stretch", config={"displayModeBar": False})
            with st.expander("Top 8 差异表达统计"):
                st.dataframe(pd.DataFrame(result["differential_genes"]), hide_index=True, width="stretch")
                st.caption("双侧 Mann–Whitney U；Log2FC 基于线性归一化丰度；FDR 对全部保留基因校正。")
        baseline = summary["models"]["baseline"]
        st.markdown('<div class="clean-card"><h3>分割边界平滑度</h3>'
                    f'<div class="boundary-row"><span>Baseline · PAS / 边界率</span><strong>{number(baseline["pas"])} / {number(baseline["boundary_rate"])}</strong></div>'
                    f'<div class="boundary-row"><span>Spatial-GAT · PAS / 边界率</span><strong>{number(gat["pas"])} / {number(gat["boundary_rate"])}</strong></div>'
                    '<div class="kpi-note">PAS：超过半数空间近邻标签不同的 Spot 占比。数值越低，空间碎片化越少。</div></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
