"""SAR Detection — Sentinel-1 Otsu Thresholding."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from skimage.filters import threshold_otsu
from utils.data_loader import (
    load_tif, EVENT_CENTERS, EVENT_ZOOM, EVENT_BOUNDS, rgba_to_b64,
)


def apply_threshold(sar, thr, perm=None):
    flood = (sar < thr).astype(np.uint8)
    if perm is not None:
        flood = flood & (~(perm > 0.5)).astype(np.uint8)
    return flood


def make_map(event, sar, flood):
    center    = EVENT_CENTERS.get(event, [0, 0])
    zoom      = EVENT_ZOOM.get(event, 9)
    bounds    = EVENT_BOUNDS.get(event, [-1, -1, 1, 1])
    img_bnds  = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

    sar_norm = np.clip((sar - (-25)) / 25, 0, 1)
    gray = (sar_norm * 200).astype(np.uint8)
    r = gray.copy(); g = gray.copy(); b = gray.copy()
    r[flood == 1] = 37;  g[flood == 1] = 99;  b[flood == 1] = 235
    a = np.full_like(r, 220)
    rgba = np.stack([r, g, b, a], axis=-1)

    m = folium.Map(location=center, zoom_start=zoom,
                   tiles="CartoDB dark_matter", prefer_canvas=True)
    folium.raster_layers.ImageOverlay(
        image=rgba_to_b64(rgba), bounds=img_bnds, opacity=0.9,
    ).add_to(m)
    return m


def render_module1(event: str):
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📡</div>
        <div>
            <div class="section-title">SAR Detection</div>
            <div class="section-desc">
                Sentinel-1 radar detects floods through clouds — and AI finds the optimal cutoff automatically.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout">
        <strong>How SAR Works</strong> &nbsp;
        Radar pulses bounce off calm water <em>away</em> from the sensor → <strong>dark (low dB)</strong> pixels.
        Rough land scatters energy back → <strong>bright (high dB)</strong> pixels.
        Dark pixels = water. That's the core idea.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading SAR data..."):
        sar_data, _ = load_tif(event, "SAR_after",           max_pixels=512)
        perm_data,_ = load_tif(event, "JRC_permanent_water", max_pixels=512)

    if sar_data is None:
        st.markdown(f"""
        <div class="callout warn">
            <strong>SAR data not found</strong><br>
            Expected: <code>data/{event}/SAR_after.tif</code><br>
            Run Cell 9 in the Colab export notebook.
        </div>
        """, unsafe_allow_html=True)
        return

    sar_arr  = sar_data[0]
    perm_arr = perm_data[0] if perm_data is not None else None

    valid    = sar_arr[~np.isnan(sar_arr)].flatten()
    valid    = valid[(valid > -35) & (valid < 5)]
    otsu_val = float(threshold_otsu(valid)) if len(valid) > 100 else -16.0

    col_ctrl, col_map = st.columns([1, 2])

    with col_ctrl:
        st.markdown('<div class="control-panel"><div class="control-title">Threshold Control</div>',
                    unsafe_allow_html=True)
        thr = st.slider("SAR VH Threshold (dB)", -30.0, -5.0,
                        round(otsu_val, 1), 0.5,
                        help="Pixels darker than this = classified as flooded")
        if st.button("Apply Otsu Auto-Threshold"):
            thr = round(otsu_val, 1)
            st.rerun()
        remove_perm = st.checkbox("Remove permanent water bodies", value=True,
                                  help="Masks pre-existing rivers/lakes using JRC data")
        st.markdown("</div>", unsafe_allow_html=True)

        flood_mask   = apply_threshold(sar_arr, thr, perm_arr if remove_perm else None)
        total_valid  = int(np.sum(~np.isnan(sar_arr)))
        flood_px     = int(np.sum(flood_mask))
        flood_pct    = flood_px / total_valid * 100 if total_valid > 0 else 0
        flood_km2    = flood_px * 0.0004

        st.markdown(f"""
        <div class="metric-row" style="flex-direction:column;gap:8px">
            <div class="metric-card blue">
                <div class="metric-label">Flooded Pixels</div>
                <div class="metric-value">{flood_px:,}</div>
            </div>
            <div class="metric-card yellow">
                <div class="metric-label">Flood Fraction</div>
                <div class="metric-value">{flood_pct:.1f}%</div>
            </div>
            <div class="metric-card green">
                <div class="metric-label">Est. Flood Area</div>
                <div class="metric-value">{flood_km2:.0f}</div>
                <div class="metric-unit">km² (approx)</div>
            </div>
            <div class="metric-card indigo">
                <div class="metric-label">Otsu Auto Value</div>
                <div class="metric-value">{otsu_val:.1f}</div>
                <div class="metric-unit">dB</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_map:
        hist_v, edges = np.histogram(valid, bins=80)
        centers = (edges[:-1] + edges[1:]) / 2
        colors  = ["#3b82f6" if c < thr else "#cbd5e1" for c in centers]

        fig = go.Figure(go.Bar(
            x=centers, y=hist_v, marker_color=colors,
            hovertemplate="%{x:.1f} dB: %{y:,} pixels<extra></extra>",
        ))
        fig.add_vline(x=thr, line_color="#ef4444", line_width=2, line_dash="dash",
                      annotation_text=f"  Threshold {thr:.1f} dB",
                      annotation_font_color="#ef4444", annotation_font_size=11)
        fig.add_vline(x=otsu_val, line_color="#16a34a", line_width=1.5, line_dash="dot",
                      annotation_text=f"  Otsu {otsu_val:.1f}",
                      annotation_font_color="#16a34a", annotation_font_size=10,
                      annotation_position="bottom right")
        fig.update_layout(
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            font=dict(color="#0f172a", family="Inter"),
            height=150, margin=dict(l=5, r=5, t=5, b=25),
            xaxis=dict(title="SAR VH (dB)", showgrid=False,
                       tickfont=dict(color="#94a3b8", size=10),
                       title_font=dict(size=10, color="#94a3b8")),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9",
                       tickfont=dict(color="#94a3b8", size=10)),
            showlegend=False,
        )
        st.markdown('<div class="card" style="padding:12px;margin-bottom:10px">',
                    unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("Rendering flood map..."):
            m = make_map(event, sar_arr, flood_mask)
        st.markdown('<div class="map-frame">', unsafe_allow_html=True)
        st_folium(m, width=None, height=350, returned_objects=[])
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("🔵 Blue = flood detected · Gray = land · Adjust the slider to explore")

    st.markdown(f"""
    <div class="callout good">
        <strong>What is Otsu's Method?</strong> &nbsp;
        The histogram shows two peaks — water (left, dark) and land (right, bright).
        Otsu's algorithm automatically finds the threshold that maximizes the variance between the two groups.
        The computer picks the best boundary on its own — the simplest form of AI we use today.
        Current Otsu value: <strong>{otsu_val:.1f} dB</strong>
    </div>
    """, unsafe_allow_html=True)
