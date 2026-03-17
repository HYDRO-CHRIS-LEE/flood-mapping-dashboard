"""Optical Before / After — Sentinel-2 + NDWI."""

import streamlit as st
import folium
from streamlit_folium import st_folium
from utils.data_loader import (
    load_tif, EVENT_CENTERS, EVENT_ZOOM, EVENT_BOUNDS,
    tif_to_rgba, rgb_tif_to_rgba, rgba_to_b64,
)

CLOUD_NOTE = {
    "harvey":   ("warn",  "Some cloud cover during Harvey's flood peak creates gaps in the optical image — this is exactly why SAR radar is needed in the next tab."),
    "pakistan": ("good",  "Sindh Province is semi-arid with minimal clouds — the NDWI contrast between flooded farmland and dry ground is striking."),
    "dubai":    ("good",  "Desert climate means near-zero cloud cover. The Before/After NDWI contrast is exceptional — you can clearly see water pooling on roads."),
    "la2025":   ("warn",  "Heavy winter cloud cover creates significant data gaps in the After image. This is intentional — it demonstrates why optical satellites alone are insufficient in rainy climates. SAR is the only solution here."),
    "germany2021": ("good","Central European summer — good optical coverage with minimal cloud interference."),
    "china2020":("warn",  "Yangtze monsoon season brings cloud cover during peak flood. SAR coverage is more reliable for this event."),
    "kerala2018":("warn", "India's southwest monsoon season creates heavy cloud cover. Optical data has gaps during peak flooding."),
}


def make_map(event: str, layer: str, period: str) -> folium.Map:
    center    = EVENT_CENTERS.get(event, [0, 0])
    zoom      = EVENT_ZOOM.get(event, 9)
    bounds    = EVENT_BOUNDS.get(event, [-1, -1, 1, 1])
    img_bnds  = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

    m = folium.Map(location=center, zoom_start=zoom,
                   tiles="CartoDB positron", prefer_canvas=True)

    if layer == "RGB (True Color)":
        data, _ = load_tif(event, f"RGB_{period}", max_pixels=600)
        if data is None: return m
        rgba = rgb_tif_to_rgba(data[0], data[1], data[2])
    elif layer == "NDWI (Water Index)":
        data, _ = load_tif(event, f"NDWI_{period}", max_pixels=600)
        if data is None: return m
        rgba = tif_to_rgba(data[0], colormap="RdYlBu")
    else:
        before, _ = load_tif(event, "NDWI_before", max_pixels=600)
        after,  _ = load_tif(event, "NDWI_after",  max_pixels=600)
        if before is None or after is None: return m
        rgba = tif_to_rgba(after[0] - before[0], colormap="RdBu")

    folium.raster_layers.ImageOverlay(
        image=rgba_to_b64(rgba), bounds=img_bnds,
        opacity=0.85, name=layer,
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m


def render_module2(event: str):
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🛰️</div>
        <div>
            <div class="section-title">Optical Before / After</div>
            <div class="section-desc">
                Compare Sentinel-2 satellite images before and after the flood,
                and discover the limitations of optical sensors.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if event in CLOUD_NOTE:
        ntype, ntext = CLOUD_NOTE[event]
        icon = "⚠️" if ntype == "warn" else "✅"
        st.markdown(f"""
        <div class="callout {ntype}">
            <strong>{icon} Event Note</strong> &nbsp; {ntext}
        </div>
        """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 2])
    with col_l:
        with st.container(border=True):
            st.markdown('<div class="control-title">Layer Settings</div>',
                        unsafe_allow_html=True)
            layer = st.radio("Band",
                             ["RGB (True Color)", "NDWI (Water Index)",
                              "NDWI Change (After − Before)"],
                             help="NDWI: blue = water, brown = dry land")
            period = "after"
            if layer != "NDWI Change (After − Before)":
                period = st.radio("Time period", ["before", "after"])

        if "NDWI" in layer:
            with st.container(border=True):
                st.markdown(
                    '<div class="card-title">NDWI Color Guide</div>'
                    '<div style="display:flex;flex-direction:column;gap:6px;line-height:1.5;font-size:var(--fs-xs)">'
                    '<div><span style="background:#2166ac;padding:1px 8px;border-radius:3px;color:white;font-size:10px">Water</span> &nbsp; NDWI &gt; 0.3</div>'
                    '<div><span style="background:#92c5de;padding:1px 8px;border-radius:3px;font-size:10px">Wet soil</span> &nbsp; shallow / wetland</div>'
                    '<div><span style="background:#f7f7f7;padding:1px 8px;border-radius:3px;border:1px solid #eee;font-size:10px">Neutral</span></div>'
                    '<div><span style="background:#d6604d;padding:1px 8px;border-radius:3px;color:white;font-size:10px">Dry land</span></div>'
                    '</div>',
                    unsafe_allow_html=True)

    with col_r:
        with st.spinner("Rendering map..."):
            m = make_map(event, layer, period)
        with st.container(border=True):
            st_folium(m, width=None, height=460, returned_objects=[])

    if layer == "NDWI Change (After − Before)":
        st.markdown("""
        <div class="callout good">
            <strong>How to read the change map</strong> &nbsp;
            Blue pixels = new water after the flood. Red pixels = areas that dried out.
            White/missing areas = cloud cover blocked the satellite view.
            → Head to <em>SAR Detection</em> to see how radar cuts through clouds.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="callout">
            <strong>Discussion</strong> &nbsp;
            Switch between Before and After. Where did flooding occur — near rivers, downtown, farmland?
            Try NDWI mode to make water pixels stand out more clearly.
        </div>
        """, unsafe_allow_html=True)
