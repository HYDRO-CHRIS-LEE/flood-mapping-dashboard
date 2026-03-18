"""
AI Flood Inundation Mapping Dashboard
UCI CHRS EarthAI Program
"""

import streamlit as st
import base64, os

st.set_page_config(
    page_title="AI Flood Mapping | UCI CHRS",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

from modules.module1_sar     import render_module1
from modules.module2_optical import render_module2
from modules.module4_rf      import render_module4
from modules.module6_gpm     import render_module6
from utils.styles            import inject_css
from utils.data_loader       import get_available_events, ALL_EVENTS

inject_css()

# ── CHRS logo ─────────────────────────────────────────────────────
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "CHRS_LOGO.png")
def get_logo_b64():
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return ""

LOGO_B64 = get_logo_b64()
LOGO_IMG  = f'<img src="{LOGO_B64}" class="sidebar-logo-img">' if LOGO_B64 else "🌊"

# ── Navigation state ──────────────────────────────────────────────
PAGES = {
    "rainfall":  {"label": "Rainfall Timeline",     "icon": "🌧️", "section": 1},
    "optical":   {"label": "Optical Before / After", "icon": "🛰️", "section": 1},
    "sar":       {"label": "SAR Detection",          "icon": "📡", "section": 1},
    "classifier":{"label": "AI Flood Classifier",    "icon": "🤖", "section": 2},
}
if "active_page" not in st.session_state:
    st.session_state.active_page = "rainfall"

active = st.session_state.active_page

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    # Logo + brand
    st.markdown(f"""
    <div class="sidebar-top">
        <div class="sidebar-logo-row">
            {LOGO_IMG}
            <div class="sidebar-brand">
                <div class="sidebar-brand-main">AI Flood Mapping</div>
                <div class="sidebar-brand-sub">UCI CHRS EarthAI Program</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 0. Team ───────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-label">0. Team</div>',
                unsafe_allow_html=True)
    team_name = st.text_input("Team Name", value="Team A",
                               placeholder="e.g. Team Alpha",
                               label_visibility="collapsed")
    st.session_state.team_name = team_name

    st.markdown("---")

    # ── 1. Flood Event ────────────────────────────────────────────
    available = get_available_events()
    if not available:
        st.error("No event data found in data/.\nRun the Colab GEE Export notebook first.")
        st.stop()

    def event_label(k):
        e = ALL_EVENTS[k]
        return f"{e['label']} ({e['year']})"

    st.markdown('<div class="sidebar-section-label">1. Flood Event</div>',
                unsafe_allow_html=True)
    selected_event = st.selectbox(
        "Event", options=available,
        format_func=event_label,
        label_visibility="collapsed"
    )
    ev    = ALL_EVENTS[selected_event]
    color = ev.get("color", "#2563eb")
    st.markdown(f"""
    <div class="event-badge" style="border-left:3px solid {color}">
        <div class="event-badge-name">{ev['label']} {ev['year']}</div>
        <div class="event-badge-sub">📍 {ev['region']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. Flood Event Detection nav ──────────────────────────────
    st.markdown('<div class="sidebar-section-label">2. Flood Event Detection</div>',
                unsafe_allow_html=True)

    for key in ["rainfall", "optical", "sar"]:
        p = PAGES[key]
        if st.button(f"{p['icon']}  {p['label']}", key=f"nav_{key}",
                     use_container_width=True):
            st.session_state.active_page = key
            st.rerun()

    st.markdown("---")

    # ── 3. AI Flood Classifier nav ────────────────────────────────
    st.markdown('<div class="sidebar-section-label">3. AI Flood Classifier</div>',
                unsafe_allow_html=True)

    p = PAGES["classifier"]
    if st.button(f"{p['icon']}  {p['label']}", key="nav_classifier",
                 use_container_width=True):
        st.session_state.active_page = "classifier"
        st.rerun()

    st.markdown("---")


# ── Main header ───────────────────────────────────────────────────
page_info = PAGES[active]

if active != "classifier":
    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">{page_info['icon']}  {page_info['label']}</div>
        <div class="page-sub">
            <strong>{ev['label']}</strong> ({ev['year']})
            &nbsp;·&nbsp; 📍 {ev['region']}
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="page-header">
        <div class="page-title">{page_info['icon']}  {page_info['label']}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr style="margin:0 0 20px 0">', unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────────────
if active == "rainfall":
    render_module6(selected_event)
elif active == "optical":
    render_module2(selected_event)
elif active == "sar":
    render_module1(selected_event)
elif active == "classifier":
    render_module4(available)
