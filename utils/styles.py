"""Global CSS — clean light dashboard theme (ecommerce-inspired)."""
import streamlit as st


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg:          #f8fafc;
        --bg2:         #ffffff;
        --bg3:         #f1f5f9;
        --sidebar-bg:  #0e1117;
        --border:      #e2e8f0;
        --border-light:#f1f5f9;
        --text:        #0f172a;
        --text-sub:    #374151;
        --text-muted:  #6b7280;
        --blue:        #2563eb;
        --blue-light:  #eff6ff;
        --cyan:        #0891b2;
        --teal:        #0d9488;
        --green:       #16a34a;
        --green-light: #f0fdf4;
        --yellow:      #d97706;
        --yellow-light:#fefce8;
        --red:         #dc2626;
        --red-light:   #fef2f2;
        --indigo:      #4f46e5;
        --indigo-light:#eef2ff;
        --font:        'Inter', -apple-system, sans-serif;
        --mono:        'JetBrains Mono', monospace;
        --radius:      10px;
        --shadow:      0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
        /* ── Font scale ── */
        --fs-xs:       11px;
        --fs-sm:       13px;
        --fs-base:     15px;
        --fs-md:       17px;
        --fs-lg:       20px;
        --fs-xl:       24px;
        /* ── Sidebar palette ── */
        --sb-text:       #94a3b8;
        --sb-text-bright:#f1f5f9;
        --sb-text-dim:   #64748b;
        --sb-card:       #1e2533;
    }

    /* ── Global ──────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: var(--font) !important;
        font-size: var(--fs-base) !important;        /* base +2pt */
        color: var(--text) !important;
    }
    .stApp { background: var(--bg) !important; }
    .main .block-container { padding-top: 1.2rem; max-width: 1400px; }

    /* Force all text dark (override any white) */
    p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
    }

    /* ── Global dropdown / popover / overlay fix ─────────────── */
    /* All baseweb overlays: selectbox, multiselect, popover, menu, tooltip */
    ul[role="listbox"],
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    div[data-baseweb="select"] ul,
    div[data-baseweb="popover"] > div,
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="select-dropdown"],
    [role="listbox"] {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
    }
    ul[role="listbox"] li,
    div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] li,
    div[data-baseweb="popover"] div[role="option"],
    [role="option"] {
        color: var(--text) !important;
        background: #ffffff !important;
    }
    ul[role="listbox"] li:hover,
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="menu"] li:hover,
    div[data-baseweb="popover"] div[role="option"]:hover,
    [role="option"]:hover {
        background: var(--bg3) !important;
    }
    ul[role="listbox"] li[aria-selected="true"],
    [role="option"][aria-selected="true"] {
        background: var(--blue-light) !important;
        color: var(--blue) !important;
    }
    /* baseweb select input — white bg + dark text in main area */
    div[data-baseweb="select"] {
        background: #ffffff !important;
        border-color: var(--border) !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: var(--text) !important;
    }
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
    }
    /* baseweb select dropdown container */
    div[data-baseweb="select"] [data-baseweb="popover"] {
        background: #ffffff !important;
    }

    /* ── Sidebar ─────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        border-right: 0.5px solid rgba(255,255,255,0.07);
    }
    /* Sidebar text stays light on dark background */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] td,
    [data-testid="stSidebar"] th,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 { color: var(--sb-text) !important; }
    [data-testid="stSidebar"] .stCaption { color: var(--sb-text-dim) !important; }

    /* Sidebar widget labels & values — light on dark */
    [data-testid="stSidebar"] .stSlider > label,
    [data-testid="stSidebar"] .stSelectbox > label,
    [data-testid="stSidebar"] .stMultiSelect > label,
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stCheckbox > label,
    [data-testid="stSidebar"] .stTextInput > label { color: #94a3b8 !important; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
    [data-testid="stSidebar"] .stCheckbox label { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] div[data-baseweb="select"],
    [data-testid="stSidebar"] div[data-baseweb="select"] > div { background: var(--sb-card) !important; border-color: rgba(255,255,255,0.1) !important; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] span { color: #f1f5f9 !important; }
    [data-testid="stSidebar"] .stTextInput input { color: #f1f5f9 !important; }
    /* Sidebar dropdown options — light text on dark background (overrides global) */
    [data-testid="stSidebar"] ul[role="listbox"],
    [data-testid="stSidebar"] div[data-baseweb="popover"],
    [data-testid="stSidebar"] div[data-baseweb="popover"] > div,
    [data-testid="stSidebar"] div[data-baseweb="menu"],
    [data-testid="stSidebar"] [role="listbox"],
    [data-testid="stSidebar"] [data-baseweb="select-dropdown"] { background: #1e2533 !important; border-color: rgba(255,255,255,0.1) !important; }
    [data-testid="stSidebar"] ul[role="listbox"] li,
    [data-testid="stSidebar"] div[data-baseweb="popover"] li,
    [data-testid="stSidebar"] div[data-baseweb="menu"] li,
    [data-testid="stSidebar"] [role="option"] { color: #f1f5f9 !important; background: #1e2533 !important; }
    [data-testid="stSidebar"] ul[role="listbox"] li:hover,
    [data-testid="stSidebar"] div[data-baseweb="popover"] li:hover,
    [data-testid="stSidebar"] div[data-baseweb="menu"] li:hover,
    [data-testid="stSidebar"] [role="option"]:hover { background: #2d3748 !important; }
    [data-testid="stSidebar"] ul[role="listbox"] li[aria-selected="true"],
    [data-testid="stSidebar"] [role="option"][aria-selected="true"] { background: #2563eb !important; }

    .sidebar-top {
        padding: 18px 14px 14px;
        border-bottom: 0.5px solid rgba(255,255,255,0.07);
        margin-bottom: 4px;
    }
    .sidebar-logo-row {
        display: flex; align-items: center; gap: 12px; margin-bottom: 10px;
    }
    .sidebar-logo-img {
        width: 44px; height: 44px; border-radius: 9px; object-fit: cover;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .sidebar-brand { line-height: 1.3; }
    .sidebar-brand-main {
        color: var(--sb-text-bright) !important; font-weight: 600; font-size: var(--fs-lg) !important;
    }
    .sidebar-brand-sub {
        color: var(--sb-text) !important; font-size: var(--fs-sm) !important; letter-spacing: 0.03em;
    }
    /* Section labels: +2pt larger than nav button text (14px → 16px) */
    .sidebar-section-label {
        padding: 12px 14px 5px;
        font-size: var(--fs-md) !important; color: var(--sb-text-dim) !important;
        letter-spacing: 0.07em; text-transform: uppercase; font-weight: 600;
        text-align: left !important;
    }
    .event-badge {
        background: var(--sb-card); border-radius: 8px;
        padding: 10px 12px; margin: 6px 0;
        text-align: left !important;
    }
    .event-badge-name { color: var(--sb-text-bright) !important; font-weight: 500; font-size: var(--fs-base) !important; text-align: left !important; }
    .event-badge-sub  { color: var(--sb-text) !important; font-size: var(--fs-xs) !important; margin-top: 3px; text-align: left !important; }

    /* Sidebar global left-align */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { text-align: left !important; }
    [data-testid="stSidebar"] .stMarkdown { text-align: left !important; }

    /* ── Sidebar nav buttons ─────────────────────────────────── */
    [data-testid="stSidebar"] .stButton {
        text-align: left !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: var(--sb-text) !important;
        border: none !important;
        border-radius: 7px !important;
        font-size: var(--fs-base) !important;
        font-weight: 400 !important;
        padding: 8px 12px !important;
        text-align: left !important;
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        transition: all 0.15s !important;
    }
    [data-testid="stSidebar"] .stButton > button > div,
    [data-testid="stSidebar"] .stButton > button > p,
    [data-testid="stSidebar"] .stButton > button > span,
    [data-testid="stSidebar"] .stButton > button * {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--sb-card) !important;
        color: var(--sb-text-bright) !important;
    }

    /* ── Main header ─────────────────────────────────────────── */
    .page-header { margin-bottom: 16px; }
    .page-title  { font-size: var(--fs-lg) !important; font-weight: 600; color: var(--text) !important; }
    .page-sub    { color: var(--text-sub) !important; font-size: var(--fs-base) !important; margin-top: 4px; }
    .header-logo {
        width: 36px; height: 36px; border-radius: 8px;
        object-fit: cover; vertical-align: middle; margin-right: 10px;
    }
    .section-label-badge {
        font-size: var(--fs-sm) !important; font-weight: 500;
        color: var(--blue) !important; margin-bottom: 4px; letter-spacing: 0.04em;
    }

    /* ── Cards ───────────────────────────────────────────────── */
    .card-title {
        font-size: var(--fs-base) !important; font-weight: 500; color: var(--text) !important;
        margin-bottom: 14px;
    }

    /* ── Metric cards ────────────────────────────────────────── */
    .metric-row  { display: flex; gap: 10px; margin: 14px 0; flex-wrap: wrap; }
    .metric-card {
        border-radius: var(--radius); padding: 13px 14px;
        min-width: 130px; flex: 1;
    }
    .metric-card.blue   { background: var(--blue-light); }
    .metric-card.green  { background: var(--green-light); }
    .metric-card.yellow { background: var(--yellow-light); }
    .metric-card.red    { background: var(--red-light); }
    .metric-card.indigo { background: var(--indigo-light); }
    .metric-label { font-size: var(--fs-xs) !important; font-weight: 500; margin-bottom: 6px; }
    .metric-card.blue   .metric-label { color: #1e40af !important; }
    .metric-card.green  .metric-label { color: #166534 !important; }
    .metric-card.yellow .metric-label { color: #92400e !important; }
    .metric-card.red    .metric-label { color: #991b1b !important; }
    .metric-card.indigo .metric-label { color: #3730a3 !important; }
    .metric-value {
        font-size: var(--fs-xl) !important; font-weight: 700; color: var(--text) !important;
        font-family: var(--mono); line-height: 1;
    }
    .metric-unit { font-size: var(--fs-xs) !important; color: var(--text-muted) !important; margin-top: 5px; }

    /* ── Callout boxes ───────────────────────────────────────── */
    .callout {
        background: var(--blue-light); border-left: 3px solid var(--blue);
        border-radius: 0 var(--radius) var(--radius) 0;
        padding: 11px 14px; font-size: var(--fs-base) !important;
        color: var(--text-sub) !important; margin: 12px 0;
    }
    .callout.warn   { background: var(--yellow-light); border-left-color: var(--yellow); }
    .callout.good   { background: var(--green-light);  border-left-color: var(--green); }
    .callout.danger { background: var(--red-light);    border-left-color: var(--red); }
    .callout strong { color: var(--text) !important; }

    /* ── Section header ──────────────────────────────────────── */
    .section-header { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
    .section-icon {
        width: 34px; height: 34px; border-radius: 8px;
        background: var(--blue-light);
        display: flex; align-items: center; justify-content: center;
        font-size: 17px; flex-shrink: 0;
    }
    .section-title { font-size: var(--fs-md) !important; font-weight: 600; color: var(--text) !important; margin: 0; }
    .section-desc  { font-size: var(--fs-sm) !important; color: var(--text-sub) !important; margin-top: 2px; }

    /* ── Container styling (st.container(border=True)) ─── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg2) !important;
        border: 0.5px solid var(--border) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow);
    }
    .control-title {
        font-size: var(--fs-xs) !important; font-weight: 600; color: var(--text-sub) !important;
        text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 10px;
    }

    /* ── Streamlit widgets ───────────────────────────────────── */
    .stSlider > label, .stSelectbox > label, .stMultiSelect > label,
    .stRadio > label, .stCheckbox > label, .stTextInput > label {
        color: var(--text-sub) !important;
        font-size: var(--fs-base) !important; font-weight: 500 !important;
    }
    .stRadio div[role="radiogroup"] label,
    .stCheckbox label {
        color: var(--text) !important; font-size: var(--fs-base) !important;
    }
    .stSelectbox div[data-baseweb="select"] span,
    .stMultiSelect div[data-baseweb="select"] span {
        color: var(--text) !important; font-size: var(--fs-base) !important;
    }
    /* Main area buttons (non-sidebar) */
    .main .stButton > button,
    [data-testid="stMainBlockContainer"] .stButton > button {
        background: var(--blue) !important; color: #ffffff !important;
        border: none !important; border-radius: 8px !important;
        font-weight: 500 !important; font-size: var(--fs-base) !important;
        font-family: var(--font) !important;
        padding: 9px 18px !important; transition: all 0.15s !important;
    }
    .main .stButton > button:hover { background: #1d4ed8 !important; }

    /* ── Leaderboard ─────────────────────────────────────────── */
    .lb-wrap { background: var(--bg2); border: 0.5px solid var(--border);
               border-radius: var(--radius); overflow: hidden; }
    .lb-head { padding: 12px 16px; border-bottom: 0.5px solid var(--border);
               font-weight: 500; font-size: var(--fs-base) !important; color: var(--text) !important; }
    .lb-row  { display: flex; align-items: center; gap: 10px; padding: 10px 16px;
               border-bottom: 0.5px solid var(--border-light);
               font-size: var(--fs-sm) !important; transition: background 0.1s; }
    .lb-row:last-child { border-bottom: none; }
    .lb-row:hover { background: var(--bg3); }
    .lb-rank  { width: 22px; font-family: var(--mono); font-size: var(--fs-xs) !important; color: var(--text-muted) !important; }
    .lb-rank.gold   { color: #d97706 !important; font-weight: 700; }
    .lb-rank.silver { color: #64748b !important; font-weight: 700; }
    .lb-rank.bronze { color: #b45309 !important; font-weight: 700; }
    .lb-team  { flex: 1; font-weight: 500; color: var(--text) !important; }
    .lb-event { font-size: var(--fs-xs) !important; color: var(--text-muted) !important; flex: 1; }
    .lb-score { font-family: var(--mono); font-weight: 600;
                font-size: var(--fs-base) !important; color: var(--green) !important; }
    .lb-bar-wrap { width: 70px; }
    .lb-bar-bg   { background: var(--bg3); border-radius: 999px; height: 4px; }
    .lb-bar-fill { background: var(--green); border-radius: 999px; height: 4px; }

    /* ── Map ─────────────────────────────────────────────────── */
    .map-frame { border: 0.5px solid var(--border); border-radius: var(--radius); overflow: hidden; }

    /* ── Chips ───────────────────────────────────────────────── */
    .chip { display: inline-block; padding: 3px 9px; border-radius: 999px;
            font-size: var(--fs-xs) !important; font-weight: 500; }
    .chip-blue   { background: var(--blue-light);   color: var(--blue) !important; }
    .chip-green  { background: var(--green-light);  color: var(--green) !important; }
    .chip-yellow { background: var(--yellow-light); color: var(--yellow) !important; }
    .chip-red    { background: var(--red-light);    color: var(--red) !important; }

    /* ── Hyperparameter guide panel ────────────────────────── */
    .hp-guide { padding: 2px 0; }
    .hp-guide-title {
        font-size: var(--fs-base) !important; font-weight: 600;
        color: var(--text) !important; margin-bottom: 12px;
    }
    .hp-section-label {
        font-size: var(--fs-xs) !important; font-weight: 600;
        color: var(--text-muted) !important; text-transform: uppercase;
        letter-spacing: 0.07em; margin: 14px 0 8px; padding-bottom: 4px;
        border-bottom: 1px solid var(--border-light);
    }
    .hp-section-label:first-of-type { margin-top: 0; }
    .hp-grid {
        display: grid; grid-template-columns: 1fr 1fr;
        gap: 6px 18px;
    }
    .hp-item {
        display: flex; align-items: baseline; gap: 6px;
        font-size: var(--fs-sm) !important; line-height: 1.5;
        padding: 3px 0;
    }
    .hp-icon { flex-shrink: 0; width: 18px; text-align: center; }
    .hp-name { font-weight: 600; color: var(--text) !important; white-space: nowrap; }
    .hp-desc { color: var(--text-sub) !important; }
    .hp-tip  { color: var(--blue) !important; font-size: var(--fs-xs) !important; }

    hr { border-color: var(--border) !important; }
    ::-webkit-scrollbar       { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg3); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
    </style>
    """, unsafe_allow_html=True)


COLORS = {
    "text": "#0f172a",
    "text_sub": "#374151",
    "text_muted": "#6b7280",
    "blue": "#2563eb",
    "blue_light": "#bfdbfe",
    "blue_dark": "#1d4ed8",
    "cyan": "#06b6d4",
    "green": "#16a34a",
    "green_light": "#10b981",
    "red": "#ef4444",
    "red_dark": "#dc2626",
    "yellow": "#d97706",
    "indigo": "#6366f1",
    "indigo_dark": "#4f46e5",
    "bg": "#ffffff",
    "bg3": "#f1f5f9",
    "border": "#e2e8f0",
    "axis": "#94a3b8",
    "bar_light": "#93c5fd",
    "bar_flood": "#ef4444",
}
