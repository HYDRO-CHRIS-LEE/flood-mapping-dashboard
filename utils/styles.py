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
    }

    /* ── Global ──────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: var(--font) !important;
        font-size: 15px !important;        /* base +2pt */
        color: var(--text) !important;
    }
    .stApp { background: var(--bg) !important; }
    .main .block-container { padding-top: 1.2rem; max-width: 1400px; }

    /* Force all text dark (override any white) */
    p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
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
    [data-testid="stSidebar"] label { color: #94a3b8 !important; }
    [data-testid="stSidebar"] .stCaption { color: #64748b !important; }

    .sidebar-top {
        padding: 16px 14px 12px;
        border-bottom: 0.5px solid rgba(255,255,255,0.07);
        margin-bottom: 4px;
    }
    .sidebar-logo-row {
        display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
    }
    .sidebar-logo-img {
        width: 36px; height: 36px; border-radius: 8px; object-fit: cover;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .sidebar-brand { line-height: 1.3; }
    .sidebar-brand-main {
        color: #f1f5f9 !important; font-weight: 600; font-size: 15px !important;
    }
    .sidebar-brand-sub {
        color: #94a3b8 !important; font-size: 11px !important; letter-spacing: 0.03em;
    }
    .sidebar-section-label {
        padding: 12px 14px 5px;
        font-size: 11px !important; color: #64748b !important;
        letter-spacing: 0.07em; text-transform: uppercase; font-weight: 600;
    }
    .event-badge {
        background: #1e2533; border-radius: 8px;
        padding: 10px 12px; margin: 6px 0;
    }
    .event-badge-name { color: #f1f5f9 !important; font-weight: 500; font-size: 14px !important; }
    .event-badge-sub  { color: #94a3b8 !important; font-size: 12px !important; margin-top: 3px; }

    /* ── Sidebar nav buttons ─────────────────────────────────── */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
        border-radius: 7px !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        padding: 8px 12px !important;
        text-align: left !important;
        width: 100% !important;
        justify-content: flex-start !important;
        transition: all 0.15s !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #1e2533 !important;
        color: #f1f5f9 !important;
    }

    /* ── Main header ─────────────────────────────────────────── */
    .page-header { margin-bottom: 16px; }
    .page-title  { font-size: 22px !important; font-weight: 600; color: var(--text) !important; }
    .page-sub    { color: var(--text-sub) !important; font-size: 14px !important; margin-top: 4px; }

    /* ── Cards ───────────────────────────────────────────────── */
    .card {
        background: var(--bg2); border-radius: var(--radius);
        border: 0.5px solid var(--border); padding: 16px;
        box-shadow: var(--shadow);
    }
    .card-title {
        font-size: 14px !important; font-weight: 500; color: var(--text) !important;
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
    .metric-label { font-size: 12px !important; font-weight: 500; margin-bottom: 6px; }
    .metric-card.blue   .metric-label { color: #1e40af !important; }
    .metric-card.green  .metric-label { color: #166534 !important; }
    .metric-card.yellow .metric-label { color: #92400e !important; }
    .metric-card.red    .metric-label { color: #991b1b !important; }
    .metric-card.indigo .metric-label { color: #3730a3 !important; }
    .metric-value {
        font-size: 22px !important; font-weight: 700; color: var(--text) !important;
        font-family: var(--mono); line-height: 1;
    }
    .metric-unit { font-size: 12px !important; color: var(--text-muted) !important; margin-top: 5px; }

    /* ── Callout boxes ───────────────────────────────────────── */
    .callout {
        background: var(--blue-light); border-left: 3px solid var(--blue);
        border-radius: 0 var(--radius) var(--radius) 0;
        padding: 11px 14px; font-size: 14px !important;
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
    .section-title { font-size: 16px !important; font-weight: 600; color: var(--text) !important; margin: 0; }
    .section-desc  { font-size: 13px !important; color: var(--text-sub) !important; margin-top: 2px; }

    /* ── Control panel ───────────────────────────────────────── */
    .control-panel {
        background: var(--bg2); border: 0.5px solid var(--border);
        border-radius: var(--radius); padding: 14px 16px; margin-bottom: 12px;
    }
    .control-title {
        font-size: 12px !important; font-weight: 600; color: var(--text-sub) !important;
        text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 10px;
    }

    /* ── Streamlit widgets ───────────────────────────────────── */
    .stSlider > label, .stSelectbox > label, .stMultiSelect > label,
    .stRadio > label, .stCheckbox > label, .stTextInput > label {
        color: var(--text-sub) !important;
        font-size: 14px !important; font-weight: 500 !important;
    }
    .stRadio div[role="radiogroup"] label,
    .stCheckbox label {
        color: var(--text) !important; font-size: 14px !important;
    }
    .stSelectbox div[data-baseweb="select"] span,
    .stMultiSelect div[data-baseweb="select"] span {
        color: var(--text) !important; font-size: 14px !important;
    }
    /* Main area buttons (non-sidebar) */
    .main .stButton > button,
    [data-testid="stMainBlockContainer"] .stButton > button {
        background: var(--blue) !important; color: #ffffff !important;
        border: none !important; border-radius: 8px !important;
        font-weight: 500 !important; font-size: 14px !important;
        font-family: var(--font) !important;
        padding: 9px 18px !important; transition: all 0.15s !important;
    }
    .main .stButton > button:hover { background: #1d4ed8 !important; }

    /* ── Leaderboard ─────────────────────────────────────────── */
    .lb-wrap { background: var(--bg2); border: 0.5px solid var(--border);
               border-radius: var(--radius); overflow: hidden; }
    .lb-head { padding: 12px 16px; border-bottom: 0.5px solid var(--border);
               font-weight: 500; font-size: 14px !important; color: var(--text) !important; }
    .lb-row  { display: flex; align-items: center; gap: 10px; padding: 10px 16px;
               border-bottom: 0.5px solid var(--border-light);
               font-size: 13px !important; transition: background 0.1s; }
    .lb-row:last-child { border-bottom: none; }
    .lb-row:hover { background: var(--bg3); }
    .lb-rank  { width: 22px; font-family: var(--mono); font-size: 12px !important; color: var(--text-muted) !important; }
    .lb-rank.gold   { color: #d97706 !important; font-weight: 700; }
    .lb-rank.silver { color: #64748b !important; font-weight: 700; }
    .lb-rank.bronze { color: #b45309 !important; font-weight: 700; }
    .lb-team  { flex: 1; font-weight: 500; color: var(--text) !important; }
    .lb-event { font-size: 12px !important; color: var(--text-muted) !important; flex: 1; }
    .lb-score { font-family: var(--mono); font-weight: 600;
                font-size: 14px !important; color: var(--green) !important; }
    .lb-bar-wrap { width: 70px; }
    .lb-bar-bg   { background: var(--bg3); border-radius: 999px; height: 4px; }
    .lb-bar-fill { background: var(--green); border-radius: 999px; height: 4px; }

    /* ── Map ─────────────────────────────────────────────────── */
    .map-frame { border: 0.5px solid var(--border); border-radius: var(--radius); overflow: hidden; }

    /* ── Chips ───────────────────────────────────────────────── */
    .chip { display: inline-block; padding: 3px 9px; border-radius: 999px;
            font-size: 12px !important; font-weight: 500; }
    .chip-blue   { background: var(--blue-light);   color: var(--blue) !important; }
    .chip-green  { background: var(--green-light);  color: var(--green) !important; }
    .chip-yellow { background: var(--yellow-light); color: var(--yellow) !important; }
    .chip-red    { background: var(--red-light);    color: var(--red) !important; }

    hr { border-color: var(--border) !important; }
    ::-webkit-scrollbar       { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg3); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
    </style>
    """, unsafe_allow_html=True)
