"""Global CSS — Cupertino Editorial Precision design system (MD3 tokens)."""
import streamlit as st


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* ══════════════════════════════════════════════════════════
       Material Design 3 color tokens
       ══════════════════════════════════════════════════════════ */
    :root {
        /* ── MD3 palette ── */
        --primary:                    #0058bc;
        --primary-container:          #0070eb;
        --surface:                    #faf9fe;
        --surface-container-low:      #f4f3f8;
        --surface-container-lowest:   #ffffff;
        --surface-container-high:     #e9e7ed;
        --surface-container-highest:  #e3e2e7;
        --on-surface:                 #1a1b1f;
        --on-surface-variant:         #414755;
        --secondary:                  #405e96;
        --outline-variant:            #c1c6d7;
        --error:                      #ba1a1a;
        --tertiary:                   #9e3d00;

        /* ── Backward-compatible aliases ── */
        --bg:           var(--surface);
        --bg2:          var(--surface-container-lowest);
        --bg3:          var(--surface-container-low);
        --text:         var(--on-surface);
        --text-sub:     var(--on-surface-variant);
        --text-muted:   #717786;
        --blue:         var(--primary);
        --blue-light:   #d8e2ff;
        --cyan:         #0891b2;
        --teal:         #0d9488;
        --green:        #16a34a;
        --green-light:  #dcfce7;
        --yellow:       #d97706;
        --yellow-light: #fef9c3;
        --red:          var(--error);
        --red-light:    #ffdad6;
        --indigo:       #4f46e5;
        --indigo-light: #e0e0ff;
        --border:       var(--outline-variant);
        --border-light: var(--surface-container-low);

        /* ── Sidebar palette ── */
        --sidebar-bg:    #020617;
        --sb-text:       #94a3b8;
        --sb-text-bright:#f1f5f9;
        --sb-text-dim:   #64748b;
        --sb-card:       rgba(255,255,255,0.06);
        --sb-active:     rgba(255,255,255,0.10);

        /* ── Typography ── */
        --font:   'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --mono:   'JetBrains Mono', ui-monospace, monospace;
        --radius: 10px;
        --radius-xl: 2rem;
        --shadow: 0 1px 2px rgba(0,0,0,0.04);
        /* ── Font scale ── */
        --fs-xs:   11px;
        --fs-sm:   13px;
        --fs-base: 15px;
        --fs-md:   17px;
        --fs-lg:   20px;
        --fs-xl:   24px;
        --fs-2xl:  32px;
        --fs-3xl:  40px;
    }

    /* ══════════════════════════════════════════════════════════
       Hide default Streamlit header
       ══════════════════════════════════════════════════════════ */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* ══════════════════════════════════════════════════════════
       Global base
       ══════════════════════════════════════════════════════════ */
    html, body, [class*="css"] {
        font-family: var(--font) !important;
        font-size: var(--fs-base) !important;
        color: var(--on-surface) !important;
    }
    .stApp { background: var(--surface) !important; }
    .main .block-container { padding-top: 1.2rem; max-width: 1400px; }

    /* Force all text to on-surface */
    p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6 {
        color: var(--on-surface) !important;
    }

    /* ══════════════════════════════════════════════════════════
       Editorial typography — negative tracking on headlines
       ══════════════════════════════════════════════════════════ */
    h1, h2, h3 {
        letter-spacing: -0.04em !important;
        font-weight: 700 !important;
    }

    /* ══════════════════════════════════════════════════════════
       Bento grid utility
       ══════════════════════════════════════════════════════════ */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(12, 1fr);
        gap: 1.5rem;
    }
    .bento-span-1  { grid-column: span 1; }
    .bento-span-2  { grid-column: span 2; }
    .bento-span-3  { grid-column: span 3; }
    .bento-span-4  { grid-column: span 4; }
    .bento-span-5  { grid-column: span 5; }
    .bento-span-6  { grid-column: span 6; }
    .bento-span-7  { grid-column: span 7; }
    .bento-span-8  { grid-column: span 8; }
    .bento-span-9  { grid-column: span 9; }
    .bento-span-10 { grid-column: span 10; }
    .bento-span-11 { grid-column: span 11; }
    .bento-span-12 { grid-column: span 12; }

    /* ══════════════════════════════════════════════════════════
       Global dropdown / popover / overlay fix
       ══════════════════════════════════════════════════════════ */
    ul[role="listbox"],
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    div[data-baseweb="select"] ul,
    div[data-baseweb="popover"] > div,
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="select-dropdown"],
    [role="listbox"] {
        background: var(--surface-container-lowest) !important;
        border: 1px solid var(--outline-variant) !important;
    }
    ul[role="listbox"] li,
    div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] li,
    div[data-baseweb="popover"] div[role="option"],
    [role="option"] {
        color: var(--on-surface) !important;
        background: var(--surface-container-lowest) !important;
    }
    ul[role="listbox"] li:hover,
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="menu"] li:hover,
    div[data-baseweb="popover"] div[role="option"]:hover,
    [role="option"]:hover {
        background: var(--surface-container-low) !important;
    }
    ul[role="listbox"] li[aria-selected="true"],
    [role="option"][aria-selected="true"] {
        background: var(--blue-light) !important;
        color: var(--primary) !important;
    }
    /* baseweb select input */
    div[data-baseweb="select"] {
        background: var(--surface-container-lowest) !important;
        border-color: var(--outline-variant) !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: var(--on-surface) !important;
    }
    div[data-baseweb="select"] > div {
        background: var(--surface-container-lowest) !important;
    }
    div[data-baseweb="select"] [data-baseweb="popover"] {
        background: var(--surface-container-lowest) !important;
    }

    /* ══════════════════════════════════════════════════════════
       Glass sidebar (dark bg-slate-950)
       ══════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        border-right: none !important;
    }
    /* Sidebar collapse/expand toggle */
    [data-testid="stSidebar"] button[kind="header"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebar"] [data-testid="baseButton-header"] {
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] button[kind="header"] svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebar"] [data-testid="baseButton-header"] svg {
        fill: #f1f5f9 !important; stroke: #f1f5f9 !important;
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

    /* Sidebar widget labels & values */
    [data-testid="stSidebar"] .stSlider > label,
    [data-testid="stSidebar"] .stSelectbox > label,
    [data-testid="stSidebar"] .stMultiSelect > label,
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stCheckbox > label,
    [data-testid="stSidebar"] .stTextInput > label { color: #94a3b8 !important; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
    [data-testid="stSidebar"] .stCheckbox label { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] div[data-baseweb="select"],
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: var(--sb-card) !important;
        border-color: rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] span { color: #f1f5f9 !important; }
    [data-testid="stSidebar"] .stTextInput input {
        color: #f1f5f9 !important;
        background: var(--sb-card) !important;
        border-color: rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }
    /* Sidebar dropdown options (overrides global) */
    [data-testid="stSidebar"] ul[role="listbox"],
    [data-testid="stSidebar"] div[data-baseweb="popover"],
    [data-testid="stSidebar"] div[data-baseweb="popover"] > div,
    [data-testid="stSidebar"] div[data-baseweb="menu"],
    [data-testid="stSidebar"] [role="listbox"],
    [data-testid="stSidebar"] [data-baseweb="select-dropdown"] {
        background: #1e293b !important;
        border-color: rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] ul[role="listbox"] li,
    [data-testid="stSidebar"] div[data-baseweb="popover"] li,
    [data-testid="stSidebar"] div[data-baseweb="menu"] li,
    [data-testid="stSidebar"] [role="option"] {
        color: #f1f5f9 !important;
        background: #1e293b !important;
    }
    [data-testid="stSidebar"] ul[role="listbox"] li:hover,
    [data-testid="stSidebar"] div[data-baseweb="popover"] li:hover,
    [data-testid="stSidebar"] div[data-baseweb="menu"] li:hover,
    [data-testid="stSidebar"] [role="option"]:hover {
        background: #334155 !important;
    }
    [data-testid="stSidebar"] ul[role="listbox"] li[aria-selected="true"],
    [data-testid="stSidebar"] [role="option"][aria-selected="true"] {
        background: var(--primary) !important;
    }

    /* ── Sidebar branding ── */
    .sidebar-top {
        padding: 18px 14px 14px;
        margin-bottom: 4px;
    }
    .sidebar-logo-row {
        display: flex; align-items: center; gap: 12px; margin-bottom: 10px;
    }
    .sidebar-logo-img {
        width: 44px; height: 44px; border-radius: 12px; object-fit: cover;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .sidebar-brand { line-height: 1.3; }
    .sidebar-brand-main {
        color: var(--sb-text-bright) !important;
        font-weight: 700; font-size: var(--fs-lg) !important;
        letter-spacing: -0.02em;
    }
    .sidebar-brand-sub {
        color: var(--sb-text) !important;
        font-size: var(--fs-sm) !important;
        letter-spacing: 0.06em; text-transform: uppercase;
    }
    /* Section labels */
    .sidebar-section-label {
        padding: 12px 14px 5px;
        font-size: 10px !important; color: var(--sb-text-dim) !important;
        letter-spacing: 0.12em; text-transform: uppercase; font-weight: 600;
        text-align: left !important;
    }
    .event-badge {
        background: var(--sb-card); border-radius: 12px;
        padding: 10px 12px; margin: 6px 0;
        text-align: left !important;
    }
    .event-badge-name { color: var(--sb-text-bright) !important; font-weight: 500; font-size: var(--fs-base) !important; text-align: left !important; }
    .event-badge-sub  { color: var(--sb-text) !important; font-size: var(--fs-xs) !important; margin-top: 3px; text-align: left !important; }

    /* Sidebar global left-align */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { text-align: left !important; }
    [data-testid="stSidebar"] .stMarkdown { text-align: left !important; }

    /* ── Sidebar nav buttons (rounded-xl active items) ── */
    [data-testid="stSidebar"] .stButton {
        text-align: left !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: var(--sb-text) !important;
        border: none !important;
        border-radius: 12px !important;
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
        background: var(--sb-active) !important;
        color: var(--sb-text-bright) !important;
    }

    /* ══════════════════════════════════════════════════════════
       Hero header
       ══════════════════════════════════════════════════════════ */
    .hero-header { margin-bottom: 24px; }
    .hero-overline {
        font-size: 10px !important; font-weight: 600;
        color: var(--secondary) !important;
        text-transform: uppercase; letter-spacing: 0.12em;
        margin-bottom: 6px;
    }
    .hero-title {
        font-size: var(--fs-3xl) !important; font-weight: 900 !important;
        color: var(--on-surface) !important;
        letter-spacing: -0.04em !important; line-height: 1.1 !important;
        margin: 0 !important;
    }
    .hero-subtitle {
        font-size: var(--fs-md) !important; font-weight: 400;
        color: var(--on-surface-variant) !important;
        margin-top: 8px;
    }

    /* ── Page header (backward compat) ── */
    .page-header { margin-bottom: 16px; }
    .page-title  {
        font-size: var(--fs-lg) !important; font-weight: 700;
        color: var(--on-surface) !important; letter-spacing: -0.04em;
    }
    .page-sub {
        color: var(--on-surface-variant) !important;
        font-size: var(--fs-base) !important; margin-top: 4px;
    }
    .header-logo {
        width: 36px; height: 36px; border-radius: 10px;
        object-fit: cover; vertical-align: middle; margin-right: 10px;
    }
    .section-label-badge {
        font-size: 10px !important; font-weight: 600;
        color: var(--secondary) !important; margin-bottom: 4px;
        letter-spacing: 0.10em; text-transform: uppercase;
    }

    /* ══════════════════════════════════════════════════════════
       Cards — surface hierarchy (no borders, background shifts)
       ══════════════════════════════════════════════════════════ */
    .card-title {
        font-size: var(--fs-base) !important; font-weight: 600;
        color: var(--on-surface) !important;
        margin-bottom: 14px;
    }

    /* ── Container styling (st.container(border=True)) ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface-container-lowest) !important;
        border: none !important;
        border-radius: var(--radius-xl) !important;
        box-shadow: var(--shadow);
    }
    .control-title {
        font-size: 10px !important; font-weight: 600;
        color: var(--on-surface-variant) !important;
        text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 10px;
    }

    /* ══════════════════════════════════════════════════════════
       Metric cards — rounded-[2rem]
       ══════════════════════════════════════════════════════════ */
    .metric-row  { display: flex; gap: 12px; margin: 14px 0; flex-wrap: wrap; }
    .metric-card {
        border-radius: var(--radius-xl); padding: 16px 18px;
        min-width: 130px; flex: 1;
    }
    .metric-card.blue   { background: var(--blue-light); }
    .metric-card.green  { background: var(--green-light); }
    .metric-card.yellow { background: var(--yellow-light); }
    .metric-card.red    { background: var(--red-light); }
    .metric-card.indigo { background: var(--indigo-light); }
    .metric-label {
        font-size: var(--fs-xs) !important; font-weight: 600; margin-bottom: 6px;
        text-transform: uppercase; letter-spacing: 0.04em;
    }
    .metric-card.blue   .metric-label { color: #004493 !important; }
    .metric-card.green  .metric-label { color: #166534 !important; }
    .metric-card.yellow .metric-label { color: #92400e !important; }
    .metric-card.red    .metric-label { color: #93000a !important; }
    .metric-card.indigo .metric-label { color: #3730a3 !important; }
    .metric-value {
        font-size: var(--fs-xl) !important; font-weight: 700;
        color: var(--on-surface) !important;
        font-family: var(--mono); line-height: 1;
    }
    .metric-unit {
        font-size: var(--fs-xs) !important; color: var(--text-muted) !important; margin-top: 5px;
    }

    /* ══════════════════════════════════════════════════════════
       Callout boxes — rounded ends
       ══════════════════════════════════════════════════════════ */
    .callout {
        background: var(--blue-light); border-left: 3px solid var(--primary);
        border-radius: 0 var(--radius-xl) var(--radius-xl) 0;
        padding: 14px 18px; font-size: var(--fs-base) !important;
        color: var(--on-surface-variant) !important; margin: 12px 0;
    }
    .callout.warn   { background: var(--yellow-light); border-left-color: var(--yellow); }
    .callout.good   { background: var(--green-light);  border-left-color: var(--green); }
    .callout.danger { background: var(--red-light);    border-left-color: var(--error); }
    .callout strong { color: var(--on-surface) !important; }

    /* ══════════════════════════════════════════════════════════
       Section header
       ══════════════════════════════════════════════════════════ */
    .section-header { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
    .section-icon {
        width: 36px; height: 36px; border-radius: 12px;
        background: var(--blue-light);
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; flex-shrink: 0;
    }
    .section-title {
        font-size: var(--fs-md) !important; font-weight: 700;
        color: var(--on-surface) !important; margin: 0;
        letter-spacing: -0.02em;
    }
    .section-desc {
        font-size: var(--fs-sm) !important; color: var(--on-surface-variant) !important; margin-top: 2px;
    }

    /* ══════════════════════════════════════════════════════════
       Leaderboard
       ══════════════════════════════════════════════════════════ */
    .lb-wrap {
        background: var(--surface-container-lowest);
        border-radius: var(--radius-xl); overflow: hidden;
        box-shadow: var(--shadow);
    }
    .lb-head {
        padding: 14px 18px;
        background: var(--surface-container-low);
        font-weight: 600; font-size: var(--fs-base) !important;
        color: var(--on-surface) !important;
    }
    .lb-row  {
        display: flex; align-items: center; gap: 10px; padding: 10px 18px;
        font-size: var(--fs-sm) !important; transition: background 0.1s;
    }
    .lb-row:nth-child(even) { background: var(--surface-container-low); }
    .lb-row:hover { background: var(--surface-container-high); }
    .lb-rank  { width: 22px; font-family: var(--mono); font-size: var(--fs-xs) !important; color: var(--text-muted) !important; }
    .lb-rank.gold   { color: var(--tertiary) !important; font-weight: 700; }
    .lb-rank.silver { color: #64748b !important; font-weight: 700; }
    .lb-rank.bronze { color: #b45309 !important; font-weight: 700; }
    .lb-team  { flex: 1; font-weight: 500; color: var(--on-surface) !important; }
    .lb-event { font-size: var(--fs-xs) !important; color: var(--text-muted) !important; flex: 1; }
    .lb-score { font-family: var(--mono); font-weight: 600;
                font-size: var(--fs-base) !important; color: var(--green) !important; }
    .lb-bar-wrap { width: 70px; }
    .lb-bar-bg   { background: var(--surface-container-high); border-radius: 999px; height: 4px; }
    .lb-bar-fill { background: var(--primary); border-radius: 999px; height: 4px; }

    /* ══════════════════════════════════════════════════════════
       Footer metadata bar with animated dot
       ══════════════════════════════════════════════════════════ */
    .footer-bar {
        display: flex; align-items: center; gap: 12px;
        padding: 12px 0; margin-top: 32px;
        font-size: 10px !important; font-weight: 500;
        color: var(--text-muted) !important;
        text-transform: uppercase; letter-spacing: 0.12em;
    }
    .footer-dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #16a34a;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.3; }
    }

    /* ══════════════════════════════════════════════════════════
       Map frame
       ══════════════════════════════════════════════════════════ */
    .map-frame {
        border: none;
        border-radius: var(--radius-xl); overflow: hidden;
        box-shadow: var(--shadow);
    }

    /* ══════════════════════════════════════════════════════════
       Chips
       ══════════════════════════════════════════════════════════ */
    .chip { display: inline-block; padding: 4px 10px; border-radius: 999px;
            font-size: var(--fs-xs) !important; font-weight: 500; }
    .chip-blue   { background: var(--blue-light);   color: var(--primary) !important; }
    .chip-green  { background: var(--green-light);  color: var(--green) !important; }
    .chip-yellow { background: var(--yellow-light); color: var(--yellow) !important; }
    .chip-red    { background: var(--red-light);    color: var(--error) !important; }

    /* ══════════════════════════════════════════════════════════
       Hyperparameter guide panel
       ══════════════════════════════════════════════════════════ */
    .hp-guide {
        background: var(--surface-container-low);
        border-radius: var(--radius-xl); padding: 18px 20px;
    }
    .hp-guide-title {
        font-size: var(--fs-base) !important; font-weight: 700;
        color: var(--on-surface) !important; margin-bottom: 12px;
        letter-spacing: -0.02em;
    }
    .hp-section-label {
        font-size: 10px !important; font-weight: 600;
        color: var(--text-muted) !important; text-transform: uppercase;
        letter-spacing: 0.10em; margin: 14px 0 8px; padding-bottom: 4px;
        border-bottom: 1px solid var(--surface-container-high);
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
    .hp-name { font-weight: 600; color: var(--on-surface) !important; white-space: nowrap; }
    .hp-desc { color: var(--on-surface-variant) !important; }
    .hp-tip  { color: var(--primary) !important; font-size: var(--fs-xs) !important; }

    /* ══════════════════════════════════════════════════════════
       Flappy Bird stage badges
       ══════════════════════════════════════════════════════════ */
    .stage-row { display: flex; gap: 8px; margin: 14px 0; flex-wrap: wrap; }
    .stage-badge {
        padding: 8px 14px; border-radius: 999px;
        font-size: var(--fs-sm) !important; font-weight: 500;
        border: none; cursor: default;
        display: flex; align-items: center; gap: 6px;
    }
    .stage-badge.locked    { background: var(--surface-container-high); color: var(--text-muted) !important; opacity: 0.6; }
    .stage-badge.available { background: var(--blue-light); color: var(--primary) !important; }
    .stage-badge.passed    { background: var(--green-light); color: var(--green) !important; }
    .stage-badge.active    { background: var(--yellow-light); color: var(--yellow) !important; }

    /* ══════════════════════════════════════════════════════════
       Streamlit widgets
       ══════════════════════════════════════════════════════════ */
    .stSlider > label, .stSelectbox > label, .stMultiSelect > label,
    .stRadio > label, .stCheckbox > label, .stTextInput > label {
        color: var(--on-surface-variant) !important;
        font-size: var(--fs-base) !important; font-weight: 500 !important;
    }
    .stRadio div[role="radiogroup"] label,
    .stCheckbox label {
        color: var(--on-surface) !important; font-size: var(--fs-base) !important;
    }
    .stSelectbox div[data-baseweb="select"] span,
    .stMultiSelect div[data-baseweb="select"] span {
        color: var(--on-surface) !important; font-size: var(--fs-base) !important;
    }
    /* Main area buttons */
    .main .stButton > button,
    [data-testid="stMainBlockContainer"] .stButton > button {
        background: var(--primary) !important; color: #ffffff !important;
        border: none !important; border-radius: 12px !important;
        font-weight: 500 !important; font-size: var(--fs-base) !important;
        font-family: var(--font) !important;
        padding: 9px 18px !important; transition: all 0.15s !important;
    }
    .main .stButton > button:hover {
        background: #004493 !important;
    }

    /* ══════════════════════════════════════════════════════════
       Misc
       ══════════════════════════════════════════════════════════ */
    hr { border-color: var(--surface-container-high) !important; }
    ::-webkit-scrollbar       { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--surface-container-low); }
    ::-webkit-scrollbar-thumb { background: var(--outline-variant); border-radius: 2px; }
    </style>
    """, unsafe_allow_html=True)


COLORS = {
    "text": "#1a1b1f",
    "text_sub": "#414755",
    "text_muted": "#717786",
    "blue": "#0058bc",
    "blue_light": "#d8e2ff",
    "blue_dark": "#004493",
    "cyan": "#06b6d4",
    "green": "#16a34a",
    "green_light": "#dcfce7",
    "red": "#ba1a1a",
    "red_dark": "#93000a",
    "yellow": "#d97706",
    "indigo": "#4f46e5",
    "indigo_dark": "#3730a3",
    "bg": "#faf9fe",
    "bg3": "#f4f3f8",
    "border": "#c1c6d7",
    "axis": "#717786",
    "bar_light": "#adc6ff",
    "bar_flood": "#ba1a1a",
}
