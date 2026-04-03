# Cupertino Design System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the "Editorial Precision" / Cupertino-inspired design system from the stitch HTML mockups to the existing Streamlit EarthAI app, matching the visual language as closely as Streamlit allows.

**Architecture:** Rewrite `utils/styles.py` with the full Material Design 3 color token system and bento-grid aesthetic. Update `app.py` sidebar/header to match the dark glass sidebar with EarthAI branding. Update each module's HTML templates to use hero headers, rounded bento cards, and editorial typography. Add a custom HTML top bar via `st.components.v1.html()` for profile/search/notifications.

**Tech Stack:** Streamlit, CSS custom properties, Material Symbols Outlined font, Inter font, `st.markdown(unsafe_allow_html=True)`, `st.components.v1.html()`

---

### Task 1: Rewrite `utils/styles.py` — Full CSS overhaul

**Files:**
- Modify: `utils/styles.py`

This is the foundation — all other tasks depend on this completing first.

- [ ] **Step 1: Replace the CSS root variables and global styles**

Replace the entire `inject_css()` function with the new design system. The new CSS must include:

1. Material Design 3 color tokens matching the stitch HTML files
2. Surface hierarchy system (surface, surface-container-low, surface-container-lowest, etc.)
3. Editorial typography with negative tracking on headlines
4. Material Symbols Outlined font import
5. Bento grid utility class
6. Glass sidebar styling
7. Updated metric cards with 2rem border-radius
8. Updated callout boxes
9. Updated leaderboard styles
10. Hero header styles
11. Top bar styles
12. Footer metadata styles
13. No-line rule: borders replaced by background shifts where possible

Key color mappings from stitch HTML → CSS custom properties:
- `--primary: #0058bc` (was `#2563eb`)
- `--primary-container: #0070eb`
- `--surface: #faf9fe` (was `#f8fafc`)
- `--surface-container-low: #f4f3f8`
- `--surface-container-lowest: #ffffff`
- `--surface-container-high: #e9e7ed`
- `--surface-container-highest: #e3e2e7`
- `--on-surface: #1a1b1f`
- `--on-surface-variant: #414755`
- `--secondary: #405e96`
- `--outline-variant: #c1c6d7`
- `--error: #ba1a1a`
- `--tertiary: #9e3d00`

```python
"""Global CSS — Cupertino Editorial Precision design system."""
import streamlit as st


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

    :root {
        /* ── Material Design 3 Color Tokens ── */
        --primary:                 #0058bc;
        --primary-container:       #0070eb;
        --on-primary:              #ffffff;
        --on-primary-container:    #fefcff;
        --secondary:               #405e96;
        --secondary-container:     #a1befd;
        --on-secondary:            #ffffff;
        --tertiary:                #9e3d00;
        --tertiary-container:      #c64f00;
        --error:                   #ba1a1a;
        --error-container:         #ffdad6;
        --on-error:                #ffffff;

        /* ── Surface Hierarchy ── */
        --surface:                 #faf9fe;
        --surface-dim:             #dad9df;
        --surface-bright:          #faf9fe;
        --surface-container-lowest:#ffffff;
        --surface-container-low:   #f4f3f8;
        --surface-container:       #eeedf3;
        --surface-container-high:  #e9e7ed;
        --surface-container-highest:#e3e2e7;
        --surface-variant:         #e3e2e7;

        /* ── Text ── */
        --on-surface:              #1a1b1f;
        --on-surface-variant:      #414755;
        --on-background:           #1a1b1f;
        --outline:                 #717786;
        --outline-variant:         #c1c6d7;
        --inverse-surface:         #2f3034;
        --inverse-on-surface:      #f1f0f5;
        --inverse-primary:         #adc6ff;

        /* ── Semantic ── */
        --emerald:                 #10b981;
        --amber:                   #f59e0b;

        /* ── Legacy compat aliases ── */
        --bg:                      var(--surface);
        --bg2:                     var(--surface-container-lowest);
        --bg3:                     var(--surface-container-low);
        --text:                    var(--on-surface);
        --text-sub:                var(--on-surface-variant);
        --text-muted:              var(--outline);
        --blue:                    var(--primary);
        --blue-light:              #d8e2ff;
        --border:                  var(--outline-variant);
        --border-light:            var(--surface-container-low);
        --red:                     var(--error);
        --red-light:               var(--error-container);
        --green:                   #16a34a;
        --green-light:             #f0fdf4;
        --yellow:                  #d97706;
        --yellow-light:            #fefce8;
        --indigo:                  #4f46e5;
        --indigo-light:            #eef2ff;
        --cyan:                    #0891b2;
        --teal:                    #0d9488;

        /* ── Typography ── */
        --font:                    'Inter', -apple-system, sans-serif;
        --mono:                    'JetBrains Mono', ui-monospace, monospace;
        --fs-xs:                   11px;
        --fs-sm:                   13px;
        --fs-base:                 15px;
        --fs-md:                   17px;
        --fs-lg:                   20px;
        --fs-xl:                   24px;
        --fs-2xl:                  30px;
        --fs-3xl:                  36px;

        /* ── Shape ── */
        --radius:                  10px;
        --radius-lg:               16px;
        --radius-xl:               24px;
        --radius-2xl:              32px;
        --radius-full:             9999px;

        /* ── Sidebar ── */
        --sb-bg:                   #0a0a0f;
        --sb-text:                 #94a3b8;
        --sb-text-bright:          #f1f5f9;
        --sb-text-dim:             #64748b;
        --sb-card:                 rgba(255,255,255,0.1);
        --sb-card-hover:           rgba(255,255,255,0.05);
    }

    /* ── Material Symbols ── */
    .material-symbols-outlined {
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
    }

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: var(--font) !important;
        font-size: var(--fs-base) !important;
        color: var(--on-surface) !important;
    }
    .stApp { background: var(--surface) !important; }
    .main .block-container {
        padding-top: 0 !important;
        max-width: 1280px;
    }

    /* Force all text dark */
    p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6 {
        color: var(--on-surface) !important;
    }

    /* ── No-Line Rule: borders → background shifts ── */
    hr { display: none !important; }

    /* ── Negative tracking utility ── */
    .neg-track { letter-spacing: -0.04em; }

    /* ── Bento grid ── */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(12, 1fr);
        gap: 1.5rem;
    }

    /* ── Top App Bar (custom HTML component) ── */
    .earthai-topbar {
        position: sticky; top: 0; z-index: 999;
        background: rgba(255,255,255,0.80);
        backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
        display: flex; justify-content: space-between; align-items: center;
        padding: 0 2rem; height: 56px;
        border-bottom: 1px solid rgba(193,198,215,0.2);
        font-family: 'Inter', sans-serif;
    }
    .earthai-topbar-title {
        font-weight: 700; font-size: 18px;
        letter-spacing: -0.04em; color: #1a1b1f;
    }
    .earthai-topbar-right {
        display: flex; align-items: center; gap: 12px;
    }
    .earthai-topbar-search {
        background: var(--surface-container-low);
        border: none; border-radius: 9999px;
        padding: 6px 16px 6px 36px;
        font-size: 13px; width: 220px;
        outline: none; transition: all 0.15s;
    }
    .earthai-topbar-search:focus {
        box-shadow: 0 0 0 3px rgba(0,88,188,0.12);
    }
    .earthai-topbar-btn {
        width: 36px; height: 36px; border-radius: 50%;
        border: none; background: transparent; cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        color: #717786; transition: background 0.15s;
    }
    .earthai-topbar-btn:hover { background: rgba(0,0,0,0.04); }
    .earthai-topbar-user {
        display: flex; align-items: center; gap: 8px;
        padding-left: 12px; border-left: 1px solid #e3e2e7;
    }
    .earthai-topbar-username {
        font-size: 13px; font-weight: 700; color: #1a1b1f;
    }
    .earthai-topbar-avatar {
        width: 32px; height: 32px; border-radius: 50%;
        background: var(--primary); color: white;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 13px;
    }

    /* ── Hero Header ── */
    .hero-header { margin-bottom: 2.5rem; }
    .hero-overline {
        color: var(--secondary) !important;
        font-weight: 600; font-size: var(--fs-sm) !important;
        letter-spacing: 0.15em; text-transform: uppercase;
        margin-bottom: 6px;
    }
    .hero-title {
        font-size: clamp(2rem, 4vw, 3rem) !important;
        font-weight: 900 !important;
        letter-spacing: -0.04em !important;
        color: var(--on-surface) !important;
        line-height: 1.1 !important;
        margin: 0 !important;
    }
    .hero-subtitle {
        color: var(--on-surface-variant) !important;
        font-size: var(--fs-md) !important;
        font-weight: 500;
        opacity: 0.8;
        margin-top: 10px;
    }

    /* ── Section header (updated) ── */
    .section-header {
        display: flex; align-items: center; gap: 12px;
        margin-bottom: 20px;
    }
    .section-icon {
        width: 40px; height: 40px; border-radius: 12px;
        background: var(--blue-light);
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; flex-shrink: 0;
        color: var(--primary) !important;
    }
    .section-title {
        font-size: var(--fs-lg) !important; font-weight: 700;
        color: var(--on-surface) !important;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .section-desc {
        font-size: var(--fs-sm) !important;
        color: var(--on-surface-variant) !important;
        margin-top: 2px;
    }

    /* ── Global dropdown fix ── */
    ul[role="listbox"],
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    div[data-baseweb="select"] ul,
    div[data-baseweb="popover"] > div,
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="select-dropdown"],
    [role="listbox"] {
        background: #ffffff !important;
        border: 1px solid var(--outline-variant) !important;
        border-radius: 16px !important;
    }
    ul[role="listbox"] li,
    div[data-baseweb="popover"] li,
    div[data-baseweb="menu"] li,
    div[data-baseweb="popover"] div[role="option"],
    [role="option"] {
        color: var(--on-surface) !important;
        background: #ffffff !important;
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
    div[data-baseweb="select"] {
        background: #ffffff !important;
        border-color: var(--outline-variant) !important;
        border-radius: 12px !important;
    }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {
        color: var(--on-surface) !important;
    }
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border-radius: 12px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--sb-bg) !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    [data-testid="stSidebar"] button[kind="header"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebar"] [data-testid="baseButton-header"] {
        color: var(--sb-text-bright) !important;
    }
    [data-testid="stSidebar"] button[kind="header"] svg,
    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebar"] [data-testid="baseButton-header"] svg {
        fill: var(--sb-text-bright) !important;
        stroke: var(--sb-text-bright) !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: var(--sb-text) !important;
    }

    /* Sidebar brand */
    .sidebar-top {
        padding: 18px 12px 16px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 8px;
    }
    .sidebar-logo-row {
        display: flex; align-items: center; gap: 12px;
    }
    .sidebar-logo-icon {
        width: 40px; height: 40px; border-radius: 12px;
        background: var(--primary);
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .sidebar-logo-icon .material-symbols-outlined {
        color: white !important; font-size: 22px;
    }
    .sidebar-brand { line-height: 1.3; }
    .sidebar-brand-main {
        color: var(--sb-text-bright) !important;
        font-weight: 900; font-size: var(--fs-lg) !important;
        letter-spacing: -0.04em;
    }
    .sidebar-brand-sub {
        color: var(--sb-text-dim) !important;
        font-size: 10px !important;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        font-weight: 700;
    }

    /* Sidebar section labels */
    .sidebar-section-label {
        padding: 14px 12px 6px;
        font-size: var(--fs-xs) !important;
        color: var(--sb-text-dim) !important;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        font-weight: 700;
        text-align: left !important;
    }
    .event-badge {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 10px 12px; margin: 6px 0;
        text-align: left !important;
        border-left: 3px solid var(--primary);
    }
    .event-badge-name {
        color: var(--sb-text-bright) !important;
        font-weight: 600; font-size: var(--fs-base) !important;
    }
    .event-badge-sub {
        color: var(--sb-text) !important;
        font-size: var(--fs-xs) !important;
        margin-top: 3px;
    }

    /* Sidebar left-align */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { text-align: left !important; }
    [data-testid="stSidebar"] .stMarkdown { text-align: left !important; }

    /* Sidebar widget overrides */
    [data-testid="stSidebar"] .stSlider > label,
    [data-testid="stSidebar"] .stSelectbox > label,
    [data-testid="stSidebar"] .stMultiSelect > label,
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stCheckbox > label,
    [data-testid="stSidebar"] .stTextInput > label {
        color: var(--sb-text) !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"],
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] span {
        color: var(--sb-text-bright) !important;
    }
    [data-testid="stSidebar"] .stTextInput input {
        color: var(--sb-text-bright) !important;
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }
    /* Sidebar dropdown items */
    [data-testid="stSidebar"] ul[role="listbox"],
    [data-testid="stSidebar"] div[data-baseweb="popover"],
    [data-testid="stSidebar"] div[data-baseweb="popover"] > div,
    [data-testid="stSidebar"] div[data-baseweb="menu"],
    [data-testid="stSidebar"] [role="listbox"],
    [data-testid="stSidebar"] [data-baseweb="select-dropdown"] {
        background: #1a1f2e !important;
        border-color: rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
    }
    [data-testid="stSidebar"] ul[role="listbox"] li,
    [data-testid="stSidebar"] div[data-baseweb="popover"] li,
    [data-testid="stSidebar"] div[data-baseweb="menu"] li,
    [data-testid="stSidebar"] [role="option"] {
        color: var(--sb-text-bright) !important;
        background: #1a1f2e !important;
    }
    [data-testid="stSidebar"] ul[role="listbox"] li:hover,
    [data-testid="stSidebar"] div[data-baseweb="popover"] li:hover,
    [data-testid="stSidebar"] div[data-baseweb="menu"] li:hover,
    [data-testid="stSidebar"] [role="option"]:hover {
        background: #2d3748 !important;
    }
    [data-testid="stSidebar"] ul[role="listbox"] li[aria-selected="true"],
    [data-testid="stSidebar"] [role="option"][aria-selected="true"] {
        background: var(--primary) !important;
    }

    /* ── Sidebar nav buttons ── */
    [data-testid="stSidebar"] .stButton { text-align: left !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        color: var(--sb-text) !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: var(--fs-sm) !important;
        font-weight: 500 !important;
        padding: 10px 12px !important;
        text-align: left !important;
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        transition: all 0.15s !important;
        letter-spacing: -0.01em !important;
    }
    [data-testid="stSidebar"] .stButton > button > div,
    [data-testid="stSidebar"] .stButton > button > p,
    [data-testid="stSidebar"] .stButton > button > span,
    [data-testid="stSidebar"] .stButton > button * {
        text-align: left !important;
        justify-content: flex-start !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--sb-card-hover) !important;
        color: var(--sb-text-bright) !important;
    }

    /* ── Hide default Streamlit header/footer ── */
    header[data-testid="stHeader"] { display: none !important; }
    .stDeployButton { display: none !important; }

    /* ── Cards (bento style) ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface-container-lowest) !important;
        border: none !important;
        border-radius: var(--radius-2xl) !important;
        box-shadow: 0 1px 3px rgba(26,27,31,0.04) !important;
    }
    .control-title {
        font-size: 10px !important; font-weight: 700;
        color: var(--secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 12px;
    }
    .card-title {
        font-size: var(--fs-base) !important;
        font-weight: 700;
        color: var(--on-surface) !important;
        letter-spacing: -0.02em;
        margin-bottom: 14px;
    }

    /* ── Metric cards (bento) ── */
    .metric-row {
        display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap;
    }
    .metric-card {
        border-radius: var(--radius-2xl);
        padding: 1.25rem 1.5rem;
        min-width: 140px; flex: 1;
        position: relative; overflow: hidden;
    }
    .metric-card.blue   { background: var(--blue-light); }
    .metric-card.green  { background: var(--green-light); }
    .metric-card.yellow { background: var(--yellow-light); }
    .metric-card.red    { background: var(--red-light); }
    .metric-card.indigo { background: var(--indigo-light); }
    .metric-card.primary {
        background: var(--primary);
    }
    .metric-card.primary .metric-label,
    .metric-card.primary .metric-value,
    .metric-card.primary .metric-unit {
        color: white !important;
    }
    .metric-label {
        font-size: 10px !important; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.12em;
        margin-bottom: 8px;
    }
    .metric-card.blue   .metric-label { color: #004493 !important; }
    .metric-card.green  .metric-label { color: #166534 !important; }
    .metric-card.yellow .metric-label { color: #92400e !important; }
    .metric-card.red    .metric-label { color: #991b1b !important; }
    .metric-card.indigo .metric-label { color: #3730a3 !important; }
    .metric-value {
        font-size: var(--fs-2xl) !important;
        font-weight: 900;
        color: var(--on-surface) !important;
        font-family: var(--font);
        letter-spacing: -0.04em;
        line-height: 1;
    }
    .metric-unit {
        font-size: 10px !important;
        color: var(--outline) !important;
        margin-top: 6px;
        font-weight: 500;
    }

    /* ── Callout boxes ── */
    .callout {
        background: var(--surface-container-low);
        border-left: 3px solid var(--primary);
        border-radius: 0 var(--radius-xl) var(--radius-xl) 0;
        padding: 14px 18px;
        font-size: var(--fs-sm) !important;
        color: var(--on-surface-variant) !important;
        margin: 14px 0;
    }
    .callout.warn   { background: var(--yellow-light); border-left-color: var(--yellow); }
    .callout.good   { background: var(--green-light);  border-left-color: var(--green); }
    .callout.danger { background: var(--red-light);    border-left-color: var(--error); }
    .callout strong  { color: var(--on-surface) !important; }

    /* ── Leaderboard ── */
    .lb-wrap {
        background: var(--surface-container-lowest);
        border-radius: var(--radius-2xl);
        overflow: hidden;
    }
    .lb-head {
        padding: 16px 20px;
        border-bottom: 1px solid var(--surface-container-low);
        font-weight: 700;
        font-size: var(--fs-base) !important;
        color: var(--on-surface) !important;
        letter-spacing: -0.02em;
    }
    .lb-row {
        display: flex; align-items: center; gap: 12px;
        padding: 12px 20px;
        border-bottom: 1px solid var(--surface-container-low);
        font-size: var(--fs-sm) !important;
        transition: background 0.1s;
    }
    .lb-row:last-child { border-bottom: none; }
    .lb-row:hover { background: var(--surface-container-low); }
    .lb-rank {
        width: 24px; font-family: var(--font);
        font-size: var(--fs-xs) !important;
        color: var(--outline) !important;
        font-weight: 900; font-style: italic;
    }
    .lb-rank.gold   { color: #d97706 !important; }
    .lb-rank.silver { color: #64748b !important; }
    .lb-rank.bronze { color: #b45309 !important; }
    .lb-team {
        flex: 1; font-weight: 700;
        color: var(--on-surface) !important;
        letter-spacing: -0.01em;
    }
    .lb-event {
        font-size: var(--fs-xs) !important;
        color: var(--outline) !important; flex: 1;
    }
    .lb-score {
        font-family: var(--font); font-weight: 900;
        font-size: var(--fs-base) !important;
        color: var(--primary) !important;
        letter-spacing: -0.02em;
    }
    .lb-bar-wrap { width: 70px; }
    .lb-bar-bg {
        background: var(--surface-container-high);
        border-radius: var(--radius-full); height: 4px;
    }
    .lb-bar-fill {
        background: var(--primary);
        border-radius: var(--radius-full); height: 4px;
    }

    /* ── Hyperparameter guide panel ── */
    .hp-guide {
        padding: 14px 16px;
        background: var(--surface-container-low);
        border-radius: var(--radius-2xl);
    }
    .hp-guide-title {
        font-size: var(--fs-base) !important; font-weight: 700;
        color: var(--on-surface) !important;
        letter-spacing: -0.02em;
        margin-bottom: 12px;
    }
    .hp-section-label {
        font-size: 10px !important; font-weight: 700;
        color: var(--outline) !important; text-transform: uppercase;
        letter-spacing: 0.12em; margin: 14px 0 8px; padding-bottom: 4px;
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
    .hp-name { font-weight: 700; color: var(--on-surface) !important; white-space: nowrap; }
    .hp-desc { color: var(--on-surface-variant) !important; }
    .hp-tip  { color: var(--primary) !important; font-size: var(--fs-xs) !important; }

    /* ── Stage badges (Flappy) ── */
    .stage-row { display: flex; gap: 8px; margin: 14px 0; flex-wrap: wrap; }
    .stage-badge {
        padding: 10px 16px; border-radius: var(--radius-xl);
        font-size: var(--fs-sm) !important; font-weight: 600;
        border: 1px solid var(--outline-variant);
        cursor: default;
        display: flex; align-items: center; gap: 6px;
    }
    .stage-badge.locked {
        background: var(--surface-container-low);
        color: var(--outline) !important; opacity: 0.6;
    }
    .stage-badge.available {
        background: var(--blue-light);
        color: var(--primary) !important;
        border-color: var(--primary);
    }
    .stage-badge.passed {
        background: var(--green-light);
        color: var(--green) !important;
        border-color: var(--green);
    }
    .stage-badge.active {
        background: var(--yellow-light);
        color: var(--yellow) !important;
        border-color: var(--yellow);
    }

    /* ── Chips ── */
    .chip {
        display: inline-block; padding: 4px 12px;
        border-radius: var(--radius-full);
        font-size: 10px !important; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.08em;
    }
    .chip-blue   { background: var(--blue-light);   color: var(--primary) !important; }
    .chip-green  { background: var(--green-light);  color: var(--green) !important; }
    .chip-yellow { background: var(--yellow-light); color: var(--yellow) !important; }
    .chip-red    { background: var(--red-light);    color: var(--error) !important; }

    /* ── Footer metadata ── */
    .earthai-footer {
        margin-top: 3rem; padding-top: 1rem;
        display: flex; justify-content: space-between; align-items: center;
        font-size: 10px; font-weight: 700;
        color: var(--outline) !important;
        text-transform: uppercase; letter-spacing: 0.15em;
    }
    .earthai-footer-left { display: flex; gap: 2rem; }
    .earthai-footer-right {
        display: flex; align-items: center; gap: 6px;
    }
    .earthai-footer-dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: var(--emerald);
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* ── Streamlit widgets ── */
    .stSlider > label, .stSelectbox > label, .stMultiSelect > label,
    .stRadio > label, .stCheckbox > label, .stTextInput > label {
        color: var(--on-surface-variant) !important;
        font-size: var(--fs-sm) !important; font-weight: 600 !important;
    }
    .stRadio div[role="radiogroup"] label,
    .stCheckbox label {
        color: var(--on-surface) !important; font-size: var(--fs-base) !important;
    }
    /* Main area buttons */
    .main .stButton > button,
    [data-testid="stMainBlockContainer"] .stButton > button {
        background: var(--primary) !important; color: #ffffff !important;
        border: none !important; border-radius: var(--radius-xl) !important;
        font-weight: 700 !important; font-size: var(--fs-sm) !important;
        font-family: var(--font) !important;
        padding: 10px 20px !important; transition: all 0.15s !important;
        letter-spacing: -0.01em !important;
    }
    .main .stButton > button:hover {
        filter: brightness(1.1) !important;
    }

    /* ── Map ── */
    .map-frame {
        border-radius: var(--radius-2xl); overflow: hidden;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar       { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--surface-container-low); }
    ::-webkit-scrollbar-thumb { background: var(--outline-variant); border-radius: 2px; }

    /* ── Page header (legacy compat) ── */
    .page-header { display: none; }
    .section-label-badge {
        font-size: var(--fs-sm) !important; font-weight: 600;
        color: var(--primary) !important; margin-bottom: 4px; letter-spacing: 0.04em;
    }
    </style>
    """, unsafe_allow_html=True)


COLORS = {
    "text":       "#1a1b1f",
    "text_sub":   "#414755",
    "text_muted": "#717786",
    "blue":       "#0058bc",
    "blue_light": "#d8e2ff",
    "blue_dark":  "#004493",
    "cyan":       "#0891b2",
    "green":      "#16a34a",
    "green_light":"#10b981",
    "red":        "#ba1a1a",
    "red_dark":   "#93000a",
    "yellow":     "#d97706",
    "indigo":     "#4f46e5",
    "indigo_dark":"#3730a3",
    "bg":         "#faf9fe",
    "bg3":        "#f4f3f8",
    "border":     "#c1c6d7",
    "axis":       "#717786",
    "bar_light":  "#adc6ff",
    "bar_flood":  "#ba1a1a",
}
```

- [ ] **Step 2: Verify styles.py has no syntax errors**

Run: `python -c "from utils.styles import inject_css, COLORS; print('OK', len(COLORS))" `
Expected: `OK 16`

- [ ] **Step 3: Commit**

```bash
git add utils/styles.py
git commit -m "style: rewrite CSS to Cupertino Editorial Precision design system"
```

---

### Task 2: Update `app.py` — Sidebar branding and hero headers

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update sidebar branding to EarthAI with Material icon**

Replace the logo rendering and sidebar brand section. Remove the CHRS logo image, use a Material Symbols `water_drop` icon in a primary-color box instead. Change the brand text to "EarthAI" / "Planetary Intelligence".

Replace lines 26–35 (logo loading) and lines 51–63 (sidebar brand rendering):

```python
# Remove LOGO_PATH, get_logo_b64, LOGO_B64, LOGO_IMG constants entirely

# Replace sidebar brand with:
with st.sidebar:
    st.markdown("""
    <div class="sidebar-top">
        <div class="sidebar-logo-row">
            <div class="sidebar-logo-icon">
                <span class="material-symbols-outlined">water_drop</span>
            </div>
            <div class="sidebar-brand">
                <div class="sidebar-brand-main">EarthAI</div>
                <div class="sidebar-brand-sub">Planetary Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

- [ ] **Step 2: Update navigation button labels to use Material icons**

Replace the nav button format strings to embed Material Symbols spans. Change each `st.button()` call's label from emoji-prefixed to icon-prefixed using HTML in a surrounding markdown:

For each nav button section, replace the emoji with Material Symbol names:
- rainfall: `water_drop` icon → "Rainfall Analysis"
- optical: `visibility` icon → "Optical Detection"
- sar: `radar` icon → "SAR Detection"
- classifier: `psychology` icon → "AI Flood Classifier"
- flappybird: `sports_esports` icon → "Flappy Bird Competition"

Note: Streamlit buttons don't render HTML, so keep the emoji prefix for now but add a comment about this limitation. The sidebar CSS will handle the visual style.

- [ ] **Step 3: Replace the main header with a hero-style header**

Replace lines 141–161 (the page header section) with a hero header component. For event-specific pages, show an overline with the event name. For non-event pages, show a module-specific overline.

```python
# ── Hero header ──────────────────────────────────────────────────
page_info = PAGES[active]

if active not in ("classifier", "flappybird"):
    overline = f"{ev['label']} ({ev['year']}) &middot; {ev['region']}"
    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-overline">{overline}</div>
        <h1 class="hero-title">{page_info['label']}</h1>
        <div class="hero-subtitle">
            Real-time satellite intelligence and flood detection analytics.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    overlines = {
        "classifier": "Classifier Engine",
        "flappybird": "Agent Registry v4.2",
    }
    subtitles = {
        "classifier": "Multimodal spatial-temporal segmentation workbench.",
        "flappybird": "Reinforcement learning PPO simulation environment.",
    }
    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-overline">{overlines.get(active, '')}</div>
        <h1 class="hero-title">{page_info['label']}</h1>
        <div class="hero-subtitle">{subtitles.get(active, '')}</div>
    </div>
    """, unsafe_allow_html=True)
```

- [ ] **Step 4: Add a footer component function and render after page routing**

Add a footer after the page routing block:

```python
# After page routing
st.markdown("""
<div class="earthai-footer">
    <div class="earthai-footer-left">
        <span>Last Synced: 2 mins ago</span>
        <span>Source: EarthAI Cloud</span>
    </div>
    <div class="earthai-footer-right">
        <div class="earthai-footer-dot"></div>
        <span>Live Satellite Feed Active</span>
    </div>
</div>
""", unsafe_allow_html=True)
```

- [ ] **Step 5: Remove the old `import base64, os` and logo code**

Remove: `import base64` (keep `os`), `LOGO_PATH`, `get_logo_b64()`, `LOGO_B64`, `LOGO_IMG`.

- [ ] **Step 6: Verify app.py loads without errors**

Run: `cd /Users/chris/EarthAI && python -c "import app" 2>&1 | head -5`
(Streamlit apps may not import cleanly outside streamlit, but check for syntax errors)

Alternatively: `python -m py_compile app.py`

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "style: update sidebar branding and hero headers to Cupertino design"
```

---

### Task 3: Update `modules/module6_gpm.py` — Rainfall Analysis

**Files:**
- Modify: `modules/module6_gpm.py`

- [ ] **Step 1: Remove the section-header block at top of render_module6**

The hero header is now rendered by `app.py`. Remove the `st.markdown("""<div class="section-header">...""")` block (lines 67–77). Keep the callout block.

- [ ] **Step 2: Update chart colors to use new COLORS dict**

The COLORS dict has been updated. Verify that chart references like `COLORS["red"]`, `COLORS["bar_light"]`, etc. still work with the new values. They will since we kept compatible key names.

- [ ] **Step 3: Add a footer source line specific to this module**

Replace the closing callout's footer or add after it:

```python
st.markdown("""
<div class="earthai-footer" style="margin-top:1.5rem">
    <div class="earthai-footer-left">
        <span>Source: GPM/IMERG Final Run</span>
    </div>
    <div class="earthai-footer-right">
        <div class="earthai-footer-dot"></div>
        <span>Precipitation Data Active</span>
    </div>
</div>
""", unsafe_allow_html=True)
```

- [ ] **Step 4: Commit**

```bash
git add modules/module6_gpm.py
git commit -m "style: update rainfall module to Cupertino design"
```

---

### Task 4: Update `modules/module2_optical.py` — Optical Detection

**Files:**
- Modify: `modules/module2_optical.py`

- [ ] **Step 1: Remove the section-header block**

Remove the `st.markdown("""<div class="section-header">...""")` block at the start of `render_module2()` (lines 54–65).

- [ ] **Step 2: Commit**

```bash
git add modules/module2_optical.py
git commit -m "style: update optical module to Cupertino design"
```

---

### Task 5: Update `modules/module1_sar.py` — SAR Detection

**Files:**
- Modify: `modules/module1_sar.py`

- [ ] **Step 1: Remove the section-header block**

Remove the `st.markdown("""<div class="section-header">...""")` block at the start of `render_module1()` (lines 47–57).

- [ ] **Step 2: Commit**

```bash
git add modules/module1_sar.py
git commit -m "style: update SAR module to Cupertino design"
```

---

### Task 6: Update `modules/module4_rf.py` — AI Flood Classifier

**Files:**
- Modify: `modules/module4_rf.py`

- [ ] **Step 1: Remove the section-header block**

Remove the `st.markdown("""<div class="section-header">...""")` block at the start of `render_module4()` (lines 287–298).

- [ ] **Step 2: Commit**

```bash
git add modules/module4_rf.py
git commit -m "style: update AI classifier module to Cupertino design"
```

---

### Task 7: Update `modules/module5_flappy.py` — Flappy Bird Competition

**Files:**
- Modify: `modules/module5_flappy.py`

- [ ] **Step 1: Remove the section-header block**

Remove the `st.markdown("""<div class="section-header">...""")` block at the start of `render_module5()` (lines 103–117).

- [ ] **Step 2: Commit**

```bash
git add modules/module5_flappy.py
git commit -m "style: update flappy bird module to Cupertino design"
```

---

### Task 8: Visual verification

**Files:** None (verification only)

- [ ] **Step 1: Run the Streamlit app and check each page**

Run: `cd /Users/chris/EarthAI && streamlit run app.py --server.headless true --server.port 8501`

Check each page in the browser:
1. Rainfall Analysis — hero header, metric cards with 2rem radius, chart, footer
2. Optical Detection — hero header, map, callouts
3. SAR Detection — hero header, histogram, map, metrics
4. AI Flood Classifier — hero header, confusion matrix, leaderboard
5. Flappy Bird — hero header, stage badges, leaderboard

- [ ] **Step 2: Fix any visual issues found**

Address any CSS conflicts, missing styles, or broken layouts discovered during verification.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "style: fix visual issues from Cupertino design verification"
```
