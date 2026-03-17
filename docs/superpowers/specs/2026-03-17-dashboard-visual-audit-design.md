# Dashboard Visual Audit & Fix Design

**Date:** 2026-03-17
**Scope:** Full visual/structural cleanup across all dashboard files

## Problem

The dashboard has accumulated visual inconsistencies:
- Font sizes used without a consistent scale (11px to 22px, ad hoc)
- Sidebar colors hardcoded in 10+ places instead of CSS variables
- HTML div wrapping pattern (`st.markdown('<div>')` ... widgets ... `st.markdown('</div>')`) does not work in Streamlit — each `st.markdown` is an independent element, so widgets are never inside the div
- Plotly chart colors hardcoded per-module with no shared constants
- Inline styles in `app.py` that should be CSS classes

## Design

### 1. CSS Variable System Expansion (`styles.py`)

#### Font Size Scale (6 steps)

| Variable | Size | Usage |
|----------|------|-------|
| `--fs-xs` | 11px | captions, auxiliary text |
| `--fs-sm` | 13px | labels, badge subtitles |
| `--fs-base` | 15px | body text |
| `--fs-md` | 17px | sidebar section labels (0-3), card titles |
| `--fs-lg` | 20px | sidebar brand, page title |
| `--fs-xl` | 24px | metric values |

#### Sidebar Color Variables

| Variable | Value | Usage |
|----------|-------|-------|
| `--sb-text` | `#94a3b8` | sidebar default text |
| `--sb-text-bright` | `#f1f5f9` | sidebar bright text (brand, active) |
| `--sb-text-dim` | `#64748b` | sidebar dim labels |
| `--sb-card` | `#1e2533` | sidebar card/badge backgrounds |

All hardcoded sidebar hex values in styles.py replaced with these variables.

### 2. HTML Div Wrapping Fix (All Modules)

**Current (broken):**
```python
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
st.slider(...)
st.markdown('</div>', unsafe_allow_html=True)
```

**Fixed:**
```python
with st.container(border=True, key="control_panel"):
    st.slider(...)
```

CSS targets `st.container(border=True)` via Streamlit's `data-testid` attribute to apply `control-panel` styling:
```css
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg2);
    border: 0.5px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
}
```

Same approach for `.card` div wrappers.

**Affected locations:**
- `module1_sar.py`: control-panel (1), card (2)
- `module2_optical.py`: control-panel (1), card (1)
- `module4_rf.py`: control-panel (1), card (3)
- `module6_gpm.py`: control-panel (1), card (1)

### 3. Plotly Color Constants (`styles.py`)

Add a `COLORS` dict exported from `styles.py`:
```python
COLORS = {
    "text": "#0f172a", "text_sub": "#374151", "text_muted": "#6b7280",
    "blue": "#2563eb", "green": "#16a34a", "red": "#dc2626",
    "yellow": "#d97706", "indigo": "#4f46e5", "border": "#e2e8f0",
    "bg": "#ffffff", "bg3": "#f1f5f9", "axis": "#94a3b8",
}
```

Each module imports `from utils.styles import COLORS` and references `COLORS["text"]` instead of literal hex strings.

### 4. Inline Style Cleanup (`app.py`, `styles.py`)

- Main header section label: `font-size:12px` -> `var(--fs-sm)`, `color:#2563eb` -> `var(--blue)`
- Logo inline style -> CSS class `.header-logo`
- Event badge text `color:#475569` -> `var(--text-sub)`

### 5. Font Size Unification

Apply the 6-step scale across all elements:
- Sidebar brand main: 19px -> `var(--fs-lg)` (20px)
- Sidebar brand sub: 13px -> `var(--fs-sm)` (13px, unchanged)
- Sidebar section labels: 16px -> `var(--fs-md)` (17px)
- Page title: 22px -> `var(--fs-lg)` (20px)
- Card titles: 14px -> `var(--fs-base)` (15px)
- Metric values: 22px -> `var(--fs-xl)` (24px)
- Metric labels: 12px -> `var(--fs-xs)` (11px)
- Control titles: 12px -> `var(--fs-sm)` (13px)

## Out of Scope

- Metric card label colors (paired with backgrounds, keep hardcoded)
- Plotly chart layout structure
- Page routing logic
- Data loading / model training logic

## Files Changed

| File | Changes |
|------|---------|
| `utils/styles.py` | CSS variables, font scale, sidebar vars, container styles, `COLORS` dict |
| `app.py` | Inline styles -> CSS classes, font size variables |
| `modules/module1_sar.py` | `st.container` pattern, `COLORS` import |
| `modules/module2_optical.py` | `st.container` pattern, `COLORS` import |
| `modules/module4_rf.py` | `st.container` pattern, `COLORS` import |
| `modules/module6_gpm.py` | `st.container` pattern, `COLORS` import |
