# Dashboard Visual Audit Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all visual inconsistencies — CSS variable system, broken div wrapping, hardcoded colors, font scale.

**Architecture:** All CSS changes in `utils/styles.py`. Each module updated independently to replace broken `st.markdown('<div>') ... widget ... st.markdown('</div>')` with `st.container(border=True)`, and hardcoded Plotly colors with shared `COLORS` dict.

**Tech Stack:** Streamlit 1.45.1, Plotly, CSS

---

### Task 1: CSS Variable System + COLORS Dict (`styles.py`)

**Files:**
- Modify: `utils/styles.py`

This is the foundation — all other tasks depend on it.

- [ ] **Step 1: Add font scale + sidebar color variables to `:root`**

In `utils/styles.py`, replace the `:root` block (lines 10-36) with:

```css
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
```

- [ ] **Step 2: Replace all hardcoded sidebar colors with variables**

Throughout `styles.py`, replace:
- `#94a3b8` → `var(--sb-text)` (in sidebar rules only)
- `#f1f5f9` → `var(--sb-text-bright)` (in sidebar rules only)
- `#64748b` → `var(--sb-text-dim)` (in sidebar rules only)
- `#1e2533` → `var(--sb-card)` (in sidebar rules only)
- `#cbd5e1` → keep as-is (only used once for radio labels, no variable needed)

Specific lines to change:
- Line 70: `color: #94a3b8` → `color: var(--sb-text)`
- Line 71: `color: #64748b` → `color: var(--sb-text-dim)`
- Line 79: `color: #94a3b8` → `color: var(--sb-text)`
- Line 83: `color: #f1f5f9` → `color: var(--sb-text-bright)`
- Line 84: `color: #f1f5f9` → `color: var(--sb-text-bright)`
- Line 100: `color: #f1f5f9` → `color: var(--sb-text-bright)`, `font-size: 19px` → `font-size: var(--fs-lg)`
- Line 103: `color: #94a3b8` → `color: var(--sb-text)`, `font-size: 13px` → `font-size: var(--fs-sm)`
- Line 108: `color: #64748b` → `color: var(--sb-text-dim)`, `font-size: 16px` → `font-size: var(--fs-md)`
- Line 113: `background: #1e2533` → `background: var(--sb-card)`
- Line 117: `color: #f1f5f9` → `color: var(--sb-text-bright)`
- Line 118: `color: #94a3b8` → `color: var(--sb-text)`
- Line 127: `color: #94a3b8` → `color: var(--sb-text)`
- Line 139: `background: #1e2533` → `background: var(--sb-card)`
- Line 140: `color: #f1f5f9` → `color: var(--sb-text-bright)`

- [ ] **Step 3: Apply font scale to all font-size declarations**

Replace hardcoded font sizes with scale variables:
- Line 41: `font-size: 15px` → `font-size: var(--fs-base)`
- Line 130: `font-size: 14px` → `font-size: var(--fs-base)` (nav buttons — 15px is close enough)
- Line 145: `.page-title font-size: 22px` → `font-size: var(--fs-lg)`
- Line 146: `.page-sub font-size: 14px` → `font-size: var(--fs-base)`
- Line 155: `.card-title font-size: 14px` → `font-size: var(--fs-base)`
- Line 170: `.metric-label font-size: 12px` → `font-size: var(--fs-xs)`
- Line 177: `.metric-value font-size: 22px` → `font-size: var(--fs-xl)`
- Line 180: `.metric-unit font-size: 12px` → `font-size: var(--fs-xs)`
- Line 186: `.callout font-size: 14px` → `font-size: var(--fs-base)`
- Line 202: `.section-title font-size: 16px` → `font-size: var(--fs-md)`
- Line 203: `.section-desc font-size: 13px` → `font-size: var(--fs-sm)`
- Line 211: `.control-title font-size: 12px` → `font-size: var(--fs-sm)`
- Line 234: `.main button font-size: 14px` → `font-size: var(--fs-base)`
- Line 244: `.lb-head font-size: 14px` → `font-size: var(--fs-base)`
- Line 247: `.lb-row font-size: 13px` → `font-size: var(--fs-sm)`
- Line 250: `.lb-rank font-size: 12px` → `font-size: var(--fs-xs)`
- Line 255: `.lb-event font-size: 12px` → `font-size: var(--fs-xs)`
- Line 257: `.lb-score font-size: 14px` → `font-size: var(--fs-base)`
- Line 267: `.chip font-size: 12px` → `font-size: var(--fs-xs)`

- [ ] **Step 4: Add `st.container(border=True)` CSS override**

Replace the existing `.control-panel` CSS block (lines 206-213) with:

```css
/* ── Container styling (st.container(border=True)) ─── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg2) !important;
    border: 0.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow);
}
.control-title {
    font-size: var(--fs-sm) !important; font-weight: 600; color: var(--text-sub) !important;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 10px;
}
```

Also remove the `.card` CSS block (lines 149-157) since `st.container(border=True)` now handles card styling. Keep `.card-title`.

- [ ] **Step 5: Add `.header-logo` CSS class**

Add after the `.page-sub` rule:

```css
.header-logo {
    width: 36px; height: 36px; border-radius: 8px;
    object-fit: cover; vertical-align: middle; margin-right: 10px;
}
```

- [ ] **Step 6: Add COLORS dict after `inject_css()` function**

At the end of `utils/styles.py`, add:

```python
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
```

- [ ] **Step 7: Commit**

```bash
git add utils/styles.py
git commit -m "style: add CSS variable system, font scale, sidebar vars, COLORS dict"
```

---

### Task 2: Fix `app.py` inline styles

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace logo inline style with CSS class**

Change lines 129-131 from:
```python
logo_small = (f'<img src="{LOGO_B64}" style="width:36px;height:36px;border-radius:8px;'
              f'object-fit:cover;vertical-align:middle;margin-right:10px;">'
              if LOGO_B64 else "🌊 ")
```
To:
```python
logo_small = (f'<img src="{LOGO_B64}" class="header-logo">'
              if LOGO_B64 else "🌊 ")
```

- [ ] **Step 2: Replace inline styles in main header**

Change lines 142-143 from:
```python
<div style="font-size:12px;font-weight:500;color:#2563eb;
            margin-bottom:4px;letter-spacing:0.04em">{section_label}</div>
```
To:
```python
<div class="section-label-badge">{section_label}</div>
```

Then add the CSS class `.section-label-badge` to `styles.py` after `.page-sub`:
```css
.section-label-badge {
    font-size: var(--fs-sm) !important; font-weight: 500;
    color: var(--blue) !important; margin-bottom: 4px; letter-spacing: 0.04em;
}
```

- [ ] **Step 3: Replace hardcoded color in badge area**

Change line 160 from:
```python
<span style="font-size:12px;color:#475569">{ev['region']} · {ev['year']}</span>
```
To:
```python
<span style="font-size:var(--fs-xs);color:var(--text-sub)">{ev['region']} · {ev['year']}</span>
```

- [ ] **Step 4: Commit**

```bash
git add app.py utils/styles.py
git commit -m "style: replace app.py inline styles with CSS classes"
```

---

### Task 3: Fix `module6_gpm.py` — container + COLORS

**Files:**
- Modify: `modules/module6_gpm.py`

- [ ] **Step 1: Add COLORS import**

Change line 6 from:
```python
from utils.data_loader import load_csv, ALL_EVENTS
```
To:
```python
from utils.data_loader import load_csv, ALL_EVENTS
from utils.styles import COLORS
```

- [ ] **Step 2: Fix control-panel div wrapping (lines 103-112)**

Replace:
```python
        st.markdown('<div class="control-panel"><div class="control-title">Display Options</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            show_thr = st.checkbox("Show threshold line", value=True)
            thr = st.slider("Threshold (mm/day)", 5, 80, 20) if show_thr else 20
        with c2:
            show_fw  = st.checkbox("Highlight flood period", value=True)
            show_cum = st.checkbox("Cumulative overlay", value=False)
        st.markdown("</div>", unsafe_allow_html=True)
```
With:
```python
        with st.container(border=True):
            st.markdown('<div class="control-title">Display Options</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                show_thr = st.checkbox("Show threshold line", value=True)
                thr = st.slider("Threshold (mm/day)", 5, 80, 20) if show_thr else 20
            with c2:
                show_fw  = st.checkbox("Highlight flood period", value=True)
                show_cum = st.checkbox("Cumulative overlay", value=False)
```

- [ ] **Step 3: Fix fact card div wrapping (lines 116-121)**

Replace:
```python
        st.markdown(f"""
        <div class="card" style="height:100%;box-sizing:border-box;">
            <div class="card-title">📌 {ev.get('label','')} {ev.get('year','')}</div>
            <p style="font-size:12px;color:var(--text-sub);margin:0;line-height:1.6">{fact}</p>
        </div>
        """, unsafe_allow_html=True)
```
With:
```python
        with st.container(border=True):
            st.markdown(
                f'<div class="card-title">📌 {ev.get("label","")} {ev.get("year","")}</div>'
                f'<p style="font-size:var(--fs-xs);color:var(--text-sub);margin:0;line-height:1.6">{fact}</p>',
                unsafe_allow_html=True)
```

- [ ] **Step 4: Fix chart card div wrapping (lines 197-199)**

Replace:
```python
    st.markdown('<div class="card" style="padding:16px">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
```
With:
```python
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
```

- [ ] **Step 5: Replace hardcoded Plotly colors with COLORS dict**

Replace all hardcoded hex colors in the Plotly chart (lines 153-196):
- `"#ef4444"` → `COLORS["red"]`
- `"#93c5fd"` → `COLORS["bar_light"]`
- `"#6366f1"` → `COLORS["indigo"]`
- `"#d97706"` → `COLORS["yellow"]`
- `"#dc2626"` → `COLORS["red_dark"]`
- `"#ffffff"` / `"#fff"` → `COLORS["bg"]`
- `"#0f172a"` → `COLORS["text"]`
- `"#94a3b8"` → `COLORS["axis"]`
- `"#f1f5f9"` → `COLORS["bg3"]`
- `"#e2e8f0"` → `COLORS["border"]`

- [ ] **Step 6: Commit**

```bash
git add modules/module6_gpm.py
git commit -m "style: fix module6_gpm container wrapping + COLORS"
```

---

### Task 4: Fix `module1_sar.py` — container + COLORS

**Files:**
- Modify: `modules/module1_sar.py`

- [ ] **Step 1: Add COLORS import**

After line 11, add:
```python
from utils.styles import COLORS
```

- [ ] **Step 2: Fix control-panel div wrapping (lines 88-98)**

Replace:
```python
        st.markdown('<div class="control-panel"><div class="control-title">Threshold Control</div>',
                    unsafe_allow_html=True)
        thr = st.slider(...)
        if st.button("Apply Otsu Auto-Threshold"):
            ...
        remove_perm = st.checkbox(...)
        st.markdown("</div>", unsafe_allow_html=True)
```
With:
```python
        with st.container(border=True):
            st.markdown('<div class="control-title">Threshold Control</div>',
                        unsafe_allow_html=True)
            thr = st.slider(...)
            if st.button("Apply Otsu Auto-Threshold"):
                ...
            remove_perm = st.checkbox(...)
```

- [ ] **Step 3: Fix histogram card div wrapping (lines 156-159)**

Replace:
```python
        st.markdown('<div class="card" style="padding:12px;margin-bottom:10px">',
                    unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
```
With:
```python
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
```

- [ ] **Step 4: Fix map-frame div wrapping (lines 163-165)**

Replace:
```python
        st.markdown('<div class="map-frame">', unsafe_allow_html=True)
        st_folium(m, width=None, height=350, returned_objects=[])
        st.markdown("</div>", unsafe_allow_html=True)
```
With:
```python
        with st.container(border=True):
            st_folium(m, width=None, height=350, returned_objects=[])
```

- [ ] **Step 5: Replace hardcoded Plotly colors with COLORS dict**

In the histogram chart (lines 130-155):
- `"#3b82f6"` → `COLORS["blue"]`
- `"#cbd5e1"` → `"#cbd5e1"` (keep — unique to histogram, not in COLORS)
- `"#ef4444"` → `COLORS["red"]`
- `"#16a34a"` → `COLORS["green"]`
- `"#fff"` / `"#ffffff"` → `COLORS["bg"]`
- `"#0f172a"` → `COLORS["text"]`
- `"#94a3b8"` → `COLORS["axis"]`
- `"#f1f5f9"` → `COLORS["bg3"]`

- [ ] **Step 6: Commit**

```bash
git add modules/module1_sar.py
git commit -m "style: fix module1_sar container wrapping + COLORS"
```

---

### Task 5: Fix `module2_optical.py` — container + COLORS

**Files:**
- Modify: `modules/module2_optical.py`

- [ ] **Step 1: Fix control-panel div wrapping (lines 78-87)**

Replace:
```python
        st.markdown('<div class="control-panel"><div class="control-title">Layer Settings</div>',
                    unsafe_allow_html=True)
        layer = st.radio(...)
        period = "after"
        if layer != "NDWI Change (After − Before)":
            period = st.radio("Time period", ["before", "after"])
        st.markdown("</div>", unsafe_allow_html=True)
```
With:
```python
        with st.container(border=True):
            st.markdown('<div class="control-title">Layer Settings</div>',
                        unsafe_allow_html=True)
            layer = st.radio(...)
            period = "after"
            if layer != "NDWI Change (After − Before)":
                period = st.radio("Time period", ["before", "after"])
```

- [ ] **Step 2: Fix NDWI color guide card (lines 90-100)**

The NDWI color guide uses a self-contained `st.markdown` with all HTML in one call — this is correct and does NOT need `st.container`. However, replace the `.card` wrapper with `st.container`:

Replace:
```python
            st.markdown("""
            <div class="card" style="font-size:12px">
                <div class="card-title">NDWI Color Guide</div>
                ...
            </div>
            """, unsafe_allow_html=True)
```
With:
```python
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
```

(Note: The NDWI guide hardcoded colors `#2166ac`, `#92c5de`, `#f7f7f7`, `#d6604d` are colormap-specific legend colors — NOT dashboard theme colors. Keep them hardcoded.)

- [ ] **Step 3: Fix map-frame div wrapping (lines 105-107)**

Replace:
```python
        st.markdown('<div class="map-frame">', unsafe_allow_html=True)
        st_folium(m, width=None, height=460, returned_objects=[])
        st.markdown("</div>", unsafe_allow_html=True)
```
With:
```python
        with st.container(border=True):
            st_folium(m, width=None, height=460, returned_objects=[])
```

- [ ] **Step 4: Commit**

```bash
git add modules/module2_optical.py
git commit -m "style: fix module2_optical container wrapping"
```

---

### Task 6: Fix `module4_rf.py` — container + COLORS

**Files:**
- Modify: `modules/module4_rf.py`

- [ ] **Step 1: Add COLORS import**

Change line 13:
```python
from utils.data_loader import load_all_rf_samples, ALL_EVENTS
```
To:
```python
from utils.data_loader import load_all_rf_samples, ALL_EVENTS
from utils.styles import COLORS
```

- [ ] **Step 2: Fix control-panel div wrapping (lines 259-293)**

Replace:
```python
        st.markdown('<div class="control-panel"><div class="control-title">Model Parameters</div>',
                    unsafe_allow_html=True)
        ...all widgets...
        st.markdown("</div>", unsafe_allow_html=True)
```
With:
```python
        with st.container(border=True):
            st.markdown('<div class="control-title">Model Parameters</div>',
                        unsafe_allow_html=True)
            ...all widgets (unchanged)...
```
(Remove `st.markdown("</div>", unsafe_allow_html=True)` on line 293)

- [ ] **Step 3: Fix Performance card div wrapping (lines 373-396)**

Replace:
```python
                st.markdown('<div class="card"><div class="card-title">Performance</div>',
                            unsafe_allow_html=True)
                fig = go.Figure(...)
                st.plotly_chart(fig, ...)
                st.markdown("</div>", unsafe_allow_html=True)
```
With:
```python
                with st.container(border=True):
                    st.markdown('<div class="card-title">Performance</div>',
                                unsafe_allow_html=True)
                    fig = go.Figure(...)
                    st.plotly_chart(fig, ...)
```

- [ ] **Step 4: Fix Confusion Matrix card div wrapping (lines 399-402)**

Replace:
```python
                st.markdown('<div class="card"><div class="card-title">Confusion Matrix</div>',
                            unsafe_allow_html=True)
                render_confusion_matrix(m["cm"])
                st.markdown("</div>", unsafe_allow_html=True)
```
With:
```python
                with st.container(border=True):
                    st.markdown('<div class="card-title">Confusion Matrix</div>',
                                unsafe_allow_html=True)
                    render_confusion_matrix(m["cm"])
```

- [ ] **Step 5: Fix Feature Importance card div wrapping (lines 405-408)**

Replace:
```python
            st.markdown('<div class="card"><div class="card-title">Feature Importance — What the AI relied on most</div>',
                        unsafe_allow_html=True)
            render_feature_importance(res["importance"])
            st.markdown("</div>", unsafe_allow_html=True)
```
With:
```python
            with st.container(border=True):
                st.markdown('<div class="card-title">Feature Importance — What the AI relied on most</div>',
                            unsafe_allow_html=True)
                render_feature_importance(res["importance"])
```

- [ ] **Step 6: Replace hardcoded Plotly colors with COLORS dict**

In `render_feature_importance` (lines 133-153):
- `"#bfdbfe"` → `COLORS["blue_light"]`
- `"#3b82f6"` → `COLORS["blue"]`
- `"#1d4ed8"` → `COLORS["blue_dark"]`
- `"#475569"` → `COLORS["text_sub"]`
- `"#ffffff"` → `COLORS["bg"]`
- `"#0f172a"` → `COLORS["text"]`
- `"#f1f5f9"` → `COLORS["bg3"]`
- `"#94a3b8"` → `COLORS["axis"]`

In `render_confusion_matrix` (lines 161-177):
- `"#eff6ff"` → `"#eff6ff"` (keep — blue-light variant)
- `"#1d4ed8"` → `COLORS["blue_dark"]`
- `"#fff"` → `COLORS["bg"]`
- `"#0f172a"` → `COLORS["text"]`
- `"#475569"` → `COLORS["text_sub"]`

In `render_module4` performance bar chart (lines 375-394):
- `"#3b82f6"` → `COLORS["blue"]`
- `"#06b6d4"` → `COLORS["cyan"]`
- `"#10b981"` → `COLORS["green_light"]`
- `"#6366f1"` → `COLORS["indigo"]`
- `"#475569"` → `COLORS["text_sub"]`
- `"#d97706"` → `COLORS["yellow"]`
- `"#fff"` → `COLORS["bg"]`
- `"#f1f5f9"` → `COLORS["bg3"]`
- `"#94a3b8"` → `COLORS["axis"]`

- [ ] **Step 7: Commit**

```bash
git add modules/module4_rf.py
git commit -m "style: fix module4_rf container wrapping + COLORS"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run the Streamlit app**

```bash
cd /Users/chris/EarthAI && streamlit run app.py
```

Visually verify:
- Sidebar: all text readable on dark background, section labels larger than nav buttons
- Control panels: widgets properly contained inside bordered containers
- Cards: Plotly charts inside bordered containers
- Font sizes: consistent scale across all pages
- No raw HTML tags visible
- Leaderboard renders properly

- [ ] **Step 2: Check all 4 pages**

Navigate through each page via sidebar nav:
1. 🌧️ Rainfall Timeline — control panel, chart card, metric cards
2. 🛰️ Optical Before/After — control panel, map, NDWI guide
3. 📡 SAR Detection — control panel, histogram card, map
4. 🤖 AI Flood Classifier — control panel, results cards, leaderboard

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "style: dashboard visual audit complete"
```
