"""Rainfall Timeline — GPM daily precipitation."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data_loader import load_csv, ALL_EVENTS

FLOOD_WINDOWS = {
    "harvey":        ("2017-08-25", "2017-09-01"),
    "pakistan":      ("2022-08-15", "2022-09-15"),
    "dubai":         ("2024-04-15", "2024-04-20"),
    "la2025":        ("2025-01-17", "2025-02-10"),
    "myanmar2015":   ("2015-07-25", "2015-08-15"),
    "chennai2015":   ("2015-11-15", "2015-12-05"),
    "louisiana2016": ("2016-08-12", "2016-08-20"),
    "srilanka2017":  ("2017-05-25", "2017-06-10"),
    "bangladesh2017":("2017-08-01", "2017-08-25"),
    "kerala2018":    ("2018-08-10", "2018-08-20"),
    "japan2018":     ("2018-07-05", "2018-07-12"),
    "mozambique2019":("2019-03-14", "2019-03-20"),
    "iran2019":      ("2019-03-25", "2019-04-10"),
    "china2020":     ("2020-07-05", "2020-08-10"),
    "sudan2020":     ("2020-08-05", "2020-08-25"),
    "germany2021":   ("2021-07-14", "2021-07-16"),
    "kalimantan2021":("2021-01-12", "2021-01-25"),
    "nigeria2022":   ("2022-09-20", "2022-10-15"),
    "libya2023":     ("2023-09-11", "2023-09-14"),
    "somalia2023":   ("2023-11-05", "2023-11-20"),
    "brazil2024":    ("2024-05-01", "2024-05-15"),
    "valencia2024":  ("2024-10-29", "2024-11-01"),
    "afghanistan2024":("2024-05-10","2024-05-14"),
    "tennessee2021": ("2021-08-21", "2021-08-22"),
}

KEY_FACTS = {
    "harvey":        "Harvey dumped over 1,300 mm (51 in) across Houston in 5 days — the highest tropical rainfall total ever recorded in the U.S.",
    "pakistan":      "Pakistan received 3–4× its annual average rainfall in just two months, submerging roughly one-third of the country.",
    "dubai":         "Dubai received nearly its entire annual rainfall (~75 mm) in a single day — a city built for desert conditions with almost no storm drainage.",
    "la2025":        "A series of atmospheric rivers made landfall in rapid succession, delivering extreme precipitation across Southern California.",
    "myanmar2015":   "Cyclone Komen triggered catastrophic monsoon flooding affecting over 1.6 million people across Myanmar.",
    "chennai2015":   "Chennai recorded 344 mm in a single day — the heaviest rainfall in over 100 years, flooding the entire metro area.",
    "louisiana2016": "An unnamed storm dropped 60+ cm of rain in 48 hours over Baton Rouge, flooding over 60,000 homes.",
    "srilanka2017":  "Southwest monsoon rains triggered widespread flooding and landslides, displacing over 600,000 people.",
    "bangladesh2017":"One-third of Bangladesh was submerged as monsoon floods affected 8 million people in August 2017.",
    "kerala2018":    "Kerala's worst flooding in nearly a century — all 14 districts affected, 1.5 million displaced.",
    "japan2018":     "The 'Heisei 30 July Rains' caused record-breaking rainfall across western Japan, triggering floods and landslides.",
    "mozambique2019":"Cyclone Idai made landfall with 195 km/h winds, generating a massive inland flood over Beira.",
    "iran2019":      "Spring floods swept through 26 of Iran's 31 provinces, the country's worst flooding in 70 years.",
    "china2020":     "Record Yangtze River levels — the Poyang Lake basin saw its largest flood extent since satellite monitoring began.",
    "sudan2020":     "Sudan's worst flooding in 100 years submerged entire neighborhoods in Khartoum.",
    "germany2021":   "The Ahr Valley received a full month of rain in 24 hours, destroying hundreds of bridges and roads.",
    "kalimantan2021":"South Kalimantan saw its worst flooding in 50 years, displacing over 60,000 residents.",
    "nigeria2022":   "Flooding affected 33 of 36 states; the Anambra–Delta corridor saw the worst inundation.",
    "libya2023":     "Cyclone Daniel caused catastrophic dam failures in Derna, killing thousands in hours.",
    "somalia2023":   "Unprecedented October–November rains flooded over 1 million people across the Shabelle basin.",
    "brazil2024":    "Cyclone-driven rains submerged 90% of Rio Grande do Sul's municipalities — Brazil's worst climate disaster.",
    "valencia2024":  "A DANA (cut-off low) dropped 450 mm in 8 hours near Valencia — Spain's deadliest flash flood in decades.",
    "afghanistan2024":"Flash floods in Baghlan Province killed hundreds in minutes as normally dry riverbeds overflowed.",
    "tennessee2021": "Humphreys County received 43 cm (17 in) in 24 hours — a 1-in-1000-year event for the region.",
}


def render_module6(event: str):
    ev = ALL_EVENTS.get(event, {})

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🌧️</div>
        <div>
            <div class="section-title">Rainfall Timeline</div>
            <div class="section-desc">
                Explore GPM satellite precipitation data — find the rainfall trigger behind the flood.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout">
        <strong>Think About It</strong> &nbsp;
        If you know when and where it rained, can you predict when and where flooding will occur?
        Locate the peak rainfall in the chart, then compare it against the satellite images in the next tabs.
    </div>
    """, unsafe_allow_html=True)

    df = load_csv(event, "GPM_rainfall_daily")
    if df is None:
        st.markdown(f"""
        <div class="callout warn">
            <strong>Data not found</strong><br>
            Expected: <code>data/{event}/GPM_rainfall_daily.csv</code><br>
            Run Cells 7 and 10 in the Colab export notebook.
        </div>
        """, unsafe_allow_html=True)
        return

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["precip_mm_day"])
    df["precip_mm_day"] = df["precip_mm_day"].clip(lower=0)

    col_ctrl, col_fact = st.columns([2, 1])
    with col_ctrl:
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

    with col_fact:
        fact = KEY_FACTS.get(event, "")
        st.markdown(f"""
        <div class="card" style="height:100%;box-sizing:border-box;">
            <div class="card-title">📌 {ev.get('label','')} {ev.get('year','')}</div>
            <p style="font-size:12px;color:var(--text-sub);margin:0;line-height:1.6">{fact}</p>
        </div>
        """, unsafe_allow_html=True)

    peak_val   = df["precip_mm_day"].max()
    peak_date  = df.loc[df["precip_mm_day"].idxmax(), "date"].strftime("%b %d")
    total_mm   = df["precip_mm_day"].sum()
    rainy_days = int((df["precip_mm_day"] > thr).sum())

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card red">
            <div class="metric-label">Peak Daily Rainfall</div>
            <div class="metric-value">{peak_val:.1f}</div>
            <div class="metric-unit">mm/day — {peak_date}</div>
        </div>
        <div class="metric-card blue">
            <div class="metric-label">Total Rainfall</div>
            <div class="metric-value">{total_mm:.0f}</div>
            <div class="metric-unit">mm over period</div>
        </div>
        <div class="metric-card yellow">
            <div class="metric-label">High-Risk Days</div>
            <div class="metric-value">{rainy_days}</div>
            <div class="metric-unit">days &gt; {thr} mm/day</div>
        </div>
        <div class="metric-card green">
            <div class="metric-label">Period Length</div>
            <div class="metric-value">{len(df)}</div>
            <div class="metric-unit">days total</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    bar_colors = ["#ef4444" if v >= thr else "#93c5fd" for v in df["precip_mm_day"]]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["precip_mm_day"], marker_color=bar_colors,
        name="Daily Rainfall",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.1f} mm/day<extra></extra>",
    ))

    if show_cum:
        df["cumsum"] = df["precip_mm_day"].cumsum()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["cumsum"], mode="lines",
            name="Cumulative (mm)", yaxis="y2",
            line=dict(color="#6366f1", width=2, dash="dot"),
        ))

    if show_thr:
        fig.add_hline(y=thr, line_dash="dash", line_color="#d97706", line_width=1.5,
                      annotation_text=f"  {thr} mm threshold",
                      annotation_font_color="#d97706", annotation_font_size=11)

    if show_fw and event in FLOOD_WINDOWS:
        fig.add_vrect(x0=FLOOD_WINDOWS[event][0], x1=FLOOD_WINDOWS[event][1],
                      fillcolor="rgba(239,68,68,0.08)", layer="below", line_width=0,
                      annotation_text="🌊 Flood Period",
                      annotation_position="top left",
                      annotation_font_color="#dc2626", annotation_font_size=11)

    fig.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Inter"),
        height=360, margin=dict(l=10, r=10, t=10, b=20),
        xaxis=dict(showgrid=False, showline=True, linecolor="#e2e8f0",
                   tickfont=dict(color="#94a3b8", size=11)),
        yaxis=dict(title="mm/day", showgrid=True, gridcolor="#f1f5f9",
                   tickfont=dict(color="#94a3b8", size=11),
                   title_font=dict(size=11, color="#94a3b8")),
        yaxis2=dict(title="Cumulative (mm)", overlaying="y", side="right",
                    showgrid=False, tickfont=dict(color="#6366f1", size=10),
                    title_font=dict(size=10, color="#6366f1")) if show_cum else {},
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0",
                    borderwidth=1, font=dict(size=11)),
        bargap=0.15,
    )
    st.markdown('<div class="card" style="padding:16px">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="callout">
        <strong>Discussion</strong> &nbsp;
        How many days after the rainfall peak did flooding begin?
        Why might flooding persist even after rain stops?
        Head to <em>Optical Before/After</em> to see it in the satellite imagery.
    </div>
    """, unsafe_allow_html=True)
