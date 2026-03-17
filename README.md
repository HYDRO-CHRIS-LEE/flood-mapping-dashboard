# 🌊 AI Flood Inundation Mapping Dashboard
**UCI CHRS EarthAI Flood Lab | High School Summer Program**

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
Run the Colab GEE Export notebook first, then organize files as:
```
streamlit_app/
└── data/
    ├── harvey/
    │   ├── SAR_before.tif             ← strip the '{event}_' prefix from Drive filenames
    │   ├── SAR_after.tif
    │   ├── NDWI_before.tif
    │   ├── NDWI_after.tif
    │   ├── RGB_before.tif
    │   ├── RGB_after.tif
    │   ├── DEM_slope.tif              ← 2-band GeoTIFF: [elevation, slope]
    │   ├── JRC_permanent_water.tif
    │   ├── GPM_rainfall_daily.csv
    │   └── RF_training_samples.csv
    ├── pakistan/   (same structure)
    ├── dubai/      (same structure)
    └── la2025/     (same structure)
```

### 3. Run locally
```bash
cd streamlit_app
streamlit run app.py
```

### 4. Deploy to Streamlit Cloud (for QR code access)
1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo
3. Generate a QR code from the deployed URL → display on the classroom screen

---

## Project Structure
```
app.py                       ← Entry point: sidebar + 4-tab layout
├── modules/
│   ├── module6_gpm.py       ← M6: Plotly GPM rainfall bar chart
│   ├── module2_optical.py   ← M2: Folium map with RGB / NDWI layers
│   ├── module1_sar.py       ← M1: SAR histogram + Otsu slider + flood map
│   └── module4_rf.py        ← M4: Random Forest GUI + team leaderboard
└── utils/
    ├── styles.py             ← Global CSS (dark sci-fi theme)
    └── data_loader.py        ← GeoTIFF loader (rasterio → numpy → base64 PNG)
```

---

## Module Overview

| Tab | Module | Student Interaction | Key Concept |
|-----|--------|---------------------|-------------|
| M6 | Rainfall Timeline | Date slider, threshold line | Rainfall → flood lag time |
| M2 | Optical Before/After | Band dropdown (RGB/NDWI), before/after toggle | Optical limitations, cloud cover |
| M1 | SAR Otsu Detection | Threshold slider, auto-Otsu button | Radar physics, automatic thresholding |
| M4 | Random Forest AI | Feature checkboxes, tree count/depth sliders, leaderboard submit | Supervised learning, feature importance |

---

## Per-Event Teaching Notes

| Event | M6 Highlight | M2 Highlight | M1 Highlight | M4 Highlight |
|-------|-------------|-------------|-------------|-------------|
| Harvey | Record-breaking rainfall peak | Clouds present → motivate SAR | Strong urban flood SAR signal | Mixed LULC feature importance |
| Pakistan | Prolonged multi-peak pattern | Near-zero cloud cover, crisp NDWI | Massive flood extent | Agricultural land classification |
| Dubai | Single-day extreme spike | Desert Before/After contrast exceptional | Low baseline backscatter challenge | Urban DEM + SAR interaction |
| LA 2025 | Atmospheric river sequence | ⚠️ Intentional cloud gaps | Mixed urban/mountain terrain | Slope feature dominance |
