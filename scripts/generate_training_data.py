"""
Generate RF training data using Copernicus EMS ground truth + GEE features.

Strategy: Sample points LOCALLY using EMS polygons, then extract GEE features
at those points. This avoids uploading large polygons to GEE.
"""

import ee
import os
import re
import json
import zipfile
import tempfile
import urllib.request
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
from glob import glob

# ── Config ────────────────────────────────────────────────────────
GEE_PROJECT = "earthai-490500"
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

EVENTS = {
    "harvey":        {"emsr": "EMSR229", "aoi": [-96.0, 29.0, -94.5, 30.5],
                      "flood": ("2017-08-26","2017-09-10"), "sar_pass": "DESCENDING", "s2_cloud": 30},
    "pakistan":       {"emsr": "EMSR629", "aoi": [66.5, 25.5, 69.5, 28.0],
                      "flood": ("2022-08-15","2022-09-25"), "sar_pass": "ASCENDING", "s2_cloud": 15},
    "myanmar2015":   {"emsr": "EMSR130", "aoi": [94.5, 15.5, 96.5, 17.5],
                      "flood": ("2015-07-25","2015-08-20"), "sar_pass": "ASCENDING", "s2_cloud": 30},
    "louisiana2016": {"emsr": "EMSR176", "aoi": [-91.5, 30.0, -90.0, 31.0],
                      "flood": ("2016-08-12","2016-08-25"), "sar_pass": "DESCENDING", "s2_cloud": 30},
    "srilanka2017":  {"emsr": "EMSR205", "aoi": [80.0, 6.0, 81.5, 7.5],
                      "flood": ("2017-05-25","2017-06-15"), "sar_pass": "ASCENDING", "s2_cloud": 25},
    "mozambique2019":{"emsr": "EMSR348", "aoi": [34.0, -20.0, 35.5, -18.5],
                      "flood": ("2019-03-14","2019-04-01"), "sar_pass": "DESCENDING", "s2_cloud": 25},
    "iran2019":      {"emsr": "EMSR352", "aoi": [48.0, 31.0, 50.0, 33.0],
                      "flood": ("2019-03-25","2019-04-20"), "sar_pass": "ASCENDING", "s2_cloud": 15},
    "germany2021":   {"emsr": "EMSR517", "aoi": [6.5, 50.0, 7.5, 50.8],
                      "flood": ("2021-07-14","2021-07-25"), "sar_pass": "DESCENDING", "s2_cloud": 25},
    "nigeria2022":   {"emsr": "EMSR314", "aoi": [6.0, 5.0, 7.5, 6.5],
                      "flood": ("2022-09-20","2022-10-20"), "sar_pass": "DESCENDING", "s2_cloud": 25},
    "libya2023":     {"emsr": "EMSR696", "aoi": [22.0, 32.0, 23.5, 33.0],
                      "flood": ("2023-09-11","2023-09-25"), "sar_pass": "ASCENDING", "s2_cloud": 10},
    "somalia2023":   {"emsr": "EMSR404", "aoi": [45.0, 2.0, 46.5, 3.5],
                      "flood": ("2023-11-05","2023-11-30"), "sar_pass": "ASCENDING", "s2_cloud": 20},
    "brazil2024":    {"emsr": "EMSR720", "aoi": [-53.0, -31.0, -51.0, -29.5],
                      "flood": ("2024-05-01","2024-05-25"), "sar_pass": "ASCENDING", "s2_cloud": 25},
    "valencia2024":  {"emsr": "EMSR773", "aoi": [-1.5, 38.5, 0.5, 40.0],
                      "flood": ("2024-10-29","2024-11-10"), "sar_pass": "DESCENDING", "s2_cloud": 20},
}

NUM_SAMPLES = 400  # per class


def scrape_ems_zip_urls(emsr_code):
    """Scrape EMS activation page for vector ZIP download URLs."""
    url = f"https://mapping.emergency.copernicus.eu/activations/{emsr_code}/"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    html = resp.read().decode()
    links = re.findall(r'https://[^"]*s3[^"]*\.zip', html, re.IGNORECASE)
    monit = [l for l in links if "MONIT" in l.upper()]
    delin = [l for l in links if "DELINEATION_MAP" in l.upper()]
    best = monit if monit else delin if delin else links
    print(f"  [EMS] {len(links)} ZIPs found, using {len(best)}")
    return best


def download_ems_flood_polygons(zip_urls, aoi_bounds, tmpdir):
    """Download EMS ZIPs, extract flood polygons, clip to AOI, return GeoDataFrame."""
    aoi_box = box(*aoi_bounds)
    all_gdfs = []

    for i, url in enumerate(zip_urls):
        fname = os.path.basename(url)
        local_zip = os.path.join(tmpdir, fname)
        try:
            urllib.request.urlretrieve(url, local_zip)
            with zipfile.ZipFile(local_zip, 'r') as z:
                z.extractall(os.path.join(tmpdir, f"e_{i}"))

            shps = glob(os.path.join(tmpdir, f"e_{i}", "**", "*observedEvent*.shp"), recursive=True)
            if not shps:
                shps = glob(os.path.join(tmpdir, f"e_{i}", "**", "*observed_event*.shp"), recursive=True)
            if not shps:
                shps = glob(os.path.join(tmpdir, f"e_{i}", "**", "*.shp"), recursive=True)

            for shp in shps:
                try:
                    gdf = gpd.read_file(shp)
                    if len(gdf) == 0:
                        continue
                    if gdf.crs and gdf.crs != "EPSG:4326":
                        gdf = gdf.to_crs("EPSG:4326")
                    # Filter flood-related features
                    for col in ['event_type', 'type', 'obj_type']:
                        if col in gdf.columns:
                            flood_rows = gdf[gdf[col].str.contains('flood|water|inund', case=False, na=False)]
                            if len(flood_rows) > 0:
                                gdf = flood_rows
                                break
                    # Clip to AOI
                    gdf["geometry"] = gdf["geometry"].intersection(aoi_box)
                    gdf = gdf[~gdf.is_empty]
                    if len(gdf) > 0:
                        all_gdfs.append(gdf)
                except Exception:
                    continue
        except Exception as e:
            print(f"  [EMS] Failed: {fname}: {e}")

    if not all_gdfs:
        return None

    merged = pd.concat(all_gdfs, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs="EPSG:4326")
    # Dissolve into single multipolygon
    dissolved = merged.dissolve()
    print(f"  [EMS] {len(merged)} polygons → dissolved")
    return dissolved


def sample_points_locally(flood_gdf, aoi_bounds, n_per_class=400, seed=2024):
    """Sample random flood and non-flood points using local geometry operations."""
    rng = np.random.RandomState(seed)
    aoi_box = box(*aoi_bounds)
    flood_geom = flood_gdf.geometry.iloc[0]

    flood_area = flood_geom.area
    aoi_area = aoi_box.area
    print(f"  [Sample] Flood covers {flood_area/aoi_area*100:.1f}% of AOI")

    def sample_in_polygon(poly, n, max_attempts=50000):
        """Random point sampling within a polygon."""
        minx, miny, maxx, maxy = poly.bounds
        pts = []
        attempts = 0
        while len(pts) < n and attempts < max_attempts:
            batch = 1000
            xs = rng.uniform(minx, maxx, batch)
            ys = rng.uniform(miny, maxy, batch)
            for x, y in zip(xs, ys):
                p = Point(x, y)
                if poly.contains(p):
                    pts.append((x, y))
                    if len(pts) >= n:
                        break
            attempts += batch
        return pts

    # Sample flood points (inside EMS polygons)
    print(f"  [Sample] Sampling {n_per_class} flood points...")
    flood_pts = sample_in_polygon(flood_geom, n_per_class)
    print(f"  [Sample] Got {len(flood_pts)} flood points")

    # Sample non-flood points (inside AOI but outside flood polygons)
    print(f"  [Sample] Sampling {n_per_class} non-flood points...")
    nonflood_geom = aoi_box.difference(flood_geom)
    nonflood_pts = sample_in_polygon(nonflood_geom, n_per_class)
    print(f"  [Sample] Got {len(nonflood_pts)} non-flood points")

    return flood_pts, nonflood_pts


def extract_features_from_gee(points, labels, cfg, event_key, batch_size=200):
    """Extract satellite features at given points using GEE."""
    b = cfg["aoi"]
    aoi = ee.Geometry.Rectangle(b)

    print(f"  [GEE] Loading imagery for {event_key}...")
    # SAR
    sar = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(aoi)
           .filterDate(cfg["flood"][0], cfg["flood"][1])
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .select("VH").mean().clip(aoi))

    # S2
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi)
          .filterDate(cfg["flood"][0], cfg["flood"][1])
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg["s2_cloud"]))
          .select(["B3", "B8", "B11"]).median().clip(aoi))
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = s2.normalizedDifference(["B3", "B11"]).rename("MNDWI")

    # Terrain
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi).rename("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(aoi)
    perm = jrc.select("occurrence").gt(90).rename("permanent_water")

    stack = ee.Image.cat([sar.rename("SAR_VH"), ndwi, mndwi, dem, slope, perm])

    # Extract in batches
    all_rows = []
    total = len(points)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_pts = points[start:end]
        batch_labels = labels[start:end]

        ee_pts = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point(lon, lat), {"label": int(lbl)})
            for (lon, lat), lbl in zip(batch_pts, batch_labels)
        ])

        sampled = stack.sampleRegions(
            collection=ee_pts,
            scale=30,
            geometries=False
        )

        try:
            data = sampled.getInfo()
            rows = [f["properties"] for f in data["features"]]
            all_rows.extend(rows)
            print(f"  [GEE] Batch {start//batch_size+1}: {len(rows)}/{end-start} points extracted")
        except Exception as e:
            print(f"  [GEE] Batch {start//batch_size+1} failed: {e}")

    return all_rows


def main():
    print("Initializing GEE...")
    ee.Initialize(project=GEE_PROJECT)
    print(f"Processing {len(EVENTS)} events\n")
    os.makedirs(DATA_ROOT, exist_ok=True)

    results = {}
    for event_key, cfg in EVENTS.items():
        print(f"\n{'='*60}")
        print(f"[{event_key}] {cfg['emsr']}")
        print(f"{'='*60}")

        out_dir = os.path.join(DATA_ROOT, event_key)
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, "RF_training_samples.csv")

        # 1. Download EMS flood polygons
        try:
            print("  Scraping EMS links...")
            zip_urls = scrape_ems_zip_urls(cfg["emsr"])
            if not zip_urls:
                print(f"  SKIP — no EMS data")
                results[event_key] = "NO_EMS"
                continue
        except Exception as e:
            print(f"  SKIP — scrape failed: {e}")
            results[event_key] = f"SCRAPE_ERR: {e}"
            continue

        # 2. Extract flood polygons locally
        with tempfile.TemporaryDirectory() as tmpdir:
            print("  Downloading shapefiles...")
            flood_gdf = download_ems_flood_polygons(zip_urls, cfg["aoi"], tmpdir)

        if flood_gdf is None or len(flood_gdf) == 0:
            print(f"  SKIP — no polygons")
            results[event_key] = "NO_POLYGONS"
            continue

        # 3. Sample points locally
        try:
            flood_pts, nonflood_pts = sample_points_locally(
                flood_gdf, cfg["aoi"], NUM_SAMPLES, seed=2024)
        except Exception as e:
            print(f"  SKIP — sampling failed: {e}")
            results[event_key] = f"SAMPLE_ERR: {e}"
            continue

        if len(flood_pts) < 50 or len(nonflood_pts) < 50:
            print(f"  SKIP — too few points ({len(flood_pts)} flood, {len(nonflood_pts)} non-flood)")
            results[event_key] = "TOO_FEW_POINTS"
            continue

        # 4. Extract features from GEE
        all_pts = flood_pts + nonflood_pts
        all_labels = [1]*len(flood_pts) + [0]*len(nonflood_pts)

        try:
            rows = extract_features_from_gee(all_pts, all_labels, cfg, event_key)
        except Exception as e:
            print(f"  ERROR — GEE extraction: {e}")
            results[event_key] = f"GEE_ERR: {e}"
            continue

        if not rows:
            print(f"  SKIP — no features extracted")
            results[event_key] = "NO_FEATURES"
            continue

        # 5. Save CSV
        df = pd.DataFrame(rows)
        df["event"] = event_key
        df.to_csv(out_csv, index=False)
        n_flood = int((df["label"] == 1).sum())
        n_nonflood = int((df["label"] == 0).sum())
        print(f"  SAVED: {out_csv}")
        print(f"  → {len(df)} samples ({n_flood} flood, {n_nonflood} non-flood)")
        results[event_key] = f"OK ({len(df)} samples)"

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        icon = "✓" if v.startswith("OK") else "✗"
        print(f"  [{icon}] {k}: {v}")


if __name__ == "__main__":
    main()
