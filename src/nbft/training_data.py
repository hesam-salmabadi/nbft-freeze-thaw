"""Load the ML-ready training tables and prepare them for cross-validation and training."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.utils.class_weight import compute_sample_weight

from nbft import config, paths
from nbft.constants import LC_GROUPS, STRATIFY_COL, TARGET

# Months dropped per biome (snow-free season, no freeze-thaw signal)
SUMMER_MONTHS = [5, 6, 7, 8, 9]
SUMMER_MASKED_LCS = [5, 8, 12]

# Spatial blocks: pixels within this many grid cells are grouped (DBSCAN, EASE-Grid indices)
BLOCK_EPS_PIXELS = 20


def merge_landcover(lc):
    """Map raw land cover classes onto the five model biomes; others become NaN."""
    out = pd.Series(np.nan, index=lc.index)
    for biome, raw in LC_GROUPS.items():
        out[lc.isin(raw)] = biome
    return out


def spatial_blocks(df):
    """Cluster pixel locations into spatially separated blocks for grouped CV."""
    locs = df[["y", "x"]].drop_duplicates()
    locs["spatial_block"] = DBSCAN(eps=BLOCK_EPS_PIXELS, min_samples=1).fit(locs).labels_
    return locs


def site_weights(groups, y):
    """Balance each (spatial block, class) cell; clip at 5x the median weight."""
    w = compute_sample_weight(class_weight="balanced", y=groups.astype(str) + "_" + y.astype(str))
    return np.clip(w, None, np.median(w) * 5.0)


def build_station_metadata(meta_csv, blocks):
    """Station table (pixel, lat/lon, network, biome, spatial block) from the in situ metadata."""
    meta = pd.read_csv(meta_csv)
    meta = meta[meta["keep_pixel"]].dropna(subset=["wbf"])
    meta["land_cover_merged"] = merge_landcover(meta["lc_type_2_c"])
    meta["y"], meta["x"] = meta["row"], meta["col"]
    meta = meta[["y", "x", "lat", "long", "network", "land_cover_merged"]].drop_duplicates()
    meta = meta.merge(blocks, on=["y", "x"], how="left").dropna(subset=["spatial_block"])
    return meta


def load_training_data(write_station_metadata=False):
    """Return (am, am_trans, pm, pm_trans).

    ``am``/``pm`` hold confident labels (target 0/1);
    ``*_trans`` hold transitional samples (target 2).
    """
    parts = []
    for am_pm in ("am", "pm"):
        d = pd.read_parquet(config.path("ml_ready_out.training_table", am_pm=am_pm))
        d["source"] = am_pm
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    df["station_id"] = df["y"].astype(str) + "_" + df["x"].astype(str)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["station_id", "source", "time"])
    df["temp_3d_avg"] = df.groupby(["station_id", "source"])["t2m"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    df["qe"] = df["amsr_18h"] / df["amsr_36v"]
    df["qe1"] = df["smap_h"] / df["amsr_36v"]

    summer = df[STRATIFY_COL].isin(SUMMER_MASKED_LCS) & df["time"].dt.month.isin(SUMMER_MONTHS)
    df = df[~summer]

    df[STRATIFY_COL] = merge_landcover(df[STRATIFY_COL])
    df = df.dropna(subset=[STRATIFY_COL])
    df[STRATIFY_COL] = df[STRATIFY_COL].astype(int).astype("category")

    blocks = spatial_blocks(df)
    df = df.merge(blocks, on=["y", "x"], how="left")
    print(f"DBSCAN found {df['spatial_block'].nunique()} spatial blocks.")

    if write_station_metadata:
        meta = build_station_metadata(config.path("ml_ready.station_metadata_source"), blocks)
        paths.STATION_METADATA.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(paths.STATION_METADATA, index=False)
        n_pixels = len(meta[["y", "x"]].drop_duplicates())
        print(f"Station metadata: {n_pixels} pixels -> {paths.STATION_METADATA}")

    confident = df[TARGET].isin([0, 1])
    trans = df[TARGET] == 2
    am, pm = df["source"] == "am", df["source"] == "pm"
    return tuple(df[m].copy() for m in (confident & am, trans & am, confident & pm, trans & pm))
