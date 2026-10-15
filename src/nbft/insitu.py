"""In situ freeze-thaw labels of EASE-2 cells, read from the sfcc-label package (stage 2).

sfcc-label (model sfcc-joint-1.1) gives every topsoil sensor hourly P(frozen), P(transition) and
P(thawed), and keeps a sensor in a 9 km cell only if it passes the ESA CCI land-cover screen. Here
each sensor's probabilities are averaged over a 5-hour window centred on the overpass time in local
solar time; a cell's daily shares are the mean over its sensors, and a cell-day is frozen or thawed
when that share reaches the cut-off, otherwise partial (the sfcc-label grid-labels rule).
"""

import re

import numpy as np
import pandas as pd

from nbft import config, paths
from nbft.constants import MAX_WATER_FRACTION, TIME_END, TIME_START  # noqa: F401 (one water limit)

GRID = "N9km"
TROPIC_N = 23.44  # Asian cells must lie north of the Tropic of Cancer
GRID_SIZE = 2000
CELL_M = 9000.0
CUTOFF = 0.75  # same level as the frozen threshold of sfcc-label (75 % frozen)
CUTOFFS = (0.7, 0.75, 0.8, 0.85, 0.9)  # sensitivity range (12_cutoff_sensitivity.py)
OVERPASS_HOUR = {"am": 3, "pm": 15}  # local solar time
WINDOW_HALF_H = 2.5  # (-2.5, +2.5] h around the overpass: five hourly values
MIN_WINDOW_HOURS = 1  # one value is enough: 3-hourly loggers have only one or two in the window
MIN_SENSOR_DAYS = 30  # AM overpass days outside summer a sensor needs in the study period
SUMMER_MONTHS = (7, 8)  # dropped: fewer than 0.5 % of cell-days are frozen or partial
COLD_MONTHS = (9, 10, 11, 12, 1, 2)  # autumn and winter: the months a cell must cover
MIN_MONTH_DAYS = 15  # AM overpass days a cell needs in each cold month (pooled over years)
OWN_THRESHOLD_SOURCES = ("own_winter", "sensor_average")
LABELS = ("frozen", "partial", "thawed")
# A partial day is a clean transitional reference when at least this share of it is in transition
# (frozen and thawed days need 0.75 of their own state); other partial days are mixtures.
MIN_TRANSITION_SHARE = 0.5
SOIL_T_RANGE = (-60.0, 60.0)  # deg C; plausible soil temperatures (sfcc-label range check)
PROBS = ["p_frozen", "p_transition", "p_thawed"]

_CELL = re.compile(r"_r(\d{4})_c(\d{4})$")


def cell_rowcol(cell_ids):
    """Grid row and column of EASE2_N_9km_rXXXX_cYYYY ids."""
    rc = pd.Series(cell_ids).str.extract(_CELL).astype(int)
    return rc[0].to_numpy(), rc[1].to_numpy()


def read_cells(grid=GRID):
    return pd.read_csv(config.path("sfcc.gridded_cells", grid=grid))


def read_kept_sensors(grid=GRID):
    """Sensors that pass the land-cover screen on the grid, with their cell."""
    s = pd.read_csv(config.path("sfcc.gridded_sensors", grid=grid))
    return s[s["eligible"]].reset_index(drop=True)


def selected_sensors():
    """Sensors of the development set (02_select_cells.py): sensor_id and cell_id."""
    from nbft import paths

    s = pd.read_csv(paths.TABLES / "insitu_sensors.csv")
    return s.loc[s["selected"], ["sensor_id", "cell_id"]].reset_index(drop=True)


def outside_summer(days):
    """Rows of a daily table whose date is not in SUMMER_MONTHS."""
    return days[~pd.DatetimeIndex(days["date"]).month.isin(SUMMER_MONTHS)]


def read_sensor_days(columns=None):
    """Daily overpass probabilities (01_extract_overpass.py) of the selected sensors only, outside
    summer."""
    read = None if columns is None else list(dict.fromkeys([*columns, "date"]))
    s = pd.read_parquet(config.path("insitu.sensor_overpass", grid=GRID), columns=read)
    s = outside_summer(s[s["sensor_id"].isin(selected_sensors()["sensor_id"])])
    return s[columns if columns is not None else s.columns].reset_index(drop=True)


def read_catalog(columns=None):
    return pd.read_csv(config.path("sfcc.catalog"), usecols=columns)


def read_predictions(sensor_id):
    return pd.read_csv(
        config.path("sfcc.predictions", sensor_id=sensor_id),
        usecols=["timestamp_utc", *PROBS, "threshold_source"],
        dtype={"threshold_source": str},
    )


def overpass_window(timestamps_utc, lon, am_pm, start=TIME_START, end=TIME_END):
    """Local solar date of each timestamp and whether it lies in the 5-hour window around the
    overpass (local solar time from the longitude) within the period."""
    t = pd.to_datetime(timestamps_utc, utc=True).dt.tz_localize(None)
    lst = t + pd.to_timedelta(lon / 15.0, unit="h")
    date = lst.dt.floor("D")
    offset = (lst - date) / pd.Timedelta(hours=1) - OVERPASS_HOUR[am_pm]
    keep = (offset > -WINDOW_HALF_H) & (offset <= WINDOW_HALF_H)
    return date, keep & (date >= start) & (date < end)


def overpass_daily(pred, lon, am_pm, start=TIME_START, end=TIME_END):
    """Daily mean probabilities over the 5-hour window around the overpass in local solar time.

    Returns one row per local solar date with p_frozen, p_transition, p_thawed, the number of
    hours averaged and own_share, the fraction of those hours whose thresholds come from the
    sensor's own moisture record.
    """
    pred = pred[pred["p_frozen"].notna()]
    date, keep = overpass_window(pred["timestamp_utc"], lon, am_pm, start, end)
    d = pred.loc[keep, PROBS].assign(
        date=date[keep], own=pred.loc[keep, "threshold_source"].isin(OWN_THRESHOLD_SOURCES)
    )
    g = d.groupby("date")
    out = g[PROBS].mean().assign(n_hours=g.size(), own_share=g["own"].mean()).reset_index()
    return out[out["n_hours"] >= MIN_WINDOW_HOURS].reset_index(drop=True)


def overpass_soil_temperature(sensor_id, source, lon, am_pm="am"):
    """Daily mean soil temperature (deg C) of a sensor over the overpass window (sfcc-label
    standardized series of ``source``), one row per local solar date. Values outside
    SOIL_T_RANGE (fill values) are dropped, as in sfcc-label."""
    path = config.path("sfcc.standardized", source=source, sensor_id=sensor_id)
    x = pd.read_csv(path, usecols=["timestamp_utc", "soil_temperature_c"]).dropna()
    x = x[x["soil_temperature_c"].between(*SOIL_T_RANGE)]
    date, keep = overpass_window(x["timestamp_utc"], lon, am_pm)
    d = x.loc[keep, ["soil_temperature_c"]].assign(date=date[keep])
    return d.groupby("date")["soil_temperature_c"].mean().rename("soil_t").reset_index()


def labels(frozen_share, thawed_share, cutoff=CUTOFF):
    """frozen / thawed / partial, the sfcc-label grid-labels rule (a 1e-9 tolerance absorbs float
    round-off at the cut-off)."""
    frozen = np.asarray(frozen_share) >= cutoff - 1e-9
    thawed = np.asarray(thawed_share) >= cutoff - 1e-9
    return np.select([frozen, thawed], ["frozen", "thawed"], "partial")


def freeze_year(date):
    """Freeze year (1 August to 31 July) named by the year it starts in."""
    date = pd.DatetimeIndex(date)
    return date.year - (date.month < 8)


ROUGHNESS_BINS = [0, 25, 75, 150, np.inf]  # elevation std within the cell (m), as in Fig. 1
ROUGHNESS_NAMES = ["0-25 m", "25-75 m", "75-150 m", ">150 m"]


def roughness_class(std_m):
    return pd.cut(pd.Series(std_m), ROUGHNESS_BINS, labels=ROUGHNESS_NAMES, right=False)


def tree_cover_class(pct):
    bins = [-np.inf, 10, 30, 60, np.inf]
    names = ["<10%", "10-30%", "30-60%", ">60%"]
    return pd.cut(pd.Series(pct), bins, labels=names, right=False)


def _countries():
    import cartopy.io.shapereader as shpreader
    import geopandas as gpd

    path = shpreader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
    return gpd.read_file(path)[["ADMIN", "CONTINENT", "geometry"]]


def regions(lat, lon):
    """'north_america' (Canada and the United States without Hawaii), 'europe' (west of 60 E),
    'asia' (Asia and Russia east of 60 E, north of the Tropic of Cancer) or 'other'. Points just
    off a coastline take the nearest country within 0.3 deg."""
    import geopandas as gpd

    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326)
    countries = _countries()
    hit = gpd.sjoin(pts, countries, how="left", predicate="within")
    hit = hit[~hit.index.duplicated()]
    miss = hit["ADMIN"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss], countries, how="left", max_distance=0.3)
        near = near[~near.index.duplicated()]
        hit.loc[miss, ["ADMIN", "CONTINENT"]] = near[["ADMIN", "CONTINENT"]].to_numpy()
    admin, continent = hit["ADMIN"].to_numpy(), hit["CONTINENT"].to_numpy()
    hawaii = (lon < -150) & (lat < 30)
    na = np.isin(admin, ["Canada", "United States of America"]) & ~hawaii
    europe = (continent == "Europe") & (lon < 60)
    asia = ((continent == "Asia") | ((admin == "Russia") & (lon >= 60))) & (lat >= TROPIC_N)
    return np.select([na, europe, asia], ["north_america", "europe", "asia"], "other")


def cci_water_fraction(grid=GRID):
    """Open-water fraction (0-1) of every cell from ESA CCI 2015, as in the land-cover screen."""
    return read_band("landcover_cci2015", "water_fraction", grid) / 10000.0


def grid_lonlat():
    """Longitude and latitude of every N9km cell centre (GRID_SIZE x GRID_SIZE)."""
    from pyproj import Transformer

    xy = -9e6 + (np.arange(GRID_SIZE) + 0.5) * CELL_M
    x, y = np.meshgrid(xy, xy[::-1])
    return Transformer.from_crs(6931, 4326, always_xy=True).transform(x, y)


def north_america_mask():
    """True for N9km cells whose centre lies in Canada or the United States (without Hawaii)."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    countries = _countries()
    na = countries[countries["ADMIN"].isin(["Canada", "United States of America"])].to_crs(6931)
    transform = from_origin(-9e6, 9e6, CELL_M, CELL_M)
    mask = rasterize(na.geometry, out_shape=(GRID_SIZE, GRID_SIZE), transform=transform) == 1
    lon, lat = grid_lonlat()
    return mask & ~((lon < -150) & (lat < 30))


def read_band(name, band, grid=GRID):
    """One band of an sfcc-label wall-to-wall raster, e.g. ('terrain', 'elevation_std')."""
    import rasterio

    path = config.path("sfcc.landcover_raster", name=name, grid=grid)
    with rasterio.open(path) as src:
        return src.read(src.descriptions.index(band) + 1)


INDEX_SIGNS = {"npr": -1, "ndv": 1, "ndh": 1, "wd": 1}  # +1 if frozen soil raises the index


def add_indices(x):
    """The four model indices from the brightness temperatures of 10_extract_satellite.py."""
    from nbft import features as F

    return x.assign(
        npr=F.npr(x["smap_v"], x["smap_h"]),
        ndv=F.ndv(x["amsr_18v"], x["amsr_36v"]),
        ndh=F.ndh(x["amsr_18h"], x["amsr_36v"]),
        wd=F.wd(x["smap_v"], x["amsr_36v"]),
    )


FEATURES = ["z_npr", "z_ndv", "z_ndh", "z_wd"]  # indices standardized per pixel (thaw reference)
BLOCK_CELLS = 22  # spatial block for held-out cells: 22 x 9 km, about 200 km


def index_references(cell_ids):
    """Per-pixel thaw-season reference and std of each index at the cells (AM), as in the model."""
    from nbft.sources import INDEX_REFS, INDEX_STD, reference_field

    rows, cols = cell_rowcol(cell_ids)
    out = pd.DataFrame({"cell_id": list(cell_ids)})
    for name, legacy in INDEX_REFS.items():
        for kind, col in (("thaw_ref", f"ref_{name}"), (INDEX_STD, f"std_{name}")):
            field = reference_field(legacy, kind, "am")[legacy].values
            out[col] = field[rows, cols]
    return out


def add_zscores(x):
    """z_* features: index minus the cell's thaw reference, over its thawed-season std (0 -> 1)."""
    x = add_indices(x).merge(index_references(x["cell_id"].unique()), on="cell_id")
    for name in INDEX_SIGNS:
        std = x[f"std_{name}"].where(x[f"std_{name}"] != 0, 1.0)
        x[f"z_{name}"] = (x[name] - x[f"ref_{name}"]) / std
    return x


def spatial_blocks(cell_ids, size=BLOCK_CELLS):
    """Block id of each cell: cells in one block are always held out together."""
    rows, cols = cell_rowcol(cell_ids)
    return pd.Series(rows // size * 1000 + cols // size, index=getattr(cell_ids, "index", None))


def block_folds(y, blocks, n_splits=5, seed=0):
    """Cross-validation folds that hold out whole spatial blocks, stratified by ``y`` so that the
    folds hold similar shares of each class (StratifiedGroupKFold)."""
    from sklearn.model_selection import StratifiedGroupKFold

    folds = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
    return list(folds.split(np.zeros(len(y)), np.asarray(y), np.asarray(blocks)))


def held_out_predictions(train, test, features, n_splits=5, seed=0):
    """P(frozen) for every row of `test` from a model that never saw its spatial block.

    The model is the paper's logistic regression (model.fit_logistic) on `features`, with equal
    total weight per spatial block (model.block_weights). When `train` is `test`, folds are
    spatial blocks stratified by class (block_folds); otherwise one model is fitted on `train`
    (which must share no block with `test`).
    """
    from nbft.model import block_weights, fit_logistic, probability

    def fit(df):
        w = block_weights(spatial_blocks(df["cell_id"]))
        return fit_logistic(df[features], df["frozen"], w)

    p = pd.Series(np.nan, index=test.index)
    if train is not test:
        p[:] = probability(fit(train), test[features])
        return p
    groups = spatial_blocks(test["cell_id"]).to_numpy()
    for fit_idx, hold in block_folds(
        test["frozen"], groups, min(n_splits, len(np.unique(groups))), seed
    ):
        tr = test.iloc[fit_idx]
        if tr["frozen"].nunique() < 2:
            continue
        p.iloc[hold] = probability(fit(tr), test.iloc[hold][features]).to_numpy()
    return p


def development_days():
    """Labelled AM days of the development cells outside summer, with the model inputs.

    Columns: cell_id, date, label (frozen / partial / thawed at the 0.75 cut-off), frozen (1/0;
    NaN for partial days), the standardized indices z_npr, z_ndv, z_ndh, z_wd, tree (tree-cover
    fraction), tree_class, roughness_class (elevation std within the cell), biome (RESOLVE),
    latitude, block (spatial block), transition_share and clean (False for partial days with less
    than MIN_TRANSITION_SHARE in transition). Rows lacking any index are dropped.
    """
    sat = pd.read_parquet(config.path("insitu.cell_satellite", grid=GRID))
    labels = pd.read_parquet(config.path("insitu.cell_overpass", grid=GRID))
    columns = ["cell_id", "date", f"label_{CUTOFF}", "transition_share"]
    labels = labels.loc[labels["am_pm"] == "am", columns]
    sat["date"], labels["date"] = pd.to_datetime(sat["date"]), pd.to_datetime(labels["date"])
    sat = sat[~sat["date"].dt.month.isin(SUMMER_MONTHS)]
    x = add_zscores(sat).merge(labels.rename(columns={f"label_{CUTOFF}": "label"}))
    x = x[x["label"].isin(LABELS)].dropna(subset=list(FEATURES))
    cells = pd.read_csv(paths.TABLES / "insitu_cell_table.csv")
    cells = cells.loc[
        cells["set"] == "development",
        ["cell_id", "tree_cover_pct", "elevation_std_m", "biome", "latitude"],
    ]
    x = x.merge(cells, on="cell_id").dropna(subset=["tree_cover_pct"])
    x["frozen"] = x["label"].map({"frozen": 1.0, "thawed": 0.0})
    x["tree"] = x["tree_cover_pct"] / 100
    x["tree_class"] = tree_cover_class(x["tree_cover_pct"]).astype(str).to_numpy()
    x["roughness_class"] = roughness_class(x["elevation_std_m"]).astype(str).to_numpy()
    x["block"] = spatial_blocks(x["cell_id"]).to_numpy()
    keep = ["cell_id", "date", "label", "frozen", *FEATURES, "tree", "tree_class"]
    x["clean"] = (x["label"] != "partial") | (x["transition_share"] >= MIN_TRANSITION_SHARE)
    keep += ["roughness_class", "biome", "latitude", "block", "transition_share", "clean"]
    return x[keep].reset_index(drop=True)


def cell_scores(x, p, min_state_days=10):
    """Per cell: AUC of the held-out P(frozen) against the in situ label, and balanced accuracy
    at P = 0.5. Cells need min_state_days frozen and thawed days."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    rows = []
    for cid, g in x.assign(p=p).dropna(subset=["p"]).groupby("cell_id"):
        nf = int(g["frozen"].sum())
        if nf < min_state_days or len(g) - nf < min_state_days:
            continue
        rows.append(
            {
                "cell_id": cid,
                "frozen_days": nf,
                "thawed_days": len(g) - nf,
                "auc": roc_auc_score(g["frozen"], g["p"]),
                "balanced_accuracy": balanced_accuracy_score(g["frozen"], g["p"] >= 0.5),
            }
        )
    return pd.DataFrame(rows)
