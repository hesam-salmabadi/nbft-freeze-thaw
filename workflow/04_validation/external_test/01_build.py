"""External test data: labelled cells the model never saw (Section 3.6.4 and Supplement).

Two sets of morning cell-days in the study period outside July and August:
- external: cells outside North America, selected with the rules of the development cells
  (02_insitu_labels/02_select_cells.py): sensors kept by the sfcc-label land-cover screen, cell with
  at most 15 % open water, at least insitu.MIN_SENSOR_DAYS AM days per sensor; the cell label is
  the mean of its sensors' probabilities. Cells must freeze, with on average at least
  MIN_FROZEN_PER_YEAR frozen mornings per freeze year (1 August to 31 July) with data, because
  cells that never freeze test only the thawed class.
- arctic_sites: George River (Quebec) and Cambridge Bay (Nunavut), North American sites removed by
  the land-cover screen; labels built the same way from the sfcc-label predictions (supplement).

For every cell-day: the in situ label and transition share, the clean flag, the model inputs
(Z_NPR and Z_NDV against the thaw reference and the July-August std of the cell's own AM series,
as in 01_preprocess_satellite/01_thaw_std.py, and the tree-cover fraction), the open-water
fraction, the in situ soil temperature (cell mean of sensor means in the AM window), the FT-SMOS
morning state, the region and the spatial block.

Slow: reads the gap-filled TB cube tile by tile at the cells.

Output: validation.external_days.

Usage: python 01_build.py
"""

import numpy as np
import pandas as pd
import xarray as xr

from nbft import config, features, insitu
from nbft.grid import open_zarr
from nbft.sources import INDEX_REFS, reference_field

MIN_FROZEN_PER_YEAR = 10
ARCTIC_SITES = ["George River", "Cambridge Bay"]
CHANNELS = ["smap_v", "smap_h", "amsr_18v", "amsr_36v"]
TILE = 100
MIN_SUMMER_DAYS = 30
SMOS_NO_DATA = 255


def in_period(days):
    days = days[(days["date"] >= insitu.TIME_START) & (days["date"] < insitu.TIME_END)]
    return days[~days["date"].dt.month.isin(insitu.SUMMER_MONTHS)]


def enough_days(sensor_days):
    n = sensor_days.groupby("sensor_id").size()
    return sensor_days[sensor_days["sensor_id"].isin(n[n >= insitu.MIN_SENSOR_DAYS].index)]


def cell_days(sensor_days):
    g = sensor_days.groupby(["cell_id", "date"])
    d = g[insitu.PROBS].mean().assign(n_sensors=g.size()).reset_index()
    d["label"] = insitu.labels(d["p_frozen"], d["p_thawed"])
    d = d.rename(columns={"p_transition": "transition_share"})
    d["clean"] = (d["label"] != "partial") | (d["transition_share"] >= insitu.MIN_TRANSITION_SHARE)
    return d[["cell_id", "date", "label", "transition_share", "clean", "n_sensors"]]


def external_cells():
    so = pd.read_parquet(config.path("insitu.sensor_overpass", grid=insitu.GRID))
    so = so[so["am_pm"] == "am"].copy()
    so["date"] = pd.to_datetime(so["date"])
    so = in_period(so)
    so = so[so["sensor_id"].isin(insitu.read_kept_sensors()["sensor_id"])]
    r, c = insitu.cell_rowcol(so["cell_id"])
    so = so[~insitu.north_america_mask()[r, c]]
    r, c = insitu.cell_rowcol(so["cell_id"])
    so = enough_days(so[insitu.cci_water_fraction()[r, c] <= insitu.MAX_WATER_FRACTION])
    d = cell_days(so)
    year = d["date"].dt.year - (d["date"].dt.month < 8)
    frozen = (d["label"] == "frozen").groupby(d["cell_id"]).sum()
    per_year = frozen / year.groupby(d["cell_id"]).nunique()
    keep = per_year[per_year >= MIN_FROZEN_PER_YEAR].index
    print(f"external: {d['cell_id'].nunique()} cells pass the rules, {len(keep)} freeze")
    return d[d["cell_id"].isin(keep)].assign(set="external"), so[so["cell_id"].isin(keep)]


def arctic_sites():
    cat = insitu.read_catalog(["sensor_id", "network", "longitude"])
    grid = pd.read_csv(config.path("sfcc.grid_cells"), usecols=["sensor_id", "EASE2_N_9km"])
    sites = cat[cat["network"].isin(ARCTIC_SITES)].merge(grid, on="sensor_id")
    frames = []
    for sid, lon, cell in sites[["sensor_id", "longitude", "EASE2_N_9km"]].itertuples(False, None):
        if config.path("sfcc.predictions", sensor_id=sid).exists():
            d = insitu.overpass_daily(insitu.read_predictions(sid), lon, "am")
            frames.append(d.assign(sensor_id=sid, cell_id=cell))
    so = enough_days(in_period(pd.concat(frames, ignore_index=True)))
    print(f"arctic sites: {so['sensor_id'].nunique()} sensors, {so['cell_id'].nunique()} cells")
    return cell_days(so).assign(set="arctic_sites"), so


def tb_series(cells):
    """Full AM gap-filled TB series of the channels at the cells, read tile by tile."""
    ds = open_zarr(config.path("timeseries.tb_gapfilled"), "am")
    time = pd.DatetimeIndex(ds["time"].values)
    rows, cols = insitu.cell_rowcol(pd.Series(cells))
    frame = pd.DataFrame({"cell_id": cells, "r": rows, "c": cols})
    frame["r0"], frame["c0"] = frame["r"] // TILE * TILE, frame["c"] // TILE * TILE
    out = []
    for (r0, c0), g in frame.groupby(["r0", "c0"]):
        block = {
            ch: ds[ch]
            .isel(y=slice(r0, r0 + TILE), x=slice(c0, c0 + TILE))
            .values[:, g["r"].to_numpy() - r0, g["c"].to_numpy() - c0]
            for ch in CHANNELS
        }
        for j, cid in enumerate(g["cell_id"]):
            out.append(
                pd.DataFrame({ch: block[ch][:, j] for ch in CHANNELS}, index=time)
                .rename_axis("date")
                .reset_index()
                .assign(cell_id=cid)
            )
    return pd.concat(out, ignore_index=True)


def model_inputs(cells):
    tb = tb_series(cells)
    tb["npr"] = features.npr(tb["smap_v"], tb["smap_h"])
    tb["ndv"] = features.ndv(tb["amsr_18v"], tb["amsr_36v"])
    summer = tb["date"].dt.month.isin(insitu.SUMMER_MONTHS) & tb["date"].between(
        insitu.TIME_START, insitu.TIME_END, inclusive="left"
    )
    sd = (
        tb[summer]
        .groupby("cell_id")[["npr", "ndv"]]
        .agg(lambda v: v.std(ddof=1) if v.notna().sum() >= MIN_SUMMER_DAYS else np.nan)
    )
    rows, cols = insitu.cell_rowcol(pd.Series(cells))
    tree = insitu.read_band("treecover", "mean")
    static = pd.DataFrame(
        {
            "cell_id": cells,
            "tree": tree[rows, cols] / 100,
            "tree_class": insitu.tree_cover_class(tree[rows, cols]).astype(str).to_numpy(),
            "water": insitu.cci_water_fraction()[rows, cols],
        }
    )
    for name in ("npr", "ndv"):
        legacy = INDEX_REFS[name]
        static[f"ref_{name}"] = reference_field(legacy, "thaw_ref", "am")[legacy].values[rows, cols]
    x = tb.merge(sd.add_prefix("sd_").reset_index(), on="cell_id").merge(static, on="cell_id")
    for name in ("npr", "ndv"):
        x[f"z_{name}"] = (x[name] - x[f"ref_{name}"]) / x[f"sd_{name}"]
    return x[["cell_id", "date", "z_npr", "z_ndv", "tree", "tree_class", "water"]]


def soil_temperature(sensors):
    cat = insitu.read_catalog(["sensor_id", "source", "longitude"]).drop_duplicates("sensor_id")
    s = sensors[["sensor_id", "cell_id"]].drop_duplicates().merge(cat, on="sensor_id")
    t = pd.concat(
        [
            insitu.overpass_soil_temperature(sid, src, lon).assign(cell_id=cell)
            for sid, cell, src, lon in s[
                ["sensor_id", "cell_id", "source", "longitude"]
            ].itertuples(False, None)
        ],
        ignore_index=True,
    )
    return t.groupby(["cell_id", "date"])["soil_t"].mean().reset_index()


def smos_states(cells, dates):
    smos = xr.open_zarr(config.path("timeseries.ft_smos"))["ft_status"].sel(am_pm="am")
    r, c = insitu.cell_rowcol(pd.Series(cells))
    t = pd.DatetimeIndex(sorted(dates)).intersection(pd.DatetimeIndex(smos["time"].values))
    pts = {"y": xr.DataArray(r, dims="cell"), "x": xr.DataArray(c, dims="cell")}
    v = pd.DataFrame(smos.sel(time=t).isel(**pts).values, index=t, columns=list(cells))
    s = v.rename_axis("date").stack().rename("smos_state").reset_index()
    s = s.rename(columns={"level_1": "cell_id"})
    return s[np.isfinite(s["smos_state"]) & (s["smos_state"] != SMOS_NO_DATA)]


def regions(sensors):
    cat = insitu.read_catalog(["sensor_id", "network", "latitude", "longitude"])
    m = sensors[["sensor_id", "cell_id"]].drop_duplicates().merge(cat.drop_duplicates("sensor_id"))
    m = m.groupby("cell_id").agg(
        network=("network", lambda s: "+".join(sorted(set(s)))),
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
    )
    asia = m["lon"] > 60
    m["region"] = np.select(
        [asia & (m["lat"] < 38), asia, m["lat"] > 60, m["lon"] < -20],
        ["Tibetan Plateau", "Mongolia and northern China", "Finland", "Arctic Canada"],
        "Romania",
    )
    m.loc[m["network"].isin(ARCTIC_SITES), "region"] = m["network"]
    return m[["network", "region"]]


def main():
    ext, ext_sensors = external_cells()
    arc, arc_sensors = arctic_sites()
    days = pd.concat([ext, arc], ignore_index=True)
    sensors = pd.concat([ext_sensors, arc_sensors], ignore_index=True)
    cells = days["cell_id"].unique()
    days = days.merge(model_inputs(cells), on=["cell_id", "date"], how="left")
    days = days.merge(soil_temperature(sensors), on=["cell_id", "date"], how="left")
    days = days.merge(smos_states(cells, days["date"].unique()), on=["cell_id", "date"], how="left")
    days = days.merge(regions(sensors), left_on="cell_id", right_index=True, how="left")
    days["block"] = insitu.spatial_blocks(days["cell_id"]).to_numpy()
    out = config.path("validation.external_days")
    out.parent.mkdir(parents=True, exist_ok=True)
    days.to_parquet(out, index=False)
    summary = days.groupby(["set", "region"]).agg(
        cells=("cell_id", "nunique"), blocks=("block", "nunique"), days=("label", "size")
    )
    print(summary.to_string(), f"-> {out}", sep="\n")


if __name__ == "__main__":
    main()
