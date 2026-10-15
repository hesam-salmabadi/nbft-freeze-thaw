"""Evaluation data at the development cells (Section 3.6).

Parts (run one or all):
- soil: in situ soil temperature of each development cell-day, the mean over the cell's sensors of
  each sensor's mean over the AM overpass window (insitu.overpass_soil_temperature);
- air: ERA5-Land 10-day running mean of the daily mean 2 m air temperature at the cell
  (timeseries.era5_t2m_daily, t2m_10d), deg C;
- smos: FT-SMOS morning (ascending) state at the cell (timeseries.ft_smos; 0 thawed, 1 frozen,
  2 partially frozen; no data dropped);
- domain: the state of the maps (product.ft_states) and the ERA5-Land air temperature on
  N_DOMAIN_CELLS random cells of the product domain, every day.
Days: the study period outside July and August.

Outputs: model.eval_soil_temperature, model.eval_air_temperature, model.eval_smos,
model.eval_domain_sample.

Usage: python 08_extract_evaluation_data.py [soil|air|smos|domain ...] [--workers N]
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from nbft import config, insitu, paths

KELVIN = 273.15
SMOS_NO_DATA = 255
N_DOMAIN_CELLS = 2000
SEED = 42


def development_cells():
    cells = pd.read_csv(paths.TABLES / "insitu_cell_table.csv")
    return cells.loc[cells["set"] == "development", "cell_id"]


def days():
    t = pd.date_range(insitu.TIME_START, insitu.TIME_END, inclusive="left")
    return t[~t.month.isin(insitu.SUMMER_MONTHS)]


def _sensor_soil(task):
    sensor_id, cell_id, source, lon = task
    t = insitu.overpass_soil_temperature(sensor_id, source, lon)
    return t.assign(sensor_id=sensor_id, cell_id=cell_id)


def soil(workers):
    sensors = insitu.read_sensor_days(["sensor_id", "cell_id"]).drop_duplicates()
    sensors = sensors[sensors["cell_id"].isin(development_cells())]
    sensors = sensors.merge(insitu.read_catalog(["sensor_id", "source", "longitude"]))
    tasks = list(sensors[["sensor_id", "cell_id", "source", "longitude"]].itertuples(False, None))
    with ProcessPoolExecutor(workers) as pool:
        t = pd.concat(pool.map(_sensor_soil, tasks, chunksize=8), ignore_index=True)
    t = t[~t["date"].dt.month.isin(insitu.SUMMER_MONTHS)]
    cell = t.groupby(["cell_id", "date"]).agg(
        soil_t=("soil_t", "mean"), soil_sensors=("sensor_id", "nunique")
    )
    print(f"soil: {len(tasks)} sensors, {len(cell):,} cell-days")
    return cell.reset_index()


def at_cells(da, cells, time):
    """Values of a (time, y, x) DataArray at the cells (vectorized point selection)."""
    rows, cols = insitu.cell_rowcol(cells)
    pts = {"y": xr.DataArray(rows, dims="cell"), "x": xr.DataArray(cols, dims="cell")}
    v = da.sel(time=time).isel(**pts).values  # (time, cell)
    return pd.DataFrame(v, index=time, columns=list(cells)).rename_axis("date")


def air():
    ds = xr.open_zarr(config.path("timeseries.era5_t2m_daily"))
    cells = development_cells()
    t = days().intersection(pd.DatetimeIndex(ds["time"].values))
    v = at_cells(ds["t2m_10d"], cells, t)
    out = v.stack().rename("air_t10").reset_index().rename(columns={"level_1": "cell_id"})
    if out["air_t10"].median() > 100:  # stored in kelvin
        out["air_t10"] -= KELVIN
    print(f"air: {len(out):,} cell-days")
    return out


def smos():
    ds = xr.open_zarr(config.path("timeseries.ft_smos"))
    cells = development_cells()
    t = days().intersection(pd.DatetimeIndex(ds["time"].values))
    v = at_cells(ds["ft_status"].sel(am_pm="am"), cells, t)
    out = v.stack().rename("smos_state").reset_index().rename(columns={"level_1": "cell_id"})
    out = out[np.isfinite(out["smos_state"]) & (out["smos_state"] != SMOS_NO_DATA)]
    print(f"smos: {len(out):,} cell-days")
    return out


def long_table(wide, name):
    return wide.stack().rename(name).reset_index().rename(columns={"level_1": "cell_id"})


def domain():
    tree = insitu.read_band("treecover", "mean")
    tree = np.where(np.isfinite(tree), tree, 0.0)  # no value north of 80 N, as in 07_upscale.py
    ice = xr.open_dataset(config.path("statics.ice_fraction"))["ice_fraction"].values
    rows, cols = np.nonzero(insitu.north_america_mask() & (ice <= 0.5))  # as in 07_upscale.py
    pick = np.random.default_rng(SEED).choice(len(rows), N_DOMAIN_CELLS, replace=False)
    cells = pd.Index(
        [f"EASE2_N_9km_r{r:04d}_c{c:04d}" for r, c in zip(rows[pick], cols[pick], strict=True)]
    )

    era5 = xr.open_zarr(config.path("timeseries.era5_t2m_daily"))
    t = days().intersection(pd.DatetimeIndex(era5["time"].values))
    air_t = long_table(at_cells(era5["t2m_10d"], cells, t), "air_t10")
    if air_t["air_t10"].median() > 100:
        air_t["air_t10"] -= KELVIN

    product = xr.open_zarr(config.path("product.ft_states", am_pm="am"))
    tb = xr.open_zarr(config.path("timeseries.tb_gapfilled"))
    y_index = {v: i for i, v in enumerate(product["y"].values)}
    x_index = {v: i for i, v in enumerate(product["x"].values)}
    r, c = insitu.cell_rowcol(cells)
    py = [y_index[v] for v in tb["y"].values[r]]
    px = [x_index[v] for v in tb["x"].values[c]]
    pts = {"y": xr.DataArray(py, dims="cell"), "x": xr.DataArray(px, dims="cell")}
    tp = t.intersection(pd.DatetimeIndex(product["time"].values))
    state = product["state"].sel(time=tp).isel(**pts).values
    state = long_table(
        pd.DataFrame(state, index=tp, columns=list(cells)).rename_axis("date"), "state"
    )

    out = state.merge(air_t, on=["cell_id", "date"])
    rr, cc = insitu.cell_rowcol(out["cell_id"])
    out["tree_class"] = insitu.tree_cover_class(tree[rr, cc]).astype(str).to_numpy()
    print(f"domain: {out['cell_id'].nunique()} cells, {len(out):,} cell-days")
    return out


def main(parts, workers=8):
    builders = {"soil": lambda: soil(workers), "air": air, "smos": smos, "domain": domain}
    keys = {
        "soil": "soil_temperature",
        "air": "air_temperature",
        "smos": "smos",
        "domain": "domain_sample",
    }
    for part in parts:
        df = builders[part]()
        key = keys[part]
        out = config.path(f"model.eval_{key}")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"-> {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 8
    parts = [a for a in args if a in ("soil", "air", "smos", "domain")]
    parts = parts or ["soil", "air", "smos", "domain"]
    main(parts, workers)
