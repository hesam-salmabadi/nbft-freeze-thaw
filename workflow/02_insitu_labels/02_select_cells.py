"""Select the development cells: every North American cell with representative topsoil sensors.

Starting point: topsoil sensors of sfcc-label in North America (Canada and the United States
without Hawaii) with at least one AM overpass value in the study period. Then:
1. sfcc-label's land-cover screen: a sensor is kept only if it represents the land cover of its
   cell (ESA CCI 2015: the sensor's class is the cell's dominant class and covers at least 70 % of
   it; the cell has at most 15 % open water and 15 % "other").
2. Summer days (SUMMER_MONTHS, July and August) are dropped: almost every cell-day is thawed then.
   Sensors with fewer than MIN_SENSOR_DAYS AM overpass days left are removed.
3. Sensors of FAULTY_STATIONS are removed: their soil temperature record is physically impossible
   (found by comparing labels with the ERA5-Land air temperature of each cell).
Every remaining day of every remaining cell is kept; balancing seasons and cells is part of model
development. Each cell carries its number of freeze years with data and of complete cold seasons
(freeze year, 1 August to 31 July, in which each month from September to February has at least
MIN_MONTH_DAYS AM days). There is no terrain-roughness cut (11_roughness_effect.py reports
accuracy by roughness range), and cells that never freeze stay in.

Writes the daily cell shares and labels (all cut-offs), the cell table, and the counts after each
step (Table 1).

Usage: python 02_select_cells.py [workers]
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from nbft import config, insitu, paths

# (network, station): reason. ARM Tyro (Kansas, 37 N): all three sensors read about -10 C from
# January to May and are labelled frozen into May, while the 10-day air temperature is above 5 C.
FAULTY_STATIONS = {("ARM", "Tyro"): "soil about -10 C from January to May"}


def cell_daily(sensor_days):
    """Cell shares per day: mean over the cell's sensors, then frozen/partial/thawed."""
    g = sensor_days.groupby(["cell_id", "am_pm", "date"])
    d = g[insitu.PROBS].mean()
    d = d.assign(n_sensors=g.size(), own_share=g["own_share"].mean()).reset_index()
    d = d.rename(
        columns={
            "p_frozen": "frozen_share",
            "p_transition": "transition_share",
            "p_thawed": "thawed_share",
        }
    )
    for cutoff in insitu.CUTOFFS:
        d[f"label_{cutoff:g}"] = insitu.labels(d["frozen_share"], d["thawed_share"], cutoff)
    return d


def has_period_data(task):
    """True if the sensor has at least one AM overpass value in the study period."""
    sensor_id, lon = task
    return not insitu.overpass_daily(insitu.read_predictions(sensor_id), lon, "am").empty


def in_period(sensor_ids, workers):
    lon = insitu.read_catalog(["sensor_id", "longitude"]).set_index("sensor_id")["longitude"]
    tasks = list(zip(sensor_ids, lon.reindex(sensor_ids), strict=True))
    with ProcessPoolExecutor(workers) as pool:
        found = list(pool.map(has_period_data, tasks, chunksize=16))
    return pd.Series(found, index=sensor_ids)


def cold_seasons(am):
    """Per cell: freeze years with any AM day, and complete cold seasons (every month September to
    February with at least MIN_MONTH_DAYS AM days in that freeze year)."""
    date = pd.DatetimeIndex(am["date"])
    key = [am["cell_id"].to_numpy(), insitu.freeze_year(date), date.month]
    counts = am.groupby(key).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(insitu.COLD_MONTHS), fill_value=0)
    complete = (counts >= insitu.MIN_MONTH_DAYS).all(axis=1).groupby(level=0).sum()
    years = pd.Series(insitu.freeze_year(date), index=am.index).groupby(am["cell_id"]).nunique()
    return pd.DataFrame({"freeze_years": years, "complete_seasons": complete})


def sensor_filter(sensor_days, sensor_ids):
    """Per sensor: AM days outside summer in the study period, and whether there are too few."""
    am = insitu.outside_summer(sensor_days[sensor_days["am_pm"] == "am"])
    out = pd.DataFrame({"am_days": am.groupby("sensor_id").size()})
    out = out.reindex(sensor_ids.unique(), fill_value=0)
    out["reason"] = np.where(out["am_days"] < insitu.MIN_SENSOR_DAYS, "fewer_than_30_days", "")
    cat = insitu.read_catalog(["sensor_id", "network", "station"]).drop_duplicates("sensor_id")
    station = cat.set_index("sensor_id").reindex(out.index)
    faulty = pd.Series(list(zip(station["network"], station["station"], strict=True)), out.index)
    out.loc[faulty.isin(list(FAULTY_STATIONS)) & (out["reason"] == ""), "reason"] = "faulty_record"
    return out


def assign_sets(cells):
    rules = [
        (cells["region"] != "north_america", "excluded", "outside_north_america"),
        (cells["am_days"] == 0, "excluded", "no_sensor_left"),
    ]
    cells["set"], cells["set_reason"] = "development", "north_america"
    decided = pd.Series(False, index=cells.index)
    for cond, name, reason in rules:
        hit = cond & ~decided
        cells.loc[hit, "set"], cells.loc[hit, "set_reason"] = name, reason
        decided |= hit
    return cells


def attrition(screened, filtered, cells, sensor_days):
    """Sensors, cells, cell-years and complete cold seasons left after each selection step."""
    s = screened[screened["in_period"] & screened["north_america"]]
    kept = s["eligible"]
    rows = [
        {
            "step": "Topsoil sensors in North America with observations in the study period",
            "sensors": len(s),
            "cells": s["cell_id"].nunique(),
        }
    ]
    for reason, n in s.loc[~kept, "reason"].value_counts().items():
        rows.append({"step": f"  removed by land-cover screen: {reason}", "sensors": -n})
    rep = s.loc[kept, "sensor_id"]
    rows.append(
        {
            "step": "Representative of the land cover of their cell",
            "sensors": len(rep),
            "cells": s.loc[kept, "cell_id"].nunique(),
        }
    )
    f = filtered.reindex(rep)
    for reason, n in f.loc[f["reason"] != "", "reason"].value_counts().items():
        rows.append({"step": f"  removed: {reason}", "sensors": -n})
    dev = cells[cells["set"] == "development"]
    am = sensor_days[(sensor_days["am_pm"] == "am") & sensor_days["cell_id"].isin(dev["cell_id"])]
    rows.append(
        {
            "step": "Development set",
            "sensors": am["sensor_id"].nunique(),
            "cells": len(dev),
            "cell_years": int(dev["freeze_years"].sum()),
            "complete_cell_seasons": int(dev["complete_seasons"].sum()),
        }
    )
    return pd.DataFrame(rows)


def main(workers=8):
    cells = insitu.read_cells()
    rows, cols = insitu.cell_rowcol(cells["cell_id"])
    cells["region"] = insitu.regions(cells["latitude"], cells["longitude"])
    cells["water_fraction"] = insitu.cci_water_fraction()[rows, cols]

    sensor_days = pd.read_parquet(config.path("insitu.sensor_overpass", grid=insitu.GRID))
    filtered = sensor_filter(sensor_days, sensor_days["sensor_id"])
    keep_sensors = filtered.index[filtered["reason"] == ""]
    sensor_days = insitu.outside_summer(sensor_days[sensor_days["sensor_id"].isin(keep_sensors)])
    daily = cell_daily(sensor_days)
    out = config.path("insitu.cell_overpass", grid=insitu.GRID)
    daily.to_parquet(out, index=False)

    am = daily[daily["am_pm"] == "am"]
    per_cell = am.groupby("cell_id").agg(
        am_days=("date", "size"),
        first_day=("date", "min"),
        last_day=("date", "max"),
        max_frozen_share=("frozen_share", "max"),
        ever_frozen=(f"label_{insitu.CUTOFF:g}", lambda s: bool((s == "frozen").any())),
    )
    cells = cells.merge(per_cell, left_on="cell_id", right_index=True, how="left")
    cells["am_days"] = cells["am_days"].fillna(0).astype(int)
    seasons = cold_seasons(am)
    for col in seasons:
        cells[col] = cells["cell_id"].map(seasons[col]).fillna(0).astype(int)
    cells["ever_frozen"] = cells["ever_frozen"].astype("boolean").fillna(False)
    cells = assign_sets(cells)

    screened = pd.read_csv(config.path("sfcc.gridded_sensors", grid=insitu.GRID))
    screened["in_period"] = in_period(screened["sensor_id"], workers).to_numpy()
    loc = insitu.read_catalog(["sensor_id", "latitude", "longitude"]).set_index("sensor_id")
    loc = loc.reindex(screened["sensor_id"])
    screened["north_america"] = insitu.regions(loc["latitude"], loc["longitude"]) == "north_america"
    table = attrition(screened, filtered, cells, sensor_days)

    keep = [
        "cell_id",
        "latitude",
        "longitude",
        "region",
        "set",
        "set_reason",
        "cell_class",
        "tree_cover_pct",
        "elevation_m",
        "elevation_std_m",
        "slope_deg",
        "water_fraction",
        "n_sensors",
        "am_days",
        "freeze_years",
        "complete_seasons",
        "first_day",
        "last_day",
        "ever_frozen",
        "max_frozen_share",
    ]
    cells[keep].to_csv(paths.TABLES / "insitu_cells.csv", index=False)
    sensors = insitu.read_kept_sensors()[["sensor_id", "cell_id"]].merge(
        filtered, left_on="sensor_id", right_index=True, how="inner"
    )
    dev = set(cells.loc[cells["set"] == "development", "cell_id"])
    sensors["selected"] = (sensors["reason"] == "") & sensors["cell_id"].isin(dev)
    sensors.to_csv(paths.TABLES / "insitu_sensors.csv", index=False)
    table.to_csv(paths.TABLES / "insitu_attrition.csv", index=False)
    print(table.to_string(index=False))
    dev = cells[cells["set"] == "development"]
    print(
        f"\ndevelopment cells never frozen at cut-off {insitu.CUTOFF}: "
        f"{(~dev['ever_frozen']).sum()} of {len(dev)}"
    )
    print(f"{len(daily):,} cell-days -> {out}")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
