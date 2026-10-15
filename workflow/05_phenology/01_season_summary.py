"""Seasonal summaries of the model probabilities and of FT-SMOS, for the maps (Results 4.8).

Per freeze season (1 September to 30 June, 2015-16 to 2022-23) and domain cell:
- model: the number of mornings whose most probable class is transitional and frozen, scaled to
  the whole season when mornings are missing. (The sum of the daily probabilities is not a
  duration: the probabilities express relative evidence with equal class weights, and thawed
  mornings carry a mean P(transitional) of about 0.3.)
- FT-SMOS: the number of partially frozen and of frozen mornings, scaled the same way;
- both: the timing events of nbft.timing (first transitional, first frozen, last frozen and last
  transitional morning, in days since 1 August), from the mornings of 1 September to 30 June.
A season with data on fewer than MIN_VALID of its mornings is NaN. Also stores the model
probabilities on SNAPSHOT_DATES.

The product is read tile by tile (its chunks hold the whole time axis); FT-SMOS day by day.

Output: product.season_summary (netCDF on the product's y/x box).

The model and FT-SMOS use the same state codes (0 thawed, 1 frozen, 2 transitional or partially
frozen). With --model-only, the FT-SMOS counts are kept from the existing output.

Usage: python 01_season_summary.py [--model-only]
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr

from nbft import config
from nbft.grid import standardize
from nbft.timing import EVENTS, RunTracker

AM_PM = "am"
TILE = 100
SEASONS = range(2015, 2023)
MIN_VALID = 0.5
SNAPSHOT_DATES = pd.to_datetime(
    ["2018-10-15", "2018-11-15", "2018-12-15", "2019-03-15", "2019-04-15", "2019-05-15"]
)
SMOS_STATES = {"transitional": 2, "frozen": 1}


def season_days(time, start):
    return (time >= pd.Timestamp(f"{start}-09-01")) & (time <= pd.Timestamp(f"{start + 1}-06-30"))


def days_since_august(time, start):
    return (time - pd.Timestamp(f"{start}-08-01")).days.to_numpy()


def scaled(total, valid, n_days):
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / valid * n_days
    return np.where(valid >= MIN_VALID * n_days, out, np.nan).astype("float32")


def model_summary(product):
    time = pd.DatetimeIndex(product["time"].values)
    ny, nx = product.sizes["y"], product.sizes["x"]
    shape = (len(SEASONS), ny, nx)
    out = {f"model_{k}_days": np.full(shape, np.nan, "float32") for k in SMOS_STATES}
    out |= {f"model_{e}": np.full(shape, np.nan, "float32") for e in EVENTS}
    snaps = {k: np.full((len(SNAPSHOT_DATES), ny, nx), np.nan, "float32") for k in SMOS_STATES}
    snap_idx = time.get_indexer(SNAPSHOT_DATES)
    for y0 in range(0, ny, TILE):
        for x0 in range(0, nx, TILE):
            box = {"y": slice(y0, y0 + TILE), "x": slice(x0, x0 + TILE)}
            p = {
                "transitional": product["p_transitional"].isel(**box).values,
                "frozen": product["p_frozen"].isel(**box).values,
            }
            if not np.isfinite(p["frozen"]).any():
                continue
            state = product["state"].isel(**box).values
            for k, v in p.items():
                snaps[k][:, box["y"], box["x"]] = v[snap_idx]
            for i, start in enumerate(SEASONS):
                sel = season_days(time, start)
                valid = np.isfinite(state[sel]).sum(0)
                for k, code in SMOS_STATES.items():
                    out[f"model_{k}_days"][i, box["y"], box["x"]] = scaled(
                        (state[sel] == code).sum(0), valid, sel.sum()
                    )
                tracker = RunTracker(state.shape[1:])
                for day, values in zip(
                    days_since_august(time[sel], start), state[sel], strict=True
                ):
                    tracker.update(day, values)
                for e, v in tracker.events().items():
                    out[f"model_{e}"][i, box["y"], box["x"]] = np.where(
                        valid >= MIN_VALID * sel.sum(), v, np.nan
                    )
        print(f"  model rows {y0}-{y0 + TILE}", flush=True)
    return out, snaps


def smos_summary(box, inside):
    smos = xr.open_zarr(config.path("timeseries.ft_smos"))["ft_status"].sel(am_pm=AM_PM)
    time = pd.DatetimeIndex(smos["time"].values)
    out = {f"smos_{k}_days": [] for k in SMOS_STATES} | {f"smos_{e}": [] for e in EVENTS}
    for start in SEASONS:
        sel = np.flatnonzero(season_days(time, start))
        counts = {k: np.zeros(inside.shape, "float32") for k in SMOS_STATES}
        valid = np.zeros(inside.shape, "float32")
        tracker = RunTracker(inside.shape)
        for t, day in zip(sel, days_since_august(time[sel], start), strict=True):
            v = smos.isel(time=t, **box).values
            v = np.where(v == 255, np.nan, v)
            ok = np.isfinite(v)
            valid += ok
            for k, code in SMOS_STATES.items():
                counts[k] += ok & (v == code)
            tracker.update(day, v)
        for k in SMOS_STATES:
            out[f"smos_{k}_days"].append(
                np.where(inside, scaled(counts[k], valid, len(sel)), np.nan)
            )
        for e, v in tracker.events().items():
            out[f"smos_{e}"].append(np.where(inside & (valid >= MIN_VALID * len(sel)), v, np.nan))
        print(f"  FT-SMOS season {start}", flush=True)
    return {k: np.stack(v).astype("float32") for k, v in out.items()}


def main():
    product = standardize(xr.open_zarr(config.path("product.ft_states", am_pm=AM_PM)))
    product = product.sel(time=slice("2015-09-01", "2023-06-30"))
    model, snaps = model_summary(product)
    inside = np.isfinite(model["model_frozen_days"]).any(0)
    # the product's box in the full grid: the cube's x/y coordinates of the product
    cube = standardize(xr.open_zarr(config.path("timeseries.tb_gapfilled")))
    r0 = int(np.abs(cube["y"].values - product["y"].values[0]).argmin())
    c0 = int(np.abs(cube["x"].values - product["x"].values[0]).argmin())
    box = {"y": slice(r0, r0 + product.sizes["y"]), "x": slice(c0, c0 + product.sizes["x"])}
    out = config.path("product.season_summary", am_pm=AM_PM)
    if "--model-only" in sys.argv:
        with xr.open_dataset(out) as old:
            smos = {k: old[k].values for k in old.data_vars if k.startswith("smos_")}
    else:
        smos = smos_summary(box, inside)
    dims = ("season", "y", "x")
    ds = xr.Dataset(
        {k: (dims, v) for k, v in {**model, **smos}.items()}
        | {f"p_{k}": (("date", "y", "x"), v) for k, v in snaps.items()},
        coords={
            "season": list(SEASONS),
            "date": SNAPSHOT_DATES,
            "y": product["y"].values,
            "x": product["x"].values,
        },
    )
    ds.attrs["description"] = (
        "mornings per season in each state, and timing events in days since 1 August"
    )
    ds.to_netcdf(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
