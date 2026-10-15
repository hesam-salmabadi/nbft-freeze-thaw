"""Per-pixel spread of each index in the thawed season, the scale of the standardized indices.

For NPR, NDV, NDH and WD: the standard deviation of the AM values of July and August over the
study period, at every North American cell. It is the day-to-day variability of the thawed state,
so Z = (X - thaw reference) / thaw std measures the departure from the thawed state in units of
its own noise; unlike the whole-record std, it does not grow with the freeze-thaw swing itself.

The cube is chunked as (all days, 100 x 100 cells), so each 100 x 100 tile is read whole once per
channel. Output: {legacy}_am_thaw_std.nc next to the thaw references (sources.reference_field).

Slow: reads the North American tiles of six channels from the external drive (about 35 GB).

Usage: python 01_thaw_std.py [workers]
"""

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from nbft import config, insitu
from nbft import features as F
from nbft.grid import open_zarr
from nbft.sources import INDEX_REFS

CHANNELS = ["smap_v", "smap_h", "amsr_18v", "amsr_18h", "amsr_36v", "amsr_36h"]
TILE = 100
MIN_SUMMER_DAYS = 30  # valid July-August values a cell needs for its std


def tile_std(ds, days, r0, c0):
    """Thawed-season std of the four indices on one tile."""
    tb = {
        v: ds[v].isel(y=slice(r0, r0 + TILE), x=slice(c0, c0 + TILE)).values[days] for v in CHANNELS
    }
    index = {
        "npr": F.npr(tb["smap_v"], tb["smap_h"]),
        "ndv": F.ndv(tb["amsr_18v"], tb["amsr_36v"]),
        "ndh": F.ndh(tb["amsr_18h"], tb["amsr_36v"]),
        "wd": F.wd(tb["smap_v"], tb["amsr_36v"]),
    }
    out = {}
    for name, x in index.items():
        x = np.where(np.isfinite(x), x, np.nan)
        n = np.isfinite(x).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells
            sd = np.nanstd(x, axis=0, ddof=1)
        out[name] = np.where(n >= MIN_SUMMER_DAYS, sd, np.nan)
    return (r0, c0), out


def main(workers=4):
    ds = open_zarr(config.path("timeseries.tb_gapfilled"), "am")
    time = pd.DatetimeIndex(ds["time"].values)
    days = np.nonzero(time.month.isin(insitu.SUMMER_MONTHS))[0]
    mask = insitu.north_america_mask()
    rows, cols = np.nonzero(mask)
    tiles = sorted(set(zip(rows // TILE * TILE, cols // TILE * TILE, strict=True)))
    print(f"{len(days)} July-August days, {len(tiles)} tiles")

    shape = mask.shape
    fields = {name: np.full(shape, np.nan, dtype="float32") for name in INDEX_REFS}
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(tile_std, ds, days, r0, c0) for r0, c0 in tiles]
        for i, f in enumerate(futures, 1):
            (r0, c0), out = f.result()
            for name, sd in out.items():
                fields[name][r0 : r0 + sd.shape[0], c0 : c0 + sd.shape[1]] = sd
            print(f"  {i}/{len(tiles)} tiles")

    root = config.data_root() / "proc" / "statics"
    for name, legacy in INDEX_REFS.items():
        field = np.where(mask, fields[name], np.nan).astype("float32")
        out = xr.Dataset(
            {legacy: (("y", "x"), field)},
            coords={"y": ds["y"].values, "x": ds["x"].values, "am_pm": "am"},
            attrs={
                "description": f"std of AM {name} in July-August, {insitu.TIME_START} to "
                f"{insitu.TIME_END}; cells with < {MIN_SUMMER_DAYS} values are NaN"
            },
        )
        path = root / f"{legacy}_am_thaw_std.nc"
        out.to_netcdf(path)
        print(
            f"{name}: median {np.nanmedian(field):.4g}, {np.isfinite(field).sum()} cells -> {path}"
        )


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
