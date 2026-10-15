"""Helpers for EASE-Grid 2.0 (9 km) datasets used throughout the workflow."""

import numpy as np
import xarray as xr

from nbft.constants import TIME_END, TIME_START


def drop_meters_coords(ds):
    return ds.drop_vars(["x_meters", "y_meters"], errors="ignore")


def add_coords(ds):
    """Promote x/y to coordinates when they are stored as data variables."""
    if "x" not in ds.coords and "x" in ds.data_vars:
        ds = ds.set_coords(["x", "y"])
    return ds


def fix_am_pm(ds):
    """Lower-case string am_pm labels and squeeze a length-1 am_pm dimension."""
    if "am_pm" in ds.coords:
        vals = ds["am_pm"].values
        if vals.dtype.kind in {"U", "S", "O"}:
            if vals.ndim == 0:
                ds = ds.assign_coords(am_pm=str(vals.item()).lower())
            else:
                ds = ds.assign_coords(am_pm=[str(v).lower() for v in vals])
    if "am_pm" in ds.dims and ds.sizes["am_pm"] == 1:
        ds = ds.squeeze("am_pm", drop=False)
    return ds


def standardize(ds):
    return drop_meters_coords(fix_am_pm(add_coords(ds)))


def select_time(ds, start=TIME_START, end=TIME_END):
    if "time" in ds.dims:
        return ds.sel(time=slice(start, end))
    return ds


def select_overpass(ds, am_pm):
    return ds.sel(am_pm=am_pm) if "am_pm" in ds.dims else ds


def open_zarr(path, am_pm=None, **kwargs):
    """Open a zarr cube, standardize coords, clip to the study period, pick an overpass."""
    ds = select_time(standardize(xr.open_zarr(path, **kwargs)))
    return select_overpass(ds, am_pm) if am_pm else ds


def open_static(path, am_pm=None):
    ds = standardize(xr.open_dataset(path))
    return select_overpass(ds, am_pm) if am_pm else ds


def to_index_grid(ds):
    """Put an EASE-Grid dataset on integer row/column indices.

    Cubes on disk mix metre coordinates with grid indices, so merges align on position only
    after this. Indices (not metres) are used because downstream code treats y/x as pixel
    row/column: station metadata stores row/col, and DBSCAN spatial blocking uses a radius in
    pixels.
    """
    return ds.assign_coords(y=np.arange(ds.sizes["y"]), x=np.arange(ds.sizes["x"]))


def reference_coords(path):
    """Return the reference EASE-Grid x/y coordinates."""
    ref = xr.open_dataset(path)
    return ref.x, ref.y
