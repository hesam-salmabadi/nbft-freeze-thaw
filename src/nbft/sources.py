"""Assemble the source cubes and extract the labelled (in situ) pixels.

This is the slow path: the TB cubes are chunked as 100x100 pixel tiles holding a whole time
series, so reading a handful of scattered station pixels touches hundreds of large tiles.
``00a_extract_station_cube.py`` runs it once and caches the result.
"""

import json
import time

import xarray as xr

from nbft import config
from nbft.grid import open_static, open_zarr, standardize, to_index_grid

# Standardized indices: name -> (raw index from TB, legacy name of its reference files)
INDEX_REFS = {"npr": "NPR", "ndv": "SR", "ndh": "QER", "wd": "WD"}
# Scale of the standardized indices: std of the thawed season (July-August), from
# workflow/01_preprocess_satellite/01_thaw_std.py; the whole-record std ('std') grows with the
# freeze-thaw swing itself and shrinks the frozen signal where winters are long.
INDEX_STD = "thaw_std"
# DAV features: name -> (legacy reference name, variable inside the reference file)
DAV_REFS = {"dav_smap": ("DAV_SMAP", "smap_v"), "dav_amsr": ("DAV_AMSR", "amsr_36v")}

CUBE_KEYS = {
    "tb": "timeseries.tb_gapfilled",
    "dav": "timeseries.tb_dav",
    "labels": "timeseries.labels",
    "air": "timeseries.era5_t2m",
    "soil_era": "timeseries.era5_stl",
    "soil_insitu": "timeseries.soil_temp_insitu",
    "ft_esdr": "timeseries.ft_esdr",
    "ft_smap": "timeseries.ft_smap",
}
STATIC_KEYS = {"landcover": "statics.landcover", "water": "statics.water_fraction"}


def reference_field(name, kind, am_pm):
    """Thaw-season reference ('thaw_ref') or std ('std') field, by legacy index name."""
    root = config.data_root() / "proc" / "statics"
    suffix = "" if name.startswith("DAV") else f"_{am_pm}"
    return standardize(xr.open_dataset(root / f"{name}{suffix}_{kind}.nc"))


def open_sources(am_pm):
    """Every source dataset for one overpass, keyed by a short name."""
    out = {k: open_zarr(config.path(p), am_pm) for k, p in CUBE_KEYS.items()}
    out["ft_esdr"] = out["ft_esdr"].rename({"ft_status": "ft_amsr"})
    out["ft_smap"] = out["ft_smap"].rename({"ft_status": "ft_smap"})
    out.update({k: open_static(config.path(p), am_pm) for k, p in STATIC_KEYS.items()})

    for name, legacy in INDEX_REFS.items():
        out[f"ref_{name}"] = reference_field(legacy, "thaw_ref", am_pm).rename(
            {legacy: f"ref_{name}"}
        )
        out[f"std_{name}"] = reference_field(legacy, INDEX_STD, am_pm).rename(
            {legacy: f"std_{name}"}
        )
    for name, (legacy, var) in DAV_REFS.items():
        out[f"ref_{name}"] = reference_field(legacy, "thaw_ref", am_pm).rename({var: f"ref_{name}"})
        out[f"std_{name}"] = reference_field(legacy, "std", am_pm).rename({var: f"std_{name}"})
    return out


def provenance(am_pm):
    """Source paths and modification times, stored with the cache so staleness is detectable."""
    items = {}
    for name, key in {**CUBE_KEYS, **STATIC_KEYS}.items():
        p = config.path(key)
        items[name] = {"path": str(p), "mtime": p.stat().st_mtime if p.exists() else None}
    return json.dumps({"am_pm": am_pm, "sources": items, "created": time.time()})


def extract_station_points(am_pm, sources=None):
    """Merge all sources at pixels that carry a ground-truth label, as a (point, time) dataset."""
    sources = sources or open_sources(am_pm)

    labelled = sources["labels"]["target"].isin([0, 1, 2]).any(dim="time").compute()
    print(f"{int(labelled.sum())} pixels with labels")
    keep = labelled.stack(point=("y", "x")).values

    def at_labelled(ds):
        return to_index_grid(ds).stack(point=("y", "x")).isel(point=keep)

    ds = xr.merge([at_labelled(d) for d in sources.values()], compat="override", join="left")
    return ds.reset_index("point")
