"""Permanent snow and ice fraction of every 9 km EASE-Grid 2.0 cell, from ESA CCI Land Cover 2015.

The aggregated CCI raster of sfcc-label merges bare areas, urban areas and permanent snow and ice
into one 'other' class, so the fraction of class 220 (permanent snow and ice) is computed here from
the original 300 m map: each latitude band of North America (north of LAT_MIN, within LON_RANGE)
is first counted in blocks of BLOCK x BLOCK pixels (about 3 km), and the counts of ice pixels and of
all pixels are reprojected to the 9 km grid by summing; the fraction is their ratio. The product
masks cells in which permanent ice dominates (03_model/07_upscale.py).

Output: statics.ice_fraction (netCDF, y/x of the 9 km grid, rows and columns as in the TB cube).

Usage: python 04_ice_fraction.py
"""

import numpy as np
import rasterio
import xarray as xr
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from nbft import config

CCI = config.path("sfcc.cci_300m")
ICE_CLASS = 220
LAT_MIN = 20.0
LON_RANGE = (-180.0, -50.0)  # North America
ROWS = 2000  # source rows per band (a multiple of BLOCK)
BLOCK = 10
GRID = 2000
CELL = 9000.0
EASE2_N = "EPSG:6931"


def main():
    dst_transform = rasterio.transform.from_origin(-GRID / 2 * CELL, GRID / 2 * CELL, CELL, CELL)
    ice = np.zeros((GRID, GRID))
    total = np.zeros((GRID, GRID))
    with rasterio.open(CCI) as src:
        last_row = int((90 - LAT_MIN) / abs(src.transform.e))
        c0 = int((LON_RANGE[0] + 180) / src.transform.a)
        width = int((LON_RANGE[1] - LON_RANGE[0]) / src.transform.a)
        for r0 in range(0, last_row, ROWS):
            window = Window(c0, r0, width, min(ROWS, last_row - r0))
            lc = src.read(1, window=window)
            h, w = lc.shape[0] // BLOCK * BLOCK, lc.shape[1] // BLOCK * BLOCK
            lc = lc[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK)
            transform = src.window_transform(window) * rasterio.Affine.scale(BLOCK)
            for target, values in ((ice, lc == ICE_CLASS), (total, lc > 0)):
                part = np.zeros((GRID, GRID), "float32")
                reproject(
                    values.sum(axis=(1, 3)).astype("float32"),
                    part,
                    src_transform=transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=EASE2_N,
                    resampling=Resampling.sum,
                    src_nodata=None,
                    dst_nodata=0,
                )
                target += part
            print(f"  rows {r0}-{r0 + window.height}", flush=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(total > 0, ice / total, 0.0).astype("float32")
    cube = xr.open_zarr(config.path("timeseries.tb_gapfilled"))
    out = xr.Dataset(
        {"ice_fraction": (("y", "x"), fraction)},
        coords={"y": cube["y"].values, "x": cube["x"].values},
    )
    out["ice_fraction"].attrs["description"] = "fraction of ESA CCI 2015 class 220 in the cell"
    path = config.path("statics.ice_fraction")
    out.to_netcdf(path)
    print(f"{int((fraction > 0.5).sum())} cells more than half ice -> {path}")


if __name__ == "__main__":
    main()
