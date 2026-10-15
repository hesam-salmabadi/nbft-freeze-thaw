"""Apply the production model to the brightness-temperature cube: the non-binary FT product.

For every AM day and every North American cell (Canada and the United States, whatever its open
water or bare ground; the land-cover screen only selects training sensors) except those more than
MAX_ICE_FRACTION permanent snow and ice (statics.ice_fraction, ESA CCI): Z_NPR and Z_NDV
against the per-pixel thaw reference and thawed-season std (sources.reference_field), the
tree-cover fraction (0 where the tree-cover map has no value, north of 80 N), then the class
probabilities and the most probable state of nbft.model (0 thawed, 1 frozen, 2 transitional) from
results/tables/model_coefficients.csv. Only the bounding box of the domain is read and written.

Output: product.ft_states (zarr): p_frozen and p_transitional (float32; P(thawed) is one minus
their sum) and state (float32, NaN outside the domain or without data).

Usage: python 07_upscale.py [days]   (days: write only the first N days, for a test)
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from nbft import config, features, insitu, paths
from nbft.grid import open_zarr
from nbft.model import TERMS, class_probabilities, design
from nbft.sources import INDEX_REFS, INDEX_STD, reference_field

AM_PM = "am"
TILE = 100  # chunk size of the cube in y and x
MAX_ICE_FRACTION = 0.5


def grid_field(values, like):
    """A 2D array on the (y, x) coordinates of ``like``."""
    return xr.DataArray(values, dims=("y", "x"), coords={"y": like["y"], "x": like["x"]})


def reference(name, kind, box, like):
    legacy = INDEX_REFS[name]
    field = reference_field(legacy, kind, AM_PM)[legacy].isel(**box).values
    if kind == INDEX_STD:
        field = np.where(field == 0, 1.0, field)
    return grid_field(field, like)


def main():
    coef = pd.read_csv(paths.TABLES / "model_coefficients.csv", index_col="class")
    coef = coef[["intercept", *TERMS]]
    tree = insitu.read_band("treecover", "mean").astype("float32") / 100
    tree = np.where(np.isfinite(tree), tree, 0.0).astype("float32")
    domain = insitu.north_america_mask()
    ice = xr.open_dataset(config.path("statics.ice_fraction"))["ice_fraction"].values
    domain &= ice <= MAX_ICE_FRACTION
    rows, cols = np.nonzero(domain)
    box = {  # whole cube tiles, so the output keeps the cube's chunks
        "y": slice(rows.min() // TILE * TILE, -(-(rows.max() + 1) // TILE) * TILE),
        "x": slice(cols.min() // TILE * TILE, -(-(cols.max() + 1) // TILE) * TILE),
    }

    tb = open_zarr(config.path("timeseries.tb_gapfilled"), AM_PM).isel(**box)
    if len(sys.argv) > 1:  # quick test on the first N days
        tb = tb.isel(time=slice(0, int(sys.argv[1])))
    inside = grid_field(domain[box["y"], box["x"]], tb)
    t = grid_field(tree[box["y"], box["x"]], tb).where(inside)
    z_npr = features.zscore(
        features.npr(tb["smap_v"], tb["smap_h"]),
        reference("npr", "thaw_ref", box, tb),
        reference("npr", INDEX_STD, box, tb),
    )
    z_ndv = features.zscore(
        features.ndv(tb["amsr_18v"], tb["amsr_36v"]),
        reference("ndv", "thaw_ref", box, tb),
        reference("ndv", INDEX_STD, box, tb),
    )
    probs = class_probabilities(coef, design(z_npr, z_ndv, t))
    probs = {k: v.where(inside) for k, v in probs.items()}
    pt, pr, pf = probs["thawed"], probs["transitional"], probs["frozen"]
    s = xr.where((pf >= pr) & (pf >= pt), 1.0, xr.where(pr >= pt, 2.0, 0.0))  # most probable
    s = s.where(pf.notnull() & pr.notnull()).astype("float32")
    product = xr.Dataset(
        {
            "p_frozen": probs["frozen"].astype("float32"),
            "p_transitional": probs["transitional"].astype("float32"),
            "state": s,
        }
    )
    product["p_frozen"].attrs["description"] = "P(frozen)"
    product["p_transitional"].attrs["description"] = "P(transitional)"
    product["state"].attrs["description"] = "0 thawed, 1 frozen, 2 transitional"
    product.attrs["model"] = coef.round(6).to_json()
    for var in product.variables:
        product[var].encoding.clear()

    out = config.path("product.ft_states", am_pm=AM_PM)
    if len(sys.argv) > 1:
        out = out.with_name(out.stem + "_test.zarr")
    print(f"{int(domain.sum()):,} domain cells, box {dict(tb.sizes)}; writing {out}")
    with ProgressBar():
        product.to_zarr(out, mode="w", zarr_format=2)


if __name__ == "__main__":
    main()
