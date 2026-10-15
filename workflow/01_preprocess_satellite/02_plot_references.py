"""Maps of the per-pixel thaw reference and thawed-season std of each index (supplement).

Top row: thaw reference (median of the 50 July-August AM values most typical of thawed soil);
bottom row: std of the July-August AM values (01_thaw_std.py). North American cells only.
Colour limits are the 2nd and 98th percentiles of each map.

Usage: python 02_plot_references.py
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from nbft import insitu
from nbft.plotting import savefig, set_rse_style
from nbft.sources import INDEX_REFS, INDEX_STD, reference_field

EXTENT = [-168, -52, 24, 75]
PROJ = ccrs.LambertAzimuthalEqualArea(-100, 50)
LAEA_N = ccrs.LambertAzimuthalEqualArea(0, 90)
GRID_EXTENT = (-9e6, 9e6, -9e6, 9e6)
NAMES = {"npr": "NPR", "ndv": "NDV", "ndh": "NDH", "wd": "WD (K)"}
ROWS = [
    ("thaw_ref", r"Thaw reference $\mu_{thaw}$", "Blues"),
    (INDEX_STD, r"Std $\sigma$", "Oranges"),
]


def main():
    set_rse_style()
    mask = insitu.north_america_mask()
    fig, axes = plt.subplots(
        len(ROWS),
        len(INDEX_REFS),
        figsize=(18, 7.4),
        subplot_kw={"projection": PROJ},
        constrained_layout=True,
    )
    for row, (kind, title, cmap) in enumerate(ROWS):
        for col, (name, legacy) in enumerate(INDEX_REFS.items()):
            field = reference_field(legacy, kind, "am")[legacy].values
            field = np.where(mask, field, np.nan)
            lo, hi = np.nanpercentile(field, [2, 98])
            ax = axes[row, col]
            ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="#e8e7e2", edgecolor="none")
            im = ax.imshow(
                field,
                extent=GRID_EXTENT,
                transform=LAEA_N,
                origin="upper",
                cmap=cmap,
                vmin=lo,
                vmax=hi,
                interpolation="nearest",
                regrid_shape=1200,
            )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#8a8984")
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#8a8984")
            letter = "abcdefgh"[row * len(INDEX_REFS) + col]
            ax.set_title(f"({letter}) {NAMES[name]}: {title}", loc="left", fontsize=12)
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.02)
            cb.ax.tick_params(labelsize=9)
            cb.formatter.set_powerlimits((-2, 3))
            cb.update_ticks()
    savefig("reference_maps", fig=fig)


if __name__ == "__main__":
    main()
