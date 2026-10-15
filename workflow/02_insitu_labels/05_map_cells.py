"""Map of the development cells in North America (Fig. 1).

Each cell is one marker: colour gives its terrain roughness (GMTED2010 elevation std, the ranges
of 11_roughness_effect.py), shape whether any of its sensors records soil moisture, and size the
number of sensors. The background is the mean tree cover of each 9 km cell (sfcc-label treecover
raster) in the four classes used throughout the in situ analysis. The map is cropped in the
projection to the continent (Hawaii left out), with latitude and longitude lines; legends sit in a
row below the map.

Usage: python 05_map_cells.py
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from nbft import insitu, paths
from nbft.plotting import savefig, set_rse_style

EXTENT = (-4.0e6, 3.5e6, -2.95e6, 3.9e6)  # x0, x1, y0, y1 in PROJ (m)
PROJ = ccrs.LambertAzimuthalEqualArea(-100, 50)
TREE_BOUNDS = [0, 10, 30, 60, 100]
TREE_COLORS = ["#f4f3ee", "#dde3d3", "#bccbb2", "#93aa8b"]  # one hue, light to dark, recessive
TREE_NAMES = ["< 10 %", "10–30 %", "30–60 %", "> 60 %"]
ROUGH_BOUNDS = [0, 25, 75, 150, np.inf]
ROUGH_NAMES = ["0–25 m", "25–75 m", "75–150 m", "> 150 m"]
ROUGH_COLORS = ["#9cc3ee", "#4f95e0", "#2160b8", "#0d2f66"]  # one hue (blue), light to dark
MARKERS = {True: ("o", "Temperature and moisture"), False: ("^", "Temperature only")}
LAEA_N = ccrs.LambertAzimuthalEqualArea(0, 90)
GRID_EXTENT = (-9e6, 9e6, -9e6, 9e6)
INK = "#2b2b2b"


def size(n):
    return 16 + 20 * np.sqrt(n)


def draw(ax, cells, tree):
    ax.set_extent(EXTENT, crs=PROJ)
    ax.add_feature(cfeature.LAND, facecolor=TREE_COLORS[0], edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff")
    ax.imshow(
        tree,
        extent=GRID_EXTENT,
        transform=LAEA_N,
        origin="upper",
        cmap=ListedColormap(TREE_COLORS),
        norm=BoundaryNorm(TREE_BOUNDS, len(TREE_COLORS)),
        interpolation="nearest",
        regrid_shape=1500,
        zorder=1,
    )
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#8a8984")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#8a8984")
    grid = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        xlocs=range(-180, -29, 20),
        ylocs=range(20, 81, 10),
        linewidth=0.5,
        color="#8a8984",
        alpha=0.6,
        linestyle="--",
        zorder=2,
    )
    grid.top_labels = grid.right_labels = False
    grid.xlabel_style = grid.ylabel_style = {"size": 13, "color": INK}
    rough = pd.cut(cells["elevation_std_m"], ROUGH_BOUNDS, labels=False, right=False)
    for moist, (marker, _) in MARKERS.items():
        for i, color in enumerate(ROUGH_COLORS):
            g = cells[(cells["moisture"] == moist) & (rough == i)]
            ax.scatter(
                g["longitude"],
                g["latitude"],
                s=size(g["n_sensors"]),
                c=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.6,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )


def legend(fig, handles, title, x, ncol=1):
    return fig.legend(
        handles=handles,
        title=title,
        loc="upper left",
        bbox_to_anchor=(x, 0.19),
        ncol=ncol,
        fontsize=14,
        title_fontsize=15,
        frameon=False,
        alignment="left",
    )


def main():
    set_rse_style()
    cells = pd.read_csv(paths.TABLES / "insitu_cell_table.csv")
    cells = cells[cells["set"] == "development"]
    cells["moisture"] = cells["n_moisture_sensors"] > 0
    tree = insitu.read_band("treecover", "mean")

    fig = plt.figure(figsize=(12, 12.6))
    ax = fig.add_axes([0.05, 0.22, 0.9, 0.76], projection=PROJ)
    draw(ax, cells, tree)

    def mark(marker, color, ms, label, edge="white"):
        return Line2D(
            [],
            [],
            marker=marker,
            color="none",
            markerfacecolor=color,
            markeredgecolor=edge,
            markeredgewidth=0.6,
            markersize=ms,
            label=label,
        )

    rough = pd.cut(cells["elevation_std_m"], ROUGH_BOUNDS, labels=ROUGH_NAMES, right=False)
    counts = rough.value_counts()
    legends = [
        (
            "Roughness (elevation std)",
            [
                mark("o", c, 13, f"{n} ({counts[n]} cells)")
                for c, n in zip(ROUGH_COLORS, ROUGH_NAMES, strict=True)
            ],
        ),
        (
            "Measurement",
            [
                mark(m, "#8a8984", 13, f"{lab} ({(cells['moisture'] == k).sum()})")
                for k, (m, lab) in MARKERS.items()
            ],
        ),
        (
            "Sensors in the cell",
            [mark("o", "#8a8984", np.sqrt(size(n)), f"{n}") for n in (1, 5, 20)],
        ),
        (
            "Tree cover (background)",
            [
                mark("s", c, 14, n, edge="#8a8984")
                for c, n in zip(TREE_COLORS, TREE_NAMES, strict=True)
            ],
        ),
    ]
    for x, (title, handles) in zip((0.02, 0.29, 0.58, 0.79), legends, strict=True):
        fig.add_artist(legend(fig, handles, title, x))
    print(f"{len(cells)} cells; roughness ranges: {counts.to_dict()}")
    savefig("insitu_cells_map", fig=fig)


if __name__ == "__main__":
    main()
