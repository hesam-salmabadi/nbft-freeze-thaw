"""Maps of the model probabilities (Results 4.8), from product.season_summary
(05_phenology/01_season_summary.py).

- maps_snapshots: six mornings of the 2018-19 season; the hue of a cell is its most probable class
  and its strength the probability of that class (pale where the classes share the probability,
  full colour where one is certain), with a key of the three ramps.
- maps_season_days: mean number of mornings per season (September to June, 2015-16 to 2022-23)
  whose most probable class is transitional or frozen, against the partially frozen and frozen
  mornings of FT-SMOS, over the seasons with data where at least nbft.timing.MIN_SEASONS have
  data. Grey: FT-SMOS without data.
- maps_timing: mean first transitional morning and first frozen morning of the model, the first
  frozen morning of FT-SMOS, and the difference model minus FT-SMOS (nbft.timing).
- maps_thaw_duration: for the model and FT-SMOS, the mean last transitional morning (end of the
  thaw) and the mean length of the autumn (first transitional to first frozen morning) and spring
  (last frozen to last transitional morning) transitions, over the seasons with both events.

Usage: python 08_plot_maps.py
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import to_rgb

from nbft import config
from nbft.plotting import savefig, set_rse_style
from nbft.timing import season_mean

COLORS = {"thawed": "#d6604d", "transitional": "#f2c14e", "frozen": "#4575b4"}
EASE2_N = ccrs.LambertAzimuthalEqualArea(
    central_latitude=90, central_longitude=0, globe=ccrs.Globe(ellipse="WGS84")
)
MAP = ccrs.LambertAzimuthalEqualArea(central_latitude=55, central_longitude=-100)
EXTENT = [-168, -52, 23, 76]  # lon, lat
GREY = (0.92, 0.92, 0.92)


def shade(p_tr, p_fr):
    """RGB of every cell: the colour of the most probable class, mixed with white by how far its
    probability is from an even split (1/3: white, 1: full colour)."""
    p = np.stack([1 - p_tr - p_fr, p_tr, p_fr], axis=-1)
    colors = np.array([to_rgb(c) for c in COLORS.values()])
    best = np.nan_to_num(p, nan=0).argmax(-1)
    strength = np.clip((np.nanmax(p, axis=-1) - 1 / 3) / (2 / 3), 0, 1)[..., None]
    rgb = 1 - strength * (1 - colors[best])
    rgb[~np.isfinite(p).all(-1)] = 1.0
    return rgb


def base(ax):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=GREY, zorder=0)
    ax.coastlines(resolution="50m", color="#404040", linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#404040", zorder=3)


def extent(ds):
    dx = float(ds["x"][1] - ds["x"][0])
    dy = float(ds["y"][1] - ds["y"][0])
    return [
        float(ds["x"][0]) - dx / 2,
        float(ds["x"][-1]) + dx / 2,
        float(ds["y"][-1]) + dy / 2,
        float(ds["y"][0]) - dy / 2,
    ]


def ramp_key(ax):
    """Key: one ramp per class, from an even split (1/3) to certainty (1)."""
    q = np.linspace(1 / 3, 1, 100)
    for i, k in enumerate(COLORS):
        p_tr = q if k == "transitional" else (1 - q) / 2
        p_fr = q if k == "frozen" else (1 - q) / 2
        ax.imshow(
            shade(p_tr, p_fr)[None], extent=[1 / 3, 1, 2 - i - 0.4, 2 - i + 0.4], aspect="auto"
        )
        ax.text(0.31, 2 - i, k.capitalize(), ha="right", va="center", fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.set_xticks([1 / 3, 2 / 3, 1], ["1/3", "2/3", "1"])
    ax.tick_params(labelsize=10)
    ax.set_xlabel("Probability of the most probable class", fontsize=11)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def snapshots(ds):
    fig = plt.figure(figsize=(15, 7.4))
    for i, date in enumerate(ds["date"].values):
        ax = fig.add_subplot(2, 3, i + 1, projection=MAP)
        base(ax)
        rgb = shade(
            ds["p_transitional"].sel(date=date).values, ds["p_frozen"].sel(date=date).values
        )
        alpha = np.isfinite(ds["p_frozen"].sel(date=date).values).astype(float)
        ax.imshow(
            np.dstack([rgb, alpha]),
            extent=extent(ds),
            transform=EASE2_N,
            origin="upper",
            interpolation="nearest",
            zorder=2,
        )
        label = np.datetime_as_string(date, unit="D")
        ax.set_title(f"({'abcdef'[i]}) {label}", loc="left", fontsize=13)
    key = fig.add_axes([0.42, -0.07, 0.22, 0.09])
    ramp_key(key)
    fig.subplots_adjust(wspace=0.03, hspace=0.1, left=0.01, right=0.99, top=0.97, bottom=0.03)
    savefig("maps_snapshots", fig=fig)


def season_days(ds):
    fig = plt.figure(figsize=(12, 7.6))
    panels = [
        ("model_transitional_days", "(a) Transitional, model", "YlOrBr", 120),
        ("smos_transitional_days", "(b) Partially frozen, FT-SMOS", "YlOrBr", 120),
        ("model_frozen_days", "(c) Frozen, model", "Blues", 250),
        ("smos_frozen_days", "(d) Frozen, FT-SMOS", "Blues", 250),
    ]
    for i, (var, title, cmap, vmax) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1, projection=MAP)
        base(ax)
        im = ax.imshow(
            season_mean(ds[var]).values,
            extent=extent(ds),
            transform=EASE2_N,
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            zorder=2,
        )
        ax.set_title(title, loc="left", fontsize=13)
        if i % 2:
            cb = fig.colorbar(im, ax=fig.axes[-2:], shrink=0.8, pad=0.02)
            cb.set_label("Mornings per season", fontsize=12)
            cb.ax.tick_params(labelsize=11)
    savefig("maps_season_days", fig=fig)


MONTHS = {31: "1 Sep", 61: "1 Oct", 92: "1 Nov", 122: "1 Dec", 153: "1 Jan", 184: "1 Feb"}


def mean_event(ds, var):
    """Mean over the seasons with the event, where at least MIN_SEASONS have it."""
    return season_mean(ds[var]).values


def timing(ds):
    fig = plt.figure(figsize=(12, 9.2))
    model_ff = mean_event(ds, "model_first_frozen")
    smos_ff = mean_event(ds, "smos_first_frozen")
    panels = [
        (mean_event(ds, "model_first_transition"), "(a) Start of freeze-up, model"),
        (model_ff, "(b) First frozen morning, model"),
        (smos_ff, "(c) First frozen morning, FT-SMOS"),
        (model_ff - smos_ff, "(d) First frozen morning, model minus FT-SMOS"),
    ]
    for i, (values, title) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1, projection=MAP)
        base(ax)
        diff = i == 3
        im = ax.imshow(
            values,
            extent=extent(ds),
            transform=EASE2_N,
            origin="upper",
            interpolation="nearest",
            cmap="RdBu_r" if diff else "viridis",
            vmin=-60 if diff else 31,
            vmax=60 if diff else 184,
            zorder=2,
        )
        ax.set_title(title, loc="left", fontsize=13)
        cb = fig.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            shrink=0.7,
            pad=0.03,
            ticks=None if diff else list(MONTHS),
        )
        if diff:
            cb.set_label("Days", fontsize=11)
        else:
            cb.ax.set_xticklabels(list(MONTHS.values()))
        cb.ax.tick_params(labelsize=10)
    savefig("maps_timing", fig=fig)


SPRING = {212: "1 Mar", 243: "1 Apr", 273: "1 May", 304: "1 Jun", 334: "1 Jul"}


def thaw_duration(ds):
    fig = plt.figure(figsize=(11, 13))
    rows = []
    for p, name in (("model", "model"), ("smos", "FT-SMOS")):
        autumn = season_mean(ds[f"{p}_first_frozen"] - ds[f"{p}_first_transition"]).values
        spring = season_mean(ds[f"{p}_last_transition"] - ds[f"{p}_last_frozen"]).values
        rows.append(
            [
                (mean_event(ds, f"{p}_last_transition"), f"End of thaw, {name}"),
                (autumn, f"Autumn transition, {name}"),
                (spring, f"Spring transition, {name}"),
            ]
        )
    for j, column in enumerate(rows):
        for i, (values, title) in enumerate(column):
            ax = fig.add_subplot(3, 2, 2 * i + j + 1, projection=MAP)
            base(ax)
            date = i == 0
            im = ax.imshow(
                values,
                extent=extent(ds),
                transform=EASE2_N,
                origin="upper",
                interpolation="nearest",
                cmap="viridis" if date else "YlOrBr",
                vmin=182 if date else 0,
                vmax=334 if date else 90,
                zorder=2,
            )
            ax.set_title(f"({'abcdef'[2 * i + j]}) {title}", loc="left", fontsize=13)
            cb = fig.colorbar(
                im,
                ax=ax,
                orientation="horizontal",
                shrink=0.7,
                pad=0.03,
                ticks=list(SPRING) if date else None,
            )
            if date:
                cb.ax.set_xticklabels(list(SPRING.values()))
            else:
                cb.set_label("Days", fontsize=11)
            cb.ax.tick_params(labelsize=10)
    savefig("maps_thaw_duration", fig=fig)


def main():
    set_rse_style()
    ds = xr.open_dataset(config.path("product.season_summary", am_pm="am"))
    snapshots(ds)
    season_days(ds)
    timing(ds)
    thaw_duration(ds)


if __name__ == "__main__":
    main()
