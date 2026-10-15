"""Site examples: one freeze year per site, model against in situ labels (Results 4.3 and 4.6).

Each site panel has three rows on a common September-June axis:
- strips: the in situ label, the model state and the FT-SMOS state of every morning;
- the model probabilities of thawed, transitional and frozen ground (stacked);
- Z_NPR and Z_NDV, with the in situ soil temperature on a second axis.
North American sites use the cross-validation model of the fold that held out the site's block
(cv_coefficients.csv), so every day is a held-out prediction; external sites use the production
model. Every day of the season is shown, including the mixed partial days left out of scoring.
The scores in each title are computed on the clean days; the mean F1 averages the classes that
occur in the label or the model state that season.

Figures (paper/figs): site_examples_north_america_frozen (seasons with a frozen winter),
site_examples_north_america_near_zero (soil near 0 C for months), site_examples_external; or a
preview with --preview DIR.

Usage: python 04_plot_site_examples.py [--preview DIR]
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from nbft import config, insitu, paths
from nbft.evaluation import scores
from nbft.model import CLASSES, TERMS, class_probabilities, design, most_probable
from nbft.plotting import savefig, set_rse_style

COLORS = {"thawed": "#d6604d", "transitional": "#f2c14e", "frozen": "#4575b4"}
LABEL_CLASS = {"thawed": "thawed", "partial": "transitional", "frozen": "frozen"}
CODE = {0: "thawed", 1: "frozen", 2: "transitional"}
C_NPR, C_NDV, C_SOIL = "#2a78d6", "#eb6834", "#2b2b2b"

# (set, cell_id, freeze year start, landscape, place)
# North America, in two figures: seasons with a frozen winter, and soil near 0 C for months
NORTH_AMERICA_FROZEN = [
    ("na", "EASE2_N_9km_r0867_c0549", 2017, "Prairie", "Kenaston, Saskatchewan"),
    ("na", "EASE2_N_9km_r0888_c0574", 2018, "Boreal forest", "BERMS, Saskatchewan"),
    ("na", "EASE2_N_9km_r0751_c0828", 2018, "Subarctic, open", "Eagle Summit, Alaska"),
]
NORTH_AMERICA_NEAR_ZERO = [
    ("na", "EASE2_N_9km_r0822_c0405", 2019, "Mountain", "Cochetopa Pass, Colorado"),
    ("na", "EASE2_N_9km_r0839_c0422", 2015, "Subalpine forest", "Niwot Ridge, Colorado"),
    ("na", "EASE2_N_9km_r1115_c0567", 2018, "Boreal forest", "James Bay, Quebec"),
]
EXTERNAL = [
    ("external", "EASE2_N_9km_r1249_c1124", 2021, "Boreal", "northern Finland"),
    ("external", "EASE2_N_9km_r0745_c1519", 2018, "Steppe", "Inner Mongolia, China"),
    ("external", "EASE2_N_9km_r0861_c1651", 2017, "Alpine meadow", "Maqu, Tibetan Plateau"),
]
Z_LINEAR = 3.0  # Z axis is linear within +-Z_LINEAR and logarithmic beyond


def add_probabilities(x, coef):
    probs = class_probabilities(
        coef[["intercept", *TERMS]], design(x["z_npr"], x["z_ndv"], x["tree"])
    )
    for k in CLASSES:
        x[f"p_{k}"] = np.asarray(probs[k])
    x["state"] = most_probable(probs)
    return x


def north_america_days(cell, start):
    x = pd.read_parquet(config.path("model.training_table"))
    x = x[x["cell_id"] == cell].copy()
    fold = pd.read_parquet(config.path("model.cv_predictions"), columns=["cell_id", "fold"])
    fold = int(fold.loc[fold["cell_id"] == cell, "fold"].iloc[0])
    coefs = pd.read_csv(paths.TABLES / "cv_coefficients.csv")
    coef = coefs[coefs["fold"] == fold].set_index("class")
    x = add_probabilities(x, coef)
    for key in ("soil_temperature", "smos"):
        x = x.merge(pd.read_parquet(config.path(f"model.eval_{key}")), how="left")
    return season(x, start)


def external_days(cell, start):
    x = pd.read_parquet(config.path("validation.external_days"))
    x = x[x["cell_id"] == cell].copy()
    coef = pd.read_csv(paths.TABLES / "model_coefficients.csv", index_col="class")
    return season(add_probabilities(x, coef), start)


def stations(cell, start):
    """Number of in situ stations with AM records in the cell during the season."""
    so = pd.read_parquet(
        config.path("insitu.sensor_overpass", grid=insitu.GRID),
        columns=["sensor_id", "cell_id", "date", "am_pm"],
        filters=[("cell_id", "==", cell), ("am_pm", "==", "am")],
    )
    so = so[pd.to_datetime(so["date"]).between(f"{start}-09-01", f"{start + 1}-06-30")]
    cat = insitu.read_catalog(["sensor_id", "station"]).drop_duplicates("sensor_id")
    return cat[cat["sensor_id"].isin(so["sensor_id"])]["station"].nunique()


def season(x, start):
    """September to June of one freeze year, one row per day (missing days are NaN)."""
    x["date"] = pd.to_datetime(x["date"])
    days = pd.date_range(f"{start}-09-01", f"{start + 1}-06-30", freq="D")
    x = x.drop_duplicates("date").set_index("date").reindex(days)
    return x.rename_axis("date").reset_index()


def strip(ax, dates, classes, y, label):
    """One row of daily classes, drawn as a single mesh so adjacent days touch."""
    order = list(COLORS)
    codes = classes.map({k: i for i, k in enumerate(order)}).to_numpy(dtype=float)
    edges = mdates.date2num(pd.DatetimeIndex(dates)) - 0.5
    edges = np.append(edges, edges[-1] + 1)
    ax.pcolormesh(
        edges,
        [y - 0.4, y + 0.4],
        np.ma.masked_invalid(codes)[None, :],
        cmap=ListedColormap([COLORS[k] for k in order]),
        vmin=-0.5,
        vmax=len(order) - 0.5,
        shading="flat",
        rasterized=True,
    )
    ax.text(
        -0.008, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=12
    )


def season_scores(x):
    clean = x.dropna(subset=["state", "label"])
    share = insitu.MIN_TRANSITION_SHARE
    clean = clean[(clean["label"] != "partial") | (clean["transition_share"] >= share)]
    sc = scores(clean["label"], clean["state"])
    # mean F1 over the classes that occur in the label or the state this season
    occur = set(clean["label"].map(LABEL_CLASS)) | set(clean["state"].map(CODE))
    sc["macro_f1"] = np.mean([sc[f"f1_{k}"] for k in CLASSES if k in occur])
    return sc


def panel(axes, x, title):
    a0, a1, a2 = axes
    d = x["date"]
    strip(a0, d, x["label"].map(LABEL_CLASS), 2, "In situ")
    strip(a0, d, x["state"].map(CODE), 1, "Model")
    strip(a0, d, x["smos_state"].map(CODE), 0, "FT-SMOS")
    a0.set_ylim(-0.6, 2.6)
    a0.set_yticks([])
    a0.set_title(title, loc="left", fontsize=13.5, weight="bold", pad=4)
    sc = season_scores(x)
    a0.text(
        1.0,
        1.04,
        f"mean F1 {sc['macro_f1']:.2f}   kappa {sc['kappa']:.2f}",
        transform=a0.transAxes,
        ha="right",
        va="bottom",
        fontsize=12.5,
    )
    bottom = np.zeros(len(x))
    for k in CLASSES:
        top = bottom + x[f"p_{k}"].to_numpy()
        a1.fill_between(
            d,
            bottom,
            top,
            where=np.isfinite(top),
            color=COLORS[k],
            lw=0,
            step="mid",
            antialiased=False,
        )
        bottom = top
    a1.set_ylim(0, 1)
    a1.set_yticks([0, 0.5, 1])
    a1.set_ylabel("P", fontsize=13)
    a2.plot(d, x["z_npr"], color=C_NPR, lw=1.1)
    a2.plot(d, x["z_ndv"], color=C_NDV, lw=1.1)
    a2.axhline(0, color="#8a8984", lw=0.6, ls=":")
    a2.set_yscale("symlog", linthresh=Z_LINEAR)
    lo, hi = a2.get_ylim()
    ticks = [v for v in (-30, -10, -3, 0, 3, 10, 30) if lo <= v <= hi]
    a2.set_yticks(ticks, [f"{v:g}" for v in ticks])
    a2.yaxis.set_minor_locator(plt.NullLocator())
    a2.set_ylabel("Z", fontsize=13)
    t = a2.twinx()
    t.plot(d, x["soil_t"], color=C_SOIL, lw=0.9, ls="--")
    t.set_ylabel("Soil T (°C)", fontsize=13)
    lo, hi = np.nanmin(x["soil_t"]), np.nanmax(x["soil_t"])
    t.set_ylim(min(lo, -1) - 1, max(hi, 1) + 1)
    t.axhline(0, color=C_SOIL, lw=0.5, alpha=0.5)
    for ax in (*axes, t):
        ax.tick_params(labelsize=12)
        ax.set_xlim(d.min(), d.max())
    a2.xaxis.set_major_locator(mdates.MonthLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))


def figure(sites, name, out_dir=None):
    set_rse_style()
    n = len(sites)
    fig = plt.figure(figsize=(11, 3.3 * n + 0.6))
    ratios = []
    for i in range(n):
        ratios += [0.75, 0.8, 1.6] + ([0.7] if i < n - 1 else [])
    height = 3.3 * n + 0.6
    gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.12, top=1 - 0.6 / height)
    row = 0
    for i, (kind, cell, start, landscape, place) in enumerate(sites):
        x = north_america_days(cell, start) if kind == "na" else external_days(cell, start)
        axes = [fig.add_subplot(gs[row + j]) for j in range(3)]
        row += 4
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        n = stations(cell, start)
        cell_text = f"9 km cell, {n} station{'s' if n > 1 else ''}"
        title = f"({'abcdefg'[i]}) {landscape}: {place} ({cell_text}), {start}–{start + 1}"
        panel(axes, x, title)
    handles = [Patch(color=COLORS[k], label=k.capitalize()) for k in CLASSES]
    handles += [
        plt.Line2D([], [], color=C_NPR, label="$Z_{NPR}$"),
        plt.Line2D([], [], color=C_NDV, label="$Z_{NDV}$"),
        plt.Line2D([], [], color=C_SOIL, ls="--", label="In situ soil temperature"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        fontsize=12.5,
        bbox_to_anchor=(0.5, 1.0),
    )
    if out_dir:
        fig.savefig(Path(out_dir) / f"{name}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    else:
        savefig(name, fig=fig)


def main():
    preview = sys.argv[sys.argv.index("--preview") + 1] if "--preview" in sys.argv else None
    figure(NORTH_AMERICA_FROZEN, "site_examples_north_america_frozen", preview)
    figure(NORTH_AMERICA_NEAR_ZERO, "site_examples_north_america_near_zero", preview)
    figure(EXTERNAL, "site_examples_external", preview)


if __name__ == "__main__":
    main()
