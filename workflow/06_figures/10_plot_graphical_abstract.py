"""Graphical abstract: the three states in the space of the two model inputs.

- Dots: SAMPLE clean training days per in situ label, drawn with probability proportional to the
  block weight of each day (model.block_weights), as in Fig. 4.
- Trajectory: the weighted medians of Z_NPR and Z_NDV of all held-out days, each day weighted by
  its block weight times its probability of each class (cv_predictions), joined from thawed to
  transitional to frozen.

Figure: paper/figs/graphical_abstract.pdf

Usage: python 10_plot_graphical_abstract.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from nbft import config
from nbft.evaluation import weighted_quantile
from nbft.model import CLASSES, block_weights
from nbft.plotting import savefig, set_rse_style

COLORS = {"thawed": "#d6604d", "transitional": "#f2c14e", "frozen": "#4575b4"}
LABEL_CLASS = {"thawed": "thawed", "partial": "transitional", "frozen": "frozen"}
NPR_RANGE, NDV_RANGE = (-6.5, 4.5), (-4.0, 150.0)
Z_LINEAR = 3.0
SAMPLE = 4000  # dots per label
SEED = 0


def symlog(v):
    v = np.asarray(v, float)
    return np.sign(v) * np.where(
        np.abs(v) <= Z_LINEAR, np.abs(v), Z_LINEAR * (1 + np.log10(np.abs(v) / Z_LINEAR))
    )


def inverse_symlog(u):
    return np.sign(u) * np.where(
        np.abs(u) <= Z_LINEAR, np.abs(u), Z_LINEAR * 10 ** (np.abs(u) / Z_LINEAR - 1)
    )


def load():
    x = pd.read_parquet(
        config.path("model.training_table"), columns=["cell_id", "date", "z_npr", "z_ndv"]
    )
    p = pd.read_parquet(config.path("model.cv_predictions"))
    d = p.merge(x, on=["cell_id", "date"])
    d["w"] = block_weights(d["block"])
    return d


def trajectory(ax, d):
    """Thawed -> transitional -> frozen through the medians of the held-out days, each weighted by
    its block weight times its probability of the class."""
    pts = []
    for k in CLASSES:
        w = d["w"] * d[f"p_{k}"]
        pts.append((weighted_quantile(d["z_npr"], w, 0.5), weighted_quantile(d["z_ndv"], w, 0.5)))
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#2b2b2b",
                "lw": 2.2,
                "shrinkA": 9,
                "shrinkB": 9,
                "mutation_scale": 18,
            },
            zorder=5,
        )
    for (px, py), k in zip(pts, CLASSES, strict=True):
        ax.scatter(px, py, s=220, color=COLORS[k], edgecolors="k", linewidths=1.5, zorder=6)


def main():
    set_rse_style()
    d = load()
    fig = plt.figure(figsize=(13.28, 5.31))
    ax = fig.add_axes([0.06, 0.13, 0.48, 0.82])
    dots = pd.concat(
        [
            g.sample(min(SAMPLE, len(g)), weights=g["w"], random_state=SEED)
            for _, g in d.groupby("label")
        ]
    ).sample(frac=1, random_state=SEED)  # mixed drawing order: no class hides another
    ax.scatter(
        dots["z_npr"],
        dots["z_ndv"],
        s=16,
        color=dots["label"].map(LABEL_CLASS).map(COLORS),
        edgecolors="none",
        alpha=0.35,
        rasterized=True,
        zorder=3,
    )
    trajectory(ax, d)
    ax.set_yscale("symlog", linthresh=Z_LINEAR)
    ax.set_xlim(*NPR_RANGE)
    ax.set_ylim(*NDV_RANGE)
    ticks = [-3, 0, 3, 10, 30, 100]
    ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(r"$Z_{NPR}$  (SMAP L-band polarization ratio)", fontsize=13)
    ax.set_ylabel(r"$Z_{NDV}$  (AMSR2 18.7/36.5 GHz difference)", fontsize=13)
    ax.tick_params(labelsize=12)

    states = [
        Line2D([], [], marker="o", ls="", ms=11, color=COLORS[k], mec="k", label=k.capitalize())
        for k in CLASSES
    ]
    ax.legend(
        handles=states,
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        fontsize=13,
        title="Soil state",
        title_fontsize=13,
        alignment="left",
    )
    lines = [
        (r"$NPR = \dfrac{TB_V - TB_H}{TB_V + TB_H}$", "SMAP, 1.41 GHz"),
        (
            r"$NDV = \dfrac{TB_{18.7V} - TB_{36.5V}}{TB_{18.7V} + TB_{36.5V}}$",
            "AMSR2, 18.7 and 36.5 GHz",
        ),
    ]
    y = 0.9
    for formula, source in lines:
        fig.text(0.59, y, formula, fontsize=15, va="top", ha="left")
        fig.text(0.59, y - 0.12, source, fontsize=12, va="top", ha="left", color="#55534f")
        y -= 0.26
    text = (
        "$Z$: departure from the thawed summer state of\n"
        "the cell, in units of its summer variability.\n\n"
        "NPR falls as soil water freezes;\n"
        "NDV rises with a frozen or snow-covered\n"
        "surface."
    )
    fig.text(0.59, 0.08, text, fontsize=13, va="bottom", ha="left", linespacing=1.4)
    savefig("graphical_abstract", fig=fig)


if __name__ == "__main__":
    main()
