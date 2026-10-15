"""Temperature of each model class (Results 4.5).

Three panels from eval_temperature.csv (03_model/09_evaluate.py): the in situ soil temperature and
the ERA5-Land 10-day mean air temperature of the held-out classes at the development cells, and the
air temperature of the map classes on the domain sample. Per panel, one group of three boxes for all
cells and one per tree-cover class: median (line), quartiles (box), 5th and 95th percentiles
(whiskers).

Figure: paper/figs/class_temperature.pdf

Usage: python 07_plot_temperature.py
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from nbft import paths
from nbft.plotting import savefig, set_rse_style

COLORS = {"thawed": "#d6604d", "transitional": "#f2c14e", "frozen": "#4575b4"}
GROUPS = ["all", "<10%", "10-30%", "30-60%", ">60%"]
GROUP_LABELS = ["All", "<10%", "10–30%", "30–60%", ">60%"]
PANELS = [
    ("development cells", "in situ soil", "(a) In situ soil, development cells"),
    (
        "development cells",
        "ERA5-Land air (10-day)",
        "(b) ERA5-Land air (10-day), development cells",
    ),
    ("domain sample", "ERA5-Land air (10-day)", "(c) ERA5-Land air (10-day), domain sample"),
]
WIDTH = 0.26


def boxes(ax, t):
    for i, group in enumerate(GROUPS):
        for j, k in enumerate(COLORS):
            r = t[(t["tree_class"] == group) & (t["class"] == k)]
            if r.empty:
                continue
            r = r.iloc[0]
            x = i + (j - 1) * (WIDTH + 0.03)
            ax.plot([x, x], [r["p05"], r["p25"]], color="#444444", lw=0.9)
            ax.plot([x, x], [r["p75"], r["p95"]], color="#444444", lw=0.9)
            ax.add_patch(
                Rectangle(
                    (x - WIDTH / 2, r["p25"]),
                    WIDTH,
                    r["p75"] - r["p25"],
                    facecolor=COLORS[k],
                    edgecolor="#444444",
                    lw=0.8,
                )
            )
            ax.plot([x - WIDTH / 2, x + WIDTH / 2], [r["median"]] * 2, color="k", lw=1.4)
    ax.axhline(0, color="#8a8984", lw=0.7, ls=":")
    ax.set_xticks(range(len(GROUPS)), GROUP_LABELS)
    ax.set_xlim(-0.6, len(GROUPS) - 0.4)


def main():
    set_rse_style()
    t = pd.read_csv(paths.TABLES / "eval_temperature.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (sample, variable, title) in zip(axes, PANELS, strict=True):
        boxes(ax, t[(t["sample"] == sample) & (t["variable"] == variable)])
        ax.set_title(title, fontsize=13, loc="left")
        ax.set_xlabel("Tree cover", fontsize=13)
        ax.tick_params(labelsize=11)
    axes[0].set_ylabel("Temperature (°C)", fontsize=13)
    handles = [
        Patch(facecolor=c, edgecolor="#444444", label=k.capitalize()) for k, c in COLORS.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.tight_layout()
    savefig("class_temperature", fig=fig)


if __name__ == "__main__":
    main()
