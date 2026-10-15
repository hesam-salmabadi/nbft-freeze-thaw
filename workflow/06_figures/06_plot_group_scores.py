"""Held-out scores of the model and FT-SMOS by tree cover, roughness and biome (Results 4.3, 4.4).

One row per group, four columns: macro F1, Cohen's kappa, and the recall of the transitional and
the frozen class. Markers: the model on all its held-out days (cv_metrics.csv), and the model and
FT-SMOS on the days both have a state (eval_smos.csv).

Figure: paper/figs/group_scores.pdf

Usage: python 06_plot_group_scores.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from nbft import paths
from nbft.plotting import savefig, set_rse_style

METRICS = {
    "macro_f1": "Macro F1",
    "kappa": "Kappa",
    "recall_transitional": "Recall, transitional",
    "recall_frozen": "Recall, frozen",
}
GROUPS = {
    "all": ("All", ["all"]),
    "tree_class": ("Tree cover", ["<10%", "10-30%", "30-60%", ">60%"]),
    "roughness_class": ("Roughness", ["0-25 m", "25-75 m", "75-150 m", ">150 m"]),
    "biome_group": (
        "Biome",
        [
            "Tundra",
            "Boreal Forests/Taiga",
            "Temperate Grasslands, Savannas & Shrublands",
            "Temperate Broadleaf & Mixed Forests",
            "Temperate Conifer Forests",
            "Deserts & Xeric Shrublands",
        ],
    ),
}
SHORT = {
    "Boreal Forests/Taiga": "Boreal forest",
    "Temperate Grasslands, Savannas & Shrublands": "Temperate grassland",
    "Temperate Broadleaf & Mixed Forests": "Temperate broadleaf",
    "Temperate Conifer Forests": "Temperate conifer",
    "Deserts & Xeric Shrublands": "Desert and xeric",
}
MARKERS = [
    ("model, all days", "#2b2b2b", "o", "none"),
    ("model, FT-SMOS days", "#2b2b2b", "o", "full"),
    ("FT-SMOS", "#2a9d8f", "D", "full"),
]


def rows():
    cv = pd.read_csv(paths.TABLES / "cv_metrics.csv").assign(retrieval="model, all days")
    smos = pd.read_csv(paths.TABLES / "eval_smos.csv")
    smos["retrieval"] = smos["retrieval"].map(
        {"model": "model, FT-SMOS days", "FT-SMOS": "FT-SMOS"}
    )
    d = pd.concat([cv, smos], ignore_index=True)
    order = []
    for group, (title, classes) in GROUPS.items():
        for c in classes:
            name = "All days" if group == "all" else SHORT.get(c, c)
            order.append((group, c, name, title))
    return d, order


def main():
    set_rse_style()
    d, order = rows()
    fig, axes = plt.subplots(1, len(METRICS), figsize=(13, 6.2), sharey=True)
    ypos = {}
    y, last = 0, None
    for group, c, _, _ in order:
        if last is not None and group != last:
            y += 0.7
        ypos[(group, c)] = y
        y += 1
        last = group
    for ax, (metric, mtitle) in zip(axes, METRICS.items(), strict=True):
        for (group, c), yy in ypos.items():
            for j, (ret, color, marker, fill) in enumerate(MARKERS):
                v = d[(d["group"] == group) & (d["class"] == c) & (d["retrieval"] == ret)]
                if v.empty:
                    continue
                ax.plot(
                    v[metric].iloc[0],
                    yy + (j - 1) * 0.22,
                    marker=marker,
                    ms=6,
                    color=color,
                    mfc=color if fill == "full" else "white",
                    ls="none",
                    label=ret if (group, c) == ("all", "all") else None,
                )
        ax.set_title(mtitle, fontsize=13)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        ax.tick_params(labelsize=11)
        ax.set_axisbelow(True)
    ticks = [ypos[(g, c)] for g, c, _, _ in order]
    names = [n for _, _, n, _ in order]
    axes[0].set_yticks(ticks, names)
    axes[0].invert_yaxis()
    for group, (title, classes) in GROUPS.items():
        if group == "all":
            continue
        top = ypos[(group, classes[0])]
        axes[0].annotate(
            title,
            xy=(0, top - 0.75),
            xycoords=("axes fraction", "data"),
            xytext=(-150, 0),
            textcoords="offset points",
            fontsize=12,
            weight="bold",
            ha="left",
            va="center",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.55, 1.04),
    )
    fig.tight_layout()
    savefig("group_scores", fig=fig)


if __name__ == "__main__":
    main()
