"""Timing of freeze-up and thaw at the development cells: model and FT-SMOS against the in situ
labels, after Fig. 5 of Rautiainen et al. (2025) (Results 4.5).

Rows: first transitional, first frozen and last transitional morning (nbft.timing); columns: the
model (held out) and FT-SMOS. One point per cell-season on which the labels and the product have
the event, with the 1:1 line; the number of points and the block-weighted bias, Pearson correlation
and standard deviation of the difference (SDD) come from timing_scores.csv, the points from
timing_events.csv (05_phenology/02_timing_insitu.py).

Figure: paper/figs/timing_scatter.pdf

Usage: python 09_plot_timing_scatter.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from nbft import paths
from nbft.plotting import savefig, set_rse_style

EVENTS = {
    "first_transition": ("Start of freeze-up", {31: "1 Sep", 92: "1 Nov", 153: "1 Jan"}),
    "first_frozen": ("First frozen morning", {31: "1 Sep", 92: "1 Nov", 153: "1 Jan"}),
    "last_transition": ("End of thaw", {212: "1 Mar", 273: "1 May", 334: "1 Jul"}),
}
LIMITS = {"first_transition": (31, 214), "first_frozen": (31, 214), "last_transition": (153, 334)}
PRODUCTS = {"model": ("Model", "#2b2b2b"), "smos": ("FT-SMOS", "#2a9d8f")}


def main():
    set_rse_style()
    e = pd.read_csv(paths.TABLES / "timing_events.csv")
    e = e[e["set"] == "development"]
    sc = pd.read_csv(paths.TABLES / "timing_scores.csv")
    sc = sc[(sc["set"] == "development") & (sc["group_by"] == "all") & (sc["pairs"] == "own")]
    sc = sc.set_index(["event", "retrieval"])
    wide = e.set_index(["cell_id", "season", "source"])[list(EVENTS)].unstack("source")
    fig, axes = plt.subplots(3, 2, figsize=(9, 13))
    for i, (event, (title, ticks)) in enumerate(EVENTS.items()):
        lo, hi = LIMITS[event]
        for j, (product, (name, color)) in enumerate(PRODUCTS.items()):
            ax = axes[i, j]
            d = wide[event][["insitu", product]].dropna()
            ax.scatter(d["insitu"], d[product], s=6, color=color, alpha=0.35, lw=0, rasterized=True)
            ax.plot([lo, hi], [lo, hi], color="#8a8984", lw=1, ls="--")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_xticks(list(ticks), list(ticks.values()))
            ax.set_yticks(list(ticks), list(ticks.values()))
            ax.tick_params(labelsize=11)
            st = sc.loc[(event, product)]
            ax.text(
                0.03,
                0.97,
                f"N = {int(st['n'])}\nbias = {st['bias']:.1f} d\nR = {st['r']:.2f}\n"
                f"SDD = {st['sdd']:.1f} d",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
            )
            ax.set_title(f"({'abcdef'[2 * i + j]}) {title}, {name}", loc="left", fontsize=12)
            ax.set_xlabel("In situ", fontsize=12)
            ax.set_ylabel(name, fontsize=12)
    fig.tight_layout()
    savefig("timing_scatter", fig=fig)


if __name__ == "__main__":
    main()
