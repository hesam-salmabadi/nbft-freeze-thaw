"""Does a shared freeze-thaw model work as well in rough terrain as on flat ground?

Reference: the in situ label of the cell-day (cut-off insitu.CUTOFF). Samples: AM days from October
to April that are frozen or thawed (partial days dropped) in development cells. Features: the four
indices standardized against each pixel's thaw reference (insitu.FEATURES, as in the model) and
the cell's tree cover. Every prediction comes from a model that never saw the cell: cells are
held out in spatial blocks of about 200 km (insitu.BLOCK_CELLS), so neither the cell's own days
nor its neighbours' are in training. Three designs:
- all: one model trained on cells of every roughness (5 block folds), scored on each held-out cell;
- within_range: the same, but trained and tested only inside each roughness range;
- flat_only: trained on flat cells (roughness <= FLAT_M) and tested on rougher ranges (training
  cells that share a block with a test cell are left out); the flat range itself is scored by
  block folds among flat cells.
Per held-out cell: AUC of P(frozen) against the in situ label, and balanced accuracy at P = 0.5
(this one also penalizes offsets in calibration). Cells need MIN_STATE_DAYS frozen and thawed days.
Ranges are summarized by the median over cells.

Usage: python 11_roughness_effect.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nbft import config, insitu, paths
from nbft.plotting import savefig, set_rse_style

MONTHS = (10, 11, 12, 1, 2, 3, 4)
MIN_STATE_DAYS = 10
FLAT_M = 25.0
BINS = [0, 25, 75, 150, 300, np.inf]
NAMES = ["0-25", "25-75", "75-150", "150-300", ">300"]
FEATURES = [*insitu.FEATURES, "tree"]
DESIGNS = {  # validated categorical slots 1-3 (dataviz reference palette)
    "all": ("Trained on all roughness", "#2a78d6", "o"),
    "within_range": ("Trained within the range", "#eb6834", "s"),
    "flat_only": (f"Trained on flat cells (<= {FLAT_M:.0f} m)", "#1baf7a", "^"),
}


def samples(cells):
    lab = f"label_{insitu.CUTOFF:g}"
    d = pd.read_parquet(config.path("insitu.cell_overpass", grid=insitu.GRID))
    d = d[(d["am_pm"] == "am") & d["cell_id"].isin(cells["cell_id"])][["cell_id", "date", lab]]
    d = d[d[lab] != "partial"]
    s = pd.read_parquet(config.path("insitu.cell_satellite", grid=insitu.GRID))
    d["date"], s["date"] = pd.to_datetime(d["date"]), pd.to_datetime(s["date"])
    x = d.merge(s, on=["cell_id", "date"])
    x = x[pd.DatetimeIndex(x["date"]).month.isin(MONTHS)]
    x = insitu.add_zscores(x).merge(cells[["cell_id", "range", "tree_cover_pct"]], on="cell_id")
    x["tree"] = x["tree_cover_pct"] / 100
    x["frozen"] = (x[lab] == "frozen").astype(int)
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    return x.reset_index(drop=True)


def score(x, p, design):
    return insitu.cell_scores(x, p, MIN_STATE_DAYS).assign(design=design)


def run(x):
    out = [score(x, insitu.held_out_predictions(x, x, FEATURES), "all")]
    for name in NAMES:
        xb = x[x["range"] == name]
        out.append(score(xb, insitu.held_out_predictions(xb, xb, FEATURES), "within_range"))
    flat = x[x["range"] == NAMES[0]]
    out.append(score(flat, insitu.held_out_predictions(flat, flat, FEATURES), "flat_only"))
    blocks = insitu.spatial_blocks(x["cell_id"])
    for name in NAMES[1:]:
        test = x[x["range"] == name]
        train = flat[~blocks[flat.index].isin(blocks[test.index])]
        out.append(score(test, insitu.held_out_predictions(train, test, FEATURES), "flat_only"))
    return pd.concat(out, ignore_index=True)


def figure(summary):
    set_rse_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    pos = np.arange(len(NAMES))
    for ax, metric, ylab in [
        (axes[0], "auc", "AUC on held-out cells (median)"),
        (axes[1], "balanced_accuracy", "Balanced accuracy at P = 0.5 (median)"),
    ]:
        for k, (lab, color, marker) in DESIGNS.items():
            s = summary[summary["design"] == k].set_index("range").reindex(NAMES)
            ax.fill_between(
                pos, s[f"{metric}_q25"], s[f"{metric}_q75"], color=color, alpha=0.10, lw=0
            )
            ax.plot(pos, s[f"{metric}_median"], color=color, marker=marker, ms=8, lw=2, label=lab)
        ax.set_xticks(pos, NAMES)
        ax.set_xlabel("Roughness range (elevation std, m)")
        ax.set_ylabel(ylab)
    n = summary[summary["design"] == "all"].set_index("range").reindex(NAMES)
    for i, (c, sz) in enumerate(zip(n["cells"], n["cell_seasons"], strict=True)):
        axes[0].annotate(
            f"{c:.0f} cells\n{sz:.0f} seasons",
            (i, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=11,
        )
    axes[0].legend(frameon=False, fontsize=12, loc="lower left", bbox_to_anchor=(0, 0.12))
    axes[0].set_title("(a)", loc="left")
    axes[1].set_title("(b)", loc="left")
    savefig("insitu_roughness_effect", fig=fig)


def main():
    cells = pd.read_csv(paths.TABLES / "insitu_cells.csv")
    cells = cells[cells["set"] == "development"].copy()
    cells["range"] = pd.cut(cells["elevation_std_m"], BINS, labels=NAMES, right=True)
    cells["range"] = cells["range"].astype(str)
    x = samples(cells)
    print(f"{len(x):,} cell-days from {x['cell_id'].nunique()} cells")

    scores = run(x).merge(cells[["cell_id", "range", "complete_seasons"]], on="cell_id")
    scores.to_csv(paths.TABLES / "insitu_roughness_cells.csv", index=False)
    rows = []
    for (design, rng), g in scores.groupby(["design", "range"]):
        row = {"design": design, "range": rng, "cells": len(g)}
        row["cell_seasons"] = int(g["complete_seasons"].sum())
        for m in ("auc", "balanced_accuracy"):
            q = g[m].quantile([0.25, 0.5, 0.75])
            row |= {f"{m}_q25": q[0.25], f"{m}_median": q[0.5], f"{m}_q75": q[0.75]}
        rows.append(row)
    summary = pd.DataFrame(rows).round(3)
    summary["range"] = pd.Categorical(summary["range"], NAMES, ordered=True)
    summary = summary.sort_values(["design", "range"])
    summary.to_csv(paths.TABLES / "insitu_roughness_effect.csv", index=False)
    print(summary.to_string(index=False))
    figure(summary)


if __name__ == "__main__":
    main()
