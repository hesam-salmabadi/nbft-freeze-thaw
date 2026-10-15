"""Sensitivity of the daily AM cell labels to the cut-off (insitu.CUTOFFS, 0.70 to 0.90).

An AM cell-day is frozen when the mean sensor P(frozen) reaches the cut-off, thawed when P(thawed)
does, partial otherwise. A lower cut-off gives more frozen and thawed labels but lets less certain
days in. For every cut-off, from the AM cell-days of the development set (study period, outside
summer):
- per cell: the share of its days that are frozen / partial, summarized (median and IQR) across
  the cells that are frozen on at least one day at the lowest cut-off, so that cells that never
  freeze do not pull the medians to zero;
- in total: frozen and partial AM cell-days, and the cells frozen on at least one day.

Usage: python 12_cutoff_sensitivity.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from nbft import config, insitu, paths
from nbft.plotting import savefig, set_rse_style

BLUE, ORANGE = "#2a78d6", "#eb6834"


def cell_shares(daily):
    rows = []
    for cutoff in insitu.CUTOFFS:
        lab = daily[f"label_{cutoff:g}"]
        g = daily.assign(frozen=lab == "frozen", partial=lab == "partial").groupby("cell_id")
        rows.append(
            pd.DataFrame(
                {
                    "cutoff": cutoff,
                    "days": g.size(),
                    "frozen_days": g["frozen"].sum(),
                    "frozen_share": g["frozen"].mean(),
                    "partial_share": g["partial"].mean(),
                }
            ).reset_index()
        )
    return pd.concat(rows, ignore_index=True)


def summarize(cells):
    low = cells[cells["cutoff"] == min(insitu.CUTOFFS)]
    freezing = set(low.loc[low["frozen_days"] > 0, "cell_id"])
    rows = []
    for cutoff, c in cells.groupby("cutoff"):
        row = {
            "cutoff": cutoff,
            "cells": len(c),
            "cells_ever_frozen": int((c["frozen_days"] > 0).sum()),
            "frozen_am_days": int(c["frozen_days"].sum()),
            "partial_am_days": int((c["partial_share"] * c["days"]).round().sum()),
            "share_cells": len(freezing),
        }
        ever = c[c["cell_id"].isin(freezing)]
        for col in ("frozen_share", "partial_share"):
            q = ever[col].quantile([0.25, 0.5, 0.75])
            row |= {f"{col}_q25": q[0.25], f"{col}_median": q[0.5], f"{col}_q75": q[0.75]}
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def figure(summary):
    set_rse_style()
    s = summary.set_index("cutoff")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax = axes[0]
    for col, color, marker, name in [
        ("frozen_share", BLUE, "o", "Frozen"),
        ("partial_share", ORANGE, "s", "Partial"),
    ]:
        ax.fill_between(
            s.index, s[f"{col}_q25"] * 100, s[f"{col}_q75"] * 100, color=color, alpha=0.15, lw=0
        )
        ax.plot(
            s.index, s[f"{col}_median"] * 100, color=color, marker=marker, lw=2, ms=7, label=name
        )
    ax.set_xlabel("Cut-off")
    ax.set_ylabel("Share of the cell's morning days (%)")
    ax.legend(frameon=False, fontsize=14)
    ax.set_title("(a) Cells that freeze: median and IQR", loc="left", fontsize=14)

    ax = axes[1]
    ax.plot(s.index, s["frozen_am_days"] / 1000, color=BLUE, marker="o", lw=2, ms=7)
    for xv, yv, n in zip(s.index, s["frozen_am_days"] / 1000, s["cells_ever_frozen"], strict=True):
        ax.annotate(
            f"{n} cells",
            (xv, yv),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=12,
        )
    ax.set_xlabel("Cut-off")
    ax.margins(x=0.08, y=0.15)
    ax.set_ylabel("Frozen morning cell-days (thousands)")
    ax.set_title("(b) Frozen cell-days; cells ever frozen", loc="left", fontsize=14)
    savefig("insitu_cutoff_sensitivity", fig=fig)


def main():
    dev = pd.read_csv(paths.TABLES / "insitu_cells.csv")
    dev = dev.loc[dev["set"] == "development", "cell_id"]
    daily = pd.read_parquet(config.path("insitu.cell_overpass", grid=insitu.GRID))
    daily = daily[(daily["am_pm"] == "am") & daily["cell_id"].isin(dev)]
    cells = cell_shares(daily)
    cells.to_csv(paths.TABLES / "insitu_cutoff_cells.csv", index=False)
    summary = summarize(cells)
    summary.to_csv(paths.TABLES / "insitu_cutoff_sensitivity.csv", index=False)
    print(summary.to_string(index=False))
    figure(summary)


if __name__ == "__main__":
    main()
