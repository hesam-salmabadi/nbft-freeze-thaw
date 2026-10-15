"""How far sensors in one 9 km cell agree, and why a cell-day is partial.

Each sensor-day is labelled on its own (frozen if P(frozen) >= cut-off, thawed if P(thawed) >=
cut-off, otherwise uncertain). A partial cell-day is then one of:
- disagreement: every sensor is confident, but some say frozen and others thawed (the cell is
  spatially mixed: the footprint-versus-point problem);
- uncertain: no sensor is confident (a genuine transition, or thresholds too uncertain to decide);
- mixed: some sensors are confident, others are not.
Fractions are computed per cell, then summarized across cells (development cells, AM),
by the number of sensors that report that day.

Usage: python 06_within_cell_agreement.py
"""

import numpy as np
import pandas as pd

from nbft import insitu, paths

SENSOR_BINS = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, np.inf, "5+")]


def classify(sensor_days, cutoff=insitu.CUTOFF):
    s = sensor_days
    s = s.assign(
        conf_frozen=s["p_frozen"] >= cutoff - 1e-9, conf_thawed=s["p_thawed"] >= cutoff - 1e-9
    )
    g = s.groupby(["cell_id", "date"])
    d = pd.DataFrame(
        {
            "n": g.size(),
            "n_frozen": g["conf_frozen"].sum(),
            "n_thawed": g["conf_thawed"].sum(),
            "frozen_share": g["p_frozen"].mean(),
            "thawed_share": g["p_thawed"].mean(),
        }
    ).reset_index()
    d["label"] = insitu.labels(d["frozen_share"], d["thawed_share"], cutoff)
    n_unc = d["n"] - d["n_frozen"] - d["n_thawed"]
    d["partial_type"] = np.select(
        [d["label"] != "partial", n_unc == 0, n_unc == d["n"]],
        ["", "disagreement", "uncertain"],
        "mixed",
    )
    d["all_agree"] = (d["n_frozen"] == d["n"]) | (d["n_thawed"] == d["n"])
    return d


def per_cell(d):
    rows = []
    for (cell, nbin), c in d.groupby(["cell_id", "sensor_bin"], observed=True):
        partial = c[c["label"] == "partial"]
        confident = c[c["n_frozen"] + c["n_thawed"] == c["n"]]
        row = {
            "cell_id": cell,
            "sensor_bin": nbin,
            "days": len(c),
            "partial_fraction": len(partial) / len(c),
            "confident_agreement": confident["all_agree"].mean() if len(confident) else np.nan,
        }
        for t in ("disagreement", "mixed", "uncertain"):
            row[f"partial_{t}"] = (partial["partial_type"] == t).mean() if len(partial) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    cells = pd.read_csv(paths.TABLES / "insitu_cells.csv")
    cells = cells[cells["set"] != "excluded"]
    cells["group"] = cells["set"]
    s = insitu.read_sensor_days()
    s = s[(s["am_pm"] == "am") & s["cell_id"].isin(cells["cell_id"])]

    d = classify(s)
    bins = pd.cut(d["n"], [0, 1, 2, 4, np.inf], labels=[b[2] for b in SENSOR_BINS])
    d["sensor_bin"] = bins
    cell_stats = per_cell(d).merge(
        cells[["cell_id", "group", "elevation_std_m", "tree_cover_pct"]], on="cell_id"
    )
    cell_stats.to_csv(paths.TABLES / "insitu_within_cell_cells.csv", index=False)

    cols = [
        "partial_fraction",
        "confident_agreement",
        "partial_disagreement",
        "partial_mixed",
        "partial_uncertain",
    ]
    summary = (
        cell_stats.groupby(["group", "sensor_bin"], observed=True)[cols]
        .median()
        .assign(n_cells=cell_stats.groupby(["group", "sensor_bin"], observed=True).size())
        .round(3)
        .reset_index()
    )
    summary.to_csv(paths.TABLES / "insitu_within_cell_summary.csv", index=False)
    print("median across cells (a cell appears once per sensor-count bin it has days in):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
