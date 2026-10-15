"""Share of frozen, partial and thawed AM cell-days, averaged per cell before it is summarized.

Records differ in length and season coverage, so pooling all cell-days would let long or
winter-heavy records dominate. Each cell is therefore reduced to one value per label:
1. label fractions per calendar month, pooled over the cell's years;
2. the cell's fraction is the mean of its monthly fractions over a season: 'cold' (September to
   February, which every cell covers; see 02_select_cells.py) and 'year' (all twelve months, only
   for cells with at least insitu.MIN_MONTH_DAYS days in every month).
Groups (season x set x tree-cover class) are then summarized across cells (median and quartiles),
at each cut-off. The monthly fractions averaged across cells give the seasonal profile.

Usage: python 04_label_composition.py
"""

import pandas as pd

from nbft import config, insitu, paths


def monthly_fractions(daily, cutoff):
    d = daily.assign(month=pd.DatetimeIndex(daily["date"]).month)
    counts = pd.crosstab([d["cell_id"], d["month"]], d[f"label_{cutoff:g}"])
    counts = counts.reindex(columns=list(insitu.LABELS), fill_value=0)
    n = counts.sum(axis=1)
    frac = counts.div(n, axis=0)[n >= insitu.MIN_MONTH_DAYS]
    return frac.reset_index()


SEASONS = {"cold": insitu.COLD_MONTHS, "year": tuple(range(1, 13))}


def cell_fractions(monthly, months):
    m = monthly[monthly["month"].isin(months)]
    g = m.groupby("cell_id")
    out = g[list(insitu.LABELS)].mean()
    return out[g.size() == len(months)]


def main():
    cells = pd.read_csv(paths.TABLES / "insitu_cells.csv")
    cells = cells[cells["set"] != "excluded"]
    cells["tree_class"] = insitu.tree_cover_class(cells["tree_cover_pct"]).to_numpy()
    cells["group"] = cells["set"]

    daily = pd.read_parquet(config.path("insitu.cell_overpass", grid=insitu.GRID))
    daily = daily[(daily["am_pm"] == "am") & daily["cell_id"].isin(cells["cell_id"])]

    per_cell, profiles = [], []
    for cutoff in insitu.CUTOFFS:
        monthly = monthly_fractions(daily, cutoff)
        for season, months in SEASONS.items():
            frac = cell_fractions(monthly, months).assign(season=season, cutoff=cutoff)
            per_cell.append(frac.reset_index())
        m = monthly.merge(cells[["cell_id", "group", "tree_class"]], on="cell_id")
        profiles.append(
            m.groupby(["group", "tree_class", "month"], observed=True)[list(insitu.LABELS)]
            .mean()
            .assign(
                cutoff=cutoff,
                n_cells=m.groupby(["group", "tree_class", "month"], observed=True).size(),
            )
            .reset_index()
        )
    per_cell = pd.concat(per_cell).merge(cells[["cell_id", "group", "tree_class"]], on="cell_id")
    per_cell.to_csv(paths.TABLES / "insitu_label_composition_cells.csv", index=False)
    pd.concat(profiles).to_csv(paths.TABLES / "insitu_label_monthly_profile.csv", index=False)

    rows = []
    keys = ["season", "cutoff", "group", "tree_class"]
    for (season, cutoff, group, tree), d in per_cell.groupby(keys, observed=True):
        row = {"season": season, "cutoff": cutoff, "group": group, "tree_class": tree}
        row["n_cells"] = len(d)
        for lab in insitu.LABELS:
            q = d[lab].quantile([0.25, 0.5, 0.75])
            row |= {f"{lab}_median": q[0.5], f"{lab}_q25": q[0.25], f"{lab}_q75": q[0.75]}
        rows.append(row)
    summary = pd.DataFrame(rows).round(3)
    summary.to_csv(paths.TABLES / "insitu_label_summary.csv", index=False)
    n_total = cells.groupby("group").size()
    full = per_cell[(per_cell["cutoff"] == insitu.CUTOFF) & (per_cell["season"] == "year")]
    n_full = full.groupby("group").size()
    print("cells with all twelve months:", (n_full / n_total).round(2).to_dict())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
