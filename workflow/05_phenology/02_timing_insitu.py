"""Timing of the freeze season: model and FT-SMOS against the in situ labels (Results 4.7).

The events of nbft.timing (first transitional, first frozen, last frozen and last transitional
morning, each the start or end of a run of at least 5 mornings) are found per cell and freeze
season in three daily series of morning states, from 1 September to 30 June:
- in situ: the cell label (thawed, partial = transitional, frozen), every partial day included;
- model: at the development cells the held-out state (the fold model that did not see the cell's
  block, cv_coefficients.csv), at the external cells the production model; every day included;
- FT-SMOS: its morning state, partially frozen = transitional; days without data skipped.
A cell-season is used when the in situ label covers at least MIN_DAYS of its mornings.

For each event and retrieval, overall and per roughness and tree-cover class, on the cell-seasons on
which the in situ series and the retrieval have the event, and on those on which all three have it,
with every cell-season weighted so that each spatial block carries the same total weight (as in
training; unweighted in timing_scores_unweighted.csv):
number, bias (retrieval minus in situ, days), median absolute difference, Pearson correlation, and
standard deviation of the difference (SDD), as in Rautiainen et al. (2025); and the share of
cell-seasons on which the in situ series has the event and the retrieval does not, and the reverse.

Outputs: results/tables/timing_events.csv (every cell-season), timing_scores.csv,
timing_scores_unweighted.csv, timing_durations.csv (median transition lengths),
long_transition_seasons.csv (cell-seasons transitional on most November-March mornings).

Usage: python 02_timing_insitu.py
"""

import numpy as np
import pandas as pd

from nbft import config, paths
from nbft.evaluation import weighted_quantile
from nbft.model import TERMS, block_weights, class_probabilities, design, most_probable
from nbft.timing import EVENTS, season_start, series_events

MIN_DAYS = 200
LABEL_STATE = {"thawed": 0, "frozen": 1, "partial": 2}
SOURCES = ["insitu", "model", "smos"]


def states(x, coef_of):
    """Most probable state of every row, with the coefficients chosen by coef_of(rows)."""
    out = pd.Series(np.nan, index=x.index)
    for coef, rows in coef_of(x):
        probs = class_probabilities(
            coef[["intercept", *TERMS]], design(rows["z_npr"], rows["z_ndv"], rows["tree"])
        )
        out[rows.index] = np.asarray(most_probable(probs), dtype=float)
    return out


def development():
    x = pd.read_parquet(config.path("model.training_table"))
    folds = pd.read_parquet(config.path("model.cv_predictions"), columns=["cell_id", "fold"])
    x["fold"] = x["cell_id"].map(folds.drop_duplicates("cell_id").set_index("cell_id")["fold"])
    coefs = pd.read_csv(paths.TABLES / "cv_coefficients.csv")

    def by_fold(x):
        for f, rows in x.groupby("fold"):
            yield coefs[coefs["fold"] == f].set_index("class"), rows

    x["model"] = states(x, by_fold)
    x = x.merge(pd.read_parquet(config.path("model.eval_smos")), how="left")
    return x.assign(set="development")


def external():
    x = pd.read_parquet(config.path("validation.external_days"))
    coef = pd.read_csv(paths.TABLES / "model_coefficients.csv", index_col="class")
    x["model"] = states(x, lambda x: [(coef, x)])
    return x


def events(x):
    x = x.copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x[~x["date"].dt.month.isin([7, 8])]
    x["insitu"] = x["label"].map(LABEL_STATE)
    x["smos"] = x["smos_state"]
    x["season"] = season_start(x["date"]).dt.year.to_numpy()
    rows = []
    for (s, season), g in x.groupby(["set", "season"]):
        n = g.groupby("cell_id")["insitu"].count()
        g = g[g["cell_id"].isin(n[n >= MIN_DAYS].index)]
        if g.empty:
            continue
        for source in SOURCES:
            wide = g.pivot_table(index="date", columns="cell_id", values=source, dropna=False)
            full = pd.date_range(f"{season}-09-01", f"{season + 1}-06-30")
            e = series_events(wide.reindex(full))
            rows.append(e.assign(set=s, season=season, source=source).rename_axis("cell_id"))
    e = pd.concat(rows).reset_index()
    meta = x.drop_duplicates("cell_id").set_index("cell_id")
    for col in ("tree_class", "roughness_class", "region", "block"):
        if col in meta:
            e[col] = e["cell_id"].map(meta[col])
    return e


def weighted_stats(x, y, w):
    """Bias, median absolute difference, Pearson correlation and SDD of y - x, weighted by w."""
    w = np.asarray(w, float) / np.sum(w)
    d = y - x
    bias = np.sum(w * d)
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    r = cov / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2))
    return {
        "bias": bias,
        "median_difference": weighted_quantile(d, w, 0.5),
        "mad": weighted_quantile(np.abs(d), w, 0.5),
        "r": r,
        "sdd": np.sqrt(np.sum(w * (d - bias) ** 2)),
    }


def pair_scores(g, source, common, blocks, weighted):
    """Scores of one retrieval for one event; every cell-season weighted so that each spatial
    block carries the same total weight within the set scored (model.block_weights), or not."""
    has = g.notna()
    both = (has["insitu"] & has[source]) if not common else has.all(axis=1)
    block = blocks.reindex(g.index.get_level_values("cell_id")).to_numpy()

    def w(mask):
        mask = np.asarray(mask)
        return block_weights(block[mask]) if weighted else np.ones(mask.sum())

    def share(base, hit):
        base, hit = np.asarray(base), np.asarray(hit)
        return float(np.sum(w(base) * hit[base]) / max(base.sum(), 1)) if base.any() else np.nan

    out = {
        "retrieval": source,
        "pairs": "common" if common else "own",
        "seasons_insitu": int(has["insitu"].sum()),
        "n": int(both.sum()),
        "missed": share(has["insitu"], ~has[source]),
        "extra": share(~has["insitu"], has[source]),
    }
    if both.sum() > 2:
        out |= weighted_stats(
            g.loc[both, "insitu"].to_numpy(), g.loc[both, source].to_numpy(), w(both)
        )
    return out


def scores(e, weighted=True):
    """Per set, group, event and retrieval: on the cell-seasons where the in situ series and the
    retrieval have the event ('own'), and where all three have it ('common')."""
    blocks = e.drop_duplicates("cell_id").set_index("cell_id")["block"]
    keys = ["set", "group_by", "group", "cell_id", "season", "source"]
    wide = e.set_index(keys)[EVENTS].unstack("source")
    rows = []
    for (s, by, grp), g in wide.groupby(level=["set", "group_by", "group"]):
        for event in EVENTS:
            for source in ("model", "smos"):
                for common in (False, True):
                    meta = {"set": s, "group_by": by, "group": grp, "event": event}
                    rows.append(meta | pair_scores(g[event], source, common, blocks, weighted))
    return pd.DataFrame(rows)


def durations(e):
    """Median length of the autumn and spring transitions on the cell-seasons on which all three
    series have both events, block-weighted and unweighted."""
    blocks = e.drop_duplicates("cell_id").set_index("cell_id")["block"]
    wide = e.set_index(["set", "cell_id", "season", "source"])[EVENTS].unstack("source")
    rows = []
    for name, a, b in (
        ("autumn", "first_transition", "first_frozen"),
        ("spring", "last_frozen", "last_transition"),
    ):
        length = (wide[b] - wide[a]).dropna()
        for s, d in length.groupby(level="set"):
            w = block_weights(blocks.reindex(d.index.get_level_values("cell_id")).to_numpy())
            for source in SOURCES:
                rows.append(
                    {
                        "set": s,
                        "transition": name,
                        "source": source,
                        "seasons": len(d),
                        "median_weighted": weighted_quantile(d[source], w, 0.5),
                        "median_unweighted": d[source].median(),
                    }
                )
    return pd.DataFrame(rows)


def grouped(e):
    """The events once for all cells and once per roughness and tree-cover class."""
    parts = [e.assign(group_by="all", group="all")]
    for col in ("roughness_class", "tree_class"):
        parts.append(e.dropna(subset=[col]).assign(group_by=col, group=e[col]))
    return pd.concat(parts, ignore_index=True)


LONG_SHARE = 0.75
MIN_COLD_DAYS = 120


def long_transition(x):
    """Development cell-seasons on which at least LONG_SHARE of the labelled November-March
    mornings are transitional (partial), and the share of their clean transitional mornings that
    the held-out model calls transitional; block-weighted."""
    x = x[x["set"] == "development"].copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x[x["date"].dt.month.isin([11, 12, 1, 2, 3])]
    x["season"] = season_start(x["date"]).dt.year.to_numpy()
    x["partial"] = x["label"] == "partial"
    x["clean"] = x["clean"].astype(bool)
    x["hit"] = (x["model"] == 2) & x["partial"] & x["clean"]
    g = x.groupby(["cell_id", "season"]).agg(
        days=("label", "count"),
        share=("partial", "mean"),
        clean_partial=("clean", lambda c: (c & x.loc[c.index, "partial"]).sum()),
        hits=("hit", "sum"),
        block=("block", "first"),
        roughness=("roughness_class", "first"),
    )
    g = g[g["days"] >= MIN_COLD_DAYS]
    g["recall"] = g["hits"] / g["clean_partial"].where(g["clean_partial"] > 0)
    w = pd.Series(block_weights(g["block"]), index=g.index)
    long = g["share"] >= LONG_SHARE
    lw = pd.Series(block_weights(g.loc[long, "block"]), index=g.index[long])
    rough = g.loc[long, "roughness"].isin(["75-150 m", ">150 m"])
    ok = g.loc[long, "recall"].notna()
    return pd.DataFrame(
        [
            {
                "seasons": len(g),
                "long_seasons": int(long.sum()),
                "long_share_weighted": float(np.average(long, weights=w)),
                "long_rough_share_weighted": float(np.average(rough, weights=lw)),
                "recall_median_weighted": weighted_quantile(g.loc[long, "recall"][ok], lw[ok], 0.5),
                "recall_median_unweighted": g.loc[long, "recall"].median(),
            }
        ]
    )


def main():
    x = pd.concat([development(), external()], ignore_index=True)
    x = x[x["set"].isin(["development", "external"])]
    e = events(x)
    e.to_csv(paths.TABLES / "timing_events.csv", index=False)
    g = grouped(e)
    s = scores(g)
    s.round(3).to_csv(paths.TABLES / "timing_scores.csv", index=False)
    scores(g, weighted=False).round(3).to_csv(
        paths.TABLES / "timing_scores_unweighted.csv", index=False
    )
    d = durations(e)
    d.round(1).to_csv(paths.TABLES / "timing_durations.csv", index=False)
    lt = long_transition(x)
    lt.round(3).to_csv(paths.TABLES / "long_transition_seasons.csv", index=False)
    print(lt.round(3).to_string(index=False))
    pd.set_option("display.width", 220)
    cols = ["set", "group", "event", "retrieval", "pairs", "n", "bias", "mad", "r", "sdd", "missed"]
    sel = s["group_by"].isin(["all", "roughness_class"]) & (s["pairs"] == "own")
    print(s[sel][cols].round(2).to_string(index=False))
    print(d.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
