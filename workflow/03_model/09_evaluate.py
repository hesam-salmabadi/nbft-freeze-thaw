"""Evaluation of the three-class model against FT-SMOS and temperature (Section 3.6).

Uses the held-out predictions of 05_evaluate_cv.py (clean labelled AM days of the development
cells outside July and August) and the data of 08_extract_evaluation_data.py.

1. FT-SMOS: the model and FT-SMOS scored against the same in situ labels on the cell-days both
   cover (evaluation.scores; FT-SMOS partially frozen = transitional), overall and per tree-cover
   class, roughness class and biome. Table eval_smos.csv (block-weighted), eval_smos_unweighted.csv,
   confusion eval_smos_confusion.csv (block-weighted). On the cell-days on which the two give
   different states, per in situ label: the share of such days, and the shares on which the
   model, FT-SMOS, or neither matches the label (eval_smos_disagreement.csv, block-weighted over
   the days both cover).
2. Physical consistency: for each held-out model class, mean, quartiles and 5th and 95th
   percentiles of the in situ soil temperature and the ERA5-Land 10-day mean air temperature at
   the development cells, and of the air temperature for the classes of the maps on the domain
   sample (overall and per tree-cover class). Table eval_temperature.csv.
At the development cells every count and quantile is weighted so that each spatial block carries
the same total weight within the scored set (model.block_weights), as in training; the domain sample
is a random sample of cells and is not weighted.

Usage: python 09_evaluate.py
"""

import numpy as np
import pandas as pd

from nbft import config, paths
from nbft.evaluation import CODE_CLASS, scores, weighted_quantile
from nbft.model import CLASSES, LABEL_CLASS, STATE_CODES, block_weights

MIN_BIOME_CELLS = 10
QUANTILES = {"p05": 0.05, "p25": 0.25, "median": 0.5, "p75": 0.75, "p95": 0.95}
GROUPS = ("tree_class", "roughness_class", "biome_group")


def load():
    x = pd.read_parquet(config.path("model.cv_predictions"))
    cells = x.groupby("biome")["cell_id"].nunique()
    x["biome_group"] = x["biome"].where(x["biome"].map(cells) >= MIN_BIOME_CELLS, "other")
    for key in ("soil_temperature", "air_temperature", "smos"):
        x = x.merge(pd.read_parquet(config.path(f"model.eval_{key}")), how="left")
    return x


def block_w(df, weighted=True):
    return pd.Series(block_weights(df["block"]) if weighted else 1.0, index=df.index)


def smos_comparison(x, weighted=True):
    both = x.dropna(subset=["smos_state", "state"])
    subsets = [("all", "all", both)]
    subsets += [(group, k, sub) for group in GROUPS for k, sub in both.groupby(group)]
    rows = []
    for name, col in (("model", "state"), ("FT-SMOS", "smos_state")):
        for group, k, sub in subsets:
            w = block_w(sub, weighted)
            sc = scores(sub["label"], sub[col], weights=w)
            rows.append({"retrieval": name, "group": group, "class": k, **sc})
    table = pd.DataFrame(rows).round(3)
    w = block_w(both, weighted)
    confusion = pd.concat(
        {
            name: pd.crosstab(
                both["label"], both[col].map(CODE_CLASS), values=w, aggfunc="sum", normalize="index"
            )
            for name, col in (("model", "state"), ("FT-SMOS", "smos_state"))
        }
    ).round(3)
    return table, confusion


def disagreement(x):
    both = x.dropna(subset=["smos_state", "state"])
    w = block_w(both)
    label = both["label"].map(LABEL_CLASS).map(dict(enumerate(CLASSES))).map(STATE_CODES)
    differ = both["state"] != both["smos_state"]
    model_ok, smos_ok = both["state"] == label, both["smos_state"] == label
    rows = []
    for name, sel in [("all", both.index), *both.groupby("label").groups.items()]:
        ws, d = w[sel], differ[sel]
        wd = ws[d]
        rows.append(
            {
                "label": name,
                "days": len(sel),
                "share_differ": ws[d].sum() / ws.sum(),
                "model_matches": wd[model_ok[sel][d]].sum() / wd.sum(),
                "smos_matches": wd[smos_ok[sel][d]].sum() / wd.sum(),
                "neither": wd[~model_ok[sel][d] & ~smos_ok[sel][d]].sum() / wd.sum(),
            }
        )
    return pd.DataFrame(rows).round(3)


def summary(values, weights):
    ok = values.notna()
    q = {k: weighted_quantile(values, weights, v) for k, v in QUANTILES.items()}
    mean = np.average(values[ok], weights=weights[ok]) if ok.any() else np.nan
    return {"days": int(ok.sum()), "mean": mean, **q}


def temperatures(x):
    domain = pd.read_parquet(config.path("model.eval_domain_sample")).dropna(subset=["state"])
    rows = []
    sources = [
        ("development cells", "in situ soil", x, "soil_t", True),
        ("development cells", "ERA5-Land air (10-day)", x, "air_t10", True),
        ("domain sample", "ERA5-Land air (10-day)", domain, "air_t10", False),
    ]
    for sample, variable, df, col, weighted in sources:
        df = df.dropna(subset=[col])
        for group, sub in [("all", df), *df.groupby("tree_class")]:
            w = block_w(sub, weighted)
            for code, g in sub.groupby("state"):
                rows.append(
                    {
                        "sample": sample,
                        "variable": variable,
                        "tree_class": group,
                        "class": CODE_CLASS[int(code)],
                        **summary(g[col], w[g.index]),
                    }
                )
    return pd.DataFrame(rows).round(2)


def main():
    x = load()
    table, confusion = smos_comparison(x)
    table.to_csv(paths.TABLES / "eval_smos.csv", index=False)
    smos_comparison(x, weighted=False)[0].to_csv(
        paths.TABLES / "eval_smos_unweighted.csv", index=False
    )
    confusion.to_csv(paths.TABLES / "eval_smos_confusion.csv")
    differ = disagreement(x)
    differ.to_csv(paths.TABLES / "eval_smos_disagreement.csv", index=False)
    print(differ.to_string(index=False))
    temps = temperatures(x)
    temps.to_csv(paths.TABLES / "eval_temperature.csv", index=False)

    cols = ["retrieval", "group", "class", "days", "macro_f1", "kappa"]
    cols += [f"recall_{c}" for c in ("thawed", "transitional", "frozen")]
    print(table[cols].to_string(index=False), confusion.to_string(), sep="\n\n")
    print(temps[temps["tree_class"] == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
