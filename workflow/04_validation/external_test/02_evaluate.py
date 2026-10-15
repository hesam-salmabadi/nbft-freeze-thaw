"""External test of the production model (Section 3.6.4 and Supplement).

The production coefficients (results/tables/model_coefficients.csv), unchanged, are applied to the
cell-days of 01_build.py; the state of a day is its most probable class. Scored on the clean days
against the in situ labels:
- external cells: every count weighted so that each 200 km spatial block carries the same total
  weight (model.block_weights), as in training, overall and per region; and, per model class,
  the weighted mean, median and quartiles of the in situ soil temperature
  (external_scores.csv, external_soil_temperature.csv);
- per network, unweighted (supplement; external_networks.csv);
- George River and Cambridge Bay, each one cell, unweighted (supplement; external_arctic_sites.csv,
  with soil temperature);
- the model and FT-SMOS on the days FT-SMOS covers (supplement; external_smos.csv).

Output: validation.external_predictions and the tables above in results/tables.

Usage: python 02_evaluate.py
"""

import numpy as np
import pandas as pd

from nbft import config, paths
from nbft.evaluation import CODE_CLASS, scores, weighted_quantile
from nbft.model import TERMS, block_weights, class_probabilities, design, most_probable

MIN_SMOS_DAYS = 100


def predict(x):
    coef = pd.read_csv(paths.TABLES / "model_coefficients.csv", index_col="class")
    probs = class_probabilities(
        coef[["intercept", *TERMS]], design(x["z_npr"], x["z_ndv"], x["tree"])
    )
    for k, v in probs.items():
        x[f"p_{k}"] = np.asarray(v)
    x["state"] = most_probable(probs)
    return x.dropna(subset=["state"])


def temperature(g, weights):
    rows = []
    for code, h in g.groupby("state"):
        w, t = weights[h.index], h["soil_t"]
        ok = t.notna()
        rows.append(
            {
                "class": CODE_CLASS[int(code)],
                "days": len(h),
                "mean": np.average(t[ok], weights=w[ok]) if ok.any() else np.nan,
                "p25": weighted_quantile(t, w, 0.25),
                "median": weighted_quantile(t, w, 0.5),
                "p75": weighted_quantile(t, w, 0.75),
            }
        )
    return pd.DataFrame(rows)


def group_rows(d, by, weighted, extra=None):
    rows, temps = [], []
    for name, g in [("all", d), *d.groupby(by)]:
        w = pd.Series(block_weights(g["block"]) if weighted else 1.0, index=g.index)
        meta = {by: name, "cells": g["cell_id"].nunique(), "blocks": g["block"].nunique()}
        rows.append(meta | scores(g["label"], g["state"], g["p_frozen"], weights=w))
        temps.append(temperature(g, w).assign(**{by: name}))
    return pd.DataFrame(rows), pd.concat(temps, ignore_index=True)


def smos_rows(d):
    rows = []
    for name, g in [("all", d), *d.groupby("region")]:
        b = g.dropna(subset=["smos_state"])
        if len(b) < MIN_SMOS_DAYS:
            continue
        for retrieval, col in (("model", "state"), ("FT-SMOS", "smos_state")):
            rows.append(
                {
                    "region": name,
                    "retrieval": retrieval,
                    "cells": b["cell_id"].nunique(),
                    **scores(b["label"], b[col]),
                }
            )
    return pd.DataFrame(rows)


def main():
    days = pd.read_parquet(config.path("validation.external_days"))
    d = predict(days[days["clean"]].copy())
    d.to_parquet(config.path("validation.external_predictions"), index=False)
    ext, arc = d[d["set"] == "external"], d[d["set"] == "arctic_sites"]

    main_scores, main_temp = group_rows(ext, "region", weighted=True)
    networks, _ = group_rows(ext, "network", weighted=False)
    arctic, arctic_temp = group_rows(arc, "region", weighted=False)
    smos = smos_rows(d)
    tables = {
        "external_scores": main_scores,
        "external_soil_temperature": main_temp,
        "external_networks": networks,
        "external_arctic_sites": arctic.merge(
            arctic_temp.pivot(index="region", columns="class", values="median").add_prefix(
                "median_soil_t_"
            ),
            left_on="region",
            right_index=True,
            how="left",
        ),
        "external_smos": smos,
    }
    for name, t in tables.items():
        t.round(3).to_csv(paths.TABLES / f"{name}.csv", index=False)

    cols = [
        "region",
        "cells",
        "blocks",
        "days",
        "share_frozen",
        "share_transitional",
        "macro_f1",
        "kappa",
        "recall_thawed",
        "recall_transitional",
        "recall_frozen",
        "precision_frozen",
        "precision_transitional",
    ]
    pd.set_option("display.width", 240)
    print(main_scores[cols].round(2).to_string(index=False))
    print(main_temp.round(2).to_string(index=False))
    print(arctic[cols].round(2).to_string(index=False))
    print(
        smos[["region", "retrieval", "cells", "days", "macro_f1", "kappa", "recall_transitional"]]
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
