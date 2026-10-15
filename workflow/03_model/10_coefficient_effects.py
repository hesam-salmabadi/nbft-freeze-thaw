"""Coefficients of the production model and the effect of each index per tree-cover class
(Results 4.3).

- model_coefficient_table.csv: the twelve production coefficients with their range over the five
  cross-validation folds (cv_coefficients.csv).
- model_index_effects.csv: per tree-cover class, the change in each log-ratio eta_k for a change of
  one standard deviation in Z_NPR or Z_NDV, (b_index + b_tree_x_index * T) * SD, at the median tree
  cover T of the class's training days, with SD the standard deviation of the index on the clean
  frozen, thawed and transitional days of the class (median and SD with every day weighted so that
  each spatial block carries the same total weight, model.block_weights). Comparable across the
  two indices, whose spreads differ by about a factor of five.

Usage: python 10_coefficient_effects.py
"""

import numpy as np
import pandas as pd

from nbft import config, paths
from nbft.evaluation import weighted_quantile
from nbft.model import TERMS, block_weights

CLASSES = ["transitional", "frozen"]
TREE_ORDER = ["<10%", "10-30%", "30-60%", ">60%"]


def main():
    coef = pd.read_csv(paths.TABLES / "model_coefficients.csv", index_col="class")
    folds = pd.read_csv(paths.TABLES / "cv_coefficients.csv")
    cols = ["intercept", *TERMS]
    rows = []
    for k in CLASSES:
        f = folds[folds["class"] == k][cols]
        for term in cols:
            rows.append(
                {
                    "class": k,
                    "term": term,
                    "coefficient": coef.loc[k, term],
                    "fold_min": f[term].min(),
                    "fold_max": f[term].max(),
                }
            )
    table = pd.DataFrame(rows).round(3)
    table.to_csv(paths.TABLES / "model_coefficient_table.csv", index=False)

    x = pd.read_parquet(config.path("model.training_table"))
    x = x[x["clean"]]
    rows = []
    for tc in TREE_ORDER:
        g = x[x["tree_class"] == tc]
        w = block_weights(g["block"])
        t = weighted_quantile(g["tree"], w, 0.5)
        for k in CLASSES:
            row = {"tree_class": tc, "tree": t, "class": k, "days": len(g)}
            row["intercept"] = coef.loc[k, "intercept"] + coef.loc[k, "tree"] * t
            for i in ("npr", "ndv"):
                slope = coef.loc[k, f"z_{i}"] + coef.loc[k, f"tree_x_{i}"] * t
                v = g[f"z_{i}"].to_numpy()
                mean = np.average(v, weights=w)
                row[f"sd_z_{i}"] = np.sqrt(np.average((v - mean) ** 2, weights=w))
                row[f"slope_z_{i}"] = slope
                row[f"effect_z_{i}"] = slope * row[f"sd_z_{i}"]
            rows.append(row)
    effects = pd.DataFrame(rows).round(3)
    effects.to_csv(paths.TABLES / "model_index_effects.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False), "", effects.to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()
