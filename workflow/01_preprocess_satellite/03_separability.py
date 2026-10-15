"""How well each index separates frozen from thawed days, raw and standardized (supplement table).

For the labelled AM cell-days of the development cells outside July and August (frozen or thawed
at the 0.75 cut-off), the area under the ROC curve of each index alone: the probability that a
random frozen day has a more frozen-like value than a random thawed day (0.5: no separation,
1: full separation), with every day weighted so that each spatial block carries the same total
weight (model.block_weights). The sign of insitu.INDEX_SIGNS orients each index.

Output: results/tables/standardization_separability.csv

Usage: python 03_separability.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nbft import config, insitu, paths
from nbft.model import block_weights


def main():
    sat = pd.read_parquet(config.path("insitu.cell_satellite", grid=insitu.GRID))
    labels = pd.read_parquet(config.path("insitu.cell_overpass", grid=insitu.GRID))
    labels = labels.loc[labels["am_pm"] == "am", ["cell_id", "date", "label_0.75"]]
    sat["date"], labels["date"] = pd.to_datetime(sat["date"]), pd.to_datetime(labels["date"])
    sat = sat[~sat["date"].dt.month.isin(insitu.SUMMER_MONTHS)]
    x = insitu.add_zscores(sat).merge(labels, on=["cell_id", "date"])
    x = x[x["label_0.75"].isin(["frozen", "thawed"])]

    rows = []
    for name, sign in insitu.INDEX_SIGNS.items():
        v = x[[name, f"z_{name}", "label_0.75"]].dropna()
        frozen = v["label_0.75"] == "frozen"
        w = block_weights(insitu.spatial_blocks(x.loc[v.index, "cell_id"]))
        rows.append(
            {
                "index": name,
                "days": len(v),
                "frozen_days": int(frozen.sum()),
                "frozen_share_weighted": np.average(frozen, weights=w),
                "auc_raw": roc_auc_score(frozen, sign * v[name], sample_weight=w),
                "auc_standardized": roc_auc_score(frozen, sign * v[f"z_{name}"], sample_weight=w),
            }
        )
    out = pd.DataFrame(rows).round(3)
    out.to_csv(paths.TABLES / "standardization_separability.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
