"""Single-index check: the three-class model on Z_NPR alone, on Z_NDV alone, and on both
(Supplementary Material).

Same days, folds and weights as the cross-validation of the paper (evaluation.cross_validate on the
clean days of the development cells); each index set keeps the tree-cover terms
(model.design_frame).
Scores of the held-out states with every spatial block weighted equally (model.block_weights):
recall, precision and F1 of each class, macro F1 and kappa, overall and per tree-cover class.

Output: results/tables/single_index.csv.

Usage: python 13_single_index.py
"""

import pandas as pd

from nbft import config, paths
from nbft.evaluation import cross_validate, scores
from nbft.model import block_weights

SETS = {"NPR only": ["z_npr"], "NDV only": ["z_ndv"], "NPR and NDV": ["z_npr", "z_ndv"]}
CLASSES = ("thawed", "transitional", "frozen")


def main():
    x = pd.read_parquet(config.path("model.training_table"))
    x = x[x["clean"]].reset_index(drop=True)
    rows = []
    for name, indices in SETS.items():
        state = cross_validate(x, indices=indices)[0]["state"]
        for group, g in [("all", x), *x.groupby("tree_class")]:
            s = scores(g["label"], state[g.index], weights=block_weights(g["block"]))
            row = {"indices": name, "tree_class": group, "days": len(g)}
            row |= {k: s[k] for k in ("macro_f1", "kappa")}
            for c in CLASSES:
                row |= {f"{m}_{c}": s[f"{m}_{c}"] for m in ("recall", "precision", "f1")}
            rows.append(row)
        print(f"{name}: done", flush=True)
    table = pd.DataFrame(rows).round(3)
    table.to_csv(paths.TABLES / "single_index.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
