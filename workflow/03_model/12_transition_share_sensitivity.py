"""Sensitivity of the model to the minimum transition share of the transitional training days
(Supplementary Material).

The transitional class is trained on the partial days whose mean probability of transition over
the morning window is at least insitu.MIN_TRANSITION_SHARE (0.5). For each threshold in
THRESHOLDS, the model is cross-validated (evaluation.cross_validate) on all labelled days with the
same spatial-block folds, training only on the thawed, frozen, and partial days at or above the
threshold. Every threshold is scored on the same days, the clean days at 0.5 (as in the main
text), and also on its own clean days, with every spatial block weighted equally within the set
scored (model.block_weights).

Output: results/tables/transition_share_sensitivity.csv.

Usage: python 12_transition_share_sensitivity.py
"""

import pandas as pd

from nbft import config, insitu, paths
from nbft.evaluation import cross_validate, scores
from nbft.model import block_weights

THRESHOLDS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
COLS = ["macro_f1", "kappa", "recall_thawed", "recall_transitional", "recall_frozen"]
COLS += ["precision_transitional"]


def clean(x, threshold):
    return (x["label"] != "partial") | (x["transition_share"] >= threshold)


def score(x, pred, rows):
    s = scores(
        x.loc[rows, "label"], pred.loc[rows, "state"], weights=block_weights(x.loc[rows, "block"])
    )
    return {k: s[k] for k in ["days", *COLS]}


def main():
    x = pd.read_parquet(config.path("model.training_table")).reset_index(drop=True)
    common = clean(x, insitu.MIN_TRANSITION_SHARE)
    out = []
    for t in THRESHOLDS:
        train = clean(x, t)
        pred, _ = cross_validate(x, train_rows=train)
        partial = (x["label"] == "partial") & train
        base = {"threshold": t, "transitional_training_days": int(partial.sum())}
        out.append(base | {"scored_on": "clean days at 0.5"} | score(x, pred, common))
        out.append(base | {"scored_on": "own clean days"} | score(x, pred, train))
        print(f"threshold {t}: done", flush=True)
    table = pd.DataFrame(out).round(3)
    table.to_csv(paths.TABLES / "transition_share_sensitivity.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
