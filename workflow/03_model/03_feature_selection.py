"""Feature selection: correlation screen of the standardized indices, with two checks.

Data: the labelled AM days of the development cells outside July and August
(insitu.development_days): frozen and thawed days for the screen and the winter check, and the
clean days of all three classes for the comparison.

1. Screen: indices ranked by how well each alone separates frozen from thawed days (ROC AUC); each
   is kept unless its Spearman correlation with an index already kept exceeds MAX_CORRELATION
   (Dormann et al., 2013). Tables feature_correlation.csv and feature_screen.csv.
2. Winter check: each index with its own best cut-off (Youden), the share of thawed days it calls
   frozen in December-February and in the other months. Table feature_winter_check.csv.
All statistics weight every day so that each spatial block carries the same total weight
(model.block_weights), as in training and evaluation.
3. Comparison: the three-class model (nbft.model) on the kept indices against the same model on all
   four, by the macro F1 and kappa of their held-out states (evaluation.cross_validate, the
   cross-validation of the paper), with every spatial block weighted equally (model.block_weights),
   overall and per tree-cover class. Table feature_comparison.csv.
4. Fold check: the screen repeated on the training blocks of each cross-validation fold, to show
   that the selection does not depend on the held-out blocks. Table feature_screen_folds.csv.

Usage: python 03_feature_selection.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from nbft import insitu, paths
from nbft.evaluation import cross_validate, scores, weighted_spearman
from nbft.model import LABEL_CLASS, block_weights

MAX_CORRELATION = 0.7
WINTER_MONTHS = (12, 1, 2)
INDICES = list(insitu.INDEX_SIGNS)  # npr, ndv, ndh, wd


def load():
    x = insitu.development_days()
    return x[x["label"] != "partial"].reset_index(drop=True), x[x["clean"]].reset_index(drop=True)


def oriented(x, name):
    """The index with frozen days high."""
    return insitu.INDEX_SIGNS[name] * x[f"z_{name}"]


def screen(x):
    w = block_weights(x["block"])
    z = [f"z_{n}" for n in INDICES]
    corr = pd.DataFrame(
        [[weighted_spearman(x[a], x[b], w) for b in z] for a in z], index=INDICES, columns=INDICES
    )
    auc = {n: roc_auc_score(x["frozen"], oriented(x, n), sample_weight=w) for n in INDICES}
    kept, rows = [], []
    for n in sorted(INDICES, key=auc.get, reverse=True):
        clash = [k for k in kept if abs(corr.loc[n, k]) > MAX_CORRELATION]
        if not clash:
            kept.append(n)
        rows.append(
            {"index": n, "auc": auc[n], "kept": not clash, "correlated_with": ",".join(clash)}
        )
    return corr, pd.DataFrame(rows), kept


def winter_check(x):
    w = block_weights(x["block"])
    winter = x["date"].dt.month.isin(WINTER_MONTHS).to_numpy()
    frozen = x["frozen"].to_numpy() == 1
    rows = []
    for n in INDICES:
        s = oriented(x, n).to_numpy()
        fpr, tpr, cut = roc_curve(frozen, s, sample_weight=w)
        called = s >= cut[np.argmax(tpr - fpr)]

        def share(mask, called=called):
            return np.average(called[mask], weights=w[mask])

        rows.append(
            {
                "index": n,
                "thawed_winter_called_frozen": share(~frozen & winter),
                "thawed_other_called_frozen": share(~frozen & ~winter),
                "frozen_detected": share(frozen),
            }
        )
    share = np.average(~frozen[winter], weights=w[winter])
    print(f"thawed share of December-February days: {share:.2f}")
    return pd.DataFrame(rows)


def comparison(x, kept):
    sets = {"kept": [f"z_{n}" for n in kept], "all_four": [f"z_{n}" for n in INDICES]}
    states = {name: cross_validate(x, indices=f)[0]["state"] for name, f in sets.items()}
    rows = []
    for group, g in [("all", x), *x.groupby("tree_class")]:
        row = {"tree_class": group, "days": len(g)}
        for name, state in states.items():
            sc = scores(g["label"], state.loc[g.index], weights=block_weights(g["block"]))
            row |= {f"macro_f1_{name}": sc["macro_f1"], f"kappa_{name}": sc["kappa"]}
        rows.append(row)
    out = pd.DataFrame(rows)
    out["difference_macro_f1"] = out["macro_f1_all_four"] - out["macro_f1_kept"]
    return out


def fold_screens(clean):
    y = clean["label"].map(LABEL_CLASS)
    rows = []
    for k, (fit_idx, _) in enumerate(insitu.block_folds(y, clean["block"])):
        tr = clean.iloc[fit_idx]
        _, table, kept = screen(tr[tr["label"] != "partial"].reset_index(drop=True))
        rows.append({"fold": k, "kept": "+".join(kept), "auc_order": "+".join(table["index"])})
    return pd.DataFrame(rows)


def main():
    x, clean = load()
    frozen = np.average(x["frozen"], weights=block_weights(x["block"]))
    print(f"{len(x):,} days at {x['cell_id'].nunique()} cells, {frozen:.3f} frozen (weighted)")
    corr, table, kept = screen(x)
    corr.round(2).to_csv(paths.TABLES / "feature_correlation.csv")
    table.round(3).to_csv(paths.TABLES / "feature_screen.csv", index=False)
    print(corr.round(2).to_string(), table.round(3).to_string(index=False), sep="\n\n")
    winter = winter_check(x).round(3)
    winter.to_csv(paths.TABLES / "feature_winter_check.csv", index=False)
    print(winter.to_string(index=False))
    folds = fold_screens(clean)
    folds.to_csv(paths.TABLES / "feature_screen_folds.csv", index=False)
    print(folds.to_string(index=False))
    comp = comparison(clean, kept).round(3)
    comp.to_csv(paths.TABLES / "feature_comparison.csv", index=False)
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
