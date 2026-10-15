"""Spatially independent cross-validation of the three-class freeze-thaw model.

Data: the clean days of the training table (thawed, frozen, and partial days with at least
insitu.MIN_TRANSITION_SHARE of the day in transition). Folds hold out whole spatial blocks (about
200 km), stratified so that each fold holds similar shares of the three labels
(evaluation.cross_validate). The state of a day is its most probable class.

Scores overall and per tree-cover class, roughness class and biome (RESOLVE; biomes with fewer
than MIN_BIOME_CELLS cells pooled as 'other'): recall, precision and F1 of each class, macro F1,
Cohen's kappa, and PR AUC of P(frozen) on the frozen and thawed days. Every count is weighted so
that each spatial block carries the same total weight within the scored set (model.block_weights),
as in training, because the stations cluster in a few regions (mountain cells hold about half of
the days); the unweighted scores go to the supplement.

Outputs: model.cv_predictions, results/tables/cv_metrics.csv (block-weighted),
cv_metrics_unweighted.csv, cv_confusion.csv (block-weighted shares of each label),
cv_coefficients.csv (each fold).

Usage: python 05_evaluate_cv.py
"""

import pandas as pd

from nbft import config, paths
from nbft.evaluation import CODE_CLASS, cross_validate, scores
from nbft.model import block_weights

MIN_BIOME_CELLS = 10


def group_scores(out, weighted):
    def score(g):
        w = block_weights(g["block"]) if weighted else None
        return scores(g["label"], g["state"], g["p_frozen"], weights=w)

    rows = [{"group": "all", "class": "all", **score(out)}]
    for col in ("tree_class", "roughness_class", "biome_group"):
        for k, g in out.groupby(col):
            rows.append({"group": col, "class": k, **score(g)})
    return pd.DataFrame(rows).round(3)


def main():
    x = pd.read_parquet(config.path("model.training_table"))
    x = x[x["clean"]].reset_index(drop=True)
    pred, coefs = cross_validate(x)
    keep = ["cell_id", "date", "label", "tree", "tree_class", "roughness_class", "biome"]
    out = x[keep + ["latitude", "block"]].join(pred)
    out.to_parquet(config.path("model.cv_predictions"), index=False)

    cells = out.groupby("biome")["cell_id"].nunique()
    out["biome_group"] = out["biome"].where(out["biome"].map(cells) >= MIN_BIOME_CELLS, "other")
    unweighted = group_scores(out, weighted=False)
    unweighted.to_csv(paths.TABLES / "cv_metrics_unweighted.csv", index=False)
    metrics = group_scores(out, weighted=True)
    metrics.to_csv(paths.TABLES / "cv_metrics.csv", index=False)
    w = block_weights(out["block"])
    confusion = pd.crosstab(
        out["label"], out["state"].map(CODE_CLASS), values=w, aggfunc="sum", normalize="index"
    )
    confusion.round(3).to_csv(paths.TABLES / "cv_confusion.csv")
    coefs.round(4).to_csv(paths.TABLES / "cv_coefficients.csv", index=False)

    cols = ["group", "class", "days", "macro_f1", "kappa", "pr_auc_frozen"]
    cols += [f"recall_{c}" for c in ("thawed", "transitional", "frozen")]
    print(coefs.round(3).to_string(index=False), "", metrics[cols].to_string(index=False), sep="\n")
    print(confusion.round(2).to_string())


if __name__ == "__main__":
    main()
