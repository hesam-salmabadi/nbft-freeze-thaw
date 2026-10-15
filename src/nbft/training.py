"""Model fitting shared by cross-validation, feature selection and production training."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler

from nbft.constants import (
    MODEL_FEATURES,
    NUM_FOLDS,
    TARGET_LCS,
    TRANSITION_BAND_SCALE,
    TRANSITION_PROBS,
)
from nbft.metrics import equal_error_rate_threshold
from nbft.training_data import site_weights

SEED = 42
MIN_MONTHS_PER_BLOCK = 6


def make_classifier():
    """Logistic regression without a penalty, used throughout the paper.

    Spatial blocks are balanced by sample weights (model.block_weights). With
    a few coefficients and several hundred thousand training days, an L1 or L2 penalty has no
    measurable effect (identical held-out PR AUC), so none is used.
    """
    return LogisticRegression(penalty=None, max_iter=1000, random_state=SEED)


def fit_global_scaler(df, features=MODEL_FEATURES):
    """StandardScaler fit on a block-balanced sample.

    Only spatial blocks covering at least six calendar months are used; each is resampled
    (with replacement if needed) to the median block size.
    """
    months = df["time"].dt.month.groupby(df["spatial_block"]).nunique()
    df = df[df["spatial_block"].isin(months[months >= MIN_MONTHS_PER_BLOCK].index)]
    n = int(df["spatial_block"].value_counts().median())
    balanced = df.groupby("spatial_block", group_keys=False).apply(
        lambda g: g.sample(n=n, replace=len(g) < n, random_state=SEED)
    )
    return StandardScaler().fit(balanced[features])


def logit(p):
    return np.log(p / (1 - p))


def inverse_logit(z):
    return 1 / (1 + np.exp(-z))


def transition_bounds(coef, threshold, scale=TRANSITION_BAND_SCALE, probs=TRANSITION_PROBS):
    """Bounds of the transitional band, as distances from the decision boundary.

    ``probs`` are the probability isolines bounding the band and ``scale`` narrows the distance to
    them. Returns (lower, upper) in the same units as D.
    """
    norm = np.linalg.norm(coef)
    logit_t = logit(threshold)
    return tuple((logit(p) - logit_t) / norm * scale for p in probs)


def implied_probabilities(coef, threshold, bounds):
    """The probabilities the band bounds actually sit at, after scaling.

    Scaling a distance shrinks the logit gap, not the probability gap, so a band described as
    10%-90% at scale 0.5 really runs from about 27% to 77%.
    """
    norm = np.linalg.norm(coef)
    logit_t = logit(threshold)
    return tuple(inverse_logit(logit_t + d * norm) for d in bounds)


def cross_validate(splits, features=MODEL_FEATURES):
    """Fit and evaluate one model per biome and fold.

    Each fold fits its own scaler on its own training rows, so nothing about the held-out stations
    reaches the model. The threshold of each fold is the equal-error-rate point on its training
    data. Test metrics are reported both with block/class-balanced weights (performance under a
    balanced design) and unweighted (performance at the sample's own prevalence).

    Returns (fold_results, out_of_fold_predictions).
    """
    rows, preds = [], []
    for lc in TARGET_LCS:
        for i, fold in enumerate(splits.get(lc, [])[:NUM_FOLDS]):
            scaler = fit_global_scaler(fold["X_train"], features)
            X_tr = scaler.transform(fold["X_train"][features])
            X_te = scaler.transform(fold["X_test"][features])
            y_tr, y_te = fold["y_train"], fold["y_test"]
            w_tr = fold["sample_weights"]
            w_te = site_weights(fold["spatial_block_test"], y_te)

            clf = make_classifier().fit(X_tr, y_tr, sample_weight=w_tr)
            thresh, _, _ = equal_error_rate_threshold(
                y_tr, clf.predict_proba(X_tr)[:, 1], sample_weight=w_tr
            )
            p_te = clf.predict_proba(X_te)[:, 1]
            yhat = (p_te >= thresh).astype(int)
            d_lower, d_upper = transition_bounds(clf.coef_[0], thresh)

            rows.append(
                {
                    "LC": lc,
                    "Fold": i,
                    "Is_Padding": fold.get("is_padding", False),
                    "Fold_Thresh": thresh,
                    "Intercept": clf.intercept_[0],
                    **{f"Coef_{f}": c for f, c in zip(features, clf.coef_[0], strict=True)},
                    "D_Lower": d_lower,
                    "D_Upper": d_upper,
                    "D_Width": d_upper - d_lower,
                    "F1": f1_score(y_te, yhat, sample_weight=w_te),
                    "PR_AUC": average_precision_score(y_te, p_te, sample_weight=w_te),
                    "F1_unweighted": f1_score(y_te, yhat),
                    "PR_AUC_unweighted": average_precision_score(y_te, p_te),
                    "Frozen_Rate": float(y_te.mean()),
                }
            )
            preds.append(
                pd.DataFrame(
                    {
                        "LC": lc,
                        "Fold": i,
                        "Is_Padding": fold.get("is_padding", False),
                        "y_true": y_te.to_numpy(),
                        "prob": p_te,
                        "y_pred": yhat,
                        "weight": w_te,
                    }
                )
            )
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
