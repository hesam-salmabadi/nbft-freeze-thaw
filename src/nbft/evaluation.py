"""Spatial cross-validation and scores of the three-class freeze-thaw model (nbft.model)."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, cohen_kappa_score

from nbft import insitu
from nbft.model import (
    CLASSES,
    LABEL_CLASS,
    STATE_CODES,
    class_probabilities,
    design_frame,
    fit_multinomial,
    model_weights,
    most_probable,
)

N_FOLDS = 5
CODE_CLASS = {code: k for k, code in STATE_CODES.items()}


def cross_validate(x, train_rows=None, n_folds=N_FOLDS, indices=("z_npr", "z_ndv")):
    """Held-out class probabilities and state of every row of ``x`` (the clean days of the
    training table), with folds of whole spatial blocks stratified by label.

    ``train_rows`` (boolean Series on x.index) restricts the days a fold may train on, e.g. to one
    season; every held-out row is still predicted. ``indices`` sets the standardized indices of the
    model (model.design_frame). Returns (predictions, coefficients per fold).
    """
    y = x["label"].map(LABEL_CLASS)
    terms = design_frame(x, indices)
    pred = pd.DataFrame(np.nan, index=x.index, columns=[f"p_{k}" for k in CLASSES] + ["state"])
    pred["fold"] = -1
    coefs = []
    for k, (fit_idx, hold) in enumerate(insitu.block_folds(y, x["block"], n_folds)):
        tr = x.index[fit_idx]
        if train_rows is not None:
            tr = tr[train_rows.loc[tr].to_numpy()]
        coef = fit_multinomial(terms.loc[tr], y.loc[tr], model_weights(x.loc[tr, "block"], y[tr]))
        idx = x.index[hold]
        probs = class_probabilities(coef, terms.loc[idx])
        for c in CLASSES:
            pred.loc[idx, f"p_{c}"] = np.asarray(probs[c])
        pred.loc[idx, "state"] = most_probable(probs)
        pred.loc[idx, "fold"] = k
        coefs.append(coef.assign(fold=k, train_days=len(tr)))
    return pred, pd.concat(coefs).rename_axis("class").reset_index()


def scores(label, state, p_frozen=None, weights=None):
    """Three-class scores of predicted states against in situ labels (clean days).

    Recall, precision and F1 of each class, their mean F1 (macro F1) and Cohen's kappa; with
    ``p_frozen``, also the PR AUC of P(frozen) on the frozen and thawed days alone. ``weights``
    (one per day, e.g. model.block_weights) weight every count; by default each day counts once.
    """
    truth = label.map({"thawed": "thawed", "partial": "transitional", "frozen": "frozen"})
    called = pd.Series(state, index=label.index).map(CODE_CLASS)
    w = pd.Series(1.0 if weights is None else np.asarray(weights, dtype=float), index=label.index)
    out = {"days": len(label)}
    f1s = []
    for c in CLASSES:
        tp = w[(truth == c) & (called == c)].sum()
        recall = tp / max(w[truth == c].sum(), 1e-12)
        precision = tp / max(w[called == c].sum(), 1e-12)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1s.append(f1)
        out |= {f"share_{c}": w[truth == c].sum() / w.sum(), f"recall_{c}": recall}
        out |= {f"precision_{c}": precision, f"f1_{c}": f1}
    out["macro_f1"] = float(np.mean(f1s))
    out["kappa"] = cohen_kappa_score(truth, called, sample_weight=w)
    if p_frozen is not None:
        binary = truth.isin(["frozen", "thawed"])
        y = truth[binary] == "frozen"
        out["pr_auc_frozen"] = (
            average_precision_score(y, p_frozen[binary], sample_weight=w[binary])
            if y.any()
            else np.nan
        )
    return out


def weighted_quantile(values, weights, q):
    """Quantile q of the finite values, each counted with its weight."""
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    ok = np.isfinite(values)
    v, w = values[ok], weights[ok]
    if len(v) == 0:
        return np.nan
    order = np.argsort(v)
    cum = np.cumsum(w[order]) / w.sum()
    return float(v[order][np.searchsorted(cum, q)])


def weighted_spearman(a, b, weights):
    """Spearman rank correlation of a and b with every observation counted with its weight: the
    weighted Pearson correlation of their ranks."""
    ra = pd.Series(np.asarray(a)).rank().to_numpy()
    rb = pd.Series(np.asarray(b)).rank().to_numpy()
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    ma, mb = np.sum(w * ra), np.sum(w * rb)
    cov = np.sum(w * (ra - ma) * (rb - mb))
    return cov / np.sqrt(np.sum(w * (ra - ma) ** 2) * np.sum(w * (rb - mb) ** 2))
