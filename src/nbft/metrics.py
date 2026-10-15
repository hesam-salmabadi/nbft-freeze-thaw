"""Decision-threshold selection and classification agreement."""

import numpy as np
from sklearn.metrics import roc_curve


def equal_error_rate_threshold(y_true, y_prob, sample_weight=None):
    """Threshold where sensitivity equals specificity (TPR == TNR).

    Returns (threshold, fpr, tpr) at that point.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, sample_weight=sample_weight)
    best = np.argmin(np.abs(tpr - (1 - fpr)))
    return thresholds[best], fpr[best], tpr[best]


def youden_threshold(y_true, y_prob, sample_weight=None):
    """Threshold maximizing Youden's J (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, sample_weight=sample_weight)
    best = np.argmax(tpr - fpr)
    return thresholds[best], fpr[best], tpr[best]


def agreement_proportion(df, column, label, threshold):
    """Ap: percentage of retrievals whose hard classification matches a binary reference label.

    Rows with a missing prediction or a non-binary label are dropped, so the returned count is the
    denominator of the percentage.
    """
    scored = df.dropna(subset=[column, label])
    scored = scored[scored[label].isin([0, 1])]
    if scored.empty:
        return float("nan"), 0
    predicted = (scored[column] >= threshold).astype(int)
    return float((predicted == scored[label]).mean() * 100), len(scored)
