"""The freeze-thaw model: a three-class logistic regression on Z_NPR and Z_NDV with a tree-cover
correction.

For the transitional and the frozen class k, against the thawed class:

    log(P_k / P_thawed) = b0k + b1k Z_NPR + b2k Z_NDV + b3k T + b4k T Z_NPR + b5k T Z_NDV

T is the tree-cover fraction of the cell. The model is fitted without a penalty
(training.make_classifier) on the thawed, clean partial (transitional) and frozen days of the
development cells, with weights that give every spatial block the same total weight and every class
the same total weight (model_weights). The state of a day is its most probable class.

fit_logistic and probability are the two-class version, used in the feature-selection screen.
Works on pandas objects (training, evaluation) and on xarray objects (maps) alike.
"""

import numpy as np
import pandas as pd

TERMS = ["z_npr", "z_ndv", "tree", "tree_x_npr", "tree_x_ndv"]
CLASSES = ["thawed", "transitional", "frozen"]  # model classes; transitional = in situ partial
LABEL_CLASS = {"thawed": 0, "partial": 1, "frozen": 2}  # in situ label -> class index
STATE_CODES = {"thawed": 0, "frozen": 1, "transitional": 2}  # coding of the product's state


def design(z_npr, z_ndv, tree):
    """The model terms from the two standardized indices and the tree-cover fraction."""
    return {
        "z_npr": z_npr,
        "z_ndv": z_ndv,
        "tree": tree,
        "tree_x_npr": tree * z_npr,
        "tree_x_ndv": tree * z_ndv,
    }


def design_frame(df, indices=("z_npr", "z_ndv")):
    """Model terms of a table: the standardized ``indices``, tree cover, and tree cover times each
    index. The default is the model's TERMS; other index sets serve the feature-selection check."""
    if tuple(indices) == ("z_npr", "z_ndv"):
        return pd.DataFrame(design(df["z_npr"], df["z_ndv"], df["tree"]), index=df.index)[TERMS]
    terms = {i: df[i] for i in indices} | {"tree": df["tree"]}
    terms |= {f"tree_x_{i.removeprefix('z_')}": df["tree"] * df[i] for i in indices}
    return pd.DataFrame(terms, index=df.index)


def block_weights(blocks):
    """Sample weights giving every spatial block the same total weight (mean weight 1)."""
    blocks = pd.Series(np.asarray(blocks))
    n = blocks.map(blocks.value_counts())
    return (len(blocks) / (blocks.nunique() * n)).to_numpy()


def model_weights(blocks, y):
    """Block weights (block_weights), rescaled so that every class has the same total weight."""
    w = block_weights(blocks)
    y = np.asarray(y)
    total = pd.Series(w).groupby(y).sum()
    return w * (w.sum() / (len(total) * total.loc[y].to_numpy()))


def fit_logistic(X, y, weights=None):
    """Coefficients (intercept first) of the paper's logistic regression, in the units of X.

    The columns are scaled for the optimizer only; without a penalty the fit does not depend on
    the scaling, and the coefficients are returned in the original units.
    """
    from nbft.training import make_classifier

    mu, sd = X.mean(), X.std().replace(0, 1.0)
    clf = make_classifier().fit((X - mu) / sd, y, sample_weight=weights)
    beta = pd.Series(clf.coef_[0], index=X.columns) / sd
    return pd.concat([pd.Series({"intercept": clf.intercept_[0] - (beta * mu).sum()}), beta])


def log_odds(coef, terms):
    """b0 + sum of b_k x_k over the terms present in ``coef`` (``terms`` maps name to values)."""
    out = coef["intercept"]
    for name, b in coef.drop("intercept").items():
        out = out + b * terms[name]
    return out


def probability(coef, terms):
    return 1.0 / (1.0 + np.exp(-log_odds(coef, terms)))


def fit_multinomial(X, y, weights=None):
    """Coefficients of the three-class model, one row per class of CLASSES (thawed row zero), with
    the intercept first and the terms in the units of X. ``y`` holds class indices."""
    from nbft.training import make_classifier

    mu, sd = X.mean(), X.std().replace(0, 1.0)
    clf = make_classifier().fit((X - mu) / sd, y, sample_weight=weights)
    beta = pd.DataFrame(clf.coef_, columns=X.columns) / sd
    coef = pd.concat([clf.intercept_ - beta @ mu, beta], axis=1)
    coef.columns = ["intercept", *X.columns]
    coef.index = [CLASSES[c] for c in clf.classes_]
    return coef - coef.loc["thawed"]


def class_probabilities(coef, terms):
    """P of each class (dict keyed by CLASSES) from the coefficients of fit_multinomial."""
    score = {k: log_odds(coef.loc[k], terms) for k in CLASSES if k != "thawed"}
    total = 1.0 + sum(np.exp(v) for v in score.values())
    out = {"thawed": 1.0 / total}
    out.update({k: np.exp(v) / total for k, v in score.items()})
    return out


def most_probable(probs):
    """State code (STATE_CODES) of the most probable class; NaN where the probabilities are NaN."""
    stack = np.stack([np.asarray(probs[k], dtype=float) for k in CLASSES])
    best = np.argmax(np.nan_to_num(stack, nan=-1.0), axis=0)
    codes = np.array([STATE_CODES[k] for k in CLASSES], dtype=float)[best]
    return np.where(np.isnan(stack).any(axis=0), np.nan, codes)
