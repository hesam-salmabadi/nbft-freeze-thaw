"""Two-state hidden Markov model baseline, after Walker et al. (2022).

Walker, V. A., Colliander, A., & Kimball, J. S. (2022). Satellite retrievals of probabilistic
freeze-thaw conditions from SMAP and AMSR brightness temperatures. IEEE TGRS 60, 4411211.
doi:10.1109/TGRS.2022.3174807

Their method, as described in Section II: for each grid cell independently, a hidden Markov model
with two hidden states (frozen, thawed) is fitted to the paired sequence of L-band NPR and Ka-band
36.5 GHz V-pol brightness temperature. Emissions are Gaussian mixtures with two subpopulations,
parameters are learned unsupervised with Baum-Welch, and the output is the posterior probability of
the frozen state. Training is restricted to the shoulder seasons so the model learns transitions
rather than settling into one state.

As in their study, missing days are filled before fitting (here by linear interpolation in time),
so the model sees a continuous sequence, and ascending and descending passes are treated
separately (this study uses the morning pass only). Their domain is the land north of 45 deg N where
seasonal freezing occurs.

Choices their paper does not state:
- features are standardized per pixel before fitting, because NPR (about 0.01-0.04) and TB (about
  200-280 K) differ by four orders of magnitude, which makes a shared-covariance mixture
  numerically unstable. Standardizing each feature is affine and monotone, so it does not change
  which observations are assigned to which state.
- an unsupervised model has no inherent state ordering, so the frozen state is the one with the
  lower mean of both features (frozen soil lowers NPR and the 36.5 GHz TB); a pixel whose two
  features disagree is flagged ambiguous (frozen_state_agrees).
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM

SHOULDER_MONTHS = (2, 3, 4, 5, 9, 10, 11, 12)  # F-M-A-M and S-O-N-D
N_STATES = 2
N_MIX = 2
SEED = 42
MIN_TRAIN_OBSERVATIONS = 200
# the paper does not state the covariance structure of its mixtures; diagonal is hmmlearn's default
COVARIANCE_TYPE = "diag"


def standardize_columns(x):
    """Z-score each feature over its own series, ignoring NaN."""
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (x - mean) / std


def contiguous_lengths(mask):
    """Lengths of each run of True in ``mask``, for hmmlearn's ``lengths`` argument.

    Shoulder seasons are separated by summer and winter, so the selected rows form blocks. Passing
    their lengths keeps the model from learning a transition across the gap between two blocks.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    return [len(part) for part in np.split(idx, breaks)]


def fit_pixel_hmm(features, months, seed=SEED, covariance_type=COVARIANCE_TYPE, standardize=True):
    """Fit the two-state model on the shoulder-season rows of one pixel.

    ``features`` is (time, 2) as [NPR, TB36V]; rows containing NaN are dropped. Each contiguous run
    of shoulder-season rows is passed as its own sequence. Returns the fitted model, or None when
    the pixel has too few usable shoulder-season observations.
    """
    usable = np.isfinite(features).all(axis=1)
    shoulder = usable & np.isin(months, SHOULDER_MONTHS)
    if shoulder.sum() < MIN_TRAIN_OBSERVATIONS:
        return None

    model = GMMHMM(
        n_components=N_STATES,
        n_mix=N_MIX,
        covariance_type=covariance_type,
        n_iter=100,
        random_state=seed,
    )
    prepared = standardize_columns(features) if standardize else features
    training = prepared[shoulder]
    lengths = contiguous_lengths(shoulder)
    try:
        model.fit(training, lengths=lengths)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return model if _parameters_are_finite(model) else None


def _parameters_are_finite(model):
    """Some pixels (near-constant series) fit to NaN parameters; those models are unusable."""
    return all(
        np.isfinite(np.asarray(p)).all()
        for p in (model.startprob_, model.transmat_, model.means_, model.weights_)
    )


def state_means(model):
    """Mean of each (standardized) feature per hidden state, averaged over its mixture components
    with their weights: array (state, feature)."""
    return np.array(
        [(model.weights_[s][:, None] * model.means_[s]).sum(axis=0) for s in range(N_STATES)]
    )


def frozen_state(model):
    """Index of the hidden state representing frozen soil: the lower sum of the standardized means
    of NPR and TB36V."""
    return int(np.argmin(state_means(model).sum(axis=1)))


def frozen_state_agrees(model):
    """True when NPR and TB36V both point to the same frozen state."""
    m = state_means(model)
    return int(np.argmin(m[:, 0])) == int(np.argmin(m[:, 1]))


def posterior_frozen(model, features, standardize=True):
    """Posterior probability of the frozen state for every time step (NaN where inputs are NaN)."""
    out = np.full(len(features), np.nan)
    usable = np.isfinite(features).all(axis=1)
    if not usable.any():
        return out
    prepared = standardize_columns(features) if standardize else features
    try:
        posteriors = model.predict_proba(prepared[usable])
    except ValueError:
        return out
    out[usable] = posteriors[:, frozen_state(model)]
    return out


def posteriors_by_pixel(df, covariance_type=COVARIANCE_TYPE, standardize=True, verbose=True):
    """Fit one model per pixel in ``df`` and return it with an ``hmm_prob`` column.

    ``df`` needs columns pixel, time, npr, tb36v (gap-filled). Pixels that cannot be fitted keep
    NaN; hmm_agrees is 1 where NPR and TB36V point to the same frozen state.
    """
    df = df.sort_values(["pixel", "time"]).copy()
    df["hmm_prob"] = np.nan
    df["hmm_agrees"] = np.nan
    fitted = skipped = 0
    for _pixel, group in df.groupby("pixel"):
        features = group[["npr", "tb36v"]].to_numpy()
        months = np.asarray(pd.DatetimeIndex(group["time"]).month)
        model = fit_pixel_hmm(
            features, months, covariance_type=covariance_type, standardize=standardize
        )
        if model is None:
            skipped += 1
            continue
        df.loc[group.index, "hmm_prob"] = posterior_frozen(model, features, standardize=standardize)
        df.loc[group.index, "hmm_agrees"] = float(frozen_state_agrees(model))
        fitted += 1
    if verbose:
        print(f"fitted {fitted} pixels, skipped {skipped}")
    return df, fitted, skipped
