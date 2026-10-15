import numpy as np
import pandas as pd

from nbft.hmm import fit_pixel_hmm, frozen_state, posterior_frozen


def synthetic_pixel(n_years=4, seed=0):
    """Frozen October-April with low NPR and low TB, thawed otherwise."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2015-01-01", periods=365 * n_years, freq="D")
    frozen = np.isin(np.asarray(time.month), [10, 11, 12, 1, 2, 3, 4])
    npr = np.where(frozen, 0.015, 0.035) + rng.normal(0, 0.002, len(time))
    tb = np.where(frozen, 225.0, 270.0) + rng.normal(0, 3.0, len(time))
    return np.column_stack([npr, tb]), np.asarray(time.month), np.asarray(frozen)


def test_hmm_recovers_the_frozen_season():
    features, months, frozen = synthetic_pixel()
    model = fit_pixel_hmm(features, months)
    assert model is not None
    probability = posterior_frozen(model, features)
    # the frozen state should be the one with the lower mean NPR, so probability tracks the season
    assert probability[frozen].mean() > 0.8
    assert probability[~frozen].mean() < 0.2


def test_frozen_state_picks_the_lower_npr_state():
    features, months, _ = synthetic_pixel(seed=1)
    model = fit_pixel_hmm(features, months)
    state = frozen_state(model)
    npr_means = [model.means_[s].mean(axis=0)[0] for s in range(2)]
    assert npr_means[state] == min(npr_means)


def test_returns_none_when_too_little_data():
    features, months, _ = synthetic_pixel()
    assert fit_pixel_hmm(features[:50], months[:50]) is None


def test_missing_inputs_stay_missing():
    features, months, _ = synthetic_pixel(seed=2)
    model = fit_pixel_hmm(features, months)
    features[10] = np.nan
    assert np.isnan(posterior_frozen(model, features)[10])


def test_degenerate_pixel_is_rejected():
    """A near-constant series fits to unusable (NaN) parameters; it must not be returned."""
    features = np.column_stack([np.full(2000, 0.02), np.full(2000, 250.0)])
    months = np.tile(np.arange(1, 13), 200)[:2000]
    assert fit_pixel_hmm(features, months) is None
