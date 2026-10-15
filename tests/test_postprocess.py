import numpy as np

from nbft.postprocess import no_thaw_in_winter


def test_no_thaw_only_in_winter():
    state = np.array([[2, 0, 1, 0, 0]])
    winter = np.array([[True, True, True, False, True]])
    assert no_thaw_in_winter(state, winter).tolist() == [[2, 2, 2, 0, 0]]


def test_missing_days_stay_missing_and_do_not_break_chain():
    state = np.array([[2, np.nan, 0]])
    winter = np.array([[True, True, True]])
    out = no_thaw_in_winter(state, winter)
    assert np.isnan(out[0, 1])
    assert out[0, 2] == 2
