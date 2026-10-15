"""Timing of the freeze season from a daily series of states (0 thawed, 1 frozen, 2 transitional).

Following the day of first freezing of Rautiainen et al. (2025), an event is the start or end of a
run of at least RUN consecutive observations in a state; missing mornings are skipped, so a run may
span gaps. Per freeze season (1 August to 31 July), in days since 1 August:

- first_transition: first morning of the first run of transitional or frozen mornings (autumn onset
  of the transition);
- first_frozen: first morning of the first run of frozen mornings (day of first freezing);
- last_frozen: last morning of the last run of frozen mornings;
- last_transition: last morning of the last run of transitional or frozen mornings (thaw end).

The autumn transition lasts from first_transition to first_frozen, the spring transition from
last_frozen to last_transition. For FT-SMOS, its partially frozen state counts as transitional.

RunTracker processes one morning at a time, for arrays of any shape (a map tile, a set of cells).
"""

import numpy as np
import pandas as pd

RUN = 5
EVENTS = ["first_transition", "first_frozen", "last_frozen", "last_transition"]
MIN_SEASONS = 4  # seasons with a value needed for a multi-season mean (maps)


class _Runs:
    """First start and last end of runs of at least RUN consecutive valid True observations."""

    def __init__(self, shape):
        self.count = np.zeros(shape, int)
        self.start = np.full(shape, np.nan)
        self.first = np.full(shape, np.nan)
        self.last = np.full(shape, np.nan)

    def update(self, t, cond, valid):
        hit = cond & valid
        self.start = np.where(hit & (self.count == 0), t, self.start)
        self.count = np.where(hit, self.count + 1, np.where(valid, 0, self.count))
        done = hit & (self.count >= RUN)
        self.first = np.where(done & np.isnan(self.first), self.start, self.first)
        self.last = np.where(done, t, self.last)


class RunTracker:
    """Freeze-season events of one season, fed one morning (day index t) at a time."""

    def __init__(self, shape):
        self.cold = _Runs(shape)  # transitional or frozen
        self.frozen = _Runs(shape)

    def update(self, t, state):
        valid = np.isfinite(state)
        self.cold.update(t, (state == 1) | (state == 2), valid)
        self.frozen.update(t, state == 1, valid)

    def events(self):
        return {
            "first_transition": self.cold.first,
            "first_frozen": self.frozen.first,
            "last_frozen": self.frozen.last,
            "last_transition": self.cold.last,
        }


def season_start(date):
    """1 August of the freeze season of each date."""
    date = pd.DatetimeIndex(date)
    return pd.to_datetime({"year": date.year - (date.month < 8), "month": 8, "day": 1})


def series_events(states):
    """Events of each column (cell) of a table of states indexed by the dates of one season."""
    index = pd.DatetimeIndex(states.index)
    t = (index - season_start(index[:1]).iloc[0]).days
    tracker = RunTracker(states.shape[1])
    values = states.to_numpy(dtype=float)
    for i, day in enumerate(t):
        tracker.update(day, values[i])
    return pd.DataFrame(tracker.events(), index=states.columns)


def season_mean(da, dim="season"):
    """Mean over the seasons with a value, where at least MIN_SEASONS seasons have one."""
    return da.mean(dim).where(da.notnull().sum(dim) >= MIN_SEASONS)
