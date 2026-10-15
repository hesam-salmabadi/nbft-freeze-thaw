"""The five example sites used for the site time series figure.

Each site is shown for one freeze-thaw year (August to August), chosen per network for data
coverage, then all panels are aligned onto a common August-August axis.
"""

SITES = [
    {
        "py": 934,
        "px": 752,
        "label": "(a) Tundra (CB01)",
        "network": "CB",
        "start": "2019-08-01",
        "end": "2020-08-01",
    },
    {
        "py": 722,
        "px": 824,
        "label": "(b) Boreal Tundra (AL02)",
        "network": "AL",
        "start": "2017-08-01",
        "end": "2018-08-01",
    },
    {
        "py": 1101,
        "px": 584,
        "label": "(c) Boreal Forest (KJ01)",
        "network": "KJ",
        "start": "2016-08-01",
        "end": "2017-08-01",
    },
    {
        "py": 1167,
        "px": 512,
        "label": "(d) Mixed Forest (FM01)",
        "network": "FM",
        "start": "2020-08-01",
        "end": "2021-08-01",
    },
    {
        "py": 866,
        "px": 552,
        "label": "(e) Croplands (UG01)",
        "network": "UG",
        "start": "2015-08-01",
        "end": "2016-08-01",
    },
]
