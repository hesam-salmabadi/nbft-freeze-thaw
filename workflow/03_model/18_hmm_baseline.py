"""Two-state HMM baseline after Walker et al. (2022), on the development cells (Section 3.6.3).

Their product does not appear to be publicly archived, so their method is reproduced here
(nbft.hmm) and run on the same brightness temperatures as this study, set up as in their paper:
- inputs: the L-band NPR of the rSIR-enhanced 9 km SMAP TB and the AMSR2 36.5 GHz V-pol TB, from
  the cube before gap filling and smoothing (timeseries.tb_raw), morning pass, study period;
- missing days filled by linear interpolation in time to give a continuous sequence (their gaps
  were also filled); edge gaps stay missing;
- one model per cell, fitted without labels on the shoulder seasons (Feb-May, Sep-Dec);
- their domain is the land north of 45 deg N, so results are given for those cells and for all.

Scored on the clean labelled days whose two inputs were observed (not filled), against the
held-out P(frozen) of the three-class model (05_evaluate_cv.py):
- PR AUC on the frozen and thawed days (a retrieval without skill scores the share of frozen days);
- quartiles of P(frozen) on thawed, partial and frozen days;
- agreement with the in situ frozen/thawed label in bins of the HMM probability (<0.25, 0.25-0.75,
  >0.75), the measure Walker et al. report, as a check of the reproduction.
Every day is weighted so that each spatial block carries the same total weight within the set
scored (model.block_weights), as in the other evaluations.
Cells whose NPR and TB36V point to different frozen states (hmm.frozen_state_agrees) are counted.

Outputs: baseline.hmm_inputs_raw (cache), baseline.hmm_posteriors (pixels=insitu),
results/tables/hmm_pr_auc.csv, hmm_partial_days.csv, hmm_agreement_bins.csv.

Usage: python 18_hmm_baseline.py [--refresh]
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import average_precision_score

from nbft import config, features, insitu, paths
from nbft.evaluation import weighted_quantile
from nbft.hmm import posteriors_by_pixel
from nbft.model import block_weights

CHANNELS = ["smap_v", "smap_h", "amsr_36v"]
WALKER_MIN_LATITUDE = 45.0
LABEL_ORDER = ["thawed", "partial", "frozen"]
BINS = [(0.0, 0.25, "<0.25"), (0.25, 0.75, "0.25-0.75"), (0.75, 1.0001, ">0.75")]


def raw_inputs(refresh=False):
    """Unsmoothed AM TB at the development cells over the study period (cached)."""
    out = config.path("baseline.hmm_inputs_raw")
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    cells = pd.read_csv(paths.TABLES / "insitu_cell_table.csv")
    cells = cells.loc[cells["set"] == "development", "cell_id"]
    rows, cols = insitu.cell_rowcol(cells)
    ds = xr.open_zarr(config.path("timeseries.tb_raw")).sel(am_pm="am")
    ds = ds.sel(time=slice(insitu.TIME_START, pd.Timestamp(insitu.TIME_END) - pd.Timedelta("1D")))
    pts = {"y": xr.DataArray(rows, dims="cell"), "x": xr.DataArray(cols, dims="cell")}
    frames = []
    for ch in CHANNELS:
        v = ds[ch].isel(**pts).values  # (time, cell)
        frames.append(
            pd.DataFrame(v, index=ds["time"].values, columns=list(cells))
            .stack(future_stack=True)
            .rename(ch)
        )
        print(f"  {ch} read")
    x = pd.concat(frames, axis=1).rename_axis(["date", "cell_id"]).reset_index()
    out.parent.mkdir(parents=True, exist_ok=True)
    x.to_parquet(out, index=False)
    return x


def hmm_posteriors(raw):
    df = pd.DataFrame(
        {
            "pixel": raw["cell_id"],
            "time": pd.to_datetime(raw["date"]),
            "npr": features.npr(raw["smap_v"], raw["smap_h"]),
            "tb36v": raw["amsr_36v"],
        }
    ).replace([np.inf, -np.inf], np.nan)
    df = df.sort_values(["pixel", "time"])
    # the HMM treats consecutive rows as consecutive days: every cell needs a complete daily index
    span = df.groupby("pixel")["time"].agg(["min", "max", "size"])
    assert ((span["max"] - span["min"]).dt.days + 1 == span["size"]).all(), "missing dates"
    df["observed"] = df[["npr", "tb36v"]].notna().all(axis=1)
    for col in ("npr", "tb36v"):  # fill gaps inside each cell's record
        df[col] = df.groupby("pixel")[col].transform(lambda s: s.interpolate(limit_area="inside"))
    df, fitted, skipped = posteriors_by_pixel(df)
    return df.rename(columns={"pixel": "cell_id", "time": "date"})


def pr_auc(g, column):
    s = g[g["label"] != "partial"]
    y = s["label"] == "frozen"
    w = block_weights(s["block"])
    return (
        average_precision_score(y, s[column], sample_weight=w) if 0 < y.sum() < len(y) else np.nan
    )


def score(x, name):
    """PR AUC, agreement by probability range and quantiles by label, every day weighted so that
    each spatial block carries the same total weight within the set scored."""
    rows, bins = [], []
    for group, g in [("all", x), *x.groupby("tree_class")]:
        b = g[g["label"] != "partial"]
        rows.append(
            {
                "cells": name,
                "tree_class": group,
                "n_cells": g["cell_id"].nunique(),
                "days": len(b),
                "frozen_share": np.average(
                    b["label"] == "frozen", weights=block_weights(b["block"])
                ),
                "pr_auc_hmm": pr_auc(g, "hmm_prob"),
                "pr_auc_model": pr_auc(g, "p_frozen"),
            }
        )
    binary = x[x["label"] != "partial"]
    w = pd.Series(block_weights(binary["block"]), index=binary.index)
    truth = binary["label"] == "frozen"
    for low, high, label in BINS:
        inside = binary["hmm_prob"].between(low, high, inclusive="left")
        called = binary.loc[inside, "hmm_prob"] >= 0.5
        bins.append(
            {
                "cells": name,
                "hmm_bin": label,
                "days": int(inside.sum()),
                "share_of_days": w[inside].sum() / w.sum(),
                "agreement": np.average(called == truth[inside], weights=w[inside])
                if inside.any()
                else np.nan,
            }
        )
    xw = pd.Series(block_weights(x["block"]), index=x.index)
    q = []
    for retrieval, col in (("hmm", "hmm_prob"), ("model", "p_frozen")):
        for label in LABEL_ORDER:
            m = x["label"] == label
            q.append(
                {
                    "retrieval": retrieval,
                    "label": label,
                    **{
                        k: weighted_quantile(x.loc[m, col], xw[m], v)
                        for k, v in (("p25", 0.25), ("median", 0.5), ("p75", 0.75))
                    },
                    "cells": name,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(bins), pd.DataFrame(q)


def main(refresh=False):
    raw = raw_inputs(refresh)
    hmm = hmm_posteriors(raw)
    out = config.path("baseline.hmm_posteriors", pixels="insitu", am_pm="am")
    out.parent.mkdir(parents=True, exist_ok=True)
    hmm.to_parquet(out, index=False)
    agree = hmm.groupby("cell_id")["hmm_agrees"].first()
    n = agree.notna().sum()
    print(f"cells where NPR and TB36V agree on the frozen state: {agree.mean():.2f} of {n}")

    pred = pd.read_parquet(config.path("model.cv_predictions"))
    x = pred.merge(
        hmm[["cell_id", "date", "hmm_prob", "hmm_agrees", "observed"]], on=["cell_id", "date"]
    )
    x = x[x["observed"]].dropna(subset=["hmm_prob", "p_frozen"])
    north = x[x["latitude"] >= WALKER_MIN_LATITUDE]
    parts = [
        score(north, "north of 45N"),
        score(x, "all cells"),
        score(north[north["hmm_agrees"] == 1], "north of 45N, agreeing cells"),
    ]
    auc, bins, q = (pd.concat(p, ignore_index=True).round(3) for p in zip(*parts, strict=True))
    auc.to_csv(paths.TABLES / "hmm_pr_auc.csv", index=False)
    bins.to_csv(paths.TABLES / "hmm_agreement_bins.csv", index=False)
    q.to_csv(paths.TABLES / "hmm_partial_days.csv", index=False)
    print(
        auc[auc["tree_class"] == "all"].to_string(index=False),
        bins.to_string(index=False),
        q.to_string(index=False),
        sep="\n\n",
    )


if __name__ == "__main__":
    main("--refresh" in sys.argv)
