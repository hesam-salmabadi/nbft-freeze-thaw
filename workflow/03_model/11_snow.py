"""Snow and the model classes at the development and external cells (Discussion, Supplementary
Material).

The ERA5-Land snow depth water equivalent (SWE, mm; timeseries.era5_snow_depth) is read at every
development and external cell-day outside July and August and cached (model.eval_snow). The days
are grouped by SWE (SNOW_BINS) and, for each in situ label and SWE group:
- share of the label's days in the group;
- median Z_NDV and Z_NPR and median in situ soil temperature;
- share of the days the model calls thawed, transitional and frozen (held-out states at the
  development cells, production model at the external cells), and at the development cells the
  same for FT-SMOS (partially frozen = transitional) on the days it has a state.
Every day is weighted so that each spatial block carries the same total weight within the set
scored (model.block_weights), as in the scores of 05_evaluate_cv.py: all development cell-days,
and at the external cells the cell-days of each region.

SWE gives the amount of snow; whether snow is on the ground is taken from the observation-based IMS
snow cover (timeseries.ims_snow). At the development cells, the held-out days are split by phase
of the freeze season (PHASES) and by condition: no IMS snow, IMS snow with an ERA5-Land daily mean
air temperature at or below 0 deg C, and IMS snow with air above 0 deg C (a proxy for wet snow).
For each phase and condition: share of the phase's days, error rate (state differs from the
label), and share of all errors (snow_errors.csv); and the share of the phase's errors of each
type, label and called class (snow_error_types.csv).

Outputs: results/tables/snow_classes.csv (development), snow_external.csv (per region),
snow_errors.csv, snow_error_types.csv.

Usage: python 11_snow.py [--extract]
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr

from nbft import config, insitu, paths
from nbft.evaluation import CODE_CLASS, weighted_quantile
from nbft.model import (
    CLASSES,
    LABEL_CLASS,
    STATE_CODES,
    TERMS,
    block_weights,
    class_probabilities,
    design,
    most_probable,
)

SNOW_BINS = [-np.inf, 1, 30, 100, np.inf]
SNOW_LABELS = ["none (<1 mm)", "1-30 mm", "30-100 mm", ">100 mm"]
LABELS = ["thawed", "partial", "frozen"]
PHASES = {"autumn": [9, 10, 11, 12], "winter": [1, 2], "spring": [3, 4, 5, 6]}


def at_cells(da, cells, t):
    rows, cols = insitu.cell_rowcol(cells)
    pts = {"y": xr.DataArray(rows, dims="cell"), "x": xr.DataArray(cols, dims="cell")}
    v = da.sel(time=t).isel(**pts).values
    return pd.DataFrame(v, index=t, columns=list(cells)).rename_axis("date").stack()


def extract():
    """ERA5-Land SWE (mm), IMS snow cover (1 snow, 0 none; no data dropped) and ERA5-Land daily
    mean 2 m air temperature (deg C) at the development and external cell-days."""
    dev = pd.read_parquet(config.path("model.cv_predictions"), columns=["cell_id", "date"])
    ext = pd.read_parquet(config.path("validation.external_days"), columns=["cell_id", "date"])
    cells = pd.Index(pd.concat([dev, ext])["cell_id"].unique())
    sources = {
        "swe": xr.open_zarr(config.path("timeseries.era5_snow_depth"))["sd"],
        "ims_snow": xr.open_zarr(config.path("timeseries.ims_snow"))["snow_cover"],
        "t2m": xr.open_zarr(config.path("timeseries.era5_t2m_daily"))["t2m"],
    }
    t = pd.date_range(insitu.TIME_START, insitu.TIME_END, inclusive="left")
    t = t[~t.month.isin(insitu.SUMMER_MONTHS)]
    for da in sources.values():
        t = t.intersection(pd.DatetimeIndex(da["time"].values))
    out = pd.concat({k: at_cells(da, cells, t) for k, da in sources.items()}, axis=1)
    out = out.reset_index().rename(columns={"level_1": "cell_id"})
    out["ims_snow"] = out["ims_snow"].where(out["ims_snow"].isin([0, 1]))
    if out["t2m"].median() > 100:  # stored in kelvin
        out["t2m"] -= 273.15
    path = config.path("model.eval_snow")
    out.to_parquet(path, index=False)
    print(f"snow: {len(cells)} cells, {len(out):,} cell-days -> {path}")


def development():
    x = pd.read_parquet(config.path("model.cv_predictions"))
    z = pd.read_parquet(
        config.path("model.training_table"), columns=["cell_id", "date", "z_npr", "z_ndv"]
    )
    for key in ("soil_temperature", "smos", "snow"):
        x = x.merge(pd.read_parquet(config.path(f"model.eval_{key}")), how="left")
    return x.merge(z, how="left")


def external():
    x = pd.read_parquet(config.path("validation.external_days"))
    x = x[x["clean"].astype(bool)]
    coef = pd.read_csv(paths.TABLES / "model_coefficients.csv", index_col="class")
    probs = class_probabilities(
        coef[["intercept", *TERMS]], design(x["z_npr"], x["z_ndv"], x["tree"])
    )
    x["state"] = np.asarray(most_probable(probs), dtype=float)
    return x.merge(pd.read_parquet(config.path("model.eval_snow")), how="left")


def table(x, by):
    """Per group (by), label and SWE class: share of days, medians, and the model states."""
    x = x.dropna(subset=["swe", "state"]).copy()
    x["snow"] = pd.cut(x["swe"], SNOW_BINS, labels=SNOW_LABELS)
    rows = []
    for key, g in x.groupby(by) if by else [("all", x)]:
        w_all = pd.Series(block_weights(g["block"]), index=g.index)
        for label in LABELS:
            gl = g[g["label"] == label]
            if gl.empty:
                continue
            w = w_all[gl.index]
            for snow, gs in gl.groupby("snow", observed=True):
                ws = w[gs.index]
                row = {
                    "group": key,
                    "label": label,
                    "snow": snow,
                    "days": len(gs),
                    "share": ws.sum() / w.sum(),
                }
                for col in ("z_ndv", "z_npr", "soil_t", "swe"):
                    if col in gs:
                        ok = gs[col].notna()
                        row[f"{col}_median"] = weighted_quantile(gs[col][ok], ws[ok], 0.5)
                for code, name in CODE_CLASS.items():
                    row[f"called_{name}"] = ws[gs["state"] == code].sum() / ws.sum()
                if "smos_state" in gs:
                    has = gs["smos_state"].notna()
                    for code, name in CODE_CLASS.items():
                        hit = gs["smos_state"][has] == code
                        row[f"smos_{name}"] = ws[has][hit].sum() / max(ws[has].sum(), 1e-12)
                rows.append(row)
    return pd.DataFrame(rows)


def errors(x):
    x = x.dropna(subset=["ims_snow", "t2m", "state"]).reset_index(drop=True)
    x["w"] = block_weights(x["block"])
    truth = x["label"].map(LABEL_CLASS).map(dict(enumerate(CLASSES))).map(STATE_CODES)
    x["wrong"] = x["state"] != truth
    month = pd.to_datetime(x["date"]).dt.month
    x["phase"] = month.map({m: p for p, ms in PHASES.items() for m in ms})
    snow = x["ims_snow"] == 1
    x["condition"] = np.where(
        ~snow, "no snow", np.where(x["t2m"] > 0, "snow, air > 0", "snow, air <= 0")
    )
    all_errors = x.loc[x["wrong"], "w"].sum()
    rows, types = [], []
    for phase, g in [*x.groupby("phase"), ("all", x)]:
        for cond, h in [*g.groupby("condition"), ("all", g)]:
            rows.append(
                {
                    "phase": phase,
                    "condition": cond,
                    "share_of_phase_days": h["w"].sum() / g["w"].sum(),
                    "error_rate": h.loc[h["wrong"], "w"].sum() / h["w"].sum(),
                    "share_of_all_errors": h.loc[h["wrong"], "w"].sum() / all_errors,
                }
            )
        e = g[g["wrong"]].assign(called=lambda d: d["state"].map(CODE_CLASS))
        t = e.groupby(["condition", "label", "called"])["w"].sum() / e["w"].sum()
        types.append(t.rename("share_of_phase_errors").reset_index().assign(phase=phase))
    return pd.DataFrame(rows).round(3), pd.concat(types).round(3)


def main():
    if "--extract" in sys.argv[1:]:
        extract()
    d = development()
    err, types = errors(d)
    err.to_csv(paths.TABLES / "snow_errors.csv", index=False)
    types.to_csv(paths.TABLES / "snow_error_types.csv", index=False)
    print(err.to_string(index=False))
    dev = table(d, None)
    dev.round(3).to_csv(paths.TABLES / "snow_classes.csv", index=False)
    ext = table(external(), "region")
    ext.round(3).to_csv(paths.TABLES / "snow_external.csv", index=False)
    pd.set_option("display.width", 220)
    print(dev.round(2).to_string(index=False), "", ext.round(2).to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()
