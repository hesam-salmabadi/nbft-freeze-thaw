"""Statistics of the seasonal maps (Results 4.8), from product.season_summary
(01_season_summary.py).

Every cell value is a mean over the seasons with data, where at least nbft.timing.MIN_SEASONS
seasons have data (as in the maps, 06_figures/08_plot_maps.py). Over the cells on which both the
model and FT-SMOS have such a mean:
- season_days: median (and, for all cells, mean) number of transitional (FT-SMOS: partially
  frozen) and frozen mornings per season, for all cells and per terrain-roughness and tree-cover
  class; and, per season, the median
  of that season's counts on the same cells;
- timing: median first transitional, first frozen and last transitional morning of each product
  (days since 1 August) and median length of the autumn (first transitional to first frozen
  morning) and spring (last frozen to last transitional morning) transitions, with the median and
  quartiles of the difference model minus FT-SMOS, for all cells and per roughness and tree-cover
  class;
- timing by season: the same medians for each season, on the cells on which both products have
  the event in that season.

Outputs: results/tables/map_season_days.csv, map_season_days_by_season.csv, map_timing.csv,
map_timing_by_season.csv.

Usage: python 03_map_statistics.py
"""

import numpy as np
import pandas as pd
import xarray as xr

from nbft import config, insitu, paths
from nbft.grid import standardize
from nbft.timing import season_mean

STATES = ["transitional", "frozen"]
TIMING = ["first_transition", "first_frozen", "last_transition", "autumn", "spring"]
DATES = ["first_transition", "first_frozen", "last_transition"]


def cell_classes(ds):
    """Roughness and tree-cover class of every cell of the summary's y/x box."""
    cube = standardize(xr.open_zarr(config.path("timeseries.tb_gapfilled")))
    r0 = int(np.abs(cube["y"].values - ds["y"].values[0]).argmin())
    c0 = int(np.abs(cube["x"].values - ds["x"].values[0]).argmin())
    box = (slice(r0, r0 + ds.sizes["y"]), slice(c0, c0 + ds.sizes["x"]))
    tree = insitu.read_band("treecover", "mean")[box].astype(float)
    tree = np.where(np.isfinite(tree), tree, 0.0)  # no value north of 80 N, as in the product
    rough = insitu.read_band("terrain", "elevation_std")[box].astype(float)
    return {
        "roughness_class": insitu.roughness_class(rough.ravel()).astype(str).to_numpy(),
        "tree_class": insitu.tree_cover_class(tree.ravel()).astype(str).to_numpy(),
    }


def by_group(values, both, classes):
    """Median of each column of values on the cells in both, overall and per class."""
    rows = [{"group_by": "all", "group": "all", "cells": int(both.sum())}]
    rows[0] |= {k: np.median(v[both]) for k, v in values.items()}
    rows[0] |= {f"{k}_mean": np.mean(v[both]) for k, v in values.items()}
    for col, cls in classes.items():
        for k in sorted(set(cls[both])):
            sel = both & (cls == k)
            rows.append({"group_by": col, "group": k, "cells": int(sel.sum())})
            rows[-1] |= {name: np.median(v[sel]) for name, v in values.items()}
    return pd.DataFrame(rows)


def main():
    ds = xr.open_dataset(config.path("product.season_summary", am_pm="am"))
    classes = cell_classes(ds)

    days = {
        f"{p}_{k}": season_mean(ds[f"{p}_{k}_days"]).values.ravel()
        for p in ("model", "smos")
        for k in STATES
    }
    both = np.all([np.isfinite(v) for v in days.values()], axis=0)
    season_days = by_group(days, both, classes)
    per_season = pd.DataFrame(
        [
            {"season": int(s)}
            | {
                f"{p}_{k}": float(
                    np.nanmedian(ds[f"{p}_{k}_days"].sel(season=s).values.ravel()[both])
                )
                for p in ("model", "smos")
                for k in STATES
            }
            for s in ds["season"].values
        ]
    )

    for p in ("model", "smos"):
        ds[f"{p}_autumn"] = ds[f"{p}_first_frozen"] - ds[f"{p}_first_transition"]
        ds[f"{p}_spring"] = ds[f"{p}_last_transition"] - ds[f"{p}_last_frozen"]
    by_season = []
    for season in ds["season"].values:
        for e in TIMING:
            m = ds[f"model_{e}"].sel(season=season).values.ravel()
            v = ds[f"smos_{e}"].sel(season=season).values.ravel()
            ok = np.isfinite(m) & np.isfinite(v)
            by_season.append(
                {
                    "season": int(season),
                    "event": e,
                    "cells": int(ok.sum()),
                    "model_median": np.median(m[ok]),
                    "smos_median": np.median(v[ok]),
                }
            )
    by_season = pd.DataFrame(by_season)

    events = {
        f"{p}_{e}": season_mean(ds[f"{p}_{e}"]).values.ravel()
        for p in ("model", "smos")
        for e in TIMING
    }
    rows = []
    for e in TIMING:
        m, s = events[f"model_{e}"], events[f"smos_{e}"]
        ok = np.isfinite(m) & np.isfinite(s)
        groups = [("all", "all", ok)] + [
            (col, k, ok & (cls == k)) for col, cls in classes.items() for k in sorted(set(cls[ok]))
        ]
        for col, k, sel in groups:
            d = m[sel] - s[sel]
            rows.append(
                {
                    "event": e,
                    "group_by": col,
                    "group": k,
                    "cells": int(sel.sum()),
                    "model_median": np.median(m[sel]),
                    "smos_median": np.median(s[sel]),
                    "difference_median": np.median(d),
                    "difference_p25": np.percentile(d, 25),
                    "difference_p75": np.percentile(d, 75),
                }
            )
    timing = pd.DataFrame(rows)
    for table in (timing, by_season):
        for col in ("model_median", "smos_median"):
            date = pd.Timestamp("2018-08-01") + pd.to_timedelta(table[col].round(), unit="D")
            table[f"{col}_date"] = date.dt.strftime("%d %b").where(table["event"].isin(DATES))

    season_days.round(1).to_csv(paths.TABLES / "map_season_days.csv", index=False)
    per_season.round(1).to_csv(paths.TABLES / "map_season_days_by_season.csv", index=False)
    timing.round(1).to_csv(paths.TABLES / "map_timing.csv", index=False)
    by_season.round(1).to_csv(paths.TABLES / "map_timing_by_season.csv", index=False)
    pd.set_option("display.width", 220)
    print(
        season_days.round(1).to_string(index=False),
        per_season.round(1).to_string(index=False),
        timing.round(1).to_string(index=False),
        by_season.round(1).to_string(index=False),
        sep="\n\n",
    )


if __name__ == "__main__":
    main()
