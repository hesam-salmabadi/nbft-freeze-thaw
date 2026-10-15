"""Where each label's thresholds come from, and whether borrowed thresholds change the labels.

sfcc-label fits the freezing curve only where a sensor has soil moisture (permittivity) and the
winter passes its checks. Every other sensor-winter borrows T_on / T_fr from a group average
(the sensor's other winters, its network and probe type, ..., all sensors). Sensors are typed:
- fitted: at least one fitted winter of its own;
- moisture, never fitted: has moisture but no winter passed;
- temperature only: no moisture record, always borrows.
Per AM sensor-day, own_share = 1 when the thresholds are the sensor's own (own winter or its own
average). The test: on cell-days that have both own and borrowed sensor-days, relabel the cell
from its own-threshold sensors alone and compare with the label from all sensors.

Usage: python 08_threshold_provenance.py
"""

import numpy as np
import pandas as pd

from nbft import config, insitu, paths


def sensor_types(sensor_ids):
    t = pd.read_csv(config.path("sfcc.thresholds"), usecols=["sensor_id", "status"])
    fitted = set(t.loc[t["status"] == "fitted", "sensor_id"])
    moisture = set(t["sensor_id"])
    return pd.Series(
        np.select(
            [sensor_ids.isin(fitted), sensor_ids.isin(moisture)],
            ["fitted", "moisture_never_fitted"],
            "temperature_only",
        ),
        index=sensor_ids.index,
    )


def cell_label(s):
    g = s.groupby(["cell_id", "date"])
    return pd.Series(
        insitu.labels(g["p_frozen"].mean(), g["p_thawed"].mean()), index=g.size().index
    )


def main():
    cells = pd.read_csv(paths.TABLES / "insitu_cells.csv")
    cells = cells[cells["set"] != "excluded"]
    s = insitu.read_sensor_days()
    s = s[(s["am_pm"] == "am") & s["cell_id"].isin(cells["cell_id"])].copy()
    s["type"] = sensor_types(s["sensor_id"])
    s["own"] = s["own_share"] >= 0.5
    s = s.merge(cells[["cell_id", "set"]], on="cell_id")

    by_type = (
        s.groupby(["set", "type"])
        .agg(
            sensors=("sensor_id", "nunique"), sensor_days=("date", "size"), own_days=("own", "mean")
        )
        .round(3)
        .reset_index()
    )
    print(by_type.to_string(index=False))

    g = s.groupby(["set", "cell_id", "date"])["own"]
    kind = np.select([g.all(), ~g.any()], ["all_own", "all_borrowed"], "mixed")
    cd = pd.DataFrame({"kind": kind}, index=g.size().index).reset_index()
    per_cell = (
        cd.groupby(["set", "cell_id"])["kind"].value_counts(normalize=True).unstack(fill_value=0)
    )
    kinds = per_cell.groupby("set").median().round(3)
    no_own = (per_cell.get("all_borrowed", 0) == 1).groupby("set").sum()
    print("\nmedian share of a cell's days by threshold origin:\n", kinds)
    print("cells whose labels never use their own thresholds:", no_own.to_dict())

    mixed = cd.loc[cd["kind"] == "mixed", ["cell_id", "date"]]
    sm = s.merge(mixed, on=["cell_id", "date"])
    both = pd.DataFrame({"all_sensors": cell_label(sm), "own_only": cell_label(sm[sm["own"]])})
    agree = (both["all_sensors"] == both["own_only"]).mean()
    confusion = pd.crosstab(both["all_sensors"], both["own_only"], normalize="all").round(3)
    print(
        f"\nmixed cell-days: {len(both):,}; label unchanged with own thresholds only: {agree:.3f}"
    )
    print(confusion)

    by_type.to_csv(paths.TABLES / "insitu_threshold_types.csv", index=False)
    per_cell.reset_index().to_csv(paths.TABLES / "insitu_threshold_origin_cells.csv", index=False)
    confusion.to_csv(paths.TABLES / "insitu_threshold_relabel.csv")


if __name__ == "__main__":
    main()
