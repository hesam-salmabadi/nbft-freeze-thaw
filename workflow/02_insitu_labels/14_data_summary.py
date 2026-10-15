"""Summary of the in situ data for Section 2.2: sources and sensors per set, and the attributes of
the cells in each set (Table 2 of the paper; the full cell table is insitu_cell_table.csv).

Sensors are those that contribute AM overpass days in the study period to a development
test cell. A sensor has a moisture record when sfcc-label could attempt a freezing-curve fit
(it appears in thresholds.csv).

Usage: python 14_data_summary.py
"""

import pandas as pd

from nbft import config, insitu, paths

SOURCE_NAMES = {
    "ismn": "ISMN",
    "ameriflux": "AmeriFlux",
}
CCI_ORDER = ["forest", "low_vegetation", "agriculture", "wetland", "other"]


def group(cells):
    return cells["set"]


def sensors_used(cells):
    s = insitu.read_sensor_days(["cell_id", "sensor_id", "am_pm"])
    s = s[(s["am_pm"] == "am") & s["cell_id"].isin(cells["cell_id"])]
    s = s[["cell_id", "sensor_id"]].drop_duplicates()
    s = s.merge(cells[["cell_id", "group"]], on="cell_id")
    cat = insitu.read_catalog(["sensor_id", "source", "network"])
    s = s.merge(cat, on="sensor_id")
    moisture = set(pd.read_csv(config.path("sfcc.thresholds"), usecols=["sensor_id"])["sensor_id"])
    s["moisture"] = s["sensor_id"].isin(moisture)
    s["source_group"] = s["source"].map(SOURCE_NAMES).fillna("Regional and own networks")
    return s


def sources_table(s):
    t = s.groupby(["group", "source_group"]).agg(
        sensors=("sensor_id", "nunique"),
        cells=("cell_id", "nunique"),
        with_moisture=("moisture", "sum"),
    )
    nets = (
        s[s["source"] == "ismn"]
        .groupby(["group", "network"])["sensor_id"]
        .nunique()
        .sort_values(ascending=False)
        .groupby(level=0)
        .head(6)
    )
    regional = (
        s[~s["source"].isin(SOURCE_NAMES)]
        .groupby(["group", "source"])["sensor_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    return t.reset_index(), nets.reset_index(), regional.reset_index()


def set_table(cells, s):
    rows = []
    for g, c in cells.groupby("group"):
        cs = s[s["group"] == g]
        row = {
            "group": g,
            "cells": len(c),
            "sensors": cs["sensor_id"].nunique(),
            "sensors_with_moisture": int(cs.drop_duplicates("sensor_id")["moisture"].sum()),
            "cell_years": int(c["freeze_years"].sum()),
            "complete_cold_seasons": int(c["complete_seasons"].sum()),
            "cells_with_complete_season": int((c["complete_seasons"] > 0).sum()),
            "latitude_min": c["latitude"].min(),
            "latitude_max": c["latitude"].max(),
            "first_day": c["first_day"].min(),
            "last_day": c["last_day"].max(),
            "multi_sensor_cells": int((c["n_sensors"] > 1).sum()),
        }
        for col in ("tree_cover_pct", "elevation_m", "elevation_std_m"):
            q = c[col].quantile([0.1, 0.5, 0.9])
            row |= {f"{col}_p10": q[0.1], f"{col}_median": q[0.5], f"{col}_p90": q[0.9]}
        shares = c["cell_class"].value_counts(normalize=True)
        row |= {f"cci_{k}": shares.get(k, 0.0) for k in CCI_ORDER}
        kg = c["koppen"].str[0].value_counts(normalize=True)
        row |= {f"koppen_{k}": kg.get(k, 0.0) for k in "ABCDE"}
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def main():
    cells = pd.read_csv(paths.TABLES / "insitu_cell_table.csv")
    cells["group"] = group(cells)
    s = sensors_used(cells)
    sources, ismn, regional = sources_table(s)
    sets = set_table(cells, s)
    sources.to_csv(paths.TABLES / "insitu_sources.csv", index=False)
    pd.concat([ismn.assign(kind="ismn_network"), regional.assign(kind="regional_source")]).to_csv(
        paths.TABLES / "insitu_networks.csv", index=False
    )
    sets.to_csv(paths.TABLES / "insitu_set_summary.csv", index=False)
    print(sources.to_string(index=False))
    print(ismn.to_string(index=False))
    print(regional.to_string(index=False))
    print(sets.T.to_string())


if __name__ == "__main__":
    main()
