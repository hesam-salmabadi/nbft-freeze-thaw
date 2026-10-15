"""Attributes of every development cell (supplementary table and CSV).

Per cell: location, set, ESA CCI class and its share of the cell, tree cover, elevation,
roughness and slope, RESOLVE biome, Koppen-Geiger class, SoilGrids soil of its sensors (most
common USDA texture; mean clay, sand and organic carbon), open water, sensors that contribute in
the study period and how many of them have at least one fitted winter, networks, and coverage.

Usage: python 03_cell_table.py
"""

import numpy as np
import pandas as pd

from nbft import config, insitu, paths

SOIL = ["clay_pct", "sand_pct", "soc_g_kg"]


def cci_fraction(cells):
    rows, cols = insitu.cell_rowcol(cells["cell_id"])
    out = np.full(len(cells), np.nan)
    for cls in cells["cell_class"].unique():
        band = insitu.read_band("landcover_cci2015", f"{cls}_fraction")
        hit = (cells["cell_class"] == cls).to_numpy()
        out[hit] = band[rows[hit], cols[hit]]
    return out


def sensor_summary(sensor_days):
    """Per cell: contributing sensors, fitted sensors, networks, soil."""
    sensors = sensor_days[["cell_id", "sensor_id"]].drop_duplicates()
    catalog = insitu.read_catalog(["sensor_id", "network"])
    soil = pd.read_csv(
        config.path("sfcc.sensor_soil"), usecols=["sensor_id", "texture_class_usda", *SOIL]
    )
    thresholds = pd.read_csv(config.path("sfcc.thresholds"), usecols=["sensor_id", "status"])
    fitted = set(thresholds.loc[thresholds["status"] == "fitted", "sensor_id"])
    moisture = set(thresholds["sensor_id"])  # sfcc-label attempts a fit only with a moisture record
    s = sensors.merge(catalog, on="sensor_id", how="left").merge(soil, on="sensor_id", how="left")
    s["fitted"] = s["sensor_id"].isin(fitted)
    s["moisture"] = s["sensor_id"].isin(moisture)
    g = s.groupby("cell_id")
    return pd.DataFrame(
        {
            "n_sensors": g.size(),
            "n_fitted_sensors": g["fitted"].sum(),
            "n_moisture_sensors": g["moisture"].sum(),
            "networks": g["network"].agg(lambda x: ";".join(sorted(set(x.dropna())))),
            "soil_texture": g["texture_class_usda"].agg(
                lambda x: x.mode().iloc[0] if x.notna().any() else np.nan
            ),
            **{c: g[c].mean().round(1) for c in SOIL},
        }
    )


def main():
    cells = pd.read_csv(paths.TABLES / "insitu_cells.csv")
    cells = cells[cells["set"] != "excluded"].drop(columns="n_sensors")
    extra = insitu.read_cells()[["cell_id", "biome", "koppen"]]
    cells = cells.merge(extra, on="cell_id", how="left")
    cells["cci_fraction"] = cci_fraction(cells).round(3)

    sensor_days = insitu.read_sensor_days(["cell_id", "sensor_id", "am_pm"])
    sensor_days = sensor_days[sensor_days["am_pm"] == "am"]
    cells = cells.merge(
        sensor_summary(sensor_days), left_on="cell_id", right_index=True, how="left"
    )

    cols = [
        "cell_id",
        "set",
        "set_reason",
        "region",
        "latitude",
        "longitude",
        "cell_class",
        "cci_fraction",
        "tree_cover_pct",
        "elevation_m",
        "elevation_std_m",
        "slope_deg",
        "biome",
        "koppen",
        "soil_texture",
        *SOIL,
        "water_fraction",
        "n_sensors",
        "n_fitted_sensors",
        "n_moisture_sensors",
        "networks",
        "first_day",
        "last_day",
        "am_days",
        "freeze_years",
        "complete_seasons",
        "ever_frozen",
    ]
    table = cells[cols].sort_values(["set", "region", "latitude"], ascending=[True, True, False])
    table.to_csv(paths.TABLES / "insitu_cell_table.csv", index=False)
    print(
        table.groupby(["set", "region"]).agg(
            cells=("cell_id", "size"),
            sensors=("n_sensors", "sum"),
            fitted=("n_fitted_sensors", "sum"),
        )
    )


if __name__ == "__main__":
    main()
