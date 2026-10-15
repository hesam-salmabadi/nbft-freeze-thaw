"""How long the topsoil stays in transition each autumn, from the in situ yearly dates.

sfcc-label gives each sensor-year a transition onset (first day of five or more non-thawed days)
and a freeze start (first day of five or more frozen days), both on or before 1 March. Per
sensor-year the autumn transition lasts from onset to freeze start; a year with an onset but no
freeze start stayed in transition all winter. Sensor-years are averaged per cell first, then
summarized across cells by set and tree-cover class (freeze years 2015/16 to 2022/23, sensors
with at least MIN_VALID_DAYS valid days that year).

Usage: python 09_transition_duration.py
"""

import pandas as pd

from nbft import config, insitu, paths

MIN_VALID_DAYS = 300
YEARS = range(2015, 2023)


def main():
    cells = pd.read_csv(paths.TABLES / "insitu_cells.csv")
    cells = cells[cells["set"] != "excluded"]
    cells["tree_class"] = insitu.tree_cover_class(cells["tree_cover_pct"]).to_numpy()
    sensors = insitu.selected_sensors()
    sensors = sensors[sensors["cell_id"].isin(cells["cell_id"])]

    ev = pd.read_csv(
        config.path("sfcc.yearly_events"),
        parse_dates=["transition_onset", "freeze_start", "freeze_end"],
    )
    ev = ev.merge(sensors, on="sensor_id")
    ev = ev[ev["fyear"].isin(YEARS) & (ev["valid_days"] >= MIN_VALID_DAYS)]
    ev["has_onset"] = ev["transition_onset"].notna()
    ev["froze"] = ev["freeze_start"].notna()
    ev["duration_days"] = (ev["freeze_start"] - ev["transition_onset"]).dt.days
    ev["onset_days"] = (
        ev["transition_onset"] - pd.to_datetime(ev["fyear"].astype(str) + "-08-01")
    ).dt.days
    onset = ev[ev["has_onset"]]

    per_cell = onset.groupby("cell_id").agg(
        sensor_years=("fyear", "size"),
        duration_days=("duration_days", "median"),
        never_froze_share=("froze", lambda x: 1 - x.mean()),
        onset_days=("onset_days", "median"),
    )
    per_cell["years_with_onset_share"] = ev.groupby("cell_id")["has_onset"].mean()
    per_cell = per_cell.merge(
        cells[["cell_id", "set", "region", "tree_class"]], left_index=True, right_on="cell_id"
    )
    per_cell.to_csv(paths.TABLES / "insitu_transition_duration_cells.csv", index=False)

    q = lambda p: lambda x: x.quantile(p)  # noqa: E731
    summary = (
        per_cell.groupby(["set", "region", "tree_class"], observed=True)
        .agg(
            n_cells=("cell_id", "size"),
            duration_median=("duration_days", "median"),
            duration_q25=("duration_days", q(0.25)),
            duration_q75=("duration_days", q(0.75)),
            never_froze_share_median=("never_froze_share", "median"),
            onset_days_after_1aug=("onset_days", "median"),
        )
        .round(1)
        .reset_index()
    )
    summary.to_csv(paths.TABLES / "insitu_transition_duration.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
