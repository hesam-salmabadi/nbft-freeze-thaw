"""Daily overpass-window probabilities for every sensor that passes the land-cover screen.

For each of the sfcc-label sensors kept on the grid, the hourly P(frozen), P(transition) and
P(thawed) are averaged over the 5-hour window centred on 03:00 (AM) and 15:00 (PM) local solar
time, within the study period. Local solar time uses the sensor's own longitude.

Slow: reads one predictions file per sensor (about 2,000).

Usage: python 01_extract_overpass.py [workers]
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from nbft import config, insitu


def extract(task):
    sensor_id, cell_id, lon = task
    pred = insitu.read_predictions(sensor_id)
    frames = [insitu.overpass_daily(pred, lon, am_pm).assign(am_pm=am_pm) for am_pm in ("am", "pm")]
    return pd.concat(frames).assign(sensor_id=sensor_id, cell_id=cell_id)


def main(workers=8):
    sensors = insitu.read_kept_sensors().merge(
        insitu.read_catalog(["sensor_id", "longitude"]), on="sensor_id"
    )
    tasks = list(sensors[["sensor_id", "cell_id", "longitude"]].itertuples(index=False, name=None))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        table = pd.concat(pool.map(extract, tasks, chunksize=8), ignore_index=True)
    out = config.path("insitu.sensor_overpass", grid=insitu.GRID)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sensor_id", "cell_id", "am_pm", "date", *insitu.PROBS, "n_hours", "own_share"]
    table[cols].to_parquet(out, index=False)
    print(f"{len(tasks)} sensors, {len(table):,} sensor-days -> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
