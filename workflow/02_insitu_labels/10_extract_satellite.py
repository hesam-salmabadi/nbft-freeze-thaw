"""AM brightness temperatures (SMAP and AMSR2, gap-filled cube) at every development
cell, for the label checks that need the satellite signal (11 and 12).

The cube is chunked as (all days, 1 overpass, 100 x 100 cells), so cells are read one chunk at a
time.

Slow: reads about 50 GB from the external drive.

Usage: python 10_extract_satellite.py [workers]
"""

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from nbft import config, insitu, paths
from nbft.grid import open_zarr

VARS = ["smap_v", "smap_h", "amsr_18v", "amsr_18h", "amsr_36v", "amsr_36h"]
CHUNK = 100


def read_block(ds, var, r0, c0, rows, cols):
    block = ds[var].isel(y=slice(r0, r0 + CHUNK), x=slice(c0, c0 + CHUNK)).values
    return var, (r0, c0), block[:, rows - r0, cols - c0]


def main(workers=4):
    cells = pd.read_csv(paths.TABLES / "insitu_cells.csv")
    cells = cells[cells["set"] != "excluded"].reset_index(drop=True)
    rows, cols = insitu.cell_rowcol(cells["cell_id"])
    cells["row"], cells["col"] = rows, cols
    cells["r0"], cells["c0"] = rows // CHUNK * CHUNK, cols // CHUNK * CHUNK

    ds = open_zarr(config.path("timeseries.tb_gapfilled"), "am")
    time = pd.DatetimeIndex(ds["time"].values)
    groups = {k: g for k, g in cells.groupby(["r0", "c0"])}
    tasks = [
        (var, r0, c0, g["row"].to_numpy(), g["col"].to_numpy())
        for (r0, c0), g in groups.items()
        for var in VARS
    ]
    print(f"{len(cells)} cells in {len(groups)} chunks, {len(tasks)} reads")
    values = {}
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(read_block, ds, *t) for t in tasks]
        for i, f in enumerate(futures, 1):
            var, key, v = f.result()
            values[(var, key)] = v
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)}")

    frames = []
    for (r0, c0), g in groups.items():
        n = len(g)
        frame = pd.DataFrame(
            {
                "cell_id": np.repeat(g["cell_id"].to_numpy()[None, :], len(time), 0).ravel(),
                "date": np.repeat(time.to_numpy(), n),
            }
        )
        for var in VARS:
            frame[var] = values[(var, (r0, c0))].ravel()
        frames.append(frame.dropna(subset=VARS, how="all"))
    out = pd.concat(frames, ignore_index=True)
    path = config.path("insitu.cell_satellite", grid=insitu.GRID)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"{len(out):,} cell-days -> {path}")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
