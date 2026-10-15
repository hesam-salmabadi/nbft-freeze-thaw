# Stage 2: ground-state labels of the 9 km cells

The labels start from hourly probabilities of frozen, transition, and thawed for each topsoil
sensor, from a soil freezing characteristic curve fitted per sensor and winter (method and data in
a separate paper; locations under `sfcc:` in `config/paths.yaml`). Each sensor's probabilities are
averaged over a 5-hour window around the morning overpass (03:00 local solar time); the cell's
daily shares are the mean over its sensors. A cell-day is frozen or thawed when that share reaches
0.75, otherwise partial (`src/nbft/insitu.py`).

Scripts, in order:

| Script | Output |
|---|---|
| `01_extract_overpass.py` | daily probabilities per sensor over the overpass window |
| `02_select_cells.py` | daily cell shares and labels; development cells; counts after each step (`insitu_attrition.csv`, Table 1) |
| `03_cell_table.py` | attributes of every development cell (`insitu_cell_table.csv`) |
| `04_label_composition.py` | frozen, partial, and thawed share per cell, by set and tree cover |
| `05_map_cells.py` | map of the development cells (Fig. 1) |
| `06_within_cell_agreement.py` | agreement of the sensors within a cell |
| `07_representativeness.py` | development cells against all North American land cells (Fig. S1) |
| `08_threshold_provenance.py` | labels from sensors with their own thresholds against borrowed ones |
| `09_transition_duration.py` | length of the autumn transition per cell |
| `10_extract_satellite.py` | morning SMAP and AMSR2 brightness temperatures at every in situ cell |
| `11_roughness_effect.py` | effect of terrain roughness on a shared model |
| `12_cutoff_sensitivity.py` | sensitivity of the labels to the cut-off (Fig. S2) |
| `14_data_summary.py` | sensors per source and network (`insitu_sources.csv`, `insitu_networks.csv`) |

Development set: every North American cell with representative topsoil sensors. Summer days (July
and August) are dropped, and sensors with fewer than 30 morning days left are removed. Open water
must cover at most 15 % of the cell.
