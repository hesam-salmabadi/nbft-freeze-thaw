# Data

No data are stored in this repository except the result tables in `results/tables/`. All inputs
live under the data root (`NBFT_DATA_ROOT`); their locations are listed in `config/paths.yaml`.

## Inputs

| Data | Product | Source |
|---|---|---|
| SMAP L-band brightness temperature | Enhanced 9 km TB, NSIDC-0738 V2/V3 (EASE-Grid 2.0) | NSIDC |
| AMSR2 18.7 and 36.5 GHz brightness temperature | Level-3 global 0.1° TB | JAXA |
| Air temperature and snow water equivalent | ERA5-Land, hourly 2 m air temperature and daily snow depth water equivalent | Copernicus Climate Change Service |
| Snow cover | IMS daily Northern Hemisphere snow and ice analysis (G02156) | NSIDC |
| Reference freeze–thaw product | FT-SMOS (frozen, partially frozen, thawed) | ESA |
| Tree cover | Copernicus Global Land Service land cover 2015, 100 m | Copernicus |
| Terrain | GMTED2010, 7.5 arc-seconds | USGS |
| Land cover | ESA CCI Land Cover 2015, 300 m | ESA |
| Biome | RESOLVE Ecoregions 2017 | RESOLVE |
| In situ soil temperature and moisture | ISMN, AmeriFlux, regional networks | providers of each network |

ISMN and AmeriFlux data are not redistributed here (terms of use of each provider). The sensor
counts per source and network are in `results/tables/insitu_sources.csv` and
`results/tables/insitu_networks.csv`.

## Preprocessed data

The daily brightness temperatures of both sensors, resampled to the 9 km EASE-Grid 2.0, and the
static maps on the same grid are available from the authors on request.

The in situ ground-state labels (hourly probabilities of frozen, transition, and thawed for each
sensor, and their combination per 9 km cell) are described in a separate paper and will be
released with it. Their locations are the `sfcc:` keys of `config/paths.yaml`.
