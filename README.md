# Non-binary soil freeze–thaw from multi-frequency passive microwave

Code for:

> Salmabadi, H., Berg, A., Roy, A. *Non-Binary Retrievals of Soil Freeze–Thaw Dynamics Using
> Multi-Frequency Passive Microwave Data.* Remote Sensing of Environment (under review).

A three-class multinomial logistic model gives the probability that the soil at the morning
overpass is **thawed**, **transitional**, or **frozen**. It uses two standardized microwave indices,
the SMAP L-band normalized polarization ratio (Z_NPR) and the AMSR2 18.7/36.5 GHz normalized
difference at vertical polarization (Z_NDV), with the tree cover of the 9 km cell (T) and its
interactions with both indices. The model is trained on daily ground-state labels of 539 North
American 9 km EASE-Grid 2.0 cells (1,258 topsoil sensors), evaluated on held-out spatial blocks,
on an external test outside North America, and against FT-SMOS, and mapped over North America for
2015–2023.

## Repository layout

```
config/paths.yaml          data locations (relative to the data root)
src/nbft/                  shared code: grid, labels, model, evaluation, timing, HMM baseline
workflow/
  01_preprocess_satellite/ thawed-season statistics of the indices, ice fraction, index separability
  02_insitu_labels/        daily labels of the 9 km cells from the in situ sensors
  03_model/                training table, feature selection, cross-validation, production model,
                           maps, FT-SMOS comparison, snow, sensitivity tests, HMM baseline
  04_validation/           external test outside North America
  05_phenology/            timing of freeze-up and thaw
  06_figures/              figures of the paper
results/tables/            tables behind every number in the paper
docs/                      data sources (data.md), figure and table map (figure_map.md)
tests/                     unit tests
```

## Setup

```bash
micromamba env create -f environment.yml
micromamba activate nbft
export NBFT_DATA_ROOT=/path/to/data
```

Paths can also be set in `config/paths.local.yaml` (see `config/paths.local.yaml.example`).

## Data

See [docs/data.md](docs/data.md). The satellite, reanalysis, and in situ data are available from
their providers. The preprocessed daily data cubes on the 9 km EASE-Grid 2.0 are available from
the authors on request.

The in situ ground-state labels are derived with a method that will be described in a separate
paper; its code and the labels will be released with that paper. The two figures that show the
label method itself (Fig. 3 and Fig. S3) are made with that code.

## Reproducing the paper

```bash
make statics      # thawed-season statistics of the indices
make labels       # daily labels of the 9 km cells
make model        # training table, cross-validation, production model, maps
make external     # external test
make evaluation   # FT-SMOS comparison, snow, sensitivity tests, HMM baseline
make timing       # timing of freeze-up and thaw
make figures
```

Each figure and table is mapped to its script in [docs/figure_map.md](docs/figure_map.md).

## License

MIT (see [LICENSE](LICENSE)).

## Citation

See [CITATION.cff](CITATION.cff).
