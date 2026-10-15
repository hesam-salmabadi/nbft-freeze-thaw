# Reproduce the analyses of the paper, in order. Each stage expects the outputs of the previous
# stages under $NBFT_DATA_ROOT (see config/paths.yaml). Run with PYTHONPATH=src or after `pip install -e .`.
PY := python

.PHONY: help env test lint statics labels model external evaluation timing figures all

help:
	@echo "env | test | lint | statics | labels | model | external | evaluation | timing | figures | all"

env:
	micromamba env create -f environment.yml

test:
	pytest

lint:
	ruff check src tests workflow

S := workflow/01_preprocess_satellite
L := workflow/02_insitu_labels
M := workflow/03_model
X := workflow/04_validation/external_test
T := workflow/05_phenology
F := workflow/06_figures

statics:      ## Thawed-season spread of each index and ice fraction of each cell
	$(PY) $(S)/01_thaw_std.py
	$(PY) $(S)/04_ice_fraction.py
	$(PY) $(S)/02_plot_references.py

labels:       ## Daily ground-state labels of the 9 km cells from the in situ sensor probabilities
	$(PY) $(L)/01_extract_overpass.py
	$(PY) $(L)/02_select_cells.py
	$(PY) $(L)/03_cell_table.py
	$(PY) $(L)/04_label_composition.py
	$(PY) $(L)/05_map_cells.py
	$(PY) $(L)/06_within_cell_agreement.py
	$(PY) $(L)/07_representativeness.py
	$(PY) $(L)/08_threshold_provenance.py
	$(PY) $(L)/09_transition_duration.py
	$(PY) $(L)/10_extract_satellite.py
	$(PY) $(L)/11_roughness_effect.py
	$(PY) $(L)/12_cutoff_sensitivity.py
	$(PY) $(L)/14_data_summary.py

model:        ## Training table, feature selection, cross-validation, production model, maps
	$(PY) $(M)/00_build_training_table.py
	$(PY) $(S)/03_separability.py
	$(PY) $(M)/03_feature_selection.py
	$(PY) $(M)/05_evaluate_cv.py
	$(PY) $(M)/06_train_production.py
	$(PY) $(M)/07_upscale.py
	$(PY) $(M)/08_extract_evaluation_data.py

external:     ## External test outside North America
	$(PY) $(X)/01_build.py
	$(PY) $(X)/02_evaluate.py

evaluation:   ## FT-SMOS comparison, coefficients, snow, sensitivity tests, HMM baseline
	$(PY) $(M)/09_evaluate.py
	$(PY) $(M)/10_coefficient_effects.py
	$(PY) $(M)/11_snow.py --extract
	$(PY) $(M)/12_transition_share_sensitivity.py
	$(PY) $(M)/13_single_index.py
	$(PY) $(M)/18_hmm_baseline.py

timing:       ## Timing of freeze-up and thaw at the cells and on the maps
	$(PY) $(T)/01_season_summary.py
	$(PY) $(T)/02_timing_insitu.py
	$(PY) $(T)/03_map_statistics.py

figures:
	$(PY) $(F)/03_plot_workflow.py
	$(PY) $(F)/04_plot_site_examples.py
	$(PY) $(F)/05_plot_decision_space.py
	$(PY) $(F)/06_plot_group_scores.py
	$(PY) $(F)/07_plot_temperature.py
	$(PY) $(F)/08_plot_maps.py
	$(PY) $(F)/09_plot_timing_scatter.py
	$(PY) $(F)/10_plot_graphical_abstract.py
	$(PY) $(F)/11_plot_hmm.py

all: statics labels model external evaluation timing figures
