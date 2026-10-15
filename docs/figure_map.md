# Figure and table map

The script behind each figure and table of the paper. Figures are written to `paper/figs/`,
tables to `results/tables/`. Paths are relative to `workflow/`.

## Main text

| Item | Content | Script |
|---|---|---|
| Table 1 | Sensors and cells after each selection step | `02_insitu_labels/02_select_cells.py` (`insitu_attrition.csv`) |
| Fig. 1 | Development cells | `02_insitu_labels/05_map_cells.py` |
| Fig. 2 | Processing chain | `06_figures/03_plot_workflow.py` |
| Table 2 | Candidate input indices | (text) |
| Fig. 3 | Ground-state labels by tree cover | label code (separate paper) |
| Table 3 | Production coefficients | `03_model/10_coefficient_effects.py` (`model_coefficient_table.csv`) |
| Fig. 4 | Decision space | `06_figures/05_plot_decision_space.py` |
| Fig. 5 | Held-out scores by tree cover, roughness, and biome | `03_model/05_evaluate_cv.py`, `03_model/09_evaluate.py`, `06_figures/06_plot_group_scores.py` |
| Figs. 6, 7, 10 | Site examples | `06_figures/04_plot_site_examples.py` |
| Fig. 8 | Timing at the development cells | `05_phenology/02_timing_insitu.py`, `06_figures/09_plot_timing_scatter.py` |
| Table 4 | Timing scores | `05_phenology/02_timing_insitu.py` (`timing_scores.csv`) |
| Fig. 9 | Temperature of each class | `03_model/09_evaluate.py`, `06_figures/07_plot_temperature.py` |
| Table 5 | Effect of snow | `03_model/11_snow.py` (`snow_classes.csv`) |
| Table 6 | External test by region | `04_validation/external_test/02_evaluate.py` (`external_scores.csv`) |
| Figs. 11, 12 | Maps | `05_phenology/01_season_summary.py`, `06_figures/08_plot_maps.py` |
| Graphical abstract | | `06_figures/10_plot_graphical_abstract.py` |

## Supplementary Material

| Item | Content | Script |
|---|---|---|
| Fig. S1 | Development cells against North American land | `02_insitu_labels/07_representativeness.py` |
| Table S1 | Training days by roughness class | `03_model/00_build_training_table.py` (`training_composition.csv`) |
| Fig. S2 | Cut-off sensitivity of the labels | `02_insitu_labels/12_cutoff_sensitivity.py` |
| Fig. S3 | Borrowed thresholds | label code (separate paper) |
| Fig. S4 | Thawed-season reference and spread of each index | `01_preprocess_satellite/01_thaw_std.py`, `01_preprocess_satellite/02_plot_references.py` |
| Table S2 | Separation of frozen and thawed days by each index | `01_preprocess_satellite/03_separability.py` (`standardization_separability.csv`) |
| Table S3 | Correlation and separation of the indices | `03_model/03_feature_selection.py` (`feature_correlation.csv`, `feature_screen.csv`) |
| Table S4 | Single-index models | `03_model/13_single_index.py` (`single_index.csv`) |
| Table S5 | Effect of each index by tree cover | `03_model/10_coefficient_effects.py` (`model_index_effects.csv`) |
| Table S6 | Minimum transition share | `03_model/12_transition_share_sensitivity.py` (`transition_share_sensitivity.csv`) |
| Table S7 | Errors by season phase and snow | `03_model/11_snow.py` (`snow_errors.csv`, `snow_error_types.csv`) |
| Table S8 | Unweighted scores | `03_model/05_evaluate_cv.py`, `03_model/09_evaluate.py`, `05_phenology/02_timing_insitu.py` (`*_unweighted.csv`) |
| Tables S9–S11 | External test per network, Arctic sites, FT-SMOS | `04_validation/external_test/02_evaluate.py` (`external_networks.csv`, `external_arctic_sites.csv`, `external_smos.csv`) |
| Figs. S5, S6 | Maps of timing and thaw length | `05_phenology/03_map_statistics.py`, `06_figures/08_plot_maps.py` |
| Table S12 | Timing per season | `05_phenology/03_map_statistics.py` (`map_timing_by_season.csv`) |
| Fig. S7, Tables S13, S14 | Two-state HMM baseline | `03_model/18_hmm_baseline.py`, `06_figures/11_plot_hmm.py` (`hmm_pr_auc.csv`, `hmm_partial_days.csv`, `hmm_agreement_bins.csv`) |
