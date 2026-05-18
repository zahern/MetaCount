Ex-16-3 Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 275 |
| Train rows | 165 |
| Validation rows | 55 |
| Test rows | 55 |
| Search iterations | 2100 |
| Search method | harmony |
| Harmony HMS | 12 |
| Harmony HMCR | 0.9 |
| Harmony PAR | 0.35 |
| Candidate profile | expanded |
| Final family | nb |
| Search families | nb, poisson |
| Benchmark upper vars (literature) | LOWPRE, GBRPM, FRICTION, EXPOSE, INTPM, CPM, HISNOW |
| AADT column | AADT |
| Offset used | yes |
| Enforce AADT increase | yes |
| Min AADT elasticity | 0.0 |
| Allow nonmonotonic fallback | yes |
| Pareto benchmark dominance required | yes |
| Final benchmark dominance required | yes |
| Selection benchmark BIC (train) | 1195.2218 |
| Selection benchmark RMSE (validation) | 17.63176 |
| Selected by Pareto iteration | 1278 |
| Upper candidate count | 29 |
| Lower candidate count | 15 |
| Selected upper vars | CURVES, GBRPM, MEDWIDTH, MIGRADE, SLOPE, SPEED, WIDTH |
| Selected lower vars | AVEPRE, MXMEDSH, SLOPE |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 165.00 | 17.0909 | 18.5380 | 15.5309 | 8.0752 | 1.4471 | 0.745937 | 0.458081 | 1,263.84 |
| validation (selection fit) | 55.0000 | 19.1273 | 19.3096 | 14.2129 | 9.9172 | 0.182314 | 0.772304 | 0.588507 | 497.71 |
| train+validation (final fit) | 220.00 | 17.6000 | 18.8478 | 14.8340 | 8.3427 | 1.2478 | 0.761984 | 0.518847 | 1,703.28 |
| test (held out) | 55.0000 | 13.9273 | 16.1600 | 11.6113 | 7.1498 | 2.2327 | 0.851983 | 0.693959 | 289.96 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (0.959, 2.276] | 6.0000 | 2.8333 | 1.6420 |
| (2.276, 3.357] | 5.0000 | 1.8000 | 2.9194 |
| (3.357, 4.171] | 6.0000 | 3.0000 | 3.6909 |
| (4.171, 5.343] | 5.0000 | 2.2000 | 4.7222 |
| (5.343, 8.546] | 6.0000 | 7.3333 | 6.9535 |
| (8.546, 12.514] | 5.0000 | 9.4000 | 10.0820 |
| (12.514, 14.222] | 5.0000 | 16.0000 | 13.1792 |
| (14.222, 24.95] | 6.0000 | 12.8333 | 18.2381 |
| (24.95, 41.969] | 5.0000 | 36.8000 | 31.0209 |
| (41.969, 116.965] | 6.0000 | 46.5000 | 66.0061 |

Outputs
- search_history_full.csv
- search_history_top25.md / .csv
- pareto_selection_summary.md / .csv
- harmony_search_summary.md / .csv
- refinement_phase_traces.png
- refinement_convergence_harmony.png
- refinement_convergence_sa.png
- search_phase_comparison.md / .csv / .png
- refinement_convergence.png
- model_settings.md
- validation_metrics.md / .csv
- test_calibration_deciles.md / .csv
- aadt_monotonicity_diagnostics.md / .csv
- coefficients_standardized_and_original.md / .csv
- standardization_reference.csv
- final_model_spec.json
- final_model_fit.pkl
- selection_model_fit.pkl
- validation_observed_vs_predicted.png
- test_observed_vs_predicted.png
- aadt_crash_risk_sensitivity.png
- curve_crash_risk_sensitivity.png (if curve selected)
- curve_cmf_sensitivity.png (if curve selected)
- binary_toggle_cmf_effects.png (if binary vars selected)
- hierarchical_cmf_dashboard.html
- search_convergence.html
- aadt_obs_pred.html
- random_params_summary.json (if ExperimentBuilder available)
- literature_vs_proposed_coefficients.md / .csv
- literature_benchmark_reference.md
- benchmark_metacount_summary.md
- benchmark_metacount_coefficients.md / .csv