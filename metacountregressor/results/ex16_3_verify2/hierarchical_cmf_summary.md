Ex-16-3 Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 275 |
| Train rows | 165 |
| Validation rows | 55 |
| Test rows | 55 |
| Search iterations | 5 |
| Candidate profile | core |
| Family | nb |
| AADT column | AADT |
| Offset used | yes |
| Enforce AADT increase | yes |
| Min AADT elasticity | 0.0 |
| Allow nonmonotonic fallback | yes |
| Upper candidate count | 25 |
| Lower candidate count | 8 |
| Selected upper vars | LANES, WIDTH |
| Selected lower vars | SPEED |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 165 | 17.09090909090909 | 18.91842797447749 | 16.29559872263169 | 8.180274651958307 | 1.827518883568401 | 0.7446796680768092 | 0.4034013824481534 | 1316.62339568503 |
| validation (selection fit) | 55 | 19.12727272727273 | 17.72703322332594 | 17.703780292736674 | 11.49180977876859 | -1.4002395039467879 | 0.6173586102562773 | 0.3615466160562236 | 652.6977083589481 |
| train+validation (final fit) | 220 | 17.6 | 19.694395127548358 | 17.87005216691948 | 9.314082708556974 | 2.094395127548355 | 0.7139812224398124 | 0.3017336941358041 | 1995.7728036342628 |
| test (held out) | 55 | 13.927272727272728 | 14.997763915781647 | 13.999388248599908 | 7.215803710950454 | 1.0704911885089208 | 0.7523567516528016 | 0.5551305266198465 | 299.8412902456061 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (0.811, 2.772] | 6 | 2.3333 | 1.694 |
| (2.772, 3.467] | 5 | 3.0 | 3.1151 |
| (3.467, 4.139] | 6 | 2.5 | 3.6334 |
| (4.139, 5.358] | 5 | 2.2 | 4.7381 |
| (5.358, 7.951] | 6 | 7.1667 | 6.7827 |
| (7.951, 10.828] | 5 | 10.8 | 9.3773 |
| (10.828, 14.632] | 5 | 10.2 | 12.691 |
| (14.632, 20.406] | 6 | 17.8333 | 16.9958 |
| (20.406, 37.364] | 5 | 28.6 | 29.3864 |
| (37.364, 71.114] | 6 | 52.1667 | 58.9505 |

Outputs
- search_history_full.csv
- search_history_top25.md / .csv
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