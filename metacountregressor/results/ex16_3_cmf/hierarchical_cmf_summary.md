Ex-16-3 Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 275 |
| Train rows | 165 |
| Validation rows | 55 |
| Test rows | 55 |
| Search iterations | 600 |
| Candidate profile | expanded |
| Family | nb |
| AADT column | AADT |
| Offset used | yes |
| Enforce AADT increase | yes |
| Min AADT elasticity | 0.0 |
| Allow nonmonotonic fallback | yes |
| Upper candidate count | 32 |
| Lower candidate count | 16 |
| Selected upper vars | DOUBLE, MEDWIDTH, MIMEDSH, MXGRDIFF |
| Selected lower vars | AVEPRE |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 165 | 17.09090909090909 | 18.570679880495984 | 15.809146060859698 | 8.0141223465914 | 1.4797707895868906 | 0.7411459664134352 | 0.438488800011789 | 1204.0615243318587 |
| validation (selection fit) | 55 | 19.12727272727273 | 17.169843000455657 | 17.166629047870643 | 11.257458828742678 | -1.95742972681707 | 0.6438580108921568 | 0.39970155849655276 | 707.9576088853765 |
| train+validation (final fit) | 220 | 17.6 | 19.640174615607645 | 16.804382574029816 | 9.067436797558587 | 2.0401746156076426 | 0.7275148591669969 | 0.38253184274010343 | 1903.6446653267972 |
| test (held out) | 55 | 13.927272727272728 | 14.681071799672456 | 16.177170175962058 | 7.651411417772466 | 0.7537990723997294 | 0.666000993144132 | 0.4059546540320452 | 349.8683216080811 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (0.981, 2.339] | 6 | 1.8333 | 1.5555 |
| (2.339, 3.198] | 5 | 3.2 | 2.8041 |
| (3.198, 3.877] | 6 | 2.3333 | 3.4668 |
| (3.877, 6.134] | 5 | 4.0 | 5.2682 |
| (6.134, 7.67] | 6 | 5.5 | 6.8854 |
| (7.67, 10.65] | 5 | 7.6 | 9.4951 |
| (10.65, 12.529] | 5 | 14.2 | 11.7565 |
| (12.529, 18.789] | 6 | 16.6667 | 15.5852 |
| (18.789, 35.056] | 5 | 38.0 | 29.3412 |
| (35.056, 89.182] | 6 | 45.5 | 58.1959 |

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