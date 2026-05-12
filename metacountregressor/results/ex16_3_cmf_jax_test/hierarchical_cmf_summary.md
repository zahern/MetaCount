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
| Upper candidate count | 32 |
| Lower candidate count | 9 |
| Selected upper vars | ACCESS, LENGTH, MXGRADE, MXMEDSH |
| Selected lower vars | DECLANES, SLOPE |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 165 | 17.09090909090909 | 4447993929690.939 | 57126131529807.734 | 4447993929681.959 | 4447993929673.849 | 0.17363046038505603 | -7.331810479023568e+24 | 1467837996773262.5 |
| validation (selection fit) | 55 | 19.12727272727273 | 83135.2053266522 | 426514.06429333985 | 83125.64074169133 | 83116.07805392494 | -0.022678435358336576 | -370564560.55468076 | 9139478.386114264 |
| train+validation (final fit) | 220 | 17.6 | 3301479068730.9033 | 48955794721468.71 | 3301479068721.9307 | 3301479068713.303 | 0.14662596074778683 | -5.240559492413772e+24 | 1452650790211512.5 |
| test (held out) | 55 | 13.927272727272728 | 2919101473807.5376 | 21648634116613.7 | 2919101473802.268 | 2919101473793.6104 | 0.08475719754724606 | -1.0638379790121718e+24 | 321101162114159.9 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (0.887, 3.183] | 6 | 5.1667 | 1.8999 |
| (3.183, 6.317] | 5 | 8.4 | 4.6645 |
| (6.317, 9.224] | 6 | 15.3333 | 7.6998 |
| (9.224, 16.592] | 5 | 18.8 | 13.8955 |
| (16.592, 40.542] | 6 | 33.5 | 28.2496 |
| (40.542, 190.139] | 5 | 24.0 | 118.339 |
| (190.139, 666.643] | 5 | 2.8 | 395.7703 |
| (666.643, 4371.107] | 6 | 14.3333 | 2106.7701 |
| (4371.107, 22058.647] | 5 | 4.0 | 10442.0764 |
| (22058.647, 160550567583344.6] | 6 | 11.0 | 26758430165278.855 |

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