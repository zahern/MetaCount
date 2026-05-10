Ex-16-3 Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 275 |
| Train rows | 165 |
| Validation rows | 55 |
| Test rows | 55 |
| Search iterations | 2 |
| Candidate profile | core |
| Family | nb |
| AADT column | AADT |
| Offset used | yes |
| Upper candidate count | 17 |
| Lower candidate count | 13 |
| Selected upper vars | LENGTH |
| Selected lower vars | CURVES |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 165 | 17.09090909090909 | 17.090909090909125 | 76.75245228722248 | 31.179166820081466 | 3.169451233451114e-14 | 0.02877112035035095 | -12.235066263830756 | 50252.97922647794 |
| validation (selection fit) | 55 | 19.12727272727273 | 15.722004161383893 | 70.4731352590157 | 33.25034090811363 | -3.4052685658888344 | -0.06890984015221538 | -9.11682777681535 | 19039.32842786857 |
| train+validation (final fit) | 220 | 17.6 | 17.6 | 82.59819337378966 | 32.536996495675865 | -4.134066826240583e-15 | 0.02362970505886998 | -13.917997915723134 | 69184.48025546048 |
| test (held out) | 55 | 13.927272727272728 | 15.08896221095784 | 66.37661453743671 | 26.417270231818268 | 1.1616894836851102 | 0.08313139353714015 | -9.001028974415066 | 14313.840529410603 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (-0.000851, 0.000272] | 6 | 49.1667 | 0.0002 |
| (0.000272, 0.000478] | 5 | 18.0 | 0.0003 |
| (0.000478, 0.000978] | 6 | 20.5 | 0.0008 |
| (0.000978, 0.00173] | 5 | 7.6 | 0.0013 |
| (0.00173, 0.00501] | 6 | 8.1667 | 0.0031 |
| (0.00501, 0.0223] | 5 | 2.4 | 0.0086 |
| (0.0223, 0.26] | 5 | 9.4 | 0.1172 |
| (0.26, 0.913] | 6 | 5.0 | 0.5165 |
| (0.913, 11.306] | 5 | 3.8 | 3.4807 |
| (11.306, 364.599] | 6 | 10.5 | 134.7881 |

Outputs
- search_history_top25.md / .csv
- model_settings.md
- validation_metrics.md / .csv
- test_calibration_deciles.md / .csv
- coefficients_standardized_and_original.md / .csv
- standardization_reference.csv
- validation_observed_vs_predicted.png
- test_observed_vs_predicted.png
- aadt_crash_risk_sensitivity.png
- curve_crash_risk_sensitivity.png (if curve selected)
- curve_cmf_sensitivity.png (if curve selected)
- binary_toggle_cmf_effects.png (if binary vars selected)
- hierarchical_cmf_dashboard.html