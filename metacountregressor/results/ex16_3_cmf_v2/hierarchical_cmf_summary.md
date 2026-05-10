Ex-16-3 Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 275 |
| Train rows | 165 |
| Validation rows | 55 |
| Test rows | 55 |
| Search iterations | 80 |
| Candidate profile | core |
| Family | nb |
| AADT column | AADT |
| Offset used | yes |
| Enforce AADT increase | yes |
| Min AADT elasticity | 0.0 |
| Allow nonmonotonic fallback | yes |
| Upper candidate count | 32 |
| Lower candidate count | 9 |
| Selected upper vars | GRADEBR, INTPM, LOWPRE, MXGRDIFF, TRAIN |
| Selected lower vars | DECLANES, HISNOW, INCLANES |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 165 | 17.09090909090909 | 17.090909090909076 | 70.51250978065845 | 29.33402301177523 | -1.7569784011522477e-14 | 0.026821579457427497 | -10.17053427032341 | 45655.23847322379 |
| validation (selection fit) | 55 | 19.12727272727273 | 48.98201369674982 | 350.7709625443641 | 64.59037107960006 | 29.85474096947708 | -0.10864690927922974 | -249.63657062463494 | 20345.017313225046 |
| train+validation (final fit) | 220 | 17.6 | 17.600000000000012 | 77.22715092905902 | 30.540779303972936 | 1.085192541888153e-14 | 0.03191389133476818 | -12.040957263254032 | 61352.62753979585 |
| test (held out) | 55 | 13.927272727272728 | 5294.216259963618 | 39047.91706900551 | 5304.411560479603 | 5280.288987236345 | -0.08990156023451068 | -3461067.814070099 | 593914.2202038378 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (-0.000999048, 6.82e-06] | 6 | 14.3333 | 0.0 |
| (6.82e-06, 0.000119] | 5 | 7.0 | 0.0001 |
| (0.000119, 0.000378] | 6 | 23.0 | 0.0002 |
| (0.000378, 0.000919] | 5 | 4.2 | 0.0007 |
| (0.000919, 0.00222] | 6 | 26.3333 | 0.0012 |
| (0.00222, 0.00473] | 5 | 8.8 | 0.0036 |
| (0.00473, 0.0231] | 5 | 5.2 | 0.0088 |
| (0.0231, 0.413] | 6 | 16.5 | 0.1018 |
| (0.413, 3.655] | 5 | 9.8 | 0.9066 |
| (3.655, 289584.579] | 6 | 18.3333 | 48529.4459 |

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