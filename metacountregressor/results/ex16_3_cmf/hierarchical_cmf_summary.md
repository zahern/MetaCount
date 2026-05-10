Ex-16-3 Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 275 |
| Train rows | 165 |
| Validation rows | 55 |
| Test rows | 55 |
| Search iterations | 20 |
| Candidate profile | expanded |
| Family | poisson |
| AADT column | AADT |
| Offset used | yes |
| Enforce AADT increase | yes |
| Min AADT elasticity | 0.0 |
| Allow nonmonotonic fallback | yes |
| Upper candidate count | 35 |
| Lower candidate count | 19 |
| Selected upper vars | ACCESS, FC, LNAADT, MIGRADE, TANGENT |
| Selected lower vars | HISNOW, MXMEDSH |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 165 | 17.09090909090909 | 17.090909090654133 | 69.06162319119625 | 32.295747523255315 | -2.549570305718314e-10 | -0.09839220558398915 | -9.715567157856997 | 69091.7484097862 |
| validation (selection fit) | 55 | 19.12727272727273 | 44.221330360521854 | 196.10496236102708 | 61.25960460076542 | 25.09405763324914 | -0.053157006391470585 | -77.3383351275517 | 24156.566093845486 |
| train+validation (final fit) | 220 | 17.6 | 17.600000000209068 | 72.6257824238091 | 33.289935272133064 | 2.0906596050322564e-10 | -0.09077229825515482 | -10.533233743494868 | 91503.91387585708 |
| test (held out) | 55 | 13.927272727272728 | 14.982349439658142 | 65.32513747946872 | 27.140583068652674 | 1.0550767123854154 | -0.030864181132915967 | -8.686684490867147 | 17089.327984369153 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (-0.000999333, 3.85e-05] | 6 | 28.8333 | 0.0 |
| (3.85e-05, 6.52e-05] | 5 | 38.2 | 0.0001 |
| (6.52e-05, 0.000435] | 6 | 20.6667 | 0.0002 |
| (0.000435, 0.00117] | 5 | 19.8 | 0.0008 |
| (0.00117, 0.00239] | 6 | 5.1667 | 0.0018 |
| (0.00239, 0.0154] | 5 | 3.0 | 0.0076 |
| (0.0154, 0.372] | 5 | 8.4 | 0.129 |
| (0.372, 1.202] | 6 | 6.3333 | 0.6308 |
| (1.202, 15.705] | 5 | 4.0 | 3.5557 |
| (15.705, 363.746] | 6 | 5.5 | 133.6278 |

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