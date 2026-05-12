Ex-16-3 Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 275 |
| Train rows | 165 |
| Validation rows | 55 |
| Test rows | 55 |
| Search iterations | 30 |
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
| train (selection fit) | 165 | 17.09090909090909 | 427188916350.08307 | 5485830847576.209 | 427188916341.6205 | 427188916332.9921 | 0.17363637508519422 | -6.76124111664636e+22 | 140972342372108.78 |
| validation (selection fit) | 55 | 19.12727272727273 | 110855.9985792675 | 581065.3787001639 | 110846.66969357329 | 110836.87130654023 | -0.021605162637772818 | -687776241.877129 | 12188719.480850616 |
| train+validation (final fit) | 220 | 17.6 | 3302619202542.662 | 48972702209087.64 | 3302619202533.6885 | 3302619202525.062 | 0.14662595988014088 | -5.24417990121729e+24 | 1453152449088686.8 |
| test (held out) | 55 | 13.927272727272728 | 2920089711878.32 | 21655963086352.633 | 2920089711873.052 | 2920089711864.3926 | 0.0847571975455773 | -1.0645584084019615e+24 | 321209868301946.1 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (0.887, 3.183] | 6 | 5.1667 | 1.8999 |
| (3.183, 6.317] | 5 | 8.4 | 4.6644 |
| (6.317, 9.224] | 6 | 15.3333 | 7.6997 |
| (9.224, 16.592] | 5 | 18.8 | 13.8956 |
| (16.592, 40.543] | 6 | 33.5 | 28.2495 |
| (40.543, 190.138] | 5 | 24.0 | 118.3395 |
| (190.138, 666.669] | 5 | 2.8 | 395.7512 |
| (666.669, 4371.358] | 6 | 14.3333 | 2106.8085 |
| (4371.358, 22057.953] | 5 | 4.0 | 10441.7059 |
| (22057.953, 160604920677634.44] | 6 | 11.0 | 26767489014261.316 |

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