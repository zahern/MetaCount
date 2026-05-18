Ex-16-3 Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 275 |
| Train rows | 165 |
| Validation rows | 55 |
| Test rows | 55 |
| Search iterations | 20 |
| Candidate profile | core |
| Final family | nb |
| Search families | nb |
| AADT column | AADT |
| Offset used | yes |
| Enforce AADT increase | yes |
| Min AADT elasticity | 0.0 |
| Allow nonmonotonic fallback | yes |
| Pareto benchmark dominance required | no |
| Final benchmark dominance required | no |
| Selection benchmark BIC (train) | 1165.8564 |
| Selection benchmark RMSE (validation) | 17.191314 |
| Selected by Pareto iteration | 18 |
| Upper candidate count | 26 |
| Lower candidate count | 8 |
| Selected upper vars | MIMEDSH, MXMEDSH, WIDTH |
| Selected lower vars | FRICTION |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 165 | 17.09090909090909 | 18.695797111761742 | 15.50598006871284 | 8.313871195921998 | 1.6048880208526521 | 0.7484479140538143 | 0.4598180825432865 | 1179.3944155551212 |
| validation (selection fit) | 55 | 19.12727272727273 | 17.846393914966658 | 16.984967074240224 | 11.351652306468038 | -1.2808788123060655 | 0.6625175036979235 | 0.41233937963759115 | 745.3550461736032 |
| train+validation (final fit) | 220 | 17.6 | 19.71789718064639 | 17.099819608461765 | 9.296381134807937 | 2.1178971806463904 | 0.7232234172088001 | 0.3606296341894336 | 1913.8680518985102 |
| test (held out) | 55 | 13.927272727272728 | 16.467822584959844 | 19.26342722284149 | 8.861205823281146 | 2.540549857687117 | 0.6178061896257089 | 0.15767129172873107 | 418.4583851444318 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (0.612, 2.376] | 6 | 3.0 | 1.6928 |
| (2.376, 3.197] | 5 | 1.0 | 2.8217 |
| (3.197, 3.724] | 6 | 2.8333 | 3.5189 |
| (3.724, 5.645] | 5 | 4.2 | 4.407 |
| (5.645, 7.717] | 6 | 7.6667 | 6.8638 |
| (7.717, 11.819] | 5 | 6.8 | 9.7371 |
| (11.819, 12.939] | 5 | 12.2 | 12.4079 |
| (12.939, 22.028] | 6 | 16.8333 | 16.6802 |
| (22.028, 42.421] | 5 | 30.4 | 30.9016 |
| (42.421, 119.497] | 6 | 51.8333 | 71.97 |

Outputs
- search_history_full.csv
- search_history_top25.md / .csv
- pareto_selection_summary.md / .csv
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