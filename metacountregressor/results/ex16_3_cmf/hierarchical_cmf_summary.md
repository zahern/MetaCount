Ex-16-3 Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 275 |
| Train rows | 165 |
| Validation rows | 55 |
| Test rows | 55 |
| Search iterations | 120 |
| Candidate profile | expanded |
| Family | nb |
| AADT column | AADT |
| Offset used | yes |
| Enforce AADT increase | yes |
| Min AADT elasticity | 0.0 |
| Allow nonmonotonic fallback | yes |
| Upper candidate count | 35 |
| Lower candidate count | 19 |
| Selected upper vars | LENGTH, MIMEDSH, SLOPE |
| Selected lower vars | ACCESS, ADTLANE, FRICTION, HISNOW, URB |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 165 | 17.09090909090909 | 17.090909090909125 | 67.90300752959484 | 30.95809191762048 | 3.169451233451114e-14 | 0.010638001407709891 | -9.359042620471728 | 47747.386756114916 |
| validation (selection fit) | 55 | 19.12727272727273 | 14.174490178423788 | 65.22850520105456 | 31.346886885187384 | -4.952782548848939 | -0.08058403329432114 | -7.667064388669878 | 19193.22044392642 |
| train+validation (final fit) | 220 | 17.6 | 17.599999999999984 | 70.1074089161885 | 32.13389250510162 | -1.2402200478721748e-14 | -0.01733401357532591 | -9.747248189963571 | 64789.72256214435 |
| test (held out) | 55 | 13.927272727272728 | 69.28438268320143 | 442.54038220317744 | 80.15436683479176 | 55.3571099559287 | 0.06975071735587121 | -443.5496808490676 | 18898.01947103676 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (-0.0009568000000000001, 0.000194] | 6 | 6.5 | 0.0001 |
| (0.000194, 0.000546] | 5 | 12.2 | 0.0003 |
| (0.000546, 0.000974] | 6 | 11.5 | 0.0007 |
| (0.000974, 0.00132] | 5 | 25.6 | 0.0011 |
| (0.00132, 0.00212] | 6 | 38.3333 | 0.0018 |
| (0.00212, 0.0194] | 5 | 16.0 | 0.0039 |
| (0.0194, 0.146] | 5 | 3.6 | 0.0907 |
| (0.146, 2.102] | 6 | 9.5 | 0.7629 |
| (2.102, 21.347] | 5 | 4.8 | 9.0871 |
| (21.347, 3285.368] | 6 | 10.0 | 626.6887 |

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