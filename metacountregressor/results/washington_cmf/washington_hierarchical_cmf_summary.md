Washington Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 37080 |
| Train rows | 22248 |
| Validation rows | 7416 |
| Test rows | 7416 |
| Search iterations | 200 |
| Candidate profile | core |
| Family | nb |
| AADT column | monthly_AADT |
| Offset used | yes |
| Upper candidate count | 14 |
| Lower candidate count | 10 |
| Selected upper vars | DX32, PRCP, SNOW, dummy_winter, segment_length, speed |
| Selected lower vars | left_shoulder_width, speed |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 22248 | 0.06288205681409564 | 0.0628771082071682 | 0.2656979870028179 | 0.10844804607230066 | -4.948606927446377e-06 | 0.299210893456882 | 0.08951802029666911 | 6731.623885719776 |
| validation (selection fit) | 7416 | 0.06351132686084142 | 0.06109993344352244 | 0.2653400332442381 | 0.10724640714241476 | -0.0024113934173189784 | 0.31570833985626057 | 0.0983622423801146 | 2209.6089233843327 |
| train+validation (final fit) | 29664 | 0.06303937432578209 | 0.0629954062321486 | 0.2655052492630272 | 0.10837579181110486 | -4.396809363349842e-05 | 0.30408662214326976 | 0.09244798212379102 | 8938.408169304275 |
| test (held out) | 7416 | 0.05690399137001079 | 0.06274857717887573 | 0.2482077230337236 | 0.10437468139949521 | 0.005844585808864934 | 0.2485310443560165 | 0.056010768194633465 | 2124.4504054349513 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (-0.000565, 0.00627] | 742 | 0.0013 | 0.0037 |
| (0.00627, 0.0121] | 742 | 0.0108 | 0.0091 |
| (0.0121, 0.0182] | 741 | 0.0135 | 0.0152 |
| (0.0182, 0.0266] | 742 | 0.031 | 0.0224 |
| (0.0266, 0.0362] | 741 | 0.0283 | 0.0314 |
| (0.0362, 0.0481] | 742 | 0.0512 | 0.0418 |
| (0.0481, 0.066] | 741 | 0.0445 | 0.0565 |
| (0.066, 0.0926] | 742 | 0.0701 | 0.0783 |
| (0.0926, 0.143] | 741 | 0.1053 | 0.1133 |
| (0.143, 0.879] | 742 | 0.2129 | 0.2557 |

Outputs
- search_history_top25.md / .csv
- model_settings.md
- validation_metrics.md / .csv
- test_calibration_deciles.md / .csv
- coefficients_standardized_and_original.md / .csv
- standardization_reference.csv
- validation_observed_vs_predicted.png
- test_observed_vs_predicted.png
- curve_crash_risk_sensitivity.png (if curve selected)
- curve_cmf_sensitivity.png (if curve selected)
- binary_toggle_cmf_effects.png (if binary vars selected)
- washington_hierarchical_cmf_dashboard.html