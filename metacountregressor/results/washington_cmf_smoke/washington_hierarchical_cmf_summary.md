Washington Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 37080 |
| Train rows | 22248 |
| Validation rows | 7416 |
| Test rows | 7416 |
| Search iterations | 8 |
| Family | nb |
| AADT column | monthly_AADT |
| Offset used | yes |
| Selected upper vars | DSND, DX32 |
| Selected lower vars | DP01, speed |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 22248 | 0.06288205681409564 | 0.06318853390957113 | 0.26605501758546946 | 0.1082927373140326 | 0.0003064770954754837 | 0.2956863599210018 | 0.08706946341403421 | 6767.224167603462 |
| validation (selection fit) | 7416 | 0.06351132686084142 | 0.061385853014538104 | 0.2660156457312863 | 0.10715790228528581 | -0.0021254738463033196 | 0.30635968065874836 | 0.09376487165298641 | 2228.3080381353548 |
| train+validation (final fit) | 29664 | 0.06303937432578209 | 0.06325722756780894 | 0.2659482245842883 | 0.10831756221274792 | 0.00021785324202684123 | 0.29940037402484393 | 0.08941709275099452 | 8993.43187763371 |
| test (held out) | 7416 | 0.05690399137001079 | 0.06282805004479082 | 0.2489302120571246 | 0.10423386339786404 | 0.00592405867478002 | 0.2443656928625986 | 0.05050719664840864 | 2130.727372455669 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (-0.00062, 0.00621] | 742 | 0.0013 | 0.0037 |
| (0.00621, 0.0119] | 742 | 0.0067 | 0.009 |
| (0.0119, 0.0179] | 741 | 0.0162 | 0.0148 |
| (0.0179, 0.0256] | 742 | 0.0256 | 0.0216 |
| (0.0256, 0.0348] | 741 | 0.0418 | 0.03 |
| (0.0348, 0.0465] | 742 | 0.0364 | 0.0403 |
| (0.0465, 0.0634] | 741 | 0.0486 | 0.0548 |
| (0.0634, 0.09] | 742 | 0.0768 | 0.0755 |
| (0.09, 0.144] | 741 | 0.1026 | 0.1127 |
| (0.144, 1.185] | 742 | 0.2129 | 0.2657 |

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