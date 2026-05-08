Washington Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 37080 |
| Train rows | 22248 |
| Validation rows | 7416 |
| Test rows | 7416 |
| Search iterations | 200 |
| Candidate profile | expanded |
| Family | nb |
| AADT column | monthly_AADT |
| Offset used | yes |
| Upper candidate count | 21 |
| Lower candidate count | 13 |
| Selected upper vars | DP01, DP10, SNOW, TMAX, segment_length, speed |
| Selected lower vars | DP01, DX32, left_shoulder_width, speed |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 22248 | 0.06288205681409564 | 0.06284490620877273 | 0.26513668715898553 | 0.10814751356312195 | -3.715060532291373e-05 | 0.30561580138474875 | 0.09336083134900386 | 6698.145790652445 |
| validation (selection fit) | 7416 | 0.06351132686084142 | 0.06108252614782812 | 0.26493823422778595 | 0.10693817901151123 | -0.0024288007130133086 | 0.3200695931160139 | 0.10109083832781518 | 2192.715986741655 |
| train+validation (final fit) | 29664 | 0.06303937432578209 | 0.06298507442633487 | 0.2649704744846094 | 0.10803753294597739 | -5.429989944722233e-05 | 0.31005635889182653 | 0.09610024251548921 | 8886.66112306519 |
| test (held out) | 7416 | 0.05690399137001079 | 0.06279221047492099 | 0.24756526393076744 | 0.10405232147076454 | 0.005888219104910194 | 0.2573580102338499 | 0.0608912736886319 | 2118.829384176289 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (-0.000613, 0.00597] | 742 | 0.0 | 0.0036 |
| (0.00597, 0.0118] | 742 | 0.0081 | 0.0088 |
| (0.0118, 0.018] | 741 | 0.0175 | 0.0149 |
| (0.018, 0.0257] | 742 | 0.0283 | 0.0219 |
| (0.0257, 0.0351] | 741 | 0.0378 | 0.0303 |
| (0.0351, 0.0481] | 742 | 0.0418 | 0.0413 |
| (0.0481, 0.0655] | 741 | 0.0553 | 0.0561 |
| (0.0655, 0.0917] | 742 | 0.0499 | 0.0775 |
| (0.0917, 0.147] | 741 | 0.1188 | 0.1151 |
| (0.147, 0.961] | 742 | 0.2116 | 0.2586 |

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