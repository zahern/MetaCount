Washington Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 37080 |
| Train rows | 22248 |
| Validation rows | 7416 |
| Test rows | 7416 |
| Search iterations | 8 |
| Candidate profile | expanded |
| Family | nb |
| AADT column | monthly_AADT |
| Offset used | yes |
| Upper candidate count | 21 |
| Lower candidate count | 13 |
| Selected upper vars | DX32, SNOW |
| Selected lower vars | DX32, paved_shoulder |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 22248 | 0.06288205681409564 | 0.06317505989802454 | 0.2665469228047496 | 0.10865666090004533 | 0.0002930030839289038 | 0.28996096989541215 | 0.0836905349184951 | 6777.344003010335 |
| validation (selection fit) | 7416 | 0.06351132686084142 | 0.06150857815298744 | 0.2663641140956653 | 0.1076331595688744 | -0.002002748707853989 | 0.30247373810995304 | 0.09138906319181739 | 2233.665114909688 |
| train+validation (final fit) | 29664 | 0.06303937432578209 | 0.06322606439698632 | 0.26648305830470337 | 0.10876361786490453 | 0.00018669007120422342 | 0.2931778846287431 | 0.08575096454400688 | 9009.420158971738 |
| test (held out) | 7416 | 0.05690399137001079 | 0.06307323014399578 | 0.24891952357118052 | 0.10462436905224412 | 0.006169238773984985 | 0.24255921744428785 | 0.05058873293538013 | 2132.2734692158283 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (-0.000535, 0.00616] | 742 | 0.0013 | 0.0036 |
| (0.00616, 0.0119] | 742 | 0.0108 | 0.0091 |
| (0.0119, 0.0183] | 741 | 0.0135 | 0.0149 |
| (0.0183, 0.0264] | 742 | 0.0377 | 0.0222 |
| (0.0264, 0.0358] | 741 | 0.0256 | 0.031 |
| (0.0358, 0.0478] | 742 | 0.0377 | 0.0415 |
| (0.0478, 0.0653] | 741 | 0.054 | 0.0561 |
| (0.0653, 0.0925] | 742 | 0.0728 | 0.0774 |
| (0.0925, 0.146] | 741 | 0.1026 | 0.1144 |
| (0.146, 1.313] | 742 | 0.2129 | 0.2605 |

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