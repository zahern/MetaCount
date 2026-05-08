Washington Hierarchical CMF Experiment

Configuration
| Setting | Value |
| --- | --- |
| Rows | 37080 |
| Train rows | 22248 |
| Validation rows | 7416 |
| Test rows | 7416 |
| Search iterations | 8 |
| Candidate profile | core |
| Family | nb |
| AADT column | monthly_AADT |
| Offset used | yes |
| Upper candidate count | 14 |
| Lower candidate count | 10 |
| Selected upper vars | SNOW |
| Selected lower vars | left_shoulder_width |

Validation and held-out crash-frequency metrics
| Split | n | obs_mean | pred_mean | rmse | mae | bias | corr | r2 | poisson_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train (selection fit) | 22248 | 0.06288205681409564 | 0.0631364254214734 | 0.26662477158094455 | 0.10893686519123352 | 0.00025436860737773794 | 0.28868699788139746 | 0.08315521463261255 | 6810.993087563034 |
| validation (selection fit) | 7416 | 0.06351132686084142 | 0.06180619210582479 | 0.26588974735692067 | 0.10801215013815785 | -0.0017051347550166357 | 0.3081855482263116 | 0.09462246429848209 | 2237.2485076334497 |
| train+validation (final fit) | 29664 | 0.06303937432578209 | 0.06317159485024745 | 0.2664428128890354 | 0.10898476072948202 | 0.00013222052446535272 | 0.2933943693460421 | 0.0860270913212291 | 9048.049345151638 |
| test (held out) | 7416 | 0.05690399137001079 | 0.0629411631564689 | 0.24953163082193633 | 0.10499748679479232 | 0.006037171786458113 | 0.2342451296513646 | 0.04591367939052382 | 2148.915900829259 |

Test calibration by predicted decile
| Predicted Bin | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- |
| (-0.00042, 0.00653] | 742 | 0.0013 | 0.0039 |
| (0.00653, 0.0124] | 742 | 0.0094 | 0.0095 |
| (0.0124, 0.0187] | 741 | 0.0189 | 0.0154 |
| (0.0187, 0.0267] | 742 | 0.0323 | 0.0226 |
| (0.0267, 0.0365] | 741 | 0.0283 | 0.0316 |
| (0.0365, 0.048] | 742 | 0.0458 | 0.042 |
| (0.048, 0.0654] | 741 | 0.0513 | 0.0562 |
| (0.0654, 0.0905] | 742 | 0.0701 | 0.077 |
| (0.0905, 0.145] | 741 | 0.1066 | 0.1144 |
| (0.145, 1.165] | 742 | 0.2049 | 0.2567 |

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