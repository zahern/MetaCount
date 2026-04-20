Scoping CMF vs Traditional comparison (quick task run)

Quick-run configuration

| Setting | Value |
| --- | --- |
| Search iterations (SA) | 20 |
| Search draws | 60 |
| Search mode | micro |
| Fit draws | 80 |
| Traditional search best BIC | 1955.988766515855 |
| CMF search best BIC | 1967.9840576208653 |

Key insights (Observed vs Predicted + Validation)

- Best RMSE (manual fits): Traditional baseline NB (15.1918)
- Best MAE (manual fits): Traditional baseline NB (8.4728)
- Best observed-predicted correlation: Traditional baseline NB (0.7562)
- CMF random vs Traditional random RMSE: -8.73% (decline).

Model comparison metrics

| Stage | Family | Model | BIC | AIC | Log-Likelihood | Parameters | RMSE | MAE | Bias | R2 | Corr | Poisson Dev | Search Iterations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Manual fit | Traditional | Traditional baseline NB | 1925.5156349389833 | 1900.1982372553173 | -943.0991186276586 | 7.0 | 15.191798103161824 | 8.472775074786407 | 1.6073226543767867 | 0.49402532285355816 | 0.7561662118954349 | 2205.622024209303 | nan |
| Manual fit | CMF | CMF baseline NB | 1928.4607855557178 | 1903.1433878720518 | -944.5716939360259 | 7.0 | 16.566000895261872 | 8.810043058172562 | 1.6227952627092714 | 0.39834742075612783 | 0.7165702152105278 | 2313.838416226848 | nan |
| Manual fit | Traditional | Traditional random-parameter NB | 1931.2558945026274 | 1902.3217257212948 | -943.1608628606474 | 8.0 | 15.236126613899907 | 8.490834561043068 | 1.6197031144782228 | 0.49106822363913905 | 0.7558344006691472 | 2208.9142369199053 | nan |
| Manual fit | CMF | CMF random-parameter NB | 1939.6943277508367 | 1907.1433878718376 | -944.5716939359188 | 9.0 | 16.566011604040593 | 8.81004401850847 | 1.6227990004856434 | 0.3983466429019601 | 0.7165701596872254 | 2313.83901065413 | nan |
| Search (hierarchical micro-SA) | Traditional | Traditional search best (quick) | 1955.988766515855 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 5.0 |
| Search (hierarchical micro-SA) | CMF | CMF search best (quick) | 1967.9840576208653 | nan | nan | nan | nan | nan | nan | nan | nan | nan | 5.0 |

Observed vs Predicted summary (manual fits)

| Model | N | Observed Mean | Predicted Mean | Bias | RMSE | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| Traditional baseline NB | 275 | 16.8655 | 18.4728 | 1.6073 | 15.1918 | 8.4728 |
| Traditional random-parameter NB | 275 | 16.8655 | 18.4852 | 1.6197 | 15.2361 | 8.4908 |
| CMF baseline NB | 275 | 16.8655 | 18.4882 | 1.6228 | 16.566 | 8.81 |
| CMF random-parameter NB | 275 | 16.8655 | 18.4883 | 1.6228 | 16.566 | 8.81 |

Observed vs Predicted by predicted-decile (manual fits)

| Model | Predicted Decile | N | Observed Mean | Predicted Mean |
| --- | --- | --- | --- | --- |
| Traditional baseline NB | (0.774, 3.217] | 28 | 2.3929 | 2.1162 |
| Traditional baseline NB | (3.217, 4.988] | 27 | 4.0741 | 4.1959 |
| Traditional baseline NB | (4.988, 7.068] | 28 | 6.0714 | 6.0287 |
| Traditional baseline NB | (7.068, 9.321] | 27 | 8.9259 | 8.1174 |
| Traditional baseline NB | (9.321, 11.714] | 28 | 8.7857 | 10.5841 |
| Traditional baseline NB | (11.714, 15.202] | 27 | 17.6667 | 13.5854 |
| Traditional baseline NB | (15.202, 19.126] | 27 | 18.3704 | 16.6333 |
| Traditional baseline NB | (19.126, 25.667] | 28 | 21.8571 | 22.5129 |
| Traditional baseline NB | (25.667, 41.02] | 27 | 26.6667 | 31.3498 |
| Traditional baseline NB | (41.02, 197.779] | 28 | 53.5357 | 68.9441 |
| Traditional random-parameter NB | (0.773, 3.201] | 28 | 2.5357 | 2.1196 |
| Traditional random-parameter NB | (3.201, 4.974] | 27 | 3.9259 | 4.2143 |
| Traditional random-parameter NB | (4.974, 7.034] | 28 | 6.0357 | 6.0082 |
| Traditional random-parameter NB | (7.034, 9.365] | 27 | 9.2963 | 8.1157 |
| Traditional random-parameter NB | (9.365, 11.752] | 28 | 8.4643 | 10.6241 |
| Traditional random-parameter NB | (11.752, 15.09] | 27 | 18.2593 | 13.6072 |
| Traditional random-parameter NB | (15.09, 19.092] | 27 | 17.7778 | 16.6368 |
| Traditional random-parameter NB | (19.092, 25.824] | 28 | 21.8214 | 22.3577 |
| Traditional random-parameter NB | (25.824, 42.167] | 27 | 26.7037 | 31.2461 |
| Traditional random-parameter NB | (42.167, 200.157] | 28 | 53.5357 | 69.2575 |
| CMF baseline NB | (0.686, 3.052] | 28 | 2.2857 | 2.1071 |
| CMF baseline NB | (3.052, 5.226] | 27 | 3.963 | 4.1447 |
| CMF baseline NB | (5.226, 7.178] | 28 | 5.3571 | 6.1061 |
| CMF baseline NB | (7.178, 9.608] | 27 | 12.7037 | 8.195 |
| CMF baseline NB | (9.608, 12.085] | 28 | 9.4286 | 10.6212 |
| CMF baseline NB | (12.085, 14.714] | 27 | 13.2963 | 13.3591 |
| CMF baseline NB | (14.714, 19.797] | 27 | 19.4815 | 17.0732 |
| CMF baseline NB | (19.797, 25.935] | 28 | 19.3214 | 22.6804 |
| CMF baseline NB | (25.935, 40.935] | 27 | 30.037 | 31.8526 |
| CMF baseline NB | (40.935, 241.58] | 28 | 52.6071 | 68.1068 |
| CMF random-parameter NB | (0.686, 3.052] | 28 | 2.2857 | 2.1071 |
| CMF random-parameter NB | (3.052, 5.226] | 27 | 3.963 | 4.1447 |
| CMF random-parameter NB | (5.226, 7.178] | 28 | 5.3571 | 6.1061 |
| CMF random-parameter NB | (7.178, 9.608] | 27 | 12.7037 | 8.195 |
| CMF random-parameter NB | (9.608, 12.085] | 28 | 9.4286 | 10.6212 |
| CMF random-parameter NB | (12.085, 14.714] | 27 | 13.2963 | 13.3591 |
| CMF random-parameter NB | (14.714, 19.797] | 27 | 19.4815 | 17.0732 |
| CMF random-parameter NB | (19.797, 25.935] | 28 | 19.3214 | 22.6804 |
| CMF random-parameter NB | (25.935, 40.935] | 27 | 30.037 | 31.8526 |
| CMF random-parameter NB | (40.935, 241.58] | 28 | 52.6071 | 68.1068 |

Observed vs Predicted charts (manual fits)

![Observed vs Predicted Panel](results/slide_assets/obs_vs_pred_panel_manual_fits.png)

![Traditional baseline NB](results/slide_assets/obs_vs_pred_traditional_baseline_nb.png)
![Traditional random-parameter NB](results/slide_assets/obs_vs_pred_traditional_random_parameter_nb.png)
![CMF baseline NB](results/slide_assets/obs_vs_pred_cmf_baseline_nb.png)
![CMF random-parameter NB](results/slide_assets/obs_vs_pred_cmf_random_parameter_nb.png)