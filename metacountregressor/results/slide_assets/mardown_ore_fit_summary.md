MetaCountRegressor fits for mardown_ore

Fit code used

```python
from metacountregressor import CMFExperimentBuilder, ExperimentBuilder, load_example16_3_model_data

df = load_example16_3_model_data().copy()
for col in ["CURVES", "WIDTH"]:
    df[f"{col}_Z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

trad_builder = ExperimentBuilder(
    df=df,
    id_col="ID",
    y_col="FREQ",
    offset_col="OFFSET",
    group_id_col="FC",
)
cmf_builder = CMFExperimentBuilder(
    df=df,
    y_col="FREQ",
    aadt_col="AADT",
    baseline_vars=["URB", "ACCESS", "GRADEBR", "CURVES_Z"],
    local_vars=["CURVES_Z", "WIDTH_Z"],
)

trad_baseline_spec = trad_builder.make_manual_spec(
    fixed_terms=["URB", "ACCESS", "GRADEBR", "CURVES", "LENGTH"],
    dispersion=1,
    latent_classes=1,
)
trad_random_spec = trad_builder.make_manual_spec(
    fixed_terms=["URB", "ACCESS", "GRADEBR", "LENGTH"],
    rdm_terms=["CURVES:lognormal"],
    dispersion=1,
    latent_classes=1,
)
cmf_baseline_spec = cmf_builder.make_manual_cmf_spec(
    baseline_fixed=["URB", "ACCESS", "GRADEBR"],
    local_fixed=["WIDTH_Z"],
    dispersion=1,
    latent_classes=1,
)
cmf_random_spec = cmf_builder.make_manual_cmf_spec(
    baseline_fixed=["URB", "ACCESS", "GRADEBR"],
    baseline_random=["CURVES_Z"],
    local_fixed=["WIDTH_Z"],
    dispersion=1,
    latent_classes=1,
)

fit_trad_baseline = trad_builder.fit_manual_model(manual_spec=trad_baseline_spec, model="nb", R=120)
fit_trad_random = trad_builder.fit_manual_model(manual_spec=trad_random_spec, model="nb", R=120)
fit_cmf_baseline = cmf_builder.fit_manual_cmf_model(
    id_col="ID", offset_col="OFFSET", group_id_col="FC",
    manual_spec=cmf_baseline_spec, model="nb", R=120
)
fit_cmf_random = cmf_builder.fit_manual_cmf_model(
    id_col="ID", offset_col="OFFSET", group_id_col="FC",
    manual_spec=cmf_random_spec, model="nb", R=120
)
```

Fit metrics

| Model | BIC | AIC | LogLik | RMSE | MAE | Corr | Observed mean | Predicted mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Traditional baseline NB | 1925.5156 | 1900.1982 | -943.0991 | 15.1918 | 8.4728 | 0.7562 | 16.8655 | 18.4728 |
| Hierarchical CMF baseline NB | 1928.4608 | 1903.1434 | -944.5717 | 16.566 | 8.81 | 0.7166 | 16.8655 | 18.4882 |
| Traditional random-parameter NB | 1931.2559 | 1902.3217 | -943.1609 | 15.2361 | 8.4908 | 0.7558 | 16.8655 | 18.4852 |
| Hierarchical CMF random-parameter NB | 1939.6943 | 1907.1434 | -944.5717 | 16.566 | 8.81 | 0.7166 | 16.8655 | 18.4883 |

Coefficient table

| Model | Parameter | Type | Estimate |
| --- | --- | --- | --- |
| Traditional baseline NB | __INTERCEPT__ | Fixed | 4.756231 |
| Traditional baseline NB | URB | Fixed | -0.050116 |
| Traditional baseline NB | ACCESS | Fixed | -0.213372 |
| Traditional baseline NB | GRADEBR | Fixed | 0.051251 |
| Traditional baseline NB | CURVES | Fixed | 0.007754 |
| Traditional baseline NB | LENGTH | Fixed | -0.100336 |
| Traditional baseline NB | Dispersion | Dispersion | -0.514935 |
| Traditional random-parameter NB | __INTERCEPT__ | Fixed | 4.769991 |
| Traditional random-parameter NB | URB | Fixed | -0.05029 |
| Traditional random-parameter NB | ACCESS | Fixed | -0.217246 |
| Traditional random-parameter NB | GRADEBR | Fixed | 0.051337 |
| Traditional random-parameter NB | LENGTH | Fixed | -0.09436 |
| Traditional random-parameter NB | CURVES (ind. mean) | Random-Ind | 0.006784 |
| Traditional random-parameter NB | CURVES (ind. SD) | Random-Ind | 0.011902 |
| Traditional random-parameter NB | Dispersion | Dispersion | -0.514166 |
| Hierarchical CMF baseline NB | __INTERCEPT__ | Fixed | 4.517889 |
| Hierarchical CMF baseline NB | URB | Fixed | 0.086082 |
| Hierarchical CMF baseline NB | ACCESS | Fixed | -0.154156 |
| Hierarchical CMF baseline NB | GRADEBR | Fixed | -0.003739 |
| Hierarchical CMF baseline NB | __cmf_log_aadt | Fixed | 0.006968 |
| Hierarchical CMF baseline NB | __cmf_local__WIDTH_Z | Fixed | -0.012824 |
| Hierarchical CMF baseline NB | Dispersion | Dispersion | -0.494786 |
| Hierarchical CMF random-parameter NB | __INTERCEPT__ | Fixed | 4.517878 |
| Hierarchical CMF random-parameter NB | URB | Fixed | 0.086081 |
| Hierarchical CMF random-parameter NB | ACCESS | Fixed | -0.154156 |
| Hierarchical CMF random-parameter NB | GRADEBR | Fixed | -0.003739 |
| Hierarchical CMF random-parameter NB | __cmf_log_aadt | Fixed | 0.006969 |
| Hierarchical CMF random-parameter NB | __cmf_local__WIDTH_Z | Fixed | -0.012824 |
| Hierarchical CMF random-parameter NB | CURVES_Z (ind. mean) | Random-Ind | 0.137426 |
| Hierarchical CMF random-parameter NB | CURVES_Z (ind. SD) | Random-Ind | 0.003386 |
| Hierarchical CMF random-parameter NB | Dispersion | Dispersion | -0.494786 |

Traditional NB interpretation

| Model | Parameter | Effect multiplier | Plain-English interpretation |
| --- | --- | --- | --- |
| Traditional baseline NB | URB | 0.9511 | Urban segments have about 4.9% lower expected crashes than non-urban segments, holding the other terms fixed. |
| Traditional baseline NB | ACCESS | 0.8079 | A one-step increase in the ACCESS coding is associated with about 19.2% fewer expected crashes, holding the other terms fixed. |
| Traditional baseline NB | GRADEBR | 1.0526 | A one-unit increase in the grade-break measure is associated with about 5.3% more expected crashes. |
| Traditional baseline NB | CURVES | 1.0078 | Each additional curve is associated with about 0.8% more expected crashes. |
| Traditional baseline NB | LENGTH | 0.9045 | This term implies about 9.5% fewer expected crashes per extra mile, but it should be read cautiously because the model already includes an offset. |
| Traditional random-parameter NB | CURVES (ind. mean) | 1.0068 | On average, one extra curve is still associated with about 0.7% more expected crashes, very close to the fixed-effect model. |
| Traditional random-parameter NB | CURVES (ind. SD) | 0.0119 | This random SD says the curve effect is allowed to vary across segments. Because the term is lognormal, the model keeps the curve effect non-negative while permitting some sites to be more curve-sensitive than others. |

Hierarchical CMF interpretation

| Model | Parameter | Effect multiplier | Plain-English interpretation |
| --- | --- | --- | --- |
| Hierarchical CMF baseline NB | URB | 1.0899 | Urban segments have about 9.0% higher baseline crash risk before the traffic-response block is applied. |
| Hierarchical CMF baseline NB | ACCESS | 0.8571 | A one-step increase in the ACCESS coding reduces baseline risk by about 14.3%. |
| Hierarchical CMF baseline NB | GRADEBR | 0.9963 | The fitted baseline effect of the grade-break measure is close to zero in this CMF model. |
| Hierarchical CMF baseline NB | __cmf_log_aadt | 0.007 | This is the base AADT elasticity at average width. In this fit it is near zero, so most traffic sensitivity is being expressed through the width interaction instead. |
| Hierarchical CMF baseline NB | __cmf_local__WIDTH_Z | 0.8737 | At mean AADT (37355), a one-SD wider segment (about 15.5 width units) has CMF 0.874, or about -12.6% fewer predicted crashes. At median AADT (23771), the same change gives CMF 0.879. That is roughly -0.9% per extra width unit near mean AADT. |
| Hierarchical CMF random-parameter NB | CURVES_Z (ind. mean) | 1.1473 | A one-SD increase in curves (about 2.86 extra curves) raises baseline risk by about 14.7% on average. That is roughly 4.9% per extra curve. |
| Hierarchical CMF random-parameter NB | CURVES_Z (ind. SD) | 0.0034 | This random SD is very small, so once curvature is placed in the CMF baseline block, the fitted curvature effect barely varies across segments in this run. |

Key takeaways

| Comparison | Takeaway |
| --- | --- |
| Traditional baseline vs Traditional random | Adding a random curve term slightly worsens BIC, AIC, RMSE, and MAE in this scoped fit, so the extra flexibility does not pay off on fit quality here. |
| CMF baseline vs CMF random | Adding random curvature to the CMF baseline block barely changes predictions but increases parameter count, so BIC gets worse. |
| Best predictive fit | Traditional baseline NB remains the best of the four models on BIC, RMSE, MAE, and correlation in this run. |
| Best interpretability | The CMF models still offer the clearest separation between baseline risk and traffic-response behavior, especially for width and curvature narratives. |
