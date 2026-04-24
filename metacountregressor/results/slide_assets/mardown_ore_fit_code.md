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
