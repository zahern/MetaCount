"""
Smoke test for continuous-DE warm starts across core model modes.

Runs side-by-side fits with and without continuous DE warm-start for:
1) Hierarchical CMF model
2) Latent-class count model
3) Random-parameter count model

Usage:
    python test_continuous_de_warmstart.py
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp

from experiment_package import ExperimentBuilder
from cmf_package import CMFExperimentBuilder
from main_hpc_lc_patch import mixed_model_loglik


def _loglik_from_fit(fit_result: dict) -> float:
    params = np.asarray(fit_result["result"].params)
    data = fit_result["data"]
    spec = fit_result["spec"]
    return -float(mixed_model_loglik(jnp.array(params), data, spec))


def _fit_with_toggle(builder, manual_spec, model="nb", R=300, use_de=True, **extra_fit_kwargs):
    de_maxiter = extra_fit_kwargs.pop("de_maxiter", 80)
    de_popsize = extra_fit_kwargs.pop("de_popsize", 8)
    de_rel_span = extra_fit_kwargs.pop("de_rel_span", 2.0)
    de_abs_span = extra_fit_kwargs.pop("de_abs_span", 1.0)
    de_seed = extra_fit_kwargs.pop("de_seed", 11)
    return builder.fit_manual_model(
        manual_spec=manual_spec,
        model=model,
        R=R,
        print_report=False,
        use_prefit_start=not (not use_de and extra_fit_kwargs.pop("force_random_no_de", False)),
        continuous_de_warm_start=use_de,
        de_maxiter=de_maxiter,
        de_popsize=de_popsize,
        de_rel_span=de_rel_span,
        de_abs_span=de_abs_span,
        de_seed=de_seed,
        **extra_fit_kwargs,
    )


def _print_de_report(label: str, fit_result: dict):
    report = fit_result.get("de_warm_start_report", {}) or {}
    print(f"{label} DE report:")
    print(f"  enabled: {report.get('enabled')}")

    single = report.get("single_class")
    if single is not None:
        start_obj = single.get("start_obj")
        final_obj = single.get("final_obj")
        start_ll = (-start_obj) if start_obj is not None else None
        final_ll = (-final_obj) if final_obj is not None else None
        print(
            "  single-class: "
            f"ran={single.get('ran')} accepted={single.get('accepted')} "
            f"start_obj={single.get('start_obj')} de_obj={single.get('de_obj')} "
            f"delta_obj={single.get('delta_obj')} final_obj={single.get('final_obj')} "
            f"start_ll={start_ll} final_ll={final_ll} reason={single.get('reason')}"
        )

    lc_seed = report.get("latent_class_seed")
    if lc_seed is not None:
        print(
            "  latent-class seed: "
            f"attempt={lc_seed.get('attempt')} noise={lc_seed.get('noise_scale')} "
            f"ran={lc_seed.get('ran')} accepted={lc_seed.get('accepted')} "
            f"start_obj={lc_seed.get('start_obj')} de_obj={lc_seed.get('de_obj')} "
            f"delta_obj={lc_seed.get('delta_obj')}"
        )


def _print_three_stage_ll(no_de_fit: dict, de_fit: dict):
    no_de_report = no_de_fit.get("de_warm_start_report", {}) or {}
    de_report = de_fit.get("de_warm_start_report", {}) or {}
    no_de_rep = no_de_report.get("single_class", {}) or {}
    de_rep = de_report.get("single_class", {}) or {}

    no_de_random_obj = no_de_rep.get("start_obj")
    de_seed_obj = de_rep.get("de_obj")
    if de_seed_obj is None:
        de_seed_obj = de_rep.get("start_obj")
    de_final_obj = de_rep.get("final_obj")

    # In latent random-start mode, single-class diagnostics are intentionally empty.
    if no_de_random_obj is None or de_seed_obj is None or de_final_obj is None:
        no_lc = no_de_report.get("latent_class_seed") or {}
        yes_lc = de_report.get("latent_class_seed") or {}
        no_de_random_obj = no_de_random_obj if no_de_random_obj is not None else no_lc.get("init_obj")
        de_seed_obj = de_seed_obj if de_seed_obj is not None else (yes_lc.get("seed_obj") or yes_lc.get("de_obj") or yes_lc.get("init_obj"))
        de_final_obj = de_final_obj if de_final_obj is not None else yes_lc.get("final_obj")

    no_de_random_ll = (-no_de_random_obj) if no_de_random_obj is not None else None
    de_seed_ll = (-de_seed_obj) if de_seed_obj is not None else None
    de_final_ll = (-de_final_obj) if de_final_obj is not None else None

    print("  Staged LL (requested order):")
    print(f"    1) No-DE (random start): {no_de_random_ll}")
    print(f"    2) DE-only (before optimizer): {de_seed_ll}")
    print(f"    3) DE + Optimize (final fit): {de_final_ll}")


def _print_model_variables(builder, label: str, fit_result: dict):
    spec = fit_result["spec"]
    fixed_terms = list(getattr(spec, "fixed_names", []))

    # CMF-style naming convention: local/lower-level interaction terms are prefixed.
    lower_local_terms = [t for t in fixed_terms if str(t).startswith("__cmf_local__")]
    upper_terms = [t for t in fixed_terms if t not in lower_local_terms]

    print(f"\n{label} Variables:")
    print(f"  fixed_terms      : {fixed_terms}")
    print(f"  upper_terms      : {upper_terms}")
    print(f"  lower_local_terms: {lower_local_terms}")
    print(f"  random_ind_terms : {list(getattr(spec, 'random_ind_names', []))}")
    print(f"  random_cor_terms : {list(getattr(spec, 'random_cor_names', []))}")
    print(f"  grouped_terms    : {list(getattr(spec, 'grouped_names', []))}")
    print(f"  membership_terms : {list(getattr(spec, 'membership_names', []))}")
    print(f"  latent_classes   : {getattr(spec, 'latent_classes', None)}")
    print(f"  model            : {getattr(spec, 'model', None)}")
    print(f"  total_params     : {len(np.asarray(fit_result['result'].params))}")

    print(f"\n{label} Coefficients:")
    coef_df = builder.print_coefficients(fit_result)

    disp_rows = coef_df[coef_df["Parameter"].astype(str).str.startswith("Dispersion")]
    if len(disp_rows) > 0:
        print(f"{label} Dispersion (interpretable scale):")
        print("  alpha = exp(raw_dispersion)")
        for _, row in disp_rows.iterrows():
            raw_val = float(row["Estimate"])
            alpha_val = float(np.exp(raw_val))
            print(f"  {row['Parameter']}: raw={raw_val:+.6f}, alpha={alpha_val:.6f}")


def make_synthetic_count_df(n_sites=120, t_per_site=3, seed=13):
    rng = np.random.default_rng(seed)
    rows = []

    for site in range(n_sites):
        aadt = rng.uniform(4000.0, 45000.0)
        width = rng.normal(11.0, 1.1)
        curves = rng.uniform(0.0, 12.0)
        access = rng.poisson(2.0)
        class_shift = rng.normal(0.0, 0.35)

        # Site-level random slope to make random-parameter tests meaningful.
        beta_curves_i = 0.05 + rng.normal(0.0, 0.03)

        for period in range(t_per_site):
            eta = (
                -2.6
                + 0.55 * np.log(aadt)
                + beta_curves_i * curves
                - 0.04 * width
                + 0.08 * access
                + class_shift
            )
            mu = float(np.exp(np.clip(eta, -6.0, 6.5)))
            y = rng.poisson(mu)

            rows.append(
                {
                    "SITE_ID": site,
                    "YEAR": period,
                    "CRASH": y,
                    "AADT": aadt,
                    "LENGTH": 1.0,
                    "CURVES": curves,
                    "WIDTH": width,
                    "ACCESS": access,
                    "URBAN": int(aadt > 22000.0),
                }
            )

    return pd.DataFrame(rows)


def run_hierarchical_cmf(df):
    print("\n=== Hierarchical CMF ===")
    cmf = CMFExperimentBuilder(
        df=df,
        y_col="CRASH",
        aadt_col="AADT",
        baseline_vars=["WIDTH", "ACCESS"],
        local_vars=["CURVES"],
    )

    spec = cmf.make_manual_cmf_spec(
        baseline_fixed=["WIDTH", "ACCESS"],
        local_fixed=["CURVES"],
        dispersion=1,
        latent_classes=1,
    )

    # CMF terms are encoded into transformed columns in the manual spec.
    term_map = cmf._cmf_term_map()
    df_cmf = cmf.df.copy()
    log_aadt_col = term_map[cmf.aadt_col]
    if log_aadt_col not in df_cmf.columns:
        df_cmf[log_aadt_col] = np.log(df_cmf[cmf.aadt_col].astype(float))
    for var in cmf.local_vars:
        interaction_col = term_map[var]
        if interaction_col not in df_cmf.columns:
            df_cmf[interaction_col] = df_cmf[var].astype(float) * df_cmf[log_aadt_col]

    # Route through ExperimentBuilder to test the shared fit path.
    eb = ExperimentBuilder(
        df=df_cmf,
        id_col="SITE_ID",
        y_col="CRASH",
        offset_col="LENGTH",
    )

    no_de = _fit_with_toggle(
        eb,
        spec,
        model="nb",
        R=30,
        use_de=False,
        force_random_no_de=True,
        lower_level_param_bounds=(-1.0, 1.0),
    )
    yes_de = _fit_with_toggle(
        eb,
        spec,
        model="nb",
        R=30,
        use_de=True,
        lower_level_param_bounds=(-1.0, 1.0),
    )

    ll0 = _loglik_from_fit(no_de)
    ll1 = _loglik_from_fit(yes_de)
    print(f"Final No-DE LL : {ll0:.4f}")
    print(f"Final DE-WS LL : {ll1:.4f}")
    print(f"Final Delta LL : {ll1 - ll0:+.4f}")
    _print_three_stage_ll(no_de, yes_de)
    _print_de_report("No-DE", no_de)
    _print_de_report("DE-WS", yes_de)
    _print_model_variables(eb, "Hierarchical No-DE", no_de)
    _print_model_variables(eb, "Hierarchical DE-WS", yes_de)


def run_latent_class(df):
    print("\n=== Latent-Class Count ===")
    eb = ExperimentBuilder(df=df, id_col="SITE_ID", y_col="CRASH", offset_col="LENGTH")

    spec = eb.make_manual_spec(
        fixed_terms=["AADT", "CURVES", "WIDTH", "ACCESS"],
        membership_terms=["URBAN", "WIDTH"],
        dispersion=1,
        latent_classes=2,
    )

    no_de = _fit_with_toggle(
        eb,
        spec,
        model="nb",
        R=30,
        use_de=False,
        force_random_no_de=True,
        latent_fast_mode=True,
        latent_random_start=True,
    )
    yes_de = _fit_with_toggle(
        eb,
        spec,
        model="nb",
        R=30,
        use_de=True,
        latent_fast_mode=True,
        latent_random_start=True,
        de_maxiter=12,
        de_popsize=6,
    )

    ll0 = _loglik_from_fit(no_de)
    ll1 = _loglik_from_fit(yes_de)
    print(f"Final No-DE LL : {ll0:.4f}")
    print(f"Final DE-WS LL : {ll1:.4f}")
    print(f"Final Delta LL : {ll1 - ll0:+.4f}")
    _print_three_stage_ll(no_de, yes_de)
    _print_de_report("No-DE", no_de)
    _print_de_report("DE-WS", yes_de)
    _print_model_variables(eb, "Latent-Class No-DE", no_de)
    _print_model_variables(eb, "Latent-Class DE-WS", yes_de)


def run_random_parameter_count(df):
    print("\n=== Random-Parameter Count ===")
    eb = ExperimentBuilder(df=df, id_col="SITE_ID", y_col="CRASH", offset_col="LENGTH")

    spec = eb.make_manual_spec(
        fixed_terms=["AADT", "WIDTH", "ACCESS"],
        rdm_terms=["CURVES:normal"],
        dispersion=1,
        latent_classes=1,
    )

    no_de = _fit_with_toggle(eb, spec, model="nb", R=35, use_de=False, force_random_no_de=True)
    yes_de = _fit_with_toggle(eb, spec, model="nb", R=35, use_de=True)

    ll0 = _loglik_from_fit(no_de)
    ll1 = _loglik_from_fit(yes_de)
    print(f"Final No-DE LL : {ll0:.4f}")
    print(f"Final DE-WS LL : {ll1:.4f}")
    print(f"Final Delta LL : {ll1 - ll0:+.4f}")
    _print_three_stage_ll(no_de, yes_de)
    _print_de_report("No-DE", no_de)
    _print_de_report("DE-WS", yes_de)


def main():
    df = make_synthetic_count_df()
    print(f"Data shape: {df.shape}")

    run_hierarchical_cmf(df)
    run_latent_class(df)
    run_random_parameter_count(df)


if __name__ == "__main__":
    main()
