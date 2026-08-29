"""Compare the package CMF fit with its PyMC CMF compilation."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln


def _raw_jax_coefficients(fit: dict[str, Any]) -> pd.Series:
    spec = fit["spec"]
    names = list(spec.fixed_names)
    if not names or names[0] != "__INTERCEPT__":
        raise ValueError("The comparison requires a fixed-intercept CMF specification.")

    params = np.asarray(fit["result"].params, dtype=float)
    coefficients = params[:len(names)].copy()
    intercept = 0
    for index, name in enumerate(names):
        if name == "__INTERCEPT__":
            continue
        if name not in fit["data"].get("scaler", {}):
            continue
        mean, scale = fit["data"]["scaler"][name]
        coefficients[index] /= scale
        coefficients[intercept] -= params[index] * mean / scale

    return pd.Series(coefficients, index=names, dtype=float)


def _display_name(name: str) -> str:
    if name == "__INTERCEPT__":
        return "Intercept"
    if name == "__cmf_log_aadt":
        return "log(AADT)"
    if name.startswith("__cmf_local__"):
        return name.replace("__cmf_local__", "") + " x log(AADT)"
    return name


def _nbl_logpmf(y: np.ndarray, mu: np.ndarray, theta: float) -> np.ndarray:
    mean_multiplier = (theta ** 2 + theta - 1.0) / ((theta + 1.0) * (theta - 1.0) ** 2)
    shape = mu / mean_multiplier
    beta_argument = shape + theta
    return (
        gammaln(y + shape)
        - gammaln(shape)
        - gammaln(y + 1.0)
        + 2.0 * np.log(theta)
        - np.log(theta + 1.0)
        + gammaln(beta_argument)
        + gammaln(y + 1.0)
        - gammaln(beta_argument + y + 1.0)
        + np.log1p(digamma(beta_argument + y + 1.0) - digamma(beta_argument))
    )


def run_cmf_pymc_comparison(
    *,
    df: Optional[pd.DataFrame] = None,
    baseline_fixed: tuple[str, ...] = ("URB",),
    local_fixed: tuple[str, ...] = ("CURVES",),
    id_col: str = "ID",
    jax_R: int = 20,
    draws: int = 100,
    tune: int = 100,
    chains: int = 1,
    cores: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    import pymc as pm

    try:
        from .cmf_package import CMFExperimentBuilder
        from .sample_data import load_example16_3_model_data
    except ImportError:
        from cmf_package import CMFExperimentBuilder
        from sample_data import load_example16_3_model_data

    if df is None:
        df = load_example16_3_model_data()
    else:
        df = df.copy()

    builder = CMFExperimentBuilder(
        df=df,
        y_col="FREQ",
        aadt_col="AADT",
        baseline_vars=list(baseline_fixed),
        local_vars=list(local_fixed),
    )
    manual_spec = builder.make_manual_cmf_spec(
        baseline_fixed=list(baseline_fixed),
        local_fixed=list(local_fixed),
        dispersion=1,
    )
    jax_fit = builder.fit_manual_cmf_model(
        id_col=id_col,
        manual_spec=manual_spec,
        model="nb",
        R=jax_R,
    )
    jax_coefficients = _raw_jax_coefficients(jax_fit)
    pymc_terms = list(dict.fromkeys(manual_spec["fixed_terms"]))
    pymc_coefficient_names = ["__INTERCEPT__", *pymc_terms]
    pymc_initial_beta = jax_coefficients.reindex(pymc_coefficient_names).to_numpy()
    if not np.isfinite(pymc_initial_beta).all():
        raise ValueError("The JAX and PyMC CMF coefficient names do not match.")
    jax_predictions = np.asarray(jax_fit["predictions"], dtype=float).reshape(-1)
    y = pd.to_numeric(df["FREQ"], errors="raise").to_numpy(float)

    compiled = builder.build_bayesian_model(
        {"family": "cmf", "model_spec": manual_spec},
        priors={"coef_scale": 5.0, "nbl_theta_scale": 2.0},
        initvals={
            "beta": pymc_initial_beta,
            "nbl_theta_excess": 1.0,
        },
    )
    with compiled.model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=seed,
            progressbar=False,
            target_accept=0.9,
            init="adapt_diag",
            compute_convergence_checks=False,
        )

    posterior = idata.posterior
    beta_samples = np.asarray(posterior["beta"], dtype=float).reshape(-1, len(jax_coefficients))
    beta_mean = beta_samples.mean(axis=0)
    beta_sd = beta_samples.std(axis=0, ddof=1 if beta_samples.shape[0] > 1 else 0)
    pymc_coefficients = pd.Series(beta_mean, index=pymc_coefficient_names, dtype=float)
    pymc_coefficient_sd = pd.Series(beta_sd, index=pymc_coefficient_names, dtype=float)
    theta_samples = np.asarray(posterior["nbl_theta"], dtype=float).reshape(-1)
    theta_mean = float(theta_samples.mean())
    theta_sd = float(theta_samples.std(ddof=1 if theta_samples.size > 1 else 0))

    fixed_terms = pymc_terms
    frame = compiled.data
    x_columns = [np.ones(len(frame), dtype=float)]
    for name in fixed_terms:
        x_columns.append(pd.to_numeric(frame[name], errors="raise").to_numpy(float))
    x_matrix = np.column_stack(x_columns)
    pymc_predictions = np.exp(np.clip(x_matrix @ beta_mean, -30.0, 30.0))
    pymc_loglik = float(np.sum(_nbl_logpmf(y, pymc_predictions, theta_mean)))

    coefficient_rows = []
    for index, name in enumerate(jax_coefficients.index):
        coefficient_rows.append({
            "Parameter": _display_name(name),
            "JAX CMF NB2 MLE": float(jax_coefficients.iloc[index]),
            "PyMC CMF NBL mean": float(pymc_coefficients[name]),
            "PyMC CMF NBL SD": float(pymc_coefficient_sd[name]),
            "Difference": float(pymc_coefficients[name] - jax_coefficients.iloc[index]),
        })
    coefficient_rows.extend([
        {
            "Parameter": "NB2 alpha (not NBL theta)",
            "JAX CMF NB2 MLE": float(np.log1p(np.exp(np.asarray(jax_fit["result"].params)[-1]))),
            "PyMC CMF NBL mean": np.nan,
            "PyMC CMF NBL SD": np.nan,
            "Difference": np.nan,
        },
        {
            "Parameter": "NBL theta (not NB2 alpha)",
            "JAX CMF NB2 MLE": np.nan,
            "PyMC CMF NBL mean": theta_mean,
            "PyMC CMF NBL SD": theta_sd,
            "Difference": np.nan,
        },
    ])
    coefficients = pd.DataFrame(coefficient_rows)

    jax_summary = jax_fit["summary"]
    metrics = pd.DataFrame([
        {
            "Metric": "Log likelihood (model-specific)",
            "JAX CMF NB2 MLE": float(jax_summary["loglik"]),
            "PyMC CMF NBL posterior mean": pymc_loglik,
        },
        {
            "Metric": "RMSE (mean prediction)",
            "JAX CMF NB2 MLE": float(np.sqrt(np.mean((jax_predictions - y) ** 2))),
            "PyMC CMF NBL posterior mean": float(np.sqrt(np.mean((pymc_predictions - y) ** 2))),
        },
        {
            "Metric": "AIC",
            "JAX CMF NB2 MLE": float(jax_summary["aic"]),
            "PyMC CMF NBL posterior mean": np.nan,
        },
        {
            "Metric": "BIC",
            "JAX CMF NB2 MLE": float(jax_summary["bic"]),
            "PyMC CMF NBL posterior mean": np.nan,
        },
    ])

    return {
        "builder": builder,
        "manual_spec": manual_spec,
        "jax_fit": jax_fit,
        "compiled": compiled,
        "idata": idata,
        "coefficients": coefficients,
        "metrics": metrics,
        "pymc_coefficients": pymc_coefficients,
        "pymc_coefficient_sd": pymc_coefficient_sd,
        "pymc_theta_mean": theta_mean,
        "pymc_theta_sd": theta_sd,
        "jax_predictions": jax_predictions,
        "pymc_predictions": pymc_predictions,
    }


def print_cmf_pymc_comparison(result: dict[str, Any]) -> None:
    print("\nCoefficient estimates")
    print(result["coefficients"].to_string(index=False, float_format=lambda value: f"{value: .5f}"))
    print("\nFit metrics")
    print(result["metrics"].to_string(index=False, float_format=lambda value: f"{value: .5f}"))


if __name__ == "__main__":
    print_cmf_pymc_comparison(run_cmf_pymc_comparison())
