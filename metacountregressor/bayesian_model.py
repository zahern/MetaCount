"""Compile a selected MetaCount model into an optional PyMC model.

The search chooses a model *structure*.  This module keeps that structure and
compiles it into a Bayesian likelihood without treating the MLE coefficients
as fixed values.  PyMC is imported lazily so the core package does not acquire
an unconditional Bayesian dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class BayesianModelError(ValueError):
    """Raised when a search result cannot be represented faithfully."""


@dataclass
class BayesianModel:
    """A compiled PyMC model together with its resolved search structure."""

    model: Any
    spec: Dict[str, Any]
    data: pd.DataFrame
    initvals: Optional[Dict[str, Any]] = None

    def sample(self, **kwargs):
        """Sample this model with :func:`pymc.sample`."""
        try:
            import pymc as pm
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Bayesian support requires the optional dependency. "
                "Install it with `pip install metacountregressor[bayesian]`."
            ) from exc

        with self.model:
            if self.initvals is not None and "initvals" not in kwargs:
                kwargs["initvals"] = self.initvals
            return pm.sample(**kwargs)


def _import_pymc():
    try:
        import pymc as pm
        import pytensor.tensor as pt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Bayesian model construction requires the optional dependency. "
            "Install it with `pip install metacountregressor[bayesian]`."
        ) from exc
    return pm, pt


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_") or "term"


def _field(value: Any, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_term_name(term: Any) -> str:
    name, _, _ = str(term).partition(":")
    return name


def _as_term_dist(term: Any) -> tuple[str, str]:
    name, separator, distribution = str(term).partition(":")
    return name, distribution.lower() if separator else "normal"


def _canonical_model_name(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    if normalized in {
        "nb",
        "nb2",
        "negative-binomial",
        "negative-binomial-2",
        "negativebinomial",
        "negativebinomial-2",
        "nbl",
        "nb-lindley",
        "negative-binomial-lindley",
        "negativebinomial-lindley",
    }:
        return "nbl"
    return normalized


def _unique_names(values) -> list[str]:
    return list(dict.fromkeys(_as_term_name(value) for value in (values or [])))


def _cmf_term_map(builder, metadata: Optional[dict]) -> dict[str, str]:
    if metadata and metadata.get("term_map"):
        return dict(metadata["term_map"])
    if hasattr(builder, "_cmf_term_map"):
        return dict(builder._cmf_term_map())
    aadt_col = getattr(builder, "aadt_col", "AADT")
    return {aadt_col: "__cmf_log_aadt"}


def _legacy_cmf_spec(payload, builder, metadata=None) -> dict[str, Any]:
    """Turn the legacy CMF dataclass result into the common count schema."""
    if builder is None:
        raise BayesianModelError(
            "A legacy CMF result needs its CMFExperimentBuilder so the "
            "AADT interaction columns can be reconstructed."
        )

    term_map = _cmf_term_map(builder, metadata)
    aadt_term = term_map[getattr(builder, "aadt_col")]
    selected_baseline = list(_field(payload, "selected_baseline", []) or [])
    selected_local = list(_field(payload, "selected_local", []) or [])
    rand_baseline = list(_field(payload, "rand_baseline", []) or [])
    rand_local = list(_field(payload, "rand_local", []) or [])

    fixed_terms = [aadt_term]
    rdm_terms = []
    for index, variable in enumerate(selected_baseline):
        if index < len(rand_baseline) and rand_baseline[index]:
            rdm_terms.append(f"{variable}:normal")
        else:
            fixed_terms.append(variable)
    for index, variable in enumerate(selected_local):
        interaction = term_map.get(variable, variable)
        if index < len(rand_local) and rand_local[index]:
            rdm_terms.append(f"{interaction}:normal")
        else:
            fixed_terms.append(interaction)

    return {
        "fixed_terms": list(dict.fromkeys(fixed_terms)),
        "rdm_terms": rdm_terms,
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": [],
        "zi_terms": [],
        "membership_terms": [],
        "dispersion": int(str(_field(payload, "model", "poisson")).lower() == "nb"),
        "latent_classes": 1,
        "model": str(_field(payload, "model", "poisson")).lower(),
        "group_id_col": getattr(builder, "group_id_col", None),
    }


def _unwrap_search_result(search_result):
    outer = search_result
    payload = search_result
    if isinstance(search_result, dict):
        if search_result.get("result") is not None:
            payload = search_result["result"]
        elif search_result.get("search_result") is not None:
            payload = search_result["search_result"]
    return outer, payload


def resolve_search_spec(
    search_result,
    *,
    evaluator=None,
    builder=None,
    family: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve a search result to a portable model specification.

    ``evaluator`` is needed only for older result files that predate the
    ``model_spec`` field.  New results can be converted from the saved result
    alone, provided their data and column metadata are supplied to the build
    function.
    """
    outer, payload = _unwrap_search_result(search_result)
    metadata = {}
    if isinstance(outer, dict):
        metadata.update(outer.get("cmf_metadata", {}) or {})
        metadata.update(outer.get("metadata", {}) or {})
    if isinstance(payload, dict):
        metadata.update(payload.get("cmf_metadata", {}) or {})
        metadata.update(payload.get("metadata", {}) or {})

    spec = None
    if isinstance(outer, dict):
        spec = outer.get("model_spec")
    if spec is None and isinstance(payload, dict):
        spec = payload.get("model_spec")

    if spec is None and evaluator is not None:
        decision = None
        for candidate in (outer, payload):
            if isinstance(candidate, dict):
                decision = candidate.get("best_solution", candidate.get("best_decision"))
                if decision is not None:
                    break
        if decision is not None and hasattr(evaluator, "build_spec"):
            spec = evaluator.build_spec(decision)

    inferred_family = family
    if inferred_family is None:
        for candidate in (outer, payload):
            value = _field(candidate, "family")
            if value:
                inferred_family = value
                break
    inferred_family = str(inferred_family or "count").lower()

    if spec is None and inferred_family == "count":
        if any(
            isinstance(candidate, dict)
            and ("front" in candidate or "front_records" in candidate)
            for candidate in (outer, payload)
        ):
            inferred_family = "pavement"
        elif any(
            isinstance(candidate, dict)
            and ("best_decoded" in candidate or "multivariate_metadata" in candidate)
            for candidate in (outer, payload)
        ):
            inferred_family = "multivariate"

    if spec is None and inferred_family in {"pavement", "multivariate"}:
        spec = {"family": inferred_family, "model": inferred_family}

    if spec is None and inferred_family == "cmf":
        spec = _legacy_cmf_spec(payload, builder, metadata)

    if spec is None:
        raise BayesianModelError(
            "The search result does not contain model_spec. Pass the evaluator "
            "used for the search, or rerun the search with the updated API."
        )

    if isinstance(spec, dict):
        metadata = {**(spec.get("metadata", {}) or {}), **metadata}
    resolved = dict(spec)
    model = _canonical_model_name(resolved.get("model", ""))
    if model in {"gaussian", "normal"}:
        inferred_family = "linear"
    elif model == "tobit":
        inferred_family = "tobit"
    elif model in {"lognormal", "weibull", "loglogistic"}:
        inferred_family = "duration"

    resolved["family"] = inferred_family
    default_model = {
        "duration": "lognormal",
        "linear": "gaussian",
        "tobit": "tobit",
    }.get(
        inferred_family,
        "nbl" if int(resolved.get("dispersion", 0)) else "poisson",
    )
    resolved["model"] = model or default_model
    resolved["metadata"] = metadata
    return resolved


def _prepare_frame(df: pd.DataFrame, spec: dict, builder=None) -> pd.DataFrame:
    frame = df.copy()
    if spec.get("family") != "cmf":
        return frame

    metadata = spec.get("metadata", {})
    term_map = _cmf_term_map(builder, metadata)
    aadt_col = metadata.get("aadt_col", getattr(builder, "aadt_col", None))
    if aadt_col is None or aadt_col not in frame.columns:
        raise BayesianModelError(
            "CMF conversion needs the original positive AADT column and its "
            "CMF metadata."
        )
    if (frame[aadt_col].astype(float) <= 0).any():
        raise BayesianModelError("CMF conversion requires strictly positive AADT values.")

    log_aadt = term_map.get(aadt_col, "__cmf_log_aadt")
    frame[log_aadt] = np.log(frame[aadt_col].astype(float))
    local_vars = list(metadata.get("local_vars", getattr(builder, "local_vars", [])))
    for variable in local_vars:
        if variable not in frame.columns:
            raise BayesianModelError(f"CMF local variable '{variable}' is missing from the data.")
        interaction = term_map.get(variable, f"__cmf_local__{_safe_name(variable)}")
        frame[interaction] = frame[variable].astype(float) * frame[log_aadt]
    return frame


def _matrix(frame: pd.DataFrame, names: list[str], *, intercept: bool = False) -> np.ndarray:
    selected = list(names)
    if intercept and "__INTERCEPT__" not in selected:
        selected.insert(0, "__INTERCEPT__")
    columns = []
    for name in selected:
        if name == "__INTERCEPT__":
            columns.append(np.ones(len(frame), dtype=float))
        else:
            if name not in frame.columns:
                raise BayesianModelError(f"Search specification references missing column '{name}'.")
            values = pd.to_numeric(frame[name], errors="raise").to_numpy(float)
            if not np.isfinite(values).all():
                raise BayesianModelError(f"Column '{name}' contains non-finite values.")
            columns.append(values)
    return np.column_stack(columns) if columns else np.zeros((len(frame), 0))


def _id_codes(frame: pd.DataFrame, id_col: Optional[str]):
    if id_col is None:
        return np.zeros(len(frame), dtype=int), np.array([0])
    if id_col not in frame.columns:
        raise BayesianModelError(f"id_col '{id_col}' is missing from the data.")
    codes, values = pd.factorize(frame[id_col], sort=True)
    return codes.astype(int), np.asarray(values)


def _group_means(matrix: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros((n_groups, 0))
    output = np.zeros((n_groups, matrix.shape[1]), dtype=float)
    for group in range(n_groups):
        rows = codes == group
        output[group] = matrix[rows].mean(axis=0) if rows.any() else 0.0
    return output


def _random_draw(pm, pt, name, mean, sd, distribution, size):
    distribution = distribution.lower()
    if distribution == "normal":
        return pm.Normal(name, mu=mean, sigma=sd, shape=size)
    latent = pm.Normal(f"{name}_latent", mu=0.0, sigma=1.0, shape=size)
    if distribution == "lognormal":
        return pm.Deterministic(name, pt.exp(mean + sd * latent))
    if distribution in {"triangular", "uniform"}:
        return pm.Deterministic(name, mean + sd * (2.0 * pm.math.invprobit(latent) - 1.0))
    raise BayesianModelError(
        f"Random-parameter distribution '{distribution}' is not supported by the Bayesian compiler."
    )


def _build_random_eta(
    pm, pt, model, frame, spec, id_col, group_id_col, coef_scale, sd_scale,
    name_suffix="",
):
    n = len(frame)
    id_codes, id_values = _id_codes(frame, id_col)
    n_ids = len(id_values)
    ind_terms = [_as_term_dist(term) for term in spec.get("rdm_terms", [])]
    cor_terms = [_as_term_dist(term) for term in spec.get("rdm_cor_terms", [])]
    grouped_terms = [_as_term_dist(term) for term in spec.get("grouped_terms", [])]
    total_random = len(ind_terms) + len(cor_terms) + len(grouped_terms)
    if total_random == 0:
        return pt.zeros(n), {"id_values": id_values}

    name_prefix = f"{name_suffix}_" if name_suffix else ""
    hetero_names = _unique_names(spec.get("hetro_in_means", []))
    hetero_rows = _matrix(frame, hetero_names)
    hetero_id = _group_means(hetero_rows, id_codes, n_ids)
    hetero_gamma = None
    if hetero_names:
        hetero_gamma = pm.Normal(
            f"{name_prefix}random_mean_heterogeneity",
            mu=0.0,
            sigma=coef_scale,
            shape=(len(hetero_names), total_random),
        )
    base_mean = pm.Normal(
        f"{name_prefix}random_mean", mu=0.0, sigma=coef_scale, shape=total_random
    )
    base_sd = pm.HalfNormal(
        f"{name_prefix}random_sd", sigma=sd_scale, shape=total_random
    )
    if hetero_gamma is None:
        id_means = pt.ones((n_ids, 1)) @ base_mean[None, :]
    else:
        id_means = pt.ones((n_ids, 1)) @ base_mean[None, :] + pt.dot(hetero_id, hetero_gamma)

    eta = pt.zeros(n)
    position = 0
    for index, (term, distribution) in enumerate(ind_terms):
        coefficients = _random_draw(
            pm, pt, f"{name_prefix}random_ind_{_safe_name(term)}", id_means[:, position],
            base_sd[position], distribution, n_ids,
        )
        values = _matrix(frame, [term]).ravel()
        eta = eta + values * coefficients[id_codes]
        position += 1

    if cor_terms:
        names = [term for term, _ in cor_terms]
        if any(distribution != "normal" for _, distribution in cor_terms):
            raise BayesianModelError(
                "Correlated random parameters currently require normal marginals."
            )
        chol, _, _ = pm.LKJCholeskyCov(
            f"{name_prefix}random_cor_chol",
            n=len(cor_terms),
            eta=2.0,
            sd_dist=pm.HalfNormal.dist(sigma=sd_scale),
            compute_corr=True,
        )
        cor_mean = id_means[:, position:position + len(cor_terms)]
        coefficients = pm.MvNormal(
            f"{name_prefix}random_cor", mu=cor_mean, chol=chol,
            shape=(n_ids, len(cor_terms)),
        )
        design = _matrix(frame, names)
        eta = eta + pt.sum(design * coefficients[id_codes], axis=1)
        position += len(cor_terms)

    if grouped_terms:
        if group_id_col is None or group_id_col not in frame.columns:
            raise BayesianModelError(
                "Grouped random parameters require group_id_col and that column in the data."
            )
        group_codes, group_values = _id_codes(frame, group_id_col)
        n_groups = len(group_values)
        for index, (term, distribution) in enumerate(grouped_terms):
            coefficients = _random_draw(
                pm, pt, f"{name_prefix}random_group_{_safe_name(term)}",
                base_mean[position + index],
                base_sd[position + index], distribution, n_groups,
            )
            values = _matrix(frame, [term]).ravel()
            eta = eta + values * coefficients[group_codes]

    return eta, {"id_values": id_values}


def _build_fixed_etas(pm, pt, frame, spec, random_etas, coef_scale):
    classes = max(1, int(spec.get("latent_classes", 1)))
    class_fixed = spec.get("class_fixed")
    if classes > 1 and class_fixed:
        names_per_class = [_unique_names(names) for names in class_fixed]
        if len(names_per_class) < classes:
            names_per_class.extend([_unique_names(spec.get("fixed_terms", []))] * (classes - len(names_per_class)))
    else:
        names = _unique_names(spec.get("fixed_terms", []))
        names_per_class = [names] * classes

    etas = []
    for index, names in enumerate(names_per_class):
        X = _matrix(frame, names, intercept=True)
        beta = pm.Normal(
            "beta" if classes == 1 else f"beta_class_{index + 1}",
            mu=0.0,
            sigma=coef_scale,
            shape=X.shape[1],
        )
        random_eta = random_etas[index] if index < len(random_etas) else random_etas[0]
        etas.append(pt.dot(X, beta) + random_eta)
    return etas


def _zero_inflation(pm, pt, frame, spec, coef_scale, name_suffix=""):
    names = _unique_names(spec.get("zi_terms", []))
    if not names:
        return pt.zeros(len(frame))
    X = _matrix(frame, names)
    prefix = f"{name_suffix}_" if name_suffix else ""
    beta = pm.Normal(
        f"{prefix}zero_inflation_beta", mu=0.0, sigma=coef_scale, shape=X.shape[1]
    )
    return pm.math.sigmoid(pt.dot(X, beta))


def _class_weights(pm, pt, frame, spec, coef_scale):
    classes = max(1, int(spec.get("latent_classes", 1)))
    if classes == 1:
        return None
    membership = _unique_names(spec.get("membership_terms", []))
    class_membership = spec.get("class_membership") or []
    logits = [pt.zeros(len(frame))]
    for index in range(classes - 1):
        names = _unique_names(class_membership[index]) if index < len(class_membership) else membership
        X = _matrix(frame, names)
        gamma = pm.Normal(
            f"membership_class_{index + 2}",
            mu=0.0,
            sigma=coef_scale,
            shape=X.shape[1] + 1,
        )
        logits.append(gamma[0] + pt.dot(X, gamma[1:]))
    return pm.math.softmax(pm.math.stack(logits, axis=1), axis=1)


def _duration_logp(pm, pt, family, y, eta, sigma, event):
    y_safe = pt.clip(y, 1e-12, np.inf)
    if family == "lognormal":
        distribution = pm.LogNormal.dist(mu=eta, sigma=sigma)
        observed = pm.logp(distribution, y_safe)
        censored = pm.math.log1p(-pt.exp(pm.logcdf(distribution, y_safe)))
    elif family == "weibull":
        shape = 1.0 / sigma
        scale = pt.exp(eta)
        z = pt.power(y_safe / scale, shape)
        observed = pt.log(shape) - pt.log(scale) + (shape - 1.0) * pt.log(y_safe / scale) - z
        censored = -z
    elif family == "loglogistic":
        z = (pt.log(y_safe) - eta) / sigma
        softplus_z = pt.log1p(pt.exp(pt.clip(z, -30.0, 30.0)))
        observed = z - pt.log(y_safe) - pt.log(sigma) - 2.0 * softplus_z
        censored = -softplus_z
    else:
        raise BayesianModelError(f"Unknown duration family '{family}'.")
    if event is None:
        return observed
    return pt.switch(event > 0, observed, censored)


def _nbl_logp(pm, pt, y, mu, theta):
    """Log PMF for a mean-linked negative-binomial Lindley model."""
    mean_multiplier = (
        theta ** 2 + theta - 1.0
    ) / ((theta + 1.0) * (theta - 1.0) ** 2)
    shape = mu / mean_multiplier
    beta_argument = shape + theta
    log_beta = (
        pt.gammaln(beta_argument)
        + pt.gammaln(y + 1.0)
        - pt.gammaln(beta_argument + y + 1.0)
    )
    log_lindley_factor = pt.log1p(
        pt.digamma(beta_argument + y + 1.0) - pt.digamma(beta_argument)
    )
    return (
        pt.gammaln(y + shape)
        - pt.gammaln(shape)
        - pt.gammaln(y + 1.0)
        + 2.0 * pt.log(theta)
        - pt.log(theta + 1.0)
        + log_beta
        + log_lindley_factor
    )


def _logp(pm, pt, family, model_name, y, eta, offset, frame, spec, coef_scale,
          event, likelihood_params):
    total_eta = eta + offset
    if family in {"count", "cmf"}:
        mu = pt.exp(pt.clip(total_eta, -30.0, 30.0))
        if model_name == "poisson":
            distribution = pm.Poisson.dist(mu=mu)
        elif model_name == "nbl":
            base_logp = _nbl_logp(pm, pt, y, mu, likelihood_params["nbl_theta"])
            if spec.get("zi_terms"):
                excess_zero = pt.clip(
                    likelihood_params["zero_inflation"], 1e-9, 1.0 - 1e-9
                )
                return pt.switch(
                    pt.eq(y, 0),
                    pt.logaddexp(
                        pt.log(excess_zero),
                        pt.log1p(-excess_zero) + base_logp,
                    ),
                    pt.log1p(-excess_zero) + base_logp,
                )
            return base_logp
        else:
            raise BayesianModelError(f"Unknown count model '{model_name}'.")
        if spec.get("zi_terms"):
            # PyMC's psi is the probability of the count component; the
            # search specification stores the excess-zero probability.
            psi = 1.0 - likelihood_params["zero_inflation"]
            if model_name == "poisson":
                distribution = pm.ZeroInflatedPoisson.dist(psi=psi, mu=mu)
        return pm.logp(distribution, y)
    if family == "linear":
        sigma = likelihood_params["sigma"]
        return pm.logp(pm.Normal.dist(mu=total_eta, sigma=sigma), y)
    if family == "tobit":
        sigma = likelihood_params["sigma"]
        return pm.logp(
            pm.Censored.dist(pm.Normal.dist(mu=total_eta, sigma=sigma), lower=0, upper=None),
            y,
        )
    if family == "duration":
        sigma = likelihood_params["sigma"]
        return _duration_logp(pm, pt, model_name, y, total_eta, sigma, event)
    raise BayesianModelError(
        f"Bayesian compilation for family '{family}' is not implemented. "
        "Pavement Markov/hazard and multivariate copula models need their "
        "own likelihood compiler."
    )


def _add_observed(pm, pt, model, family, model_name, y, eta, offset, frame,
                  spec, coef_scale, event, likelihood_params):
    if int(spec.get("latent_classes", 1)) > 1 or event is not None:
        return False
    total_eta = eta + offset
    if family in {"count", "cmf"}:
        mu = pt.exp(pt.clip(total_eta, -30.0, 30.0))
        if model_name == "poisson":
            distribution = pm.Poisson
            kwargs = {"mu": mu}
        elif model_name == "nbl":
            return False
        else:
            return False
        if spec.get("zi_terms"):
            # PyMC's psi is the probability of the count component; the
            # search specification stores the excess-zero probability.
            psi = 1.0 - likelihood_params["zero_inflation"]
            if model_name == "poisson":
                pm.ZeroInflatedPoisson("y_obs", psi=psi, mu=mu, observed=y)
        else:
            distribution("y_obs", observed=y, **kwargs)
        return True
    if family == "linear":
        sigma = likelihood_params["sigma"]
        pm.Normal("y_obs", mu=total_eta, sigma=sigma, observed=y)
        return True
    if family == "tobit":
        sigma = likelihood_params["sigma"]
        pm.Censored(
            "y_obs", pm.Normal.dist(mu=total_eta, sigma=sigma),
            lower=0, upper=None, observed=y,
        )
        return True
    if family == "duration" and model_name == "lognormal":
        sigma = likelihood_params["sigma"]
        pm.LogNormal("y_obs", mu=total_eta, sigma=sigma, observed=y)
        return True
    if family == "duration" and model_name == "weibull":
        sigma = likelihood_params["sigma"]
        pm.Weibull(
            "y_obs", alpha=1.0 / sigma, beta=pt.exp(total_eta), observed=y
        )
        return True
    return False


def build_bayesian_model(
    search_result,
    *,
    df: Optional[pd.DataFrame] = None,
    builder=None,
    evaluator=None,
    id_col: Optional[str] = None,
    y_col: Optional[str] = None,
    offset_col: Optional[str] = None,
    group_id_col: Optional[str] = None,
    event_col: Optional[str] = None,
    family: Optional[str] = None,
    priors: Optional[dict[str, float]] = None,
    initvals: Optional[Dict[str, Any]] = None,
) -> BayesianModel:
    """Build a PyMC model from a completed MetaCount search.

    ``builder`` is preferred because it supplies the original dataframe and
    column names.  Alternatively pass ``df``, ``id_col``, and ``y_col``.
    ``evaluator`` is only required for old search results without
    ``model_spec``.
    """
    spec = resolve_search_spec(
        search_result, evaluator=evaluator, builder=builder, family=family
    )
    if builder is not None:
        df = builder.df if df is None else df
        id_col = getattr(builder, "id_col", id_col)
        y_col = getattr(builder, "y_col", y_col)
        offset_col = getattr(builder, "offset_col", offset_col)
        group_id_col = getattr(builder, "group_id_col", group_id_col)
    group_id_col = group_id_col or spec.get("group_id_col")
    if df is None or y_col is None:
        raise BayesianModelError("Bayesian conversion requires df and y_col.")
    if y_col not in df.columns:
        raise BayesianModelError(f"y_col '{y_col}' is missing from the data.")

    family_name = spec["family"]
    if family_name in {"pavement", "multivariate"}:
        raise BayesianModelError(
            f"Family '{family_name}' is not silently approximated. Its pavement "
            "layers or copula likelihood need a dedicated Bayesian compiler."
        )
    if family_name not in {"count", "cmf", "linear", "tobit", "duration"}:
        raise BayesianModelError(f"Unknown Bayesian model family '{family_name}'.")

    frame = _prepare_frame(df, spec, builder=builder)
    y = pd.to_numeric(frame[y_col], errors="raise").to_numpy(float)
    if family_name in {"count", "cmf"}:
        if (y < 0).any() or not np.equal(y, np.floor(y)).all():
            raise BayesianModelError("Count outcomes must be non-negative integers.")
        y = y.astype(int)
    if family_name == "duration" and (y <= 0).any():
        raise BayesianModelError("Duration outcomes must be strictly positive.")

    offset = np.zeros(len(frame), dtype=float)
    if offset_col is not None:
        if offset_col not in frame.columns:
            raise BayesianModelError(f"offset_col '{offset_col}' is missing from the data.")
        offset = pd.to_numeric(frame[offset_col], errors="raise").to_numpy(float)
    event = None
    if event_col is not None:
        if event_col not in frame.columns:
            raise BayesianModelError(f"event_col '{event_col}' is missing from the data.")
        event = pd.to_numeric(frame[event_col], errors="raise").to_numpy(float)

    pm, pt = _import_pymc()
    options = {"coef_scale": 5.0, "sd_scale": 2.0, "nbl_theta_scale": 2.0}
    options.update(priors or {})
    coef_scale = float(options["coef_scale"])
    sd_scale = float(options["sd_scale"])
    nbl_theta_scale = float(options["nbl_theta_scale"])
    if coef_scale <= 0 or sd_scale <= 0 or nbl_theta_scale <= 0:
        raise BayesianModelError(
            "coef_scale, sd_scale, and nbl_theta_scale must be positive."
        )

    model_name = _canonical_model_name(spec.get("model", "poisson"))
    with pm.Model() as model:
        classes = max(1, int(spec.get("latent_classes", 1)))
        random_etas = []
        random_meta = {"id_values": np.array([0])}
        class_rdm_ind = spec.get("class_rdm_ind") or [] if classes > 1 else []
        class_rdm_cor = spec.get("class_rdm_cor") or [] if classes > 1 else []
        for class_index in range(classes):
            class_spec = dict(spec)
            if class_index < len(class_rdm_ind):
                class_spec["rdm_terms"] = class_rdm_ind[class_index]
            if class_index < len(class_rdm_cor):
                class_spec["rdm_cor_terms"] = class_rdm_cor[class_index]
            random_eta, random_meta = _build_random_eta(
                pm, pt, model, frame, class_spec, id_col, group_id_col,
                coef_scale, sd_scale,
                name_suffix=f"class_{class_index + 1}" if classes > 1 else "",
            )
            random_etas.append(random_eta)

        likelihood_params = []
        for class_index in range(classes):
            name_suffix = f"class_{class_index + 1}" if classes > 1 else ""
            params = {}
            if family_name in {"linear", "tobit"}:
                params["sigma"] = pm.HalfNormal(
                    f"{name_suffix + '_' if name_suffix else ''}sigma", sigma=10.0
                )
            elif family_name == "duration":
                params["sigma"] = pm.HalfNormal(
                    f"{name_suffix + '_' if name_suffix else ''}sigma", sigma=2.0
                )
            elif family_name in {"count", "cmf"} and model_name == "nbl":
                prefix = f"{name_suffix + '_' if name_suffix else ''}"
                theta_excess = pm.HalfNormal(
                    f"{prefix}nbl_theta_excess",
                    sigma=nbl_theta_scale,
                )
                params["nbl_theta"] = pm.Deterministic(
                    f"{prefix}nbl_theta", 2.0 + theta_excess
                )
            if spec.get("zi_terms"):
                params["zero_inflation"] = _zero_inflation(
                    pm, pt, frame, spec, coef_scale, name_suffix=name_suffix
                )
            likelihood_params.append(params)

        etas = _build_fixed_etas(pm, pt, frame, spec, random_etas, coef_scale)
        weights = _class_weights(pm, pt, frame, spec, coef_scale)
        offset_tensor = pt.as_tensor_variable(offset)
        y_tensor = pt.as_tensor_variable(y)
        event_tensor = None if event is None else pt.as_tensor_variable(event)

        if weights is None:
            if not _add_observed(
                pm, pt, model, family_name, model_name, y, etas[0],
                offset_tensor, frame, spec, coef_scale, event,
                likelihood_params[0],
            ):
                logp = _logp(
                    pm, pt, family_name, model_name, y_tensor, etas[0],
                    offset_tensor, frame, spec, coef_scale, event_tensor,
                    likelihood_params[0],
                )
                pm.Potential("observed_log_likelihood", pt.sum(logp))
        else:
            component_logps = []
            for class_index, eta in enumerate(etas):
                component_logps.append(
                    _logp(
                        pm, pt, family_name, model_name, y_tensor, eta,
                        offset_tensor, frame, spec, coef_scale, event_tensor,
                        likelihood_params[class_index],
                    )
                )
            stacked = pm.math.stack(component_logps, axis=1)
            pm.Potential(
                "observed_log_likelihood",
                pt.sum(pm.math.logsumexp(pm.math.log(weights) + stacked, axis=1)),
            )
        pm.Deterministic("linear_predictor", etas[0])

    resolved = dict(spec)
    resolved["id_values"] = random_meta.get("id_values")
    resolved["y_col"] = y_col
    resolved["id_col"] = id_col
    resolved["offset_col"] = offset_col
    resolved["event_col"] = event_col
    return BayesianModel(model=model, spec=resolved, data=frame, initvals=initvals)


__all__ = [
    "BayesianModel",
    "BayesianModelError",
    "build_bayesian_model",
    "resolve_search_spec",
]