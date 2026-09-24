"""Joint random-parameter and temporal dependence estimators."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, ndtri
from scipy.stats import nbinom


_TEMPORAL_ALIASES = {
    "ar1": "ar1",
    "ar(1)": "ar1",
    "ar2": "ar2",
    "ar(2)": "ar2",
    "random_walk": "random_walk",
    "random-walk": "random_walk",
    "rw": "random_walk",
    "random_walk_drift": "random_walk_drift",
    "random-walk-drift": "random_walk_drift",
    "rw_drift": "random_walk_drift",
    "rw+drift": "random_walk_drift",
}


def _canonical_temporal_model(value: str) -> str:
    key = str(value).strip().lower()
    try:
        return _TEMPORAL_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "temporal_model must be one of ar1, ar2, random_walk, "
            "random_walk_drift"
        ) from exc


def _design_matrix(
    frame: pd.DataFrame,
    terms: Sequence[str],
    include_intercept: bool,
) -> np.ndarray:
    columns = []
    if include_intercept:
        columns.append(np.ones((len(frame), 1), dtype=float))
    for term in terms:
        if term in ("Intercept", "__INTERCEPT__"):
            if not include_intercept:
                columns.append(np.ones((len(frame), 1), dtype=float))
            continue
        if term not in frame.columns:
            raise ValueError(f"Model term {term!r} is not a column in the data")
        values = pd.to_numeric(frame[term], errors="raise").to_numpy(dtype=float)
        columns.append(values.reshape(-1, 1))
    if not columns:
        return np.empty((len(frame), 0), dtype=float)
    return np.column_stack(columns)


def _stationary_ar2_covariance(
    phi1: float,
    phi2: float,
    innovation_variance: float,
    size: int,
) -> np.ndarray:
    system = np.array([
        [1.0 - phi2 * phi2, -phi1 * (1.0 + phi2)],
        [-phi1, 1.0 - phi2],
    ])
    gamma_zero, gamma_one = np.linalg.solve(
        system, np.array([innovation_variance, 0.0])
    )
    autocovariances = np.zeros(max(size, 2), dtype=float)
    autocovariances[0] = gamma_zero
    autocovariances[1] = gamma_one
    for lag in range(2, size):
        autocovariances[lag] = (
            phi1 * autocovariances[lag - 1]
            + phi2 * autocovariances[lag - 2]
        )
    covariance = np.empty((size, size), dtype=float)
    for row_index in range(size):
        for column_index in range(size):
            covariance[row_index, column_index] = autocovariances[
                abs(row_index - column_index)
            ]
    return covariance


def temporal_covariance(
    temporal_model: str,
    n_obs: int,
    sigma: float = 1.0,
    phi1: float = 0.0,
    phi2: float = 0.0,
) -> np.ndarray:
    """Return the innovation-scale covariance for a temporal error model."""
    model = _canonical_temporal_model(temporal_model)
    if n_obs < 1:
        raise ValueError("n_obs must be positive")
    innovation_variance = max(float(sigma) ** 2, 1e-12)
    if model == "ar1":
        if abs(phi1) >= 1.0:
            raise ValueError("AR(1) requires abs(phi1) < 1")
        lags = np.abs(np.subtract.outer(np.arange(n_obs), np.arange(n_obs)))
        covariance = innovation_variance * phi1 ** lags / (1.0 - phi1 * phi1)
    elif model == "ar2":
        covariance = _stationary_ar2_covariance(
            float(phi1), float(phi2), innovation_variance, n_obs
        )
    else:
        positions = np.arange(1, n_obs + 1, dtype=float)
        covariance = innovation_variance * np.minimum(
            positions[:, None], positions[None, :]
        )
    covariance = np.asarray(covariance, dtype=float)
    covariance.flat[:: n_obs + 1] += 1e-9
    return covariance


def _normal_draws(n_draws: int, n_random: int, seed: int) -> np.ndarray:
    if n_random == 0:
        return np.zeros((1, 0), dtype=float)
    return np.random.default_rng(seed).standard_normal((n_draws, n_random))


def _logmeanexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + log(float(np.mean(np.exp(values - maximum))))


def _nb2_logpmf(y_values: np.ndarray, means: np.ndarray, dispersion: float) -> np.ndarray:
    size = 1.0 / max(float(dispersion), 1e-10)
    means = np.clip(np.asarray(means, dtype=float), 1e-12, 1e100)
    y_values = np.asarray(y_values, dtype=float)
    return (
        gammaln(y_values + size)
        - gammaln(size)
        - gammaln(y_values + 1.0)
        + size * np.log(size / (size + means))
        + y_values * np.log(means / (size + means))
    )


@dataclass
class TemporalRandomParameterFit:
    """Fitted result for a joint random-parameter temporal model."""

    params: np.ndarray
    beta: np.ndarray
    random_covariance: np.ndarray
    temporal_parameters: dict[str, float]
    loglik: float
    penalized_objective: float
    regularization_penalty: float
    aic: float
    bic: float
    converged: bool
    message: str
    likelihood: str
    temporal_model: str
    fixed_terms: tuple[str, ...]
    random_terms: tuple[str, ...]
    id_col: str
    time_col: str
    y_col: str
    n_obs: int
    n_panels: int
    n_draws: int
    regularization: str
    regularization_strength: float
    coefficient_bound: float
    dispersion: Optional[float] = None

    def predict(self, frame: pd.DataFrame, offset_col: Optional[str] = None) -> np.ndarray:
        """Return population means, without conditioning on random draws."""
        design = _design_matrix(frame, self.fixed_terms, include_intercept=True)
        linear_predictor = design @ self.beta
        if self.temporal_model == "random_walk_drift":
            ordered_indices = frame.sort_values(
                [self.id_col, self.time_col], kind="mergesort"
            ).index
            ranks = pd.Series(
                np.arange(len(frame), dtype=float), index=ordered_indices
            ).reindex(frame.index).to_numpy()
            linear_predictor = linear_predictor + self.temporal_parameters["drift"] * ranks
        if offset_col is not None:
            linear_predictor = linear_predictor + pd.to_numeric(
                frame[offset_col], errors="raise"
            ).to_numpy(dtype=float)
        if self.likelihood == "gaussian":
            return linear_predictor
        return np.exp(np.clip(linear_predictor, -50.0, 50.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params.tolist(),
            "beta": self.beta.tolist(),
            "random_covariance": self.random_covariance.tolist(),
            "temporal_parameters": dict(self.temporal_parameters),
            "loglik": float(self.loglik),
            "penalized_objective": float(self.penalized_objective),
            "regularization_penalty": float(self.regularization_penalty),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "converged": bool(self.converged),
            "message": self.message,
            "likelihood": self.likelihood,
            "temporal_model": self.temporal_model,
            "fixed_terms": list(self.fixed_terms),
            "random_terms": list(self.random_terms),
            "n_obs": int(self.n_obs),
            "n_panels": int(self.n_panels),
            "n_draws": int(self.n_draws),
            "regularization": self.regularization,
            "regularization_strength": float(self.regularization_strength),
            "coefficient_bound": float(self.coefficient_bound),
            "dispersion": None if self.dispersion is None else float(self.dispersion),
        }


class TemporalRandomParameterRegressor:
    """Joint random-coefficient and temporal-dependence estimator.

    The Gaussian path has an exact temporal covariance and simulated
    integration over random coefficients. The count path uses NB2 marginals
    with a Gaussian-copula serial correction. Both paths use ridge
    regularization and finite optimizer bounds for coefficient stability.
    Every random term is also required to be present in the fixed-effects
    design matrix, so random coefficients are deviations on variables in X.
    """

    def __init__(
        self,
        id_col: str,
        time_col: str,
        y_col: str,
        fixed_terms: Sequence[str] = (),
        random_terms: Sequence[str] = (),
        *,
        likelihood: str = "gaussian",
        temporal_model: str = "ar1",
        random_structure: str = "independent",
        group_col: Optional[str] = None,
        offset_col: Optional[str] = None,
        n_draws: int = 64,
        seed: int = 42,
        maxiter: int = 300,
        regularization: str = "ridge",
        regularization_strength: float = 1e-3,
        coefficient_bound: float = 20.0,
        random_parameter_bound: float = 6.0,
    ) -> None:
        likelihood_key = str(likelihood).lower()
        if likelihood_key == "nb":
            likelihood_key = "nb2_gaussian_copula"
        if likelihood_key not in {"gaussian", "nb2_gaussian_copula"}:
            raise ValueError(
                "likelihood must be 'gaussian' or 'nb2_gaussian_copula'"
            )
        structure = str(random_structure).lower()
        if structure not in {"independent", "correlated"}:
            raise ValueError(
                "random_structure must be 'independent' or 'correlated'"
            )
        regularization_key = str(regularization).lower()
        if regularization_key not in {"none", "ridge"}:
            raise ValueError("regularization must be 'none' or 'ridge'")
        if regularization_strength < 0:
            raise ValueError("regularization_strength must be non-negative")
        if coefficient_bound <= 0 or random_parameter_bound <= 0:
            raise ValueError("coefficient bounds must be positive")
        if n_draws < 1:
            raise ValueError("n_draws must be positive")
        self.id_col = id_col
        self.time_col = time_col
        self.y_col = y_col
        self.fixed_terms = tuple(fixed_terms)
        self.random_terms = tuple(random_terms)
        fixed_term_set = set(self.fixed_terms)
        missing_random_terms = [
            term for term in self.random_terms
            if term not in ("Intercept", "__INTERCEPT__")
            and term not in fixed_term_set
        ]
        if missing_random_terms:
            raise ValueError(
                "random_terms must be included in fixed_terms; missing: "
                + ", ".join(missing_random_terms)
            )
        self.likelihood = likelihood_key
        self.temporal_model = _canonical_temporal_model(temporal_model)
        self.random_structure = structure
        self.group_col = group_col
        self.offset_col = offset_col
        self.n_draws = int(n_draws)
        self.seed = int(seed)
        self.maxiter = int(maxiter)
        self.regularization = regularization_key
        self.regularization_strength = float(regularization_strength)
        self.coefficient_bound = float(coefficient_bound)
        self.random_parameter_bound = float(random_parameter_bound)

    @property
    def _n_covariance_params(self) -> int:
        n_random = len(self.random_terms)
        if self.random_structure == "independent":
            return n_random
        return n_random * (n_random + 1) // 2

    def _validate_data(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = [
            self.id_col,
            self.time_col,
            self.y_col,
            *self.fixed_terms,
            *self.random_terms,
        ]
        if self.group_col is not None:
            required.append(self.group_col)
        if self.offset_col is not None:
            required.append(self.offset_col)
        missing = sorted(set(required).difference(frame.columns))
        if missing:
            raise ValueError(f"Missing model columns: {', '.join(missing)}")
        validated = frame.copy().sort_values(
            [self.id_col, self.time_col], kind="mergesort"
        ).reset_index(drop=True)
        outcome = pd.to_numeric(validated[self.y_col], errors="raise")
        if not np.isfinite(outcome).all():
            raise ValueError("The outcome contains non-finite values")
        if self.likelihood == "nb2_gaussian_copula" and (outcome < 0).any():
            raise ValueError("NB2 outcomes must be non-negative")
        return validated

    def _blocks(
        self,
        frame: pd.DataFrame,
    ) -> tuple[list[dict[str, Any]], dict[Any, list[dict[str, Any]]]]:
        fixed_design = _design_matrix(frame, self.fixed_terms, True)
        fixed_names = ["Intercept"]
        fixed_names.extend(
            term for term in self.fixed_terms
            if term not in ("Intercept", "__INTERCEPT__")
        )
        random_names = [
            "Intercept" if term in ("Intercept", "__INTERCEPT__") else term
            for term in self.random_terms
        ]
        random_indices = [fixed_names.index(term) for term in random_names]
        outcome = pd.to_numeric(frame[self.y_col], errors="raise").to_numpy(float)
        offset = np.zeros(len(frame), dtype=float)
        if self.offset_col is not None:
            offset = pd.to_numeric(frame[self.offset_col], errors="raise").to_numpy(float)
        blocks = []
        grouped: dict[Any, list[dict[str, Any]]] = {}
        for panel_id, index_values in frame.groupby(self.id_col, sort=False).groups.items():
            indices = np.asarray(index_values, dtype=int)
            group_id = (
                frame.iloc[indices[0]][self.group_col]
                if self.group_col is not None
                else panel_id
            )
            block = {
                "X": fixed_design[indices],
                "random_indices": random_indices,
                "y": outcome[indices],
                "offset": offset[indices],
                "time_index": np.arange(len(indices), dtype=float),
            }
            blocks.append(block)
            grouped.setdefault(group_id, []).append(block)
        return blocks, grouped

    def _unpack_covariance(self, raw: np.ndarray) -> np.ndarray:
        n_random = len(self.random_terms)
        if n_random == 0:
            return np.zeros((0, 0), dtype=float)
        if self.random_structure == "independent":
            return np.diag(np.exp(2.0 * raw))
        cholesky = np.zeros((n_random, n_random), dtype=float)
        cursor = 0
        for row_index in range(n_random):
            for column_index in range(row_index + 1):
                cholesky[row_index, column_index] = (
                    np.exp(raw[cursor])
                    if row_index == column_index
                    else raw[cursor]
                )
                cursor += 1
        return cholesky @ cholesky.T

    def _unpack_temporal(
        self,
        theta: np.ndarray,
        cursor: int,
    ) -> tuple[dict[str, float], int]:
        temporal: dict[str, float] = {}
        if self.temporal_model == "ar1":
            temporal["phi1"] = float(np.tanh(theta[cursor]))
            cursor += 1
        elif self.temporal_model == "ar2":
            first_partial = float(np.tanh(theta[cursor]))
            second_partial = float(np.tanh(theta[cursor + 1]))
            temporal["phi1"] = first_partial * (1.0 - second_partial)
            temporal["phi2"] = second_partial
            cursor += 2
        if self.likelihood == "gaussian":
            temporal["sigma"] = float(np.exp(theta[cursor]))
            cursor += 1
        if self.temporal_model == "random_walk_drift":
            temporal["drift"] = float(theta[cursor])
            cursor += 1
        return temporal, cursor

    def _random_penalty(self, raw: np.ndarray) -> float:
        if len(raw) == 0:
            return 0.0
        if self.random_structure == "independent":
            return float(np.sum(np.exp(2.0 * raw)))
        n_random = len(self.random_terms)
        penalty = 0.0
        cursor = 0
        for row_index in range(n_random):
            for column_index in range(row_index + 1):
                value = raw[cursor]
                penalty += np.exp(2.0 * value) if row_index == column_index else value * value
                cursor += 1
        return float(penalty)

    def _penalty(self, theta: np.ndarray, n_beta: int) -> float:
        if self.regularization == "none" or self.regularization_strength == 0:
            return 0.0
        coefficient_values = theta[:n_beta]
        slope_values = coefficient_values[1:]
        covariance_start = n_beta
        covariance_end = covariance_start + self._n_covariance_params
        return self.regularization_strength * (
            float(np.sum(slope_values * slope_values))
            + self._random_penalty(theta[covariance_start:covariance_end])
        )

    def _gaussian_loglik(
        self,
        block: dict[str, Any],
        beta: np.ndarray,
        random_effect: np.ndarray,
        temporal: dict[str, float],
    ) -> float:
        mean = block["X"] @ beta + block["offset"]
        if len(random_effect):
            mean = mean + block["X"][:, block["random_indices"]] @ random_effect
        if self.temporal_model == "random_walk_drift":
            mean = mean + temporal["drift"] * block["time_index"]
        residual = block["y"] - mean
        covariance = temporal_covariance(
            self.temporal_model,
            len(residual),
            temporal.get("sigma", 1.0),
            temporal.get("phi1", 0.0),
            temporal.get("phi2", 0.0),
        )
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0 or not np.isfinite(logdet):
            return -1e100
        solved = np.linalg.solve(covariance, residual)
        return float(
            -0.5
            * (len(residual) * log(2.0 * pi) + logdet + residual @ solved)
        )

    def _count_loglik(
        self,
        block: dict[str, Any],
        beta: np.ndarray,
        random_effect: np.ndarray,
        temporal: dict[str, float],
        dispersion: float,
    ) -> float:
        linear_predictor = block["X"] @ beta + block["offset"]
        if len(random_effect):
            linear_predictor = (
                linear_predictor
                + block["X"][:, block["random_indices"]] @ random_effect
            )
        if self.temporal_model == "random_walk_drift":
            linear_predictor = linear_predictor + temporal["drift"] * block["time_index"]
        means = np.exp(np.clip(linear_predictor, -50.0, 50.0))
        marginal = _nb2_logpmf(block["y"], means, dispersion)
        size = 1.0 / max(dispersion, 1e-10)
        probability = size / (size + means)
        pmf = np.exp(marginal)
        lower_tail = np.where(
            block["y"] > 0,
            nbinom.cdf(block["y"] - 1.0, size, probability),
            0.0,
        )
        normal_score = ndtri(np.clip(lower_tail + 0.5 * pmf, 1e-8, 1.0 - 1e-8))
        temporal_correlation = temporal_covariance(
            self.temporal_model,
            len(block["y"]),
            1.0,
            temporal.get("phi1", 0.0),
            temporal.get("phi2", 0.0),
        )
        scale = np.sqrt(np.maximum(np.diag(temporal_correlation), 1e-12))
        temporal_correlation = temporal_correlation / np.outer(scale, scale)
        temporal_correlation.flat[:: len(normal_score) + 1] += 1e-9
        sign, logdet = np.linalg.slogdet(temporal_correlation)
        if sign <= 0 or not np.isfinite(logdet):
            return -1e100
        solved = np.linalg.solve(temporal_correlation, normal_score)
        copula_adjustment = normal_score @ (solved - normal_score)
        return float(np.sum(marginal) - 0.5 * (logdet + copula_adjustment))

    def fit(self, frame: pd.DataFrame) -> TemporalRandomParameterFit:
        validated = self._validate_data(frame)
        blocks, grouped = self._blocks(validated)
        n_random = len(self.random_terms)
        n_beta = len(self.fixed_terms) + 1
        draws = _normal_draws(self.n_draws, n_random, self.seed)
        n_temporal_raw = {
            "ar1": 1,
            "ar2": 2,
            "random_walk": 0,
            "random_walk_drift": 0,
        }[self.temporal_model]
        n_temporal = n_temporal_raw + int(self.likelihood == "gaussian")
        n_temporal += int(self.temporal_model == "random_walk_drift")
        n_dispersion = int(self.likelihood == "nb2_gaussian_copula")
        n_parameters = (
            n_beta + self._n_covariance_params + n_temporal + n_dispersion
        )
        outcome = pd.to_numeric(validated[self.y_col], errors="raise").to_numpy(float)
        initial_outcome = outcome if self.likelihood == "gaussian" else np.log1p(outcome)
        initial_design = _design_matrix(validated, self.fixed_terms, True)
        beta_initial = np.linalg.lstsq(initial_design, initial_outcome, rcond=None)[0]
        theta_initial = np.zeros(n_parameters, dtype=float)
        theta_initial[:n_beta] = np.clip(
            beta_initial, -self.coefficient_bound, self.coefficient_bound
        )
        cursor = n_beta
        theta_initial[cursor:cursor + self._n_covariance_params] = -1.0
        cursor += self._n_covariance_params
        cursor += n_temporal_raw
        if self.likelihood == "gaussian":
            residual_scale = np.std(initial_outcome - initial_design @ beta_initial)
            theta_initial[cursor] = log(max(float(residual_scale), 0.1))
            cursor += 1
        if self.temporal_model == "random_walk_drift":
            theta_initial[cursor] = 0.0
            cursor += 1
        if n_dispersion:
            theta_initial[cursor] = log(0.5)

        def likelihood_value(theta: np.ndarray) -> float:
            beta = theta[:n_beta]
            covariance = self._unpack_covariance(
                theta[n_beta:n_beta + self._n_covariance_params]
            )
            temporal, temporal_cursor = self._unpack_temporal(
                theta, n_beta + self._n_covariance_params
            )
            dispersion = (
                float(np.exp(theta[temporal_cursor]))
                if n_dispersion
                else None
            )
            total = 0.0
            try:
                cholesky = (
                    np.linalg.cholesky(covariance + 1e-10 * np.eye(n_random))
                    if n_random
                    else np.zeros((0, 0))
                )
                for group_blocks in grouped.values():
                    draw_loglikelihoods = np.zeros(len(draws), dtype=float)
                    for draw_index, draw in enumerate(draws):
                        random_effect = (
                            cholesky @ draw
                            if n_random
                            else np.zeros(0, dtype=float)
                        )
                        for block in group_blocks:
                            if self.likelihood == "gaussian":
                                draw_loglikelihoods[draw_index] += self._gaussian_loglik(
                                    block, beta, random_effect, temporal
                                )
                            else:
                                draw_loglikelihoods[draw_index] += self._count_loglik(
                                    block, beta, random_effect, temporal, dispersion
                                )
                    total += _logmeanexp(draw_loglikelihoods)
            except (FloatingPointError, np.linalg.LinAlgError, ValueError, OverflowError):
                return -1e100
            return float(total) if np.isfinite(total) else -1e100

        def objective(theta: np.ndarray) -> float:
            value = likelihood_value(theta)
            penalty = self._penalty(theta, n_beta)
            return float(-value + penalty) if np.isfinite(value) else 1e100

        bounds = [
            (-self.coefficient_bound, self.coefficient_bound)
            for _ in range(n_beta)
        ]
        bounds.extend([
            (-self.random_parameter_bound, self.random_parameter_bound)
            for _ in range(self._n_covariance_params)
        ])
        bounds.extend([
            (-4.0, 4.0)
            for _ in range(1 if self.temporal_model == "ar1" else 2 if self.temporal_model == "ar2" else 0)
        ])
        if self.likelihood == "gaussian":
            bounds.append((-6.0, 5.0))
        if self.temporal_model == "random_walk_drift":
            bounds.append((-self.coefficient_bound, self.coefficient_bound))
        if n_dispersion:
            bounds.append((-6.0, 3.0))
        result = minimize(
            objective,
            theta_initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter, "ftol": 1e-9},
        )
        theta = np.asarray(result.x, dtype=float)
        beta = theta[:n_beta]
        covariance = self._unpack_covariance(
            theta[n_beta:n_beta + self._n_covariance_params]
        )
        temporal, temporal_cursor = self._unpack_temporal(
            theta, n_beta + self._n_covariance_params
        )
        dispersion = (
            float(np.exp(theta[temporal_cursor])) if n_dispersion else None
        )
        loglik = likelihood_value(theta)
        penalty = self._penalty(theta, n_beta)
        penalized_objective = objective(theta)
        aic = 2.0 * len(theta) - 2.0 * loglik
        bic = log(float(len(validated))) * len(theta) - 2.0 * loglik
        return TemporalRandomParameterFit(
            params=theta,
            beta=beta,
            random_covariance=covariance,
            temporal_parameters=temporal,
            loglik=loglik,
            penalized_objective=penalized_objective,
            regularization_penalty=penalty,
            aic=aic,
            bic=bic,
            converged=bool(result.success),
            message=str(result.message),
            likelihood=self.likelihood,
            temporal_model=self.temporal_model,
            fixed_terms=self.fixed_terms,
            random_terms=self.random_terms,
            id_col=self.id_col,
            time_col=self.time_col,
            y_col=self.y_col,
            n_obs=len(validated),
            n_panels=len(blocks),
            n_draws=len(draws),
            regularization=self.regularization,
            regularization_strength=self.regularization_strength,
            coefficient_bound=self.coefficient_bound,
            dispersion=dispersion,
        )


def fit_temporal_random_parameter_model(
    frame: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    y_col: str,
    fixed_terms: Sequence[str] = (),
    random_terms: Sequence[str] = (),
    **kwargs: Any,
) -> TemporalRandomParameterFit:
    """Fit one regularized joint random-parameter temporal model."""
    model = TemporalRandomParameterRegressor(
        id_col=id_col,
        time_col=time_col,
        y_col=y_col,
        fixed_terms=fixed_terms,
        random_terms=random_terms,
        **kwargs,
    )
    return model.fit(frame)


__all__ = [
    "TemporalRandomParameterFit",
    "TemporalRandomParameterRegressor",
    "fit_temporal_random_parameter_model",
    "temporal_covariance",
]