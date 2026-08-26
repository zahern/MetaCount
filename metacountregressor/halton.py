import numpy as np
import jax.numpy as jnp
import jax
from scipy.optimize import minimize
from scipy.stats import norm, gamma, triang
from scipy.stats.qmc import Sobol
import pandas as pd
import warnings

try:
    from ._jax_config import configure_jax
except ImportError:  # flat import (script run from inside the package dir)
    from _jax_config import configure_jax
configure_jax()


def _next_power_of_2(n: int) -> int:
    """Round n up to the nearest power of 2 (required for optimal Sobol uniformity)."""
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def generate_sobol_draws(
    dim: int,
    n_obs: int,
    n_draws: int,
    distribution: list[str],
    scramble: bool = True,
    as_jax: bool = True,
) -> np.ndarray | jnp.ndarray:
    """
    Generate quasi-random draws using a scrambled Sobol sequence and apply
    inverse-CDF transforms to match the requested marginal distributions.

    Sobol sequences have better uniformity than Halton in higher dimensions
    and their scrambled variant avoids the base-correlation artefact.

    Args:
        dim:          Number of random dimensions (one per random parameter).
        n_obs:        Number of observations.
        n_draws:      Number of simulation draws per observation.
        distribution: List of length `dim` with distribution names:
                      'normal', 'lognormal', 'gamma', 'triangular', 'uniform'.
        scramble:     Use Owen scrambling for better uniformity (default True).
        as_jax:       Return a jnp.ndarray instead of np.ndarray (default True).

    Returns:
        Array of shape (n_obs * n_draws, dim) with transformed draws.
    """
    total = n_obs * n_draws
    # Sobol works best with power-of-2 sample counts; round up then trim
    n_sobol = _next_power_of_2(total)

    engine = Sobol(d=dim, scramble=scramble, seed=42)
    k = int(np.log2(n_sobol))
    try:
        uniform = engine.random_base2(k)      # power-of-2 draw (deprecated in scipy >= 1.12)
    except (AttributeError, TypeError):
        uniform = engine.random(n_sobol)       # fallback for newer scipy
    uniform = uniform[:total]                  # trim to exact count needed

    # Apply per-dimension inverse-CDF transforms
    sample = uniform.copy()
    for i, dist in enumerate(distribution):
        u = uniform[:, i]
        if dist == "normal":
            sample[:, i] = norm.ppf(u)
        elif dist == "lognormal":
            sample[:, i] = np.exp(norm.ppf(u))
        elif dist == "gamma":
            sample[:, i] = gamma.ppf(u, a=1.0)
        elif dist in ("triangular", "triang"):
            sample[:, i] = triang.ppf(u, c=0.5)
        elif dist == "uniform":
            sample[:, i] = 2.0 * u - 1.0          # maps [0,1] â†’ [-1, 1]
        else:
            warnings.warn(
                f"Unknown distribution '{dist}' at dim {i}; keeping uniform [0,1].",
                UserWarning,
                stacklevel=2,
            )

    return jnp.array(sample, dtype=jnp.float64) if as_jax else sample


# Backward-compatible alias so existing code calling prepare_halton still works
def prepare_halton(
    dim: int,
    n_obs: int,
    n_draws: int,
    distribution: list[str],
    scramble: bool = True,
    as_jax: bool = True,
) -> np.ndarray | jnp.ndarray:
    """Alias for generate_sobol_draws (Sobol replaces legacy Halton draws)."""
    return generate_sobol_draws(dim, n_obs, n_draws, distribution, scramble, as_jax)


class count_model:
    """Mixed Poisson regression estimated by simulated maximum likelihood."""

    def __init__(self, nDraws: int, distribution: list[str] | None = None, predictor: str = "FREQ"):
        self.Ndraws = nDraws
        self.distribution = distribution or []
        self.predictor = predictor
        self._draws_cache: jnp.ndarray | None = None
        self._x_data: pd.DataFrame | None = None
        self.observations: int | None = None

    # ------------------------------------------------------------------
    # Draw management
    # ------------------------------------------------------------------

    def _get_draws(self, dim: int, n_obs: int) -> jnp.ndarray:
        """Return cached Sobol draws or generate fresh ones."""
        if self._draws_cache is None:
            self._draws_cache = generate_sobol_draws(
                dim, n_obs, self.Ndraws, self.distribution
            )
        return self._draws_cache

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def PoissonNegLogLikelihood(self, lam: np.ndarray, y: np.ndarray) -> float:
        prob = np.exp(-lam) * np.power(lam, y) / np.vectorize(np.math.factorial)(y.astype(int))
        prob = prob.reshape(self.observations, -1, order="F")
        log_lik = np.sum(np.log(np.maximum(prob.mean(axis=1), 1e-300)))
        return -log_lik

    def poissonRegressionNegLogLikelihood(
        self, b: np.ndarray, X: np.ndarray, y: np.ndarray, Xr: np.ndarray | None = None
    ) -> float:
        if Xr is not None:
            n, p   = X.shape
            _,  pr = Xr.shape

            bf  = b[:p]
            br  = b[p : p + pr]
            brstd = b[p + pr : p + pr + pr]

            draws = np.array(self._get_draws(pr, n))  # (n*R, pr)

            eta      = np.tile(X @ bf, (1, self.Ndraws))        # (n, R)
            beta     = draws * brstd + br                        # (n*R, pr)
            data_rep = np.tile(Xr, (self.Ndraws, 1))            # (n*R, pr)
            het      = eta.ravel(order="F") + (data_rep * beta).sum(axis=1)
            lam      = np.exp(het)
            y_rep    = np.tile(y, (1, self.Ndraws))
            return self.PoissonNegLogLikelihood(lam, y_rep)
        else:
            eta = X @ b
            lam = np.exp(eta)
            return self.PoissonNegLogLikelihood(lam, y.reshape(-1, 1))

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def fitPoissonRegression(
        self, X: np.ndarray, y: np.ndarray, Xr: np.ndarray | None = None
    ):
        if Xr is not None:
            n, p   = X.shape
            _,  pr = Xr.shape
            n_params = p + 2 * pr
            b0       = np.zeros(n_params)

            result = minimize(
                self.poissonRegressionNegLogLikelihood,
                b0,
                args=(X, y, Xr),
                method="SLSQP",
                tol=1e-6,
                options={"maxiter": 2000},
            )

            # Standard errors from numerical Hessian (finite difference)
            eps    = 1e-5
            n_p    = len(result.x)
            H      = np.zeros((n_p, n_p))
            f0     = self.poissonRegressionNegLogLikelihood(result.x, X, y, Xr)
            for i in range(n_p):
                ei       = np.zeros(n_p)
                ei[i]    = eps
                fi       = self.poissonRegressionNegLogLikelihood(result.x + ei, X, y, Xr)
                for j in range(i, n_p):
                    ej   = np.zeros(n_p)
                    ej[j] = eps
                    fij  = self.poissonRegressionNegLogLikelihood(result.x + ei + ej, X, y, Xr)
                    fj   = self.poissonRegressionNegLogLikelihood(result.x + ej, X, y, Xr)
                    H[i, j] = H[j, i] = (fij - fi - fj + f0) / (eps ** 2)

            try:
                cov    = np.linalg.inv(H)
                stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
            except np.linalg.LinAlgError:
                stderr = np.full(n_p, np.nan)

            z_vals  = np.where(stderr > 1e-12, result.x / stderr, 0.0)
            p_vals  = 2.0 * norm.sf(np.abs(z_vals))
            return -result.fun, result.x, stderr, p_vals
        else:
            _, p = X.shape
            b0   = np.zeros(p)
            result = minimize(
                self.poissonRegressionNegLogLikelihood,
                b0,
                args=(X, y, None),
                method="SLSQP",
                tol=1e-6,
                options={"maxiter": 2000},
            )
            return result.x

    # ------------------------------------------------------------------
    # Data utilities
    # ------------------------------------------------------------------

    def tranformer(self, transform, idc):
        if transform in (0, 1, None):
            tr = self._x_data.iloc[:, idc]
        elif transform == "ln":
            tr = np.log(self._x_data.iloc[:, idc])
        elif transform == "exp":
            tr = np.exp(self._x_data.iloc[:, idc])
        elif transform == "sqrt":
            tr = np.power(self._x_data.iloc[:, idc], 0.5)
        else:
            tr = np.power(self._x_data.iloc[:, idc], transform)

        if not np.any(np.isfinite(tr)):
            tr = self._x_data.iloc[:, idc]
        return tr
