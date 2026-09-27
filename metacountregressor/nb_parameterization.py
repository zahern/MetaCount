"""Shared, numerically stable NB2 parameterization helpers.

The optimizer works with ``log_alpha`` on the real line.  The model scale is
then ``alpha = exp(log_alpha)`` and the conditional variance is
``mu + alpha * mu**2``.  Keeping this transform in one place prevents the
JAX, NumPy, and simulation-based estimators from silently using different
NB dispersion conventions.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy as jsp

try:
    from ._jax_config import configure_jax
except ImportError:
    from _jax_config import configure_jax

configure_jax()


# These limits cover essentially all practical count-data dispersions while
# preventing overflow in exp() and pathological gamma evaluations during a
# failed optimizer trial.
LOG_ALPHA_MIN = -12.0
LOG_ALPHA_MAX = 12.0


def clipped_log_alpha(log_alpha):
    """Return a finite optimizer coordinate for the NB2 dispersion."""
    return jnp.clip(log_alpha, LOG_ALPHA_MIN, LOG_ALPHA_MAX)


def nb2_dispersion(log_alpha):
    """Map the unconstrained optimizer coordinate to positive alpha."""
    return jnp.exp(clipped_log_alpha(log_alpha))


def nb2_logpmf(y, eta, log_alpha):
    """NB2 log-PMF for ``log(mu) == eta`` and optimizer-scale dispersion.

    The denominator is evaluated as ``logaddexp(log(r), log(mu))`` rather
    than by forming ``r + mu``.  This avoids overflow when alpha is small and
    keeps the expression smooth for autodiff.
    """
    eta = jnp.clip(eta, -30.0, 30.0)
    log_alpha = clipped_log_alpha(log_alpha)
    inv_alpha = jnp.exp(-log_alpha)
    log_inv_alpha = -log_alpha
    log_denom = jnp.logaddexp(log_inv_alpha, eta)

    return (
        jsp.special.gammaln(y + inv_alpha)
        - jsp.special.gammaln(inv_alpha)
        - jsp.special.gammaln(y + 1.0)
        + inv_alpha * (log_inv_alpha - log_denom)
        + y * (eta - log_denom)
    )


def nb2_logpmf_from_mean(y, mu, log_alpha):
    """NB2 log-PMF for a positive mean and optimizer-scale dispersion."""
    log_mu = jnp.log(jnp.clip(mu, 1e-12, 1e12))
    return nb2_logpmf(y, log_mu, log_alpha)


__all__ = [
    "LOG_ALPHA_MIN",
    "LOG_ALPHA_MAX",
    "clipped_log_alpha",
    "nb2_dispersion",
    "nb2_logpmf",
    "nb2_logpmf_from_mean",
]
