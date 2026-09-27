import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax

try:
    from .nb_parameterization import nb2_logpmf
except ImportError:
    from nb_parameterization import nb2_logpmf


def loglik_gradient_clean(
    betas,
    X,
    y,
    offset,
    dispersion
):
    """
    Pure JAX log-likelihood (Poisson or NB2).
    Returns scalar negative log-likelihood.
    """

    # ---------- Linear predictor ----------
    eta = jnp.dot(X, betas[:-1]) + offset

    # ---------- Mean ----------
    mu = jnp.exp(eta)

    # ---------- Poisson case ----------
    def poisson_case(_):
        loglik = (
            y * jnp.log(mu)
            - mu
            - jsp.special.gammaln(y + 1)
        )
        return -jnp.sum(loglik)

    # ---------- Negative Binomial case ----------
    def nb_case(_):
        loglik = nb2_logpmf(y, eta, betas[-1])

        return -jnp.sum(loglik)

    loss = lax.cond(
        dispersion == 0,
        poisson_case,
        nb_case,
        operand=None
    )

    return loss

class CleanCountModel:

    def __init__(self, X, y, offset=None, dispersion=0):
        self.X = jnp.asarray(X)
        self.y = jnp.asarray(y)
        self.offset = jnp.zeros_like(y) if offset is None else jnp.asarray(offset)
        self.dispersion = dispersion

    def loss(self, params):
        return loglik_gradient_clean(
            params,
            self.X,
            self.y,
            self.offset,
            self.dispersion
        )

    def fit(self, init_params):
        solver = jaxopt.BFGS(fun=self.loss)
        result = solver.run(init_params)
        return result