"""Shim layer that re-exports the bivariate copula NB module so the
package works whether it was installed in flat-layout mode (the most
common dev setup) or via the re-export wrapper used at top level.
"""
from bivariate_copula import (
    # Main types
    BivariateCopulaFit,
    BivariateCopulaNB,
    # Copula primitives
    frank_logpdf,
    normal_logpdf,
    kimeldorf_sampson_logpdf,
    # Marginals
    nb_log_pmf,
    univariate_nb_loglik,
    # Joint log-likelihoods
    bivariate_copula_loglik,
    famoye_bivariate_nb_loglik,
    marshall_olkin_nb_loglik,
    # Fitters
    fit_copula_bivariate_nb,
    fit_famoye_nb,
    fit_marshall_olkin_nb,
    compare_bivariate_copulas,
)

__all__ = [
    "BivariateCopulaFit",
    "BivariateCopulaNB",
    "frank_logpdf",
    "normal_logpdf",
    "kimeldorf_sampson_logpdf",
    "nb_log_pmf",
    "univariate_nb_loglik",
    "bivariate_copula_loglik",
    "famoye_bivariate_nb_loglik",
    "marshall_olkin_nb_loglik",
    "fit_copula_bivariate_nb",
    "fit_famoye_nb",
    "fit_marshall_olkin_nb",
    "compare_bivariate_copulas",
]
