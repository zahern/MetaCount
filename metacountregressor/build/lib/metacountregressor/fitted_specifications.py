"""
fitted_specifications.py
------------------------
Pre-defined model specifications derived from the Example 16-3 dataset
(Washington, Karlaftis & Mannering - Statistical and Econometric Methods
for Transportation Data Analysis).

These are *structural* specifications only (which variables play which
roles, what distributions).  To recover the numerical parameter estimates
pass the spec to ``ExperimentBuilder.fit_manual_model()``.

Example
-------
>>> from metacountregressor import (
...     load_example16_3_model_data,
...     load_book_latent_class_spec,
...     ExperimentBuilder,
... )
>>> df = load_example16_3_model_data()
>>> spec = load_book_latent_class_spec()
>>> builder = ExperimentBuilder(df=df, id_col="ID", y_col="FREQ",
...                             offset_col="OFFSET")
>>> fit = builder.fit_manual_model(manual_spec=spec, model="nb", R=200)
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_spec(title: str, spec: dict[str, Any]) -> None:
    width = 68
    print("=" * width)
    print(f"  {title}")
    print("=" * width)

    _SECTION_LABELS = {
        "fixed_terms":      "Fixed-effect variables",
        "rdm_terms":        "Random-effect variables  (var:distribution)",
        "rdm_cor_terms":    "Correlated random-effects",
        "grouped_terms":    "Grouped random-effects",
        "hetro_in_means":   "Heterogeneity-in-means variables",
        "zi_terms":         "Zero-inflation variables",
        "membership_terms": "Latent-class membership variables",
        "dispersion":       "Dispersion",
        "latent_classes":   "Number of latent classes",
    }

    _DISPERSION_LABELS = {0: "Poisson", 1: "Negative Binomial"}

    for key, label in _SECTION_LABELS.items():
        val = spec.get(key)
        if val is None:
            continue
        if key == "dispersion":
            val = _DISPERSION_LABELS.get(int(val), str(val))
        if isinstance(val, list):
            if val:
                print(f"\n  {label}:")
                for item in val:
                    print(f"    - {item}")
            else:
                print(f"\n  {label}: (none)")
        else:
            print(f"\n  {label}: {val}")

    print("=" * width)
    print()


# ---------------------------------------------------------------------------
# 2-class Negative-Binomial Latent Class model (Example 16-3)
# ---------------------------------------------------------------------------

def load_book_latent_class_spec() -> dict[str, Any]:
    """
    Return the structural specification for a 2-class Negative Binomial
    latent-class crash-frequency model representative of Example 16-3.

    The specification captures:
    - Fixed geometric/environmental covariates in the outcome equation
    - A lognormal random parameter for CURVES (capturing unobserved
      heterogeneity in curvature effects across sites)
    - A normal random parameter for LENGTH (site exposure heterogeneity)
    - FC_ENCODED as the latent-class membership variable so that
      functional class of road drives class assignment probability
    - Two latent classes with Negative Binomial dispersion

    To re-estimate the model::

        from metacountregressor import (
            load_example16_3_model_data,
            load_book_latent_class_spec,
            ExperimentBuilder,
        )
        df   = load_example16_3_model_data()
        spec = load_book_latent_class_spec()
        builder = ExperimentBuilder(
            df=df, id_col="ID", y_col="FREQ", offset_col="OFFSET"
        )
        fit = builder.fit_manual_model(manual_spec=spec, model="nb", R=200)
        print(fit)

    Returns
    -------
    dict
        A manual_spec dict suitable for
        ``ExperimentBuilder.fit_manual_model()``.
    """
    return {
        # Always-included fixed-effect terms
        "fixed_terms": [
            "URB",       # urban/rural indicator (0/1)
            "ACCESS",    # access-point density
            "GRADEBR",   # presence of grade break
        ],
        # Random parameters - format: "variable:distribution"
        "rdm_terms": [
            "CURVES:lognormal",  # lognormal ensures positive effect direction
            "LENGTH:normal",     # site segment length - symmetric uncertainty
        ],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": [],
        "zi_terms": [],
        # Membership equation: FC_ENCODED shifts class-assignment probability
        "membership_terms": ["FC_ENCODED"],
        # 1 = Negative Binomial (over-dispersed count data)
        "dispersion": 1,
        # Two latent sub-populations
        "latent_classes": 2,
    }


def describe_book_latent_class_spec() -> None:
    """
    Print a human-readable description of the Example 16-3 book specification.
    """
    spec = load_book_latent_class_spec()
    _print_spec(
        "Example 16-3  -  2-Class NB Latent-Class Specification",
        spec,
    )
    print(
        "  Variable notes:\n"
        "    URB        : 1 = urban road segment, 0 = rural\n"
        "    ACCESS     : number of access points per unit length\n"
        "    GRADEBR    : 1 = grade break present\n"
        "    CURVES     : curvature (lognormal - strictly positive effect)\n"
        "    LENGTH     : segment length in miles\n"
        "    FC_ENCODED : encoded functional class (1-6 ordinal)\n"
        "                 drives latent class membership probability\n"
    )


# ---------------------------------------------------------------------------
# Simple fixed-effects NB baseline (single class)
# ---------------------------------------------------------------------------

def load_book_nb_baseline_spec() -> dict[str, Any]:
    """
    Return the structural specification for a single-class Negative Binomial
    model - the baseline against which the latent-class model is compared.

    This corresponds to the simpler regression model in Example 16-3 before
    latent classes are introduced.

    Returns
    -------
    dict
        A manual_spec dict suitable for
        ``ExperimentBuilder.fit_manual_model()``.
    """
    return {
        "fixed_terms": [
            "URB",
            "ACCESS",
            "GRADEBR",
            "CURVES",
            "LENGTH",
        ],
        "rdm_terms": [],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": [],
        "zi_terms": [],
        "membership_terms": [],
        "dispersion": 1,
        "latent_classes": 1,
    }


def describe_book_nb_baseline_spec() -> None:
    """Print a human-readable description of the baseline NB specification."""
    spec = load_book_nb_baseline_spec()
    _print_spec(
        "Example 16-3  -  Single-Class NB Baseline Specification",
        spec,
    )


# ---------------------------------------------------------------------------
# CMF specification (AADT-based, two-component)
# ---------------------------------------------------------------------------

def load_book_cmf_spec() -> dict[str, Any]:
    """
    Return a representative CMF model specification for Example 16-3
    using the two-component AADT structure::

        log(mu_i) = [alpha_0 + sum_k alpha_k * X_ki]
                  + [beta_0  + sum_k beta_k  * X_ki] * log(AADT_i)

    Use with ``CMFExperimentBuilder``::

        from metacountregressor import (
            load_example16_3_model_data,
            load_book_cmf_spec,
            CMFExperimentBuilder,
        )
        df   = load_example16_3_model_data()
        spec = load_book_cmf_spec()

        cmf = CMFExperimentBuilder(
            df=df,
            y_col="FREQ",
            aadt_col="AADT",
            baseline_vars=spec["baseline_vars"],
            local_vars=spec["local_vars"],
        )
        manual_spec = cmf.make_manual_cmf_spec(
            baseline_fixed=spec["baseline_fixed"],
            local_fixed=spec["local_fixed"],
            baseline_random=spec["baseline_random"],
        )
        fit = cmf.fit_manual_cmf_model(id_col="ID", manual_spec=manual_spec, R=200)

    Returns
    -------
    dict
        Keys: ``baseline_vars``, ``local_vars``,
        ``baseline_fixed``, ``local_fixed``, ``baseline_random``.
    """
    return {
        # Variables that enter the baseline component (outside log AADT)
        "baseline_vars": ["URB", "ACCESS", "GRADEBR", "CURVES"],
        # Variables that enter the AADT-interaction component
        "local_vars": ["CURVES", "WIDTH"],
        # Fixed-effect assignment within each component
        "baseline_fixed": ["URB", "ACCESS", "GRADEBR"],
        "local_fixed": ["WIDTH"],
        # Random parameters in the baseline component
        "baseline_random": ["CURVES"],
    }


def describe_book_cmf_spec() -> None:
    """Print a human-readable description of the book CMF specification."""
    spec = load_book_cmf_spec()
    width = 68
    print("=" * width)
    print("  Example 16-3  -  CMF (AADT Two-Component) Specification")
    print("=" * width)
    for key, val in spec.items():
        print(f"\n  {key}: {val}")
    print("=" * width)
    print(
        "\n  CMF interpretation:\n"
        "    Baseline component: CMF = exp(alpha_k)\n"
        "    Local component:    CMF = AADT_mean ^ beta_k\n"
    )


# ---------------------------------------------------------------------------
# Convenience listing
# ---------------------------------------------------------------------------

def list_book_specifications() -> None:
    """Print a summary of all available pre-defined specifications."""
    print(
        "Available book specifications\n"
        "-----------------------------------------------------------------\n"
        "  load_book_latent_class_spec()   2-class NB LC  (Example 16-3)\n"
        "  load_book_nb_baseline_spec()    Single-class NB baseline\n"
        "  load_book_cmf_spec()            CMF two-component AADT model\n"
        "\n"
        "All specs are structural only - pass to fit_manual_model() or\n"
        "CMFExperimentBuilder.fit_manual_cmf_model() to get estimates.\n"
    )
