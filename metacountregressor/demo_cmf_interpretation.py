"""
Demo: Using CMF Interpretation Output from metacountregressor Package

This script shows how to:
1. Fit a count model
2. Print standard coefficients  
3. Print CMF interpretations (NEW!)
4. Understand the HSM-style percentage changes
"""

import sys

import pandas as pd
import numpy as np
from metacountregressor import ExperimentBuilder, CMFExperimentBuilder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================================
# Example 1: Standard Count Model with CMF Interpretation
# ============================================================================

print("\n" + "=" * 100)
print("EXAMPLE 1: TRADITIONAL NEGATIVE BINOMIAL MODEL WITH CMF OUTPUT")
print("=" * 100)

# Load your data (assuming Example 16-3 or similar structure)
try:
    df = pd.read_csv("data/Ex-16-3.csv")
except FileNotFoundError:
    print("(Demo data not found; using synthetic example)")
    # Create synthetic crash data
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'ID': range(1, n + 1),
        'Crashes': np.random.poisson(5, n),
        'AADT': np.random.uniform(5000, 50000, n),
        'LENGTH': np.random.uniform(0.5, 3, n),
        'CURVES': np.random.uniform(0, 15, n),
        'WIDTH': np.random.uniform(9, 14, n),
        'ACCESS': np.random.poisson(2, n),
    })

# Initialize builder
builder = ExperimentBuilder(
    df=df,
    id_col='ID',
    y_col='Crashes',
    offset_col='LENGTH',
)

# Build a simple specification
manual_spec = {
    'fixed_terms': ['CURVES', 'WIDTH', 'ACCESS'],
    'rdm_terms': [],
    'rdm_cor_terms': [],
    'grouped_terms': [],
}

# Fit model
print("\nFitting traditional negative binomial model...")
standard_nb_fit = None
standard_nb_table = None
try:
    standard_nb_fit = builder.fit_manual_model(
        manual_spec=manual_spec,
        model="nb",
        R=32,
        print_report=False
    )
    
    # Print standard coefficients
    print("\n" + "-" * 100)
    print("STANDARD COEFFICIENT TABLE:")
    print("-" * 100)
    coef_table = builder.print_coefficients(standard_nb_fit)
    
    # NEW: Print CMF interpretations
    print("\n" + "-" * 100)
    print("CMF INTERPRETATION TABLE (NEW!):")
    print("-" * 100)
    standard_nb_table = builder.print_cmf_interpretation(
        fit_result=standard_nb_fit,
        aadt_col='AADT',
    )
    print("\nCMF Table:")
    print(standard_nb_table.to_string(index=False))
    
except Exception as e:
    print(f"Note: Could not fit model due to: {e}")
    print("This is expected if dependencies are not fully installed.")

# ============================================================================
# Example 2: Hierarchical CMF Model with CMF Interpretation
# ============================================================================

print("\n\n" + "=" * 100)
print("EXAMPLE 2: HIERARCHICAL CMF MODEL WITH CMF INTERPRETATION OUTPUT")
print("=" * 100)

try:
    # Initialize the current unified CMF builder
    cmf_builder = CMFExperimentBuilder(
        df=df,
        y_col='Crashes',
        aadt_col='AADT',
        baseline_vars=['ACCESS', 'WIDTH'],
        local_vars=['CURVES'],
    )
    
    print("\nCMFExperimentBuilder initialized with:")
    print(f"  Baseline features: {cmf_builder.baseline_vars}")
    print(f"  Traffic-response features: {cmf_builder.local_vars}")
    print(f"  Data shape: {cmf_builder.df.shape}")
    
    print("\n" + "-" * 100)
    print("POISSON VS NEGATIVE BINOMIAL: SAME HIERARCHICAL CMF SPECIFICATION")
    print("-" * 100)

    general_builder, evaluator, cmf_metadata = cmf_builder.build_jax_count_evaluator(
        id_col='ID',
        offset_col='LENGTH',
        variables=cmf_builder.baseline_vars + cmf_builder.local_vars,
        fixed_override={
            **{var: [1] for var in cmf_builder.baseline_vars},
            **{var: [1] for var in cmf_builder.local_vars},
        },
        max_latent_classes=1,
        R=32,
    )
    variable_count = len(evaluator.vars)
    base_decision = np.concatenate([
        np.ones(variable_count, dtype=int),
        np.zeros(variable_count, dtype=int),
        np.array([0], dtype=int),
    ])
    hierarchical_spec = evaluator.build_spec(base_decision)
    if hierarchical_spec is None:
        raise RuntimeError("The unified CMF evaluator rejected the hierarchical specification.")

    print("The same hierarchical structure is estimated in both models:")
    print(f"  Baseline block: {cmf_builder.baseline_vars}")
    print(f"  AADT-response block: {cmf_builder.local_vars} x log({cmf_builder.aadt_col})")
    print("  AADT term: log(AADT)")
    print(f"  Unified transformed terms: {cmf_metadata['interaction_cols']}")

    interpretation_tables = []
    fit_metrics = []
    for model_name in ("poisson", "nb"):
        print(f"\nEstimating hierarchical {model_name.upper()} model...")
        try:
            model_spec = dict(hierarchical_spec)
            model_spec["dispersion"] = int(model_name == "nb")
            fit_result = general_builder.fit_manual_model(
                manual_spec=model_spec,
                model=model_name,
                R=32,
            )
            model_table = cmf_builder.print_cmf_interpretation(
                fit_result=fit_result,
                model_label=model_name,
            )
            model_table.insert(0, "Model", model_name.upper())
            interpretation_tables.append(model_table)
            fit_metrics.append({
                "Model": model_name.upper(),
                "Log-Likelihood": fit_result["summary"]["loglik"],
                "AIC": fit_result["summary"]["aic"],
                "BIC": fit_result["summary"]["bic"],
            })
        except Exception as exc:
            print(f"Could not estimate {model_name.upper()} model: {exc}")

    if interpretation_tables:
        print("\nCASE-BY-CASE HIERARCHICAL CMF COMPARISON:")
        comparison = pd.concat(interpretation_tables, ignore_index=True)
        print(comparison[[
            "Model", "Component", "Parameter", "Coefficient",
            "CMF(+1)", "Percent Change",
        ]].to_string(index=False))

    if fit_metrics:
        print("\nMODEL FIT COMPARISON:")
        print(pd.DataFrame(fit_metrics).to_string(index=False))

    hierarchical_nb_table = next(
        (table for table in interpretation_tables
         if not table.empty and table.iloc[0]["Model"] == "NB"),
        None,
    )
    if standard_nb_table is not None and hierarchical_nb_table is not None:
        standard_curves = standard_nb_table[
            standard_nb_table["Parameter"] == "CURVES"
        ]
        hierarchical_curves = hierarchical_nb_table[
            hierarchical_nb_table["Parameter"] == "CURVES"
        ]
        if not standard_curves.empty and not hierarchical_curves.empty:
            direct_beta = float(standard_curves.iloc[0]["Coefficient"])
            hierarchical_beta = float(hierarchical_curves.iloc[0]["Coefficient"])
            median_aadt = float(df["AADT"].median())
            naive_percent = 100.0 * (
                np.exp(direct_beta * np.log(median_aadt)) - 1.0
            )
            print("\nNORMAL NB VS HIERARCHICAL NB: CURVES CASE")
            print("  Direct normal NB effect (+1 CURVES): "
                  f"{100.0 * (np.exp(direct_beta) - 1.0):+.2f}%")
            print("  Correct hierarchical NB effect at median AADT "
                  f"({median_aadt:,.0f}): "
                  f"{float(hierarchical_curves.iloc[0]['Percent Change']):+.2f}%")
            print("  Naive post-hoc transformation of the normal NB beta: "
                  f"{naive_percent:+.2f}%")
            print("  These differ because the hierarchical model estimates "
                  "CURVES x log(AADT) as a new coefficient; it cannot be "
                  "recovered from the direct CURVES coefficient after fitting.")
    
except Exception as e:
    print(f"Note: {e}")

# ============================================================================
# Example 3: Understanding CMF Output
# ============================================================================

print("\n" + "=" * 100)
print("EXAMPLE 3: INTERPRETING CMF OUTPUT")
print("=" * 100)

print("""
TRADITIONAL MODEL OUTPUT:
  Parameter          Estimate
  CURVES            +0.007754
  
CMF INTERPRETATION (what it means):
    Coefficient (beta) : +0.007754
  CMF for +1 unit   : exp(0.007754) = 1.0078
    Percent Change    : 100 x (1.0078 - 1) = +0.78%
  
  Interpretation: Adding 1 curve/mile increases crashes by 0.78%

-------------------------------------------------------------------------

HIERARCHICAL CMF MODEL OUTPUT:
  
  BASELINE BLOCK:
    Parameter: ACCESS    beta = -0.160110
  CMF for +1: exp(-0.160110) = 0.852
  Percent Change: -14.8%
  Interpretation: +1 access point reduces baseline crashes by 14.8%
  
  AADT-RESPONSE BLOCK:
    Parameter: CURVES    beta = -0.008395 (in AADT elasticity)
  CMF at median AADT (23,771): 23771^(-0.008395) = 0.9205
  Percent Change: -8.05%
  Interpretation: +1 curve/mile reduces AADT elasticity, leading to 8.05% 
                  fewer crashes at typical traffic volumes
  
  KEY INSIGHT: Traditional says curves ADD crashes; CMF says curves 
               REDUCE traffic sensitivity (possibly safer driving behavior)

-------------------------------------------------------------------------

HSM-STYLE CMF FORMULA (used internally):
  
    CMF(a -> b) = exp(beta x (b - a))
    Percent Change = 100 x (CMF - 1)
  
  For a one-unit increase (b = a + 1):
    CMF(a -> a+1) = exp(beta)
  
  This is the standard road safety formula from the Highway Safety Manual
  and AASHTO guidelines.
""")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 100)
print("SUMMARY: NEW CMF INTERPRETATION FEATURES")
print("=" * 100)

print("""
The metacountregressor package now automatically outputs CMF interpretations
when fitting count models. Two new methods are available:

1. ExperimentBuilder.print_cmf_interpretation(fit_result, aadt_col=None)
    [OK] Works with traditional and hierarchical count models
    [OK] Converts coefficients to CMF values and percent changes
    [OK] Provides HSM-style interpretation text
    [OK] Optional AADT-dependent calculations

2. CMFExperimentBuilder.print_cmf_interpretation(fit_result)
    [OK] Specific to hierarchical CMF models
    [OK] Separates baseline and AADT-response blocks
    [OK] Computed at median AADT for context
    [OK] Includes block-specific interpretation guide

USAGE:
  # After fitting a model, call:
  cmf_table = builder.print_cmf_interpretation(fit_result)
  
  # The output includes:
  # - Parameter names
    # - Fitted coefficients (beta)
  # - CMF values
  # - Percent changes (the main safety metric)
  # - Intuitive interpretations in plain language

BENEFITS:
    [OK] Coefficients are immediately translated to safety language
    [OK] Practitioners don't need to calculate 100*(exp(beta)-1) themselves
    [OK] Hierarchical model structure is visible in output
    [OK] AADT context is provided automatically
    [OK] Output matches HSM/AASHTO CMF conventions
""")

print("\n" + "=" * 100)
