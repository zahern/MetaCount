"""
Demo: Using CMF Interpretation Output from metacountregressor Package

This script shows how to:
1. Fit a count model
2. Print standard coefficients  
3. Print CMF interpretations (NEW!)
4. Understand the HSM-style percentage changes
"""

import pandas as pd
import numpy as np
from metacountregressor import ExperimentBuilder, CMFExperimentBuilder

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
    'fixd_terms': ['CURVES', 'WIDTH', 'ACCESS'],
    'rdm_terms': [],
    'rdm_cor_terms': [],
    'grouped_terms': [],
}

# Fit model
print("\nFitting traditional negative binomial model...")
try:
    fit_result = builder.fit_manual_model(
        manual_spec=manual_spec,
        model="nb",
        R=100,
        print_report=False
    )
    
    # Print standard coefficients
    print("\n" + "-" * 100)
    print("STANDARD COEFFICIENT TABLE:")
    print("-" * 100)
    coef_table = builder.print_coefficients(fit_result)
    
    # NEW: Print CMF interpretations
    print("\n" + "-" * 100)
    print("CMF INTERPRETATION TABLE (NEW!):")
    print("-" * 100)
    cmf_table = builder.print_cmf_interpretation(
        fit_result=fit_result,
        aadt_col='AADT',
    )
    print("\nCMF Table:")
    print(cmf_table.to_string(index=False))
    
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
    # Initialize CMF builder
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
    print("TYPICAL WORKFLOW:")
    print("-" * 100)
    print("""
    1. Run GA search to select best baseline + response features:
       search_result = cmf_builder.run_search(R=200)
    
    2. Fit best model found by search:
       fit_result = cmf_builder.fit_best_model(search_result, final_R=500)
    
    3. Print standard CMF results table:
       cmf_builder.print_report(search_result, fit_result)
    
    4. NEW - Print CMF interpretations with HSM-style explanations:
       cmf_table = cmf_builder.print_cmf_interpretation(fit_result)
    
    The print_cmf_interpretation() method outputs:
    ✓ Baseline block coefficients and their crash percent changes
    ✓ AADT-response block elasticity effects
    ✓ Plain-language interpretations in HSM style
    ✓ Effects computed at median AADT for context
    """)
    
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
  Coefficient (β)   : +0.007754
  CMF for +1 unit   : exp(0.007754) = 1.0078
  Percent Change    : 100 × (1.0078 - 1) = +0.78%
  
  Interpretation: Adding 1 curve/mile increases crashes by 0.78%

─────────────────────────────────────────────────────────────────────────

HIERARCHICAL CMF MODEL OUTPUT:
  
  BASELINE BLOCK:
  Parameter: ACCESS    β = -0.160110
  CMF for +1: exp(-0.160110) = 0.852
  Percent Change: -14.8%
  Interpretation: +1 access point reduces baseline crashes by 14.8%
  
  AADT-RESPONSE BLOCK:
  Parameter: CURVES    β = -0.008395 (in AADT elasticity)
  CMF at median AADT (23,771): 23771^(-0.008395) = 0.9205
  Percent Change: -8.05%
  Interpretation: +1 curve/mile reduces AADT elasticity, leading to 8.05% 
                  fewer crashes at typical traffic volumes
  
  KEY INSIGHT: Traditional says curves ADD crashes; CMF says curves 
               REDUCE traffic sensitivity (possibly safer driving behavior)

─────────────────────────────────────────────────────────────────────────

HSM-STYLE CMF FORMULA (used internally):
  
  CMF(a → b) = exp(β × (b - a))
  Percent Change = 100 × (CMF - 1)
  
  For a one-unit increase (b = a + 1):
  CMF(a → a+1) = exp(β)
  
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
   ✓ Works with traditional and hierarchical count models
   ✓ Converts coefficients to CMF values and percent changes
   ✓ Provides HSM-style interpretation text
   ✓ Optional AADT-dependent calculations

2. CMFExperimentBuilder.print_cmf_interpretation(fit_result)
   ✓ Specific to hierarchical CMF models
   ✓ Separates baseline and AADT-response blocks
   ✓ Computed at median AADT for context
   ✓ Includes block-specific interpretation guide

USAGE:
  # After fitting a model, call:
  cmf_table = builder.print_cmf_interpretation(fit_result)
  
  # The output includes:
  # - Parameter names
  # - Fitted coefficients (β)
  # - CMF values
  # - Percent changes (the main safety metric)
  # - Intuitive interpretations in plain language

BENEFITS:
  ✓ Coefficients are immediately translated to safety language
  ✓ Practitioners don't need to calculate 100*(exp(β)-1) themselves
  ✓ Hierarchical model structure is visible in output
  ✓ AADT context is provided automatically
  ✓ Output matches HSM/AASHTO CMF conventions
""")

print("\n" + "=" * 100)
