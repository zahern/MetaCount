% CMF Interpretation Features - Quick Reference
% Updated Package Features for Road Safety Analysis

# Package Updates: CMF Interpretation Output

Your `metacountregressor` package now automatically outputs **CMF (Crash Modification Factor)** interpretations alongside fitted coefficients. This makes it easy to communicate safety effects in road-safety language.

---

## What Changed?

### New Methods Added

#### 1. `ExperimentBuilder.print_cmf_interpretation()`

**Where to use:** After fitting any count model (NB, Poisson, with or without latent classes)

**What it does:**
- Converts fitted coefficients (β) into CMF values
- Calculates percent changes using: $100 \times (\exp(\beta) - 1)$
- Provides plain-language interpretations
- Optionally includes AADT-dependent calculations

**Signature:**
```python
cmf_table = builder.print_cmf_interpretation(
    fit_result=fit_result,
    aadt_col=None,                    # Optional: column with AADT values
    aadt_median=None                  # Optional: explicit median AADT
)
```

**Example:**
```python
# Fit a traditional NB model
fit_result = builder.fit_manual_model(manual_spec, model="nb")

# Get standard coefficients
coef_table = builder.print_coefficients(fit_result)

# NEW: Get CMF interpretations
cmf_table = builder.print_cmf_interpretation(fit_result, aadt_col='AADT')
```

**Output includes:**
```
Parameter           Coefficient    CMF(+1)    Percent Change    Interpretation
ACCESS              -0.160110      0.8520     -14.80%          ACCESS +1 → -14.80% crashes (safer)
CURVES              +0.007754      1.0078     +0.78%           CURVES +1 → +0.78% crashes (riskier)
WIDTH               -0.012352      0.9877     -1.24%           WIDTH +1 → -1.24% crashes (safer)
```

---

#### 2. `CMFExperimentBuilder.print_cmf_interpretation()`

**Where to use:** After fitting a hierarchical CMF model with `fit_best_model()`

**What it does:**
- Separates **baseline block** (inherent risk) from **AADT-response block** (traffic sensitivity)
- Shows which features affect crashes directly vs. how they scale with traffic
- Computed at **median AADT** for context
- Provides structured, block-specific interpretations

**Signature:**
```python
cmf_table = cmf_builder.print_cmf_interpretation(fit_result)
```

**Example Workflow:**
```python
from metacountregressor import CMFExperimentBuilder

# Initialize CMF builder
cmf_builder = CMFExperimentBuilder(
    df=data,
    y_col='Crashes',
    aadt_col='AADT',
    baseline_vars=['ACCESS', 'GRADEBR', 'URB'],      # Inherent risk
    local_vars=['CURVES', 'WIDTH']                    # Traffic-response
)

# Run search and fit best model
search_result = cmf_builder.run_search(R=200)
fit_result = cmf_builder.fit_best_model(search_result)

# OLD: Just print standard results
cmf_builder.print_report(search_result, fit_result)

# NEW: Get CMF interpretations
cmf_table = cmf_builder.print_cmf_interpretation(fit_result)
```

**Output Example:**
```
BASELINE BLOCK (Inherent Crash Risk Factors):
─────────────────────────────────────────────
ACCESS                   β = -0.160110
  CMF(+1 unit)     : 0.8520
  Effect (+1 unit) : -14.80%
  ➜ ACCESS +1 → -14.80% crashes

AADT-RESPONSE BLOCK (Traffic-Dependent Factors):
─────────────────────────────────────────────────
CURVES                   β = -0.008395
  Elasticity effect: 0.9917
  Effect at 23,771 AADT: -8.05%
  ➜ CURVES +1 → Reduces AADT elasticity; at median AADT: -8.05% crashes
```

---

## Understanding the Output

### Standard Coefficient → CMF Conversion

The math is simple:

**CMF for a one-unit change:**
$$\text{CMF} = \exp(\beta)$$

**Percent change in crashes:**
$$\text{Percent Change} = 100 \times (\exp(\beta) - 1)$$

### Example Interpretation

**Traditional Model:**
```
Parameter: CURVES
Coefficient (β) = +0.007754
CMF = exp(0.007754) = 1.0078
Percent Change = +0.78%

Meaning: "One additional curve per mile is associated with 0.78% more crashes"
```

**Hierarchical CMF Model:**
```
Component: Baseline Block
Parameter: ACCESS
Coefficient (β) = -0.160110
CMF = exp(-0.160110) = 0.852
Percent Change = -14.80%

Meaning: "Baseline crashes reduce by 14.80% for each additional access point"

Component: AADT-Response Block
Parameter: CURVES
Coefficient (β) = -0.008395
Effect at 23,771 AADT: 23771^(-0.008395) = 0.9205 → -8.05%

Meaning: "Curves reduce how sensitive crashes are to traffic volume. 
          At median traffic, this translates to 8.05% fewer crashes"
```

---

## When to Use Each Method

### Use `ExperimentBuilder.print_cmf_interpretation()` for:
- ✅ Traditional count models (single equation)
- ✅ Latent-class models
- ✅ Heterogeneous models
- ✅ When you want quick CMF conversions

### Use `CMFExperimentBuilder.print_cmf_interpretation()` for:
- ✅ Hierarchical CMF models specifically
- ✅ When you need baseline vs. response separation
- ✅ Road-safety projects requiring HSM-style documentation
- ✅ When AADT context is critical

---

## Output Format

### Column Definitions

| Column | Meaning |
|--------|---------|
| **Parameter** | Variable name (e.g., "CURVES", "ACCESS") |
| **Coefficient** | Fitted regression coefficient (β) |
| **CMF(+1)** | exp(β) - the multiplicative effect for one-unit increase |
| **Percent Change** | 100 × (CMF - 1) - the percent-change metric (main safety metric) |
| **Interpretation** | Plain-language explanation of the effect |

### Safety Interpretation Guide

```
Percent Change < 0  →  CMF < 1.0  →  SAFER (fewer crashes) ✓
Percent Change = 0  →  CMF = 1.0  →  NEUTRAL (no change)
Percent Change > 0  →  CMF > 1.0  →  RISKIER (more crashes) ✗
```

---

## Integration with Your Workflow

### Before (Old Way)
```python
fit_result = builder.fit_manual_model(spec, model="nb")
coef_table = builder.print_coefficients(fit_result)
# Manual calculation: Need to do 100*(exp(β) - 1) yourself for CMF
```

### After (New Way)
```python
fit_result = builder.fit_manual_model(spec, model="nb")
coef_table = builder.print_coefficients(fit_result)
cmf_table = builder.print_cmf_interpretation(fit_result)  # ← Automatic CMF!
# All percent changes computed and interpreted automatically
```

---

## For Presentations & Reports

The CMF output is formatted for:
- ✅ Direct inclusion in PowerPoint/Google Slides tables
- ✅ Copy-paste into Word documents
- ✅ Integration with academic papers
- ✅ Sharing with traffic engineers who understand CMF conventions
- ✅ Compliance with HSM/AASHTO standards

### Example: Creating a CMF Summary Table for Stakeholders

```python
# Fit your model
fit_result = builder.fit_manual_model(spec, model="nb")

# Get CMF interpretations
cmf_table = builder.print_cmf_interpretation(fit_result)

# Export to CSV for presentation software
cmf_table.to_csv("cmf_summary.csv", index=False)

# Or to Excel with formatting
cmf_table.to_excel("cmf_summary.xlsx", index=False)
```

---

## Technical Details

### HSM Formula (Standard Road Safety)

$$\text{CMF}(a \rightarrow b) = \exp(\beta \times (b - a))$$

For a one-unit change ($b = a + 1$):

$$\text{CMF} = \exp(\beta)$$

$$\text{Percent Change} = 100 \times (\exp(\beta) - 1)$$

### How the Package Implements This

1. Extracts fitted coefficients (β) from the model result
2. For each coefficient, computes: `cmf = exp(β)`
3. For each CMF, computes: `percent_change = 100 * (cmf - 1)`
4. Generates interpretation text based on sign and magnitude
5. If AADT column provided, computes traffic-dependent effects: `exp(β * ln(AADT))`
6. Formats everything into a readable table

---

## Examples & Demo

See **`demo_cmf_interpretation.py`** for:
- Complete working examples
- How to structure your data
- How to call the new methods
- How to interpret the output

---

## Questions?

The CMF interpretation methods are designed to make your model outputs immediately understandable to road-safety practitioners. 

**Key principle:** Every coefficient should be interpretable as a safety effect without additional manual calculation.
