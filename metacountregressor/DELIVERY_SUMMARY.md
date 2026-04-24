# CMF Methods: Complete Delivery Package

## What You Now Have

### 1. **Slides Presentation** (`cmf_technical_slides.qmd`)
- **Format:** Quarto reveal.js slides (HTML)
- **File:** `cmf_technical_slides.html`
- **Content:** 14 slides covering:
  - CMF definition and interpretation
  - Problem with traditional models
  - Hierarchical CMF solution
  - Your fitted Example 16-3 models with real coefficients
  - 5 concrete benefits of CMF approach
  - Practical workflow
  - Summary and recommendations

**To view:** Open `cmf_technical_slides.html` in a web browser
**To edit:** Edit `cmf_technical_slides.qmd`, then run `quarto render cmf_technical_slides.qmd`

### 2. **Technical Report** (`cmf_technical_report.qmd`)
- **Format:** Quarto document (HTML + PDF)
- **Files:** `cmf_technical_report.html`, `cmf_technical_report.pdf`
- **Content:** Comprehensive academic paper (~4000 words) with:
  - CMF formalization and equations
  - Problem analysis (traditional models)
  - Detailed hierarchical CMF structure
  - Empirical demonstration on Example 16-3
  - Coefficient comparison (CURVES: +0.78% vs -8.05%)
  - 5 detailed benefit sections with worked examples
  - Practical workflow with code examples
  - Sensitivity analysis
  - References

**To view:** Open either HTML (interactive) or PDF (printable)
**To edit:** Edit `cmf_technical_report.qmd`, then run `quarto render cmf_technical_report.qmd`

### 3. **Package Updates** (`experiment_package.py` & `cmf_package.py`)
**New Methods Added:**

#### A. `ExperimentBuilder.print_cmf_interpretation(fit_result, aadt_col=None, aadt_median=None)`
- Works with any count model (traditional NB, Poisson, latent-class, etc.)
- Automatically converts coefficients (β) to CMF values and percent changes
- Provides plain-language interpretations in HSM style
- Optional AADT-dependent calculations

**Example:**
```python
from metacountregressor import ExperimentBuilder

builder = ExperimentBuilder(df=data, id_col='ID', y_col='Crashes')
fit_result = builder.fit_manual_model(spec, model="nb")

# NEW: Get CMF interpretations automatically
cmf_table = builder.print_cmf_interpretation(
    fit_result, 
    aadt_col='AADT'  # Optional
)
```

**Output:**
```
Parameter           Coefficient    CMF(+1)    Percent Change    Interpretation
CURVES              +0.007754      1.0078     +0.78%           CURVES +1 → +0.78% crashes
ACCESS              -0.213372      0.8079     -19.2%           ACCESS +1 → -19.2% crashes
```

#### B. `CMFExperimentBuilder.print_cmf_interpretation(fit_result)`
- Specific to hierarchical CMF models
- Separates **baseline block** (inherent risk) from **AADT-response block** (traffic sensitivity)
- Computed at median AADT for context
- Structured, block-specific output

**Example:**
```python
from metacountregressor import CMFExperimentBuilder

cmf_builder = CMFExperimentBuilder(
    df=data, y_col='Crashes', aadt_col='AADT',
    baseline_vars=['ACCESS', 'GRADEBR'],
    local_vars=['CURVES', 'WIDTH']
)

search_result = cmf_builder.run_search()
fit_result = cmf_builder.fit_best_model(search_result)

# NEW: Get CMF interpretations with structure
cmf_table = cmf_builder.print_cmf_interpretation(fit_result)
```

**Output (structured):**
```
BASELINE BLOCK (Inherent Crash Risk Factors):
ACCESS    β = -0.160110 → CMF = 0.8520 → -14.80% crashes

AADT-RESPONSE BLOCK (Traffic-Dependent Factors):
CURVES    β = -0.008395 → Effect at 23,771 AADT = -8.05% crashes
```

### 4. **Documentation & Guides**

#### A. `CMF_INTERPRETATION_GUIDE.md`
- Complete reference for new methods
- When to use each method
- Understanding the output
- Integration with workflow
- Examples and best practices

#### B. `demo_cmf_interpretation.py`
- Working demo script
- Example 1: Traditional model with CMF output
- Example 2: Hierarchical CMF model with CMF output
- Typical workflow walkthrough
- Interpretation examples

---

## How They Work Together

### For Creating Presentations

**Workflow:**
1. **Data Analysis Phase:**
   - Fit your models using the package
   - Call `print_cmf_interpretation()` to get automatic CMF tables
   - Export tables to CSV/Excel

2. **Presentation Creation:**
   - Use `cmf_technical_slides.qmd` as template
   - Insert your CMF tables and data
   - Customize the slides with your specific findings
   - Render to HTML reveal.js

3. **Audience:**
   - Engineers: Show slides, emphasize practical benefits
   - Managers: Show percent-change tables (CMF values)
   - Academics: Show technical report, equations, rigor

### For Writing Reports

**Workflow:**
1. **Content Generation:**
   - Fit your models
   - Get CMF interpretations from package
   - Gather coefficients and percent changes

2. **Report Writing:**
   - Use `cmf_technical_report.qmd` as template
   - Replace example data with your actual models
   - Update fitted coefficients and metrics
   - Render to PDF or HTML

3. **Distribution:**
   - PDF for printing/archiving
   - HTML for interactive viewing
   - Embed tables in Word documents

---

## Key Numbers from Your Example 16-3 Data

These are baked into both the slides and technical report. You can customize them:

### Model Comparison

| Metric | Traditional NB | CMF Hierarchical | Difference |
|--------|---|---|---|
| **BIC (with offset)** | 1925.52 | 1931.82 | +6.3 (marginal) |
| **CURVES +1** | +0.78% | -8.05% | -8.83 pp (opposite!) |
| **ACCESS +1** | -19.2% | -14.8% | +4.4 pp |
| **Interpretation** | Direct effects | Baseline vs. response | Clearer mechanism |

### Key Insight
The **same feature shows opposite effects** in the two models:
- **Traditional:** CURVES increase crashes directly
- **CMF:** CURVES reduce AADT elasticity (safer at high traffic)

This shows why mechanistic separation matters.

---

## Customizing for Your Data

### Updating the Slides

Edit `cmf_technical_slides.qmd`:
- Replace "Example 16-3" with your dataset name
- Update coefficients in the tables (Slide 8-9)
- Update AADT median (23,771 in examples)
- Change fitted model specifications as needed
- Render: `quarto render cmf_technical_slides.qmd`

### Updating the Technical Report

Edit `cmf_technical_report.qmd`:
- Update all coefficient tables (Section 5)
- Replace BIC values
- Update percent-change examples
- Change AADT context values
- Render: `quarto render cmf_technical_report.qmd`

### Using in Your Code

```python
# Your data analysis
fit_result = builder.fit_manual_model(spec, model="nb")

# Get CMF interpretations
cmf_table = builder.print_cmf_interpretation(fit_result)

# Export for presentation
cmf_table.to_csv("my_project_cmf.csv")
cmf_table.to_excel("my_project_cmf.xlsx")
```

---

## File Structure

```
metacountregressor/
├── cmf_technical_slides.qmd          ← Edit to customize slides
├── cmf_technical_slides.html         ← Open in browser
├── cmf_technical_report.qmd          ← Edit to customize report
├── cmf_technical_report.html         ← View online
├── cmf_technical_report.pdf          ← Print/archive
├── CMF_INTERPRETATION_GUIDE.md       ← Reference documentation
├── demo_cmf_interpretation.py        ← Working examples
├── experiment_package.py             ← Updated (new methods)
└── cmf_package.py                    ← Updated (new methods)
```

---

## Quick Start Checklist

- [ ] **View slides:** Open `cmf_technical_slides.html` in browser
- [ ] **View report:** Open `cmf_technical_report.html` or `cmf_technical_report.pdf`
- [ ] **Read guide:** Review `CMF_INTERPRETATION_GUIDE.md` for method reference
- [ ] **Run demo:** Execute `python demo_cmf_interpretation.py`
- [ ] **Fit your model:** Use `ExperimentBuilder.fit_manual_model()`
- [ ] **Get CMF output:** Call `print_cmf_interpretation()` on fit_result
- [ ] **Export table:** Save CMF table to CSV/Excel for presentations

---

## What This Achieves

✅ **Slides Format:** Easy to present, full of content, no display issues
✅ **Technical Rigor:** Complete academic report with equations and citations
✅ **Automatic CMF Output:** Package automatically translates coefficients to safety language
✅ **HSM Compliance:** All interpretations follow Highway Safety Manual conventions
✅ **Practitioner-Ready:** Tables and interpretations ready for engineers and decision-makers
✅ **Editable:** All Quarto files (.qmd) are plain markdown - easy to customize
✅ **Reproducible:** Fits are documented, coefficients visible, percent-changes computed transparently

---

## Next Steps

1. **View the slides and report** to understand the CMF approach
2. **Read the interpretation guide** to understand the new methods
3. **Run the demo** to see working examples
4. **Fit your own models** and use `print_cmf_interpretation()`
5. **Customize slides/report** with your data and findings
6. **Share with stakeholders** in your preferred format

---

## Support

For questions on:
- **CMF methodology** → See `cmf_technical_report.qmd`
- **How to use methods** → See `CMF_INTERPRETATION_GUIDE.md`
- **Working examples** → See `demo_cmf_interpretation.py`
- **Code implementation** → See `experiment_package.py` and `cmf_package.py`

All code includes docstrings explaining parameters and return values.
