---
marp: true
title: Hierarchical CMF vs Traditional Count Models
paginate: true
math: katex
theme: default
---

# Hierarchical CMF vs Traditional Count Models
## A General-Audience Walkthrough

Washington Example 16-3 crash data

---

## Why This Matters

When agencies decide on safety investments, they need models that are:

- accurate enough for forecasting
- understandable for non-technical stakeholders
- interpretable for policy and design decisions

---

## The Two Approaches

### Traditional Count Model

Models crash counts directly with road and traffic variables.

$$\log(\mu_i)=\beta_0+\sum_k\beta_k X_{ki}+\text{offset}_i$$

### Hierarchical CMF Model

Splits the problem into baseline risk and traffic-exposure interaction.

$$\log(\mu_i)=\underbrace{\alpha_0+\sum_k\alpha_k X^{\text{baseline}}_{ki}}_{\text{base risk}} + \underbrace{\beta_0+\sum_j\beta_j X^{\text{local}}_{ji}}_{\text{AADT sensitivity}}\log(\text{AADT}_i)$$

---

## What We Removed For Clarity

To keep this comparison easy to understand, this version excludes:

- latent class modeling
- zero-inflated modeling

The focus is now on a clean, apples-to-apples comparison:

- baseline negative binomial count model
- random-parameter negative binomial count model
- baseline CMF model
- random-parameter CMF model
- automated structure search in both frameworks

---

## Data and Problem Setup

Dataset: Washington Example 16-3

- response: crash frequency (`FREQ`)
- exposure: offset and AADT
- road descriptors: curvature, access, grade, width, etc.

Key first check: crash counts are overdispersed (variance greater than mean),
which supports negative binomial over plain Poisson for count models.

---

## Step 1: Baseline Traditional Model

We fit a standard negative binomial (NB) model with fixed effects.

What this gives the audience:

- a familiar benchmark
- easy-to-explain coefficient effects
- baseline BIC and prediction performance

---

## Step 2: Random-Parameter Traditional Model

We allow selected effects to vary across segments.

Why this helps:

- captures unobserved differences between sites
- often improves fit and prediction realism
- still stays interpretable with distribution assumptions

---

## Step 3: Baseline Hierarchical CMF Model

We estimate baseline and local CMF components jointly.

Why this helps communication:

- effects can be framed as crash modification behavior
- clearer policy narrative than raw coefficient-only discussions
- traffic interaction is built into the model structure

---

## Step 4: Random-Parameter Hierarchical CMF Model

We add random parameters to CMF terms.

Benefit:

- keeps CMF interpretation
- adds flexibility for site heterogeneity
- can improve predictive performance vs fixed-only CMF

---

## Step 5: Automated Search (No LC / No ZI)

Both frameworks run a constrained model search.

Search in this version allows:

- exclude / fixed / random-independent roles
- no latent classes
- no zero-inflation

Outcome:

- best structure by BIC for each framework
- transparent comparison of selected model families

---

## How To Read The Comparison

Primary metrics:

- **BIC**: lower is better (fit with complexity penalty)
- **RMSE / MAE**: lower is better (prediction error)
- **Correlation**: higher is better (observed vs predicted alignment)

Visuals to show:

- ranked BIC bar chart
- validation metric chart
- observed vs predicted scatter

---

## What A General Audience Should Take Away

- Traditional NB is a strong baseline and easy to explain.
- Hierarchical CMF offers a more policy-aligned structure.
- Random parameters improve realism in both frameworks.
- The "best" model depends on the goal:
  - pure forecast accuracy
  - interpretability for safety interventions

---

## Reproducible Assets

- Notebook (general-audience, no latent class / no zero inflation):
  `cmf_vs_count_comparison.ipynb`
- Slide deck source:
  `cmf_vs_count_comparison_slides.md`

---

## Recommended Presentation Flow

1. Start with the policy problem and why interpretability matters.
2. Explain traditional NB baseline first.
3. Introduce CMF structure as a communication advantage.
4. Show search-based, data-driven comparison results.
5. End with practical selection guidance for analysts and decision makers.
