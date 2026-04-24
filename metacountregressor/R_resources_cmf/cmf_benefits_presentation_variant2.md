---
title: "CMF Methods: Fitted Differences And Gains"
subtitle: "Paper-style variant with coefficient evidence"
format:
  revealjs:
    theme: simple
    slide-number: true
    incremental: false
    transition: fade
    scrollable: false
slide-level: 2
editor: source
include-in-header:
  text: |
    <style>
      .reveal .slides section { font-size: 0.93em; }
      .reveal .slides section.small { font-size: 0.84em; }
      .reveal table { font-size: 0.70em; }
      .reveal img { max-height: 56vh; }
    </style>
---

## 1. Motivation

Following the reviewed CMF paper structure, this variant asks:

- How does method choice change estimated effects?
- What do we gain from hierarchical CMF fitting versus traditional NB?

## 2. Methods Compared In This Project

1. Traditional baseline NB with offset.
2. Hierarchical CMF baseline NB with offset.
3. Offset sensitivity comparison (same specs, offset removed).
4. Multi-step scenario testing (+0 to +3).

## 3. Fitted Coefficients: Direct Comparison

{{< include results/slide_assets/mardown_ore_method_fit_coefficients.md >}}

## 3.0 How To Read The Percentages (Very Simple)

{{< include results/slide_assets/mardown_ore_method_curve_percent_from_coeff.md >}}

{{< include results/slide_assets/mardown_ore_method_curve_percent_table.md >}}

## 3.0b What Delta CMF Means

{{< include results/slide_assets/mardown_ore_method_delta_cmf_explained.md >}}

## 3.0c HSM-Style CMF Explanation (Step By Step)

{{< include results/slide_assets/mardown_ore_method_hsm_style_explainer.md >}}

## 3.1 What The Fit Adds

{{< include results/slide_assets/mardown_ore_method_gain_from_fit.md >}}

## 4. Fit Quality Snapshot

{{< include results/slide_assets/mardown_ore_baseline_fit_snapshot.md >}}

![Observed vs predicted](results/slide_assets/mardown_ore_observed_vs_predicted.png)

## 5. Sensitivity And Combined-Effect Behavior

![Multi-step curve sensitivity](results/slide_assets/mardown_ore_cmf_change_map.png)

{{< include results/slide_assets/mardown_ore_offset_test.md >}}

## 5.1 Curve Shape Check

![Smoothed predicted crashes vs AADT](results/slide_assets/mardown_ore_predictions_vs_aadt.png)

{{< include results/slide_assets/mardown_ore_curve_explanation.md >}}

## 6. What Is Gained (Final)

- Coefficients become more interpretable by mechanism (baseline-risk vs traffic-response).
- Multi-treatment and multi-step scenario communication is clearer.
- Offset assumptions can be stress-tested transparently in the same framework.
- Even when predictive fit is similar, decision-quality explanation is improved.

## 7. Live Demo Files

- results/slide_assets/mardown_ore_cmf_change_explorer.html
- results/slide_assets/mardown_ore_predictions_interactive.html
