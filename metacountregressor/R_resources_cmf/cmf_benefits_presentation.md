---
title: "Why Use Hierarchical CMF?"
subtitle: "Benefits, evidence, and practical examples"
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
      .reveal .slides section { font-size: 0.94em; }
      .reveal .slides section.small { font-size: 0.84em; }
      .reveal table { font-size: 0.72em; }
      .reveal img { max-height: 58vh; }
      .reveal .tight p, .reveal .tight li { font-size: 0.90em; }
      .pill {
        display: inline-block;
        border: 1px solid #666;
        border-radius: 999px;
        padding: 2px 10px;
        margin-right: 6px;
        font-size: 0.75em;
      }
    </style>
---

## Goal

This presentation follows the review-paper logic from your PDF ([metacountregressor/mcf-main.pdf](metacountregressor/mcf-main.pdf)):

- what CMFs are used for,
- what estimation families exist,
- why method choice changes conclusions,
- how to communicate combined-treatment effects.

All evidence slides use your fitted outputs.

## Why This Matters

<span class="pill">Stakeholder clarity</span>
<span class="pill">Defensible assumptions</span>
<span class="pill">Policy simulation</span>

- Traditional NB often gives a single combined equation that is accurate enough but harder to explain in engineering terms.
- Hierarchical CMF separates baseline roadway risk from traffic-response behavior, which gives a clearer causal story for countermeasures.

## Review Paper Context

The inspected paper is:

- "A review of the state-of-the-art methods in estimating crash modification factor (CMF)"
- Transportation Research Interdisciplinary Perspectives 20 (2023) 100839

Its key message is that method choice (BA variants, EB/FB, cross-sectional) materially affects CMF consistency and precision.

## The Core Tradeoff

| Question | Traditional NB | Hierarchical CMF |
|---|---|---|
| Best compact predictive benchmark? | Strong | Usually close, not always best |
| Easy to explain baseline-risk effects? | Moderate | Strong |
| Easy to explain response under traffic/exposure change? | Moderate | Strong |
| Scenario testing for design decisions? | Possible | More structured |

## Estimation Families (Mirroring The Paper)

1. Observational before-after (BA) methods.
2. Cross-sectional methods when BA data are constrained.
3. Combined-treatment CMF calculation approaches.

This deck maps your model outputs to these three buckets.

## Why Hierarchical CMF Fits This Literature

- Like EB/FB framing in the review, hierarchical CMF explicitly separates sources of variation instead of collapsing everything into one coefficient block.
- That structure gives a more stable policy narrative when multiple safety features interact.
- It also makes combined-treatment communication easier because baseline and response effects are separated.

## Benefit 1: Structured Interpretation

- Traditional NB: one combined effect pathway.
- Hierarchical CMF: two-part story:
  1. Baseline safety effect block.
  2. Traffic-response adjustment block.
- This helps answer: "Did a design feature change baseline risk, or did it change how risk scales with traffic?"

## Benefit 1b: Strengths And Limitations (Transparent)

Strengths:
- Clearer mechanism and policy narrative.
- Better what-if decomposition for multiple treatments.

Limitations:
- Not automatically the best predictive fit.
- Requires careful offset/exposure specification.
- Can be sensitive to scaling and specification choices.

## Benefit 2: Better Decision Communication

{{< include results/slide_assets/mardown_ore_cmf_advantage.md >}}

## Benefit 3: Transparent Sensitivity Testing

![Multi-step sensitivity with and without offset](results/slide_assets/mardown_ore_cmf_change_map.png)

- This figure now tests +0, +1, +2, +3 changes and explicitly compares with-offset vs without-offset behavior.

## Combined-Treatment Communication Benefit

The review emphasizes that combining CMFs can over/underestimate if treatment effects are not independent.

How we address this in your workflow:

- We do not only multiply static factors.
- We run explicit multi-step scenario predictions (+0 to +3) under each model.
- We show with-offset and without-offset behavior side by side.

## Example 1: Offset Sensitivity (Key Finding)

{{< include results/slide_assets/mardown_ore_offset_test.md >}}

## Example 2: Predicted Curve Behavior

![Smoothed predicted crashes vs AADT](results/slide_assets/mardown_ore_predictions_vs_aadt.png)

{{< include results/slide_assets/mardown_ore_curve_explanation.md >}}

## Example 3: Fit Quality Still Matters

{{< include results/slide_assets/mardown_ore_baseline_fit_snapshot.md >}}

![Observed vs predicted](results/slide_assets/mardown_ore_observed_vs_predicted.png)

- In this dataset, predictive differences are small; the main CMF gain is interpretability structure.

## Practical Takeaway

Use traditional NB when:
- Your only objective is a compact predictive benchmark.

Use hierarchical CMF when:
- You need a policy-facing explanation of safety effects.
- You need scenario tools that separate baseline-risk shifts from traffic-response shifts.
- You need to show sensitivity to offset/exposure assumptions.

## Method-Selection Rule (Paper-Aligned)

1. If robust BA data and strong calibration are available, prioritize consistency-focused estimation logic.
2. If data constraints prevent BA-style implementation, use structured cross-sectional modeling with explicit assumptions.
3. For multiple treatments, avoid one-shot combination assumptions; validate through scenario simulation.

## Interactive Demonstration

Open and present live:

- results/slide_assets/mardown_ore_cmf_change_explorer.html
- results/slide_assets/mardown_ore_predictions_interactive.html

The first file supports:
- Feature toggles (CURVES, ACCESS, WIDTH)
- Multi-step changes (+0 to +3)
- With-offset vs without-offset comparison
- Percent-change and absolute-value views

## Closing Message

The benefit of hierarchical CMF is not "always better fit."

The benefit is "better structure for decisions":
- clearer mechanism,
- clearer what-if testing,
- clearer communication of why the estimate changes.
