---
title: "Hierarchical CMF vs Traditional NB"
subtitle: "Simple, clear comparison for design-exception decisions"
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
      .reveal .slides section { font-size: 0.95em; }
      .reveal .slides section.small { font-size: 0.86em; }
      .reveal table { font-size: 0.74em; }
      .reveal img { max-height: 58vh; }
      .reveal .tight p, .reveal .tight li { font-size: 0.92em; }
    </style>
---

## Goal

This deck answers one practical question:

- What does the CMF structure buy us if predictive fit is similar?

## Bottom Line

{{< include results/slide_assets/mardown_ore_bottom_line.md >}}

## Quick Fit Check

{{< include results/slide_assets/mardown_ore_baseline_fit_snapshot.md >}}

## Observed Vs Predicted

![Observed vs predicted](results/slide_assets/mardown_ore_observed_vs_predicted.png)

The dashed line is perfect agreement. Both models track the center of the data similarly, so the CMF case here is about interpretation structure, not a big jump in predictive accuracy.

## Why Use CMF At All?

{{< include results/slide_assets/mardown_ore_cmf_advantage.md >}}

## Smoothed AADT Curve

![Predicted crashes vs AADT](results/slide_assets/mardown_ore_predictions_vs_aadt.png)

## Why The Curves Look The Same

{{< include results/slide_assets/mardown_ore_curve_explanation.md >}}

## Offset Test

{{< include results/slide_assets/mardown_ore_offset_test.md >}}

## What Changes Under CMF

![CMF change map](results/slide_assets/mardown_ore_cmf_change_map.png)

This now shows multi-step curve changes at median AADT with and without the offset, which makes the difference in response direction easier to see.

## Interactive Views

Open this file for the smooth curve with buttons and a difference view:

- `results/slide_assets/mardown_ore_predictions_interactive.html`

Open this file for the interactive CMF change explorer with feature buttons, +0 to +3 step changes, and an offset toggle:

- `results/slide_assets/mardown_ore_cmf_change_explorer.html`

## Recommendation

- Use traditional NB if the only goal is the best compact predictive benchmark.
- Use hierarchical CMF when you want the engineering story to be clearer: what moves baseline risk, what moves the traffic response, and how design changes should be explained.
