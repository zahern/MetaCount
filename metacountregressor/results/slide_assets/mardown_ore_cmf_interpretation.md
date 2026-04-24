| Model | Parameter | Effect multiplier | Plain-English interpretation |
| --- | --- | --- | --- |
| Hierarchical CMF baseline NB | URB | 1.0899 | Urban segments have about 9.0% higher baseline crash risk before the traffic-response block is applied. |
| Hierarchical CMF baseline NB | ACCESS | 0.8571 | A one-step increase in the ACCESS coding reduces baseline risk by about 14.3%. |
| Hierarchical CMF baseline NB | GRADEBR | 0.9963 | The fitted baseline effect of the grade-break measure is close to zero in this CMF model. |
| Hierarchical CMF baseline NB | __cmf_log_aadt | 0.007 | This is the base AADT elasticity at average width. In this fit it is near zero, so most traffic sensitivity is being expressed through the width interaction instead. |
| Hierarchical CMF baseline NB | __cmf_local__WIDTH_Z | 0.8737 | At mean AADT (37355), a one-SD wider segment (about 15.5 width units) has CMF 0.874, or about -12.6% fewer predicted crashes. At median AADT (23771), the same change gives CMF 0.879. That is roughly -0.9% per extra width unit near mean AADT. |
| Hierarchical CMF random-parameter NB | CURVES_Z (ind. mean) | 1.1473 | A one-SD increase in curves (about 2.86 extra curves) raises baseline risk by about 14.7% on average. That is roughly 4.9% per extra curve. |
| Hierarchical CMF random-parameter NB | CURVES_Z (ind. SD) | 0.0034 | This random SD is very small, so once curvature is placed in the CMF baseline block, the fitted curvature effect barely varies across segments in this run. |