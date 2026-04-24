| Step | Action |
| --- | --- |
| 1 | Fit Traditional baseline and random NB models. |
| 2 | Fit Hierarchical CMF baseline and random models using raw CURVES; if non-finite, auto-fallback to internally scaled stabilization. |
| 3 | Compute fit metrics and plain-language percent effects. |
| 4 | Generate smooth model-based AADT curves from baseline fits and sample representative pointwise anchors from that curve. |
| 5 | Export markdown assets and render the deck. |