| Model | Parameter | Effect multiplier | Plain-English interpretation |
| --- | --- | --- | --- |
| Traditional baseline NB | URB | 0.9511 | Urban segments have about 4.9% lower expected crashes than non-urban segments, holding the other terms fixed. |
| Traditional baseline NB | ACCESS | 0.8079 | A one-step increase in the ACCESS coding is associated with about 19.2% fewer expected crashes, holding the other terms fixed. |
| Traditional baseline NB | GRADEBR | 1.0526 | A one-unit increase in the grade-break measure is associated with about 5.3% more expected crashes. |
| Traditional baseline NB | CURVES | 1.0078 | Each additional curve is associated with about 0.8% more expected crashes. |
| Traditional baseline NB | LENGTH | 0.9045 | This term implies about 9.5% fewer expected crashes per extra mile, but it should be read cautiously because the model already includes an offset. |
| Traditional random-parameter NB | CURVES (ind. mean) | 1.0068 | On average, one extra curve is still associated with about 0.7% more expected crashes, very close to the fixed-effect model. |
| Traditional random-parameter NB | CURVES (ind. SD) | 0.0119 | This random SD says the curve effect is allowed to vary across segments. Because the term is lognormal, the model keeps the curve effect non-negative while permitting some sites to be more curve-sensitive than others. |