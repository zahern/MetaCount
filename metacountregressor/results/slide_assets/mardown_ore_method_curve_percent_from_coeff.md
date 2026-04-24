Use this exact rule for log-link count models:

- Percent change from one coefficient is 100 * (exp(beta * delta_x) - 1).
- beta is the fitted coefficient.
- delta_x is how much the variable changed (+1, +2, +3).

Worked examples from your fitted CURVES coefficients:

- Traditional beta_CURVES = +0.007754
- CMF baseline beta_CURVES = -0.083946

So for +1 CURVES:

- Traditional: 100 * (exp(+0.007754 * 1) - 1) = +0.78%
- CMF baseline: 100 * (exp(-0.083946 * 1) - 1) = -8.05%

Important: these are coefficient-implied effects from one term.
The scenario plots are full-model effects, so they can differ when interaction and offset terms are active.
