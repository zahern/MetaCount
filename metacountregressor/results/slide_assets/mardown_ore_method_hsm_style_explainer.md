Below is the same style of explanation you asked for, using your fitted hierarchical model.

Traditional-style single-level count form:

$$
\hat N_i = E_i \times \exp(\beta_0 + \beta_1 X_{1i} + \cdots + \beta_k X_{ki})
$$

Hierarchical CMF two-level form (same fitted model, rewritten for interpretation):

$$
\hat N_i = E_i \times \exp(\beta_0 + \beta_{URB}URB_i + \beta_{ACCESS}ACCESS_i + \beta_{GRADE}GRADEBR_i + \beta_{CURVES}CURVES_i) \times AADT_i^{(\beta_{logAADT} + \beta_{WIDTH,local}WIDTH_i)}
$$

With your fitted values:

$$
\beta_0=+4.6164,\; \beta_{URB}=+0.0715,\; \beta_{ACCESS}=-0.1601,\; \beta_{GRADE}=+0.0083,\; \beta_{CURVES}=-0.0839,\; \beta_{logAADT}=-0.0056,\; \beta_{WIDTH,local}=-0.0124
$$

CMF from a coefficient (holding other factors constant):

$$
CMF(a \rightarrow b) = \frac{\hat N_b}{\hat N_a} = \exp(\alpha_1(b-a))
$$

So percent change is:

$$
100 \times (CMF-1) = 100 \times (\exp(\alpha_1(b-a)) - 1)
$$

Worked one-unit examples from your fitted CMF baseline block:

- ACCESS +1: $100 \times (\exp(-0.160110)-1) = -14.79\%$
- CURVES +1: $100 \times (\exp(-0.083946)-1) = -8.05\%$
- GRADEBR +1: $100 \times (\exp(+0.008255)-1) = +0.83\%$

For WIDTH in this hierarchical fit, WIDTH is in the local AADT-response block, so its CMF depends on AADT:

$$
CMF_{WIDTH}(a \rightarrow b \mid AADT) = \exp(\beta_{WIDTH,local}\log(AADT)(b-a))
$$

At median $AADT=23771$ and $b-a=1$ this gives approximately:

$$
100 \times \left(\exp(-0.012352\log(23771)) - 1\right) = -11.70\%
$$

Why this is easier than traditional for interpretation:

- Traditional gives one combined equation, so baseline and traffic-response effects are mixed together.
- Hierarchical CMF separates the baseline-risk block from the traffic-response block, so each percentage has a clear role in the story.
