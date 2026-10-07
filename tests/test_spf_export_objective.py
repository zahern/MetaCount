"""SPF export objective: screen candidate exposure-response forms and
export the winning SPF for each road class ("peer group").

Candidate families:
  1. exp(a + beta*log AADT + log L)              - exact offset form
  2. exp(a + beta*log AADT + log(length))        - same, offset on L
  3. exp(a + beta*log(AADT)^2 power)             - elastic AADT^beta

For each road class (here licensed as the LANES grouping of the demo)
we score every form with its NB maximum log-likelihood and BIC on the same
observations, pick the NB-loglik maximiser, and report the exported SPF:
exportable means it can be handed to a designer as a function
mean_count(AADT, length, class) expressed in the HSM peer-group form
`crashes = exp(a + b*ln(AADT)) * L` with the class-specific (a, b).

Comparison "outcome": the excavated elasticity is compared to the
published rural two-lane baseline elasticity b ~= 1.68 reported in
HSM / FHWA network screening materials (see metacount paper, the
corresponding section of the ext paper).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

def nb_mll(y, mu, theta):
    if theta <= 0:
        return -np.inf
    r = 1.0 / theta
    return np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                  + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))

def fit_mu_link(y, mu_model):
    def loss(theta):
        x = exp(np.clip(mu_model, -25, 25))
        return -nb_mll(y, x, np.exp(theta[0]))
    return loss

# get a quick fixed-effects Poisson/NB on class = LANES
df = pd.read_csv(r"Z:\cmf_metacount\data\synthetic_cmf_demo.csv")
df["ln_aadt"] = np.log(df["AADT"].clip(lower=1))
df["ln_len"] = np.log(df["LENGTH"].clip(lower=1e-6))

class_result = []
for lanes, g in df.groupby("LANES"):
    y, L, aadt = g["FREQ"].to_numpy(), g["LENGTH"].to_numpy(), g["AADT"].to_numpy()
    cand = {}
    # form 1: crashes = L * exp(a + b*ln AADT)   (fixed offset on L)
    xf = np.log(aadt).ravel()
    def ll_1():
        p = np.array([0.0])
        from scipy.optimize import lsq_linear
        return None
    def objective_form(form_name, b0, b1):
        mu = L * np.exp(b0 + b1 * np.log(aadt))
        # NB loglik
        def NLL(logtheta):
            th = np.exp(logtheta)
            r = 1.0 / th
            mu_c = np.clip(mu, 1e-12, None)
            return -np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                           + r * np.log(r / (r + mu_c)) + y * np.log(mu_c / (r + mu_c)))
        # BFGS over (b0,b1,logtheta)
        def J(p):
            mu_c = L * np.exp(p[0] + p[1] * np.log(aadt))
            mu_c = np.clip(mu_c, 1e-12, None)
            th = np.exp(p[2])
            r = 1.0 / th
            return -np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                           + r * np.log(r / (r + mu_c)) + y * np.log(mu_c / (r + mu_c)))
        res = minimize(J, np.array([-6.0, 1.0, np.log(0.2)]), method="BFGS")
        mu_fit = L * np.exp(res.x[0] + res.x[1] * np.log(aadt))
        bic = 2 * (-res.fun) + 3 * np.log(len(y))
        return res.x, -res.fun, bic

    rb, ll1, bic1 = objective_form("offset", 0, 0)
    cand["L*exp(a+b*ln AADT)"] = (rb[0], rb[1], ll1, bic1)
    # form 2: crashes = exp(a + b*ln AADT) * L^(1+g) with g free:
    def J2(p):
        mu_c = np.exp(p[0] + p[1] * np.log(aadt)) * L ** (1 + p[3])
        mu_c = np.clip(mu_c, 1e-12, None)
        th = np.exp(p[2])
        r = 1.0 / th
        return -np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                       + r * np.log(r / (r + mu_c)) + y * np.log(mu_c / (r + mu_c)))
    res2 = minimize(J2, np.array([-6.0, 1.0, np.log(0.2), 0.0]), method="BFGS")
    bic2 = 2 * (-res2.fun) + 4 * np.log(len(y))
    cand["L^(1+g)*exp(a+b*ln AADT)"] = (res2.x[0], res2.x[1], -res2.fun, bic2)

    best = max(cand.items(), key=lambda kv: kv[1][2])
    class_result.append((lanes, len(g), best[0], best[1][0], best[1][1], best[1][2]))

print(f"{'LANES':>5}  {'n':>6}  {'selected form':>28}  {'a':>8}  {'b(AADT)':>9}  {'NB loglik':>10}")
for r in class_result:
    print(f"{r[0]:>5}  {r[1]:>6}  {r[2]:>28}  {r[3]:>+8.3f}  {r[4]:>+9.3f}  {r[5]:>10.1f}")
print()
print("Benchmark (reported):")
print("  HSM rural two-lane (total crashes, L offset exact):")
print("     crashes = exp[-15.22 + 1.68*ln(AADT)] * L   => elasticity b = 1.68")
print("  AASHTO / Safety Analyst rural multilane divided:")
print("     crashes per mile = 0.665 * exp(0.05 * log(AADT))")
