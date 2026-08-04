import sys
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
sys.path.insert(0, r"C:\Users\ahernz\source\MetaCount")

from metacountregressor.GA_CMF_AADT_JAX import _fixed_ll, _param_labels
from scipy.optimize import minimize

DATA = r"C:\Users\ahernz\source\MetaCount\metacountregressor\Re_ CMF\Ex-16-3.csv"
baseline_vars = ["WIDTH_PER_LANE", "CURVES", "TANGENT", "INTECHAG", "MXGRDIFF", "SHOULDER_WIDTH"]
local_vars = ["SLOPE_FLAT"]
rand_baseline = tuple([False] * len(baseline_vars))
rand_local = tuple([False] * len(local_vars))
model = "nb"

df = pd.read_csv(DATA)
df["WIDTH_PER_LANE"] = df["WIDTH"] / (df["INCLANES"] + df["DECLANES"])
df["SLOPE_FLAT"] = (df["SLOPE"] == 0).astype(int)
df["SHOULDER_WIDTH"] = df["MIMEDSH"] / df["WIDTH"]
for col in baseline_vars + local_vars + ["FREQ", "AADT"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=baseline_vars + local_vars + ["FREQ", "AADT"])

y = np.asarray(df["FREQ"], dtype=np.float64)
AADT = np.asarray(df["AADT"], dtype=np.float64)
N = len(y)

def std(x):
    x = x.astype(np.float64)
    mu = x.mean(axis=0); sd = x.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd, mu, sd

B_raw = df[baseline_vars].values.astype(np.float64)
L_raw = df[local_vars].values.astype(np.float64)
B, B_mu, B_sd = std(B_raw)
L, L_mu, L_sd = std(L_raw)

y_jax = jnp.array(y); AADT_jax = jnp.array(AADT)
B_jax = jnp.array(B); L_jax = jnp.array(L)

def negll_std(p):
    return _fixed_ll(jnp.array(p, dtype=np.float64), y_jax, AADT_jax,
                     B_jax, L_jax, rand_baseline, rand_local, model)

names = _param_labels(baseline_vars, local_vars, list(rand_baseline), list(rand_local), model)

# Fit (standardized space) via L-BFGS-B with analytic grad
grad = jax.grad(negll_std)
rng = np.random.default_rng(0)
starts = [np.random.default_rng(s).normal(0, 0.8, len(names)) for s in range(5)]
starts += [np.zeros(len(names))]
best = None
for s in starts:
    r = minimize(lambda p: float(negll_std(p)),
                 s, jac=lambda p: np.asarray(grad(jnp.array(p, dtype=np.float64))),
                 method="L-BFGS-B",
                 options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-12})
    if best is None or r.fun < best.fun:
        best = r
pz = best.x
ll = -best.fun
print(f"MLE obj = {ll:.4f}")

# Exact Hessian in standardized space
H = np.asarray(jax.hessian(negll_std)(jnp.array(pz, dtype=np.float64)))
ev = np.linalg.eigvalsh(H)
print("Hessian eigenvalues (min/max):", round(ev.min(),6), round(ev.max(),3))
covz = np.linalg.inv(H)
sez = np.sqrt(np.diag(covz))

# Delta-method transform to raw scale. Raw eta contributions:
#   log_mu = a0 + sum_k a_raw_k*(x_k) + (b0 + sum_j b_raw_j*z_j)*log AADT
# with x = B_mu + B_sd*zx etc. Relation (std) -> (raw):
#   a_raw_k = az_k / B_sd_k ;   b_raw_j = bz_j / L_sd_j
#   a0_raw = a0z - sum az_k*B_mu_k/B_sd_k ;  b0_raw = b0z - sum bz_j*L_mu_j/L_sd_j
# Jacobian of raw vec wrt std vec is diagonal-ish; use it for SEs.
K = len(baseline_vars); J = len(local_vars)
a0z = pz[0]; az = pz[1:1+K]; b0z = pz[1+K]; bz = pz[1+K+1:1+K+1+J]; lt = pz[-1]

a0_raw = a0z - np.sum(az * B_mu / B_sd)
a_raw  = az / B_sd
b0_raw = b0z - np.sum(bz * L_mu / L_sd)
b_raw  = bz / L_sd

# Jacobian d(raw)/d(std), 10x10
P = np.zeros((len(names), len(names)))
# rows: alpha0, alpha[..], beta0, beta[..], log_theta
P[0, 0] = 1.0
for k in range(K):
    P[0, 1+k] = -B_mu[k] / B_sd[k]
    P[1+k, 1+k] = 1.0 / B_sd[k]
row = 1+K
P[row, row] = 1.0
for j in range(J):
    P[row, row+1+j] = -L_mu[j] / L_sd[j]
    P[row+1+j, row+1+j] = 1.0 / L_sd[j]
P[-1, -1] = 1.0

raw_vec = np.array([a0_raw, *a_raw, b0_raw, *b_raw, lt])
cov_raw = P @ covz @ P.T
se_raw = np.sqrt(np.clip(np.diag(cov_raw), 0, None))

print("\n=== WASHINGTON PRESPECIFIED NB  (raw coefficient scale) ===")
print(f"{'Parameter':<28}{'Estimate':>12}{'Std.Err':>10}{'z':>9}{'p-value':>9}")
rows = []
for n_, v, s in zip(names, raw_vec, se_raw):
    z_ = v / s if s > 1e-12 else np.nan
    p_ = 2*(1 - __import__('scipy').stats.norm.cdf(abs(z_)))
    rows.append((n_, v, s, z_, p_))
    print(f"{n_:<28}{v:>12.4f}{s:>10.4f}{z_:>9.3f}{p_:>9.4f}")

print(f"\nNB theta = exp(log_theta) = {np.exp(lt):.3f}")
print(f"AIC = {2*len(names) - 2*ll:.2f}   BIC = {len(names)*np.log(N) - 2*ll:.2f}")

out = r"C:\Users\ahernz\source\MetaCount\metacountregressor\Re_ CMF\washington_nb_prespecified_fit.csv"
pd.DataFrame(rows, columns=["Parameter","Estimate","Std.Err","z","p-value"]).to_csv(out, index=False)
print("saved:", out)
