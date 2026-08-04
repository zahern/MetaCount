import sys
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as jr
sys.path.insert(0, r"C:\Users\ahernz\source\MetaCount")

from metacountregressor.GA_CMF_AADT_JAX import _fixed_ll_exp, _mixed_ll_exp, _param_labels_exp
from scipy.optimize import minimize

DATA = r"C:\Users\ahernz\source\MetaCount\metacountregressor\Re_ CMF\Ex-16-3.csv"
baseline_vars = ["WIDTH_PER_LANE","CURVES","TANGENT","INTECHAG","MXGRDIFF","SHOULDER_WIDTH"]
local_vars = ["SLOPE_FLAT"]
model = "nb"
R = 400

df = pd.read_csv(DATA)
df["WIDTH_PER_LANE"] = df["WIDTH"] / (df["INCLANES"] + df["DECLANES"])
df["SLOPE_FLAT"] = (df["SLOPE"] == 0).astype(int)
df["SHOULDER_WIDTH"] = df["MIMEDSH"] / df["WIDTH"]
for col in baseline_vars + local_vars + ["FREQ","AADT"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=baseline_vars + local_vars + ["FREQ","AADT"])

y  = np.asarray(df["FREQ"], dtype=np.float64)
AADT = np.asarray(df["AADT"], dtype=np.float64)
log_gmean = float(np.log(AADT).mean())     # centering constant for log(AADT)
AADTc = AADT / np.exp(log_gmean)           # log(AADTc) = log(AADT) - log_gmean (centered)
N = len(y)

def std(x):
    x = x.astype(np.float64); mu = x.mean(axis=0); sd = x.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd); return (x-mu)/sd, mu, sd

B_raw = df[baseline_vars].values.astype(np.float64)
L_raw = df[local_vars].values.astype(np.float64)
B, B_mu, B_sd = std(B_raw)
L, L_mu, L_sd = std(L_raw)
B_jax = jnp.array(B); L_jax = jnp.array(L)
y_jax = jnp.array(y); AADT_jax = jnp.array(AADTc)   # centered channel used in fitting

def run_fit(rand_bs, rand_ls, seed=0, n_starts=8):
    rbt = tuple(rand_bs); rlt = tuple(rand_ls)
    sim = any(rbt) or any(rlt)
    if sim:
        nr = sum(rbt)+sum(rlt)
        draws = jr.normal(jr.PRNGKey(seed), (R, N, nr))
        negll = lambda p: _mixed_ll_exp(jnp.array(p,dtype=np.float64), y_jax, AADT_jax,
                                        B_jax, L_jax, rbt, rlt, draws, model)
    else:
        negll = lambda p: _fixed_ll_exp(jnp.array(p,dtype=np.float64), y_jax, AADT_jax,
                                        B_jax, L_jax, rbt, rlt, model)
    names = _param_labels_exp(baseline_vars, local_vars, list(rand_bs), list(rand_ls), model)
    grad = jax.grad(negll)
    np_ = len(names)
    starts = [np.random.default_rng(s).normal(0,0.8,np_) for s in range(n_starts)]
    starts += [np.zeros(np_)]
    if 'log_beta0' in names:
        s = np.zeros(np_); s[names.index('log_beta0')] = np.log(0.65); s[-1] = -0.6
        starts.append(s)
    best = None
    for s0 in starts:
        r = minimize(lambda p: float(negll(p)),
                     s0, jac=lambda p: np.asarray(grad(jnp.array(p,dtype=np.float64))),
                     method="L-BFGS-B",
                     options={"maxiter":4000, "ftol":1e-14, "gtol":1e-12})
        if best is None or r.fun < best.fun:
            best = r
    pz = best.x
    ll = -best.fun
    H = np.asarray(jax.hessian(negll)(jnp.array(pz,dtype=np.float64)))
    ev = np.linalg.eigvalsh(H)
    return names, pz, ll, H, ev

def report(names, pz, ll, H, ev, rand_bs, rand_ls):
    print(f"  objective (-LL) = {-ll:.4f}   N = {N}   npar = {len(names)}")
    print("  Hessian min eig:", round(float(ev.min()),6), " max:", round(float(ev.max()),3),
          " pos-def:", bool(float(ev.min()) > 0))
    idx = {nm: i for i, nm in enumerate(names)}
    covz = np.linalg.inv(H + 1e-12*np.eye(len(names)))
    sez = np.sqrt(np.clip(np.diag(covz),0,None))

    P = np.eye(len(names))
    raw = pz.copy()
    cols = [idx[f'alpha[{v}]'] for v in baseline_vars]
    for k, c in enumerate(cols):
        P[idx['alpha0'], c] = -B_mu[k]/B_sd[k]
    raw[idx['alpha0']] = pz[idx['alpha0']] - np.sum(pz[cols]*B_mu/B_sd)
    for k, c in enumerate(cols):
        P[c, c] = 1.0/B_sd[k]
        raw[c] = pz[c]/B_sd[k]
    ib = idx['log_beta0']
    jcols = [idx[f'beta[{v}]'] for v in local_vars]
    for j, c in enumerate(jcols):
        P[ib, c] = -L_mu[j]/L_sd[j]
    raw[ib] = pz[ib] - np.sum(pz[jcols]*L_mu/L_sd)
    for j, c in enumerate(jcols):
        P[c, c] = 1.0/L_sd[j]
        raw[c] = pz[c]/L_sd[j]
    # ── Fold log(AADT) centering back at reference (SLOPE_FLAT=0) ──
    # log_mu = alpha0c + ... + beta0*exp(Σβz)·log(AADT/log_gmean)
    #        = [alpha0c - beta0*exp(Σβz)*log_gmean] + ... + beta0*exp(Σβz)*log(AADT)
    # At reference z2=0, exp(Σβz)=1, so alpha0_raw = alpha0c - beta0*log_gmean.
    beta0_rep = np.exp(raw[ib])
    d_alpha0 = -beta0_rep * log_gmean
    raw[idx['alpha0']] += d_alpha0
    P[idx['alpha0'], ib] = d_alpha0   # d alpha0_raw / d log_beta0
    cov_raw = P @ covz @ P.T
    se_raw = np.sqrt(np.clip(np.diag(cov_raw),0,None))

    print(f"\n{'Parameter':<26}{'Estimate':>12}{'Std.Err':>10}{'z':>9}{'p-value':>9}")
    rows = []
    for n_, v, s in zip(names, raw, se_raw):
        z_ = v/s if s > 1e-12 else np.nan
        p_ = 2*(1-__import__('scipy').stats.norm.cdf(abs(z_)))
        rows.append((n_, v, s, z_, p_))
        print(f"{n_:<30}{v:>12.4f}{s:>10.4f}{z_:>9.3f}{p_:>9.4f}")

    lb0 = raw[ib]; b1 = raw[jcols[0]]; beta0 = np.exp(lb0)
    slb0 = se_raw[ib]; sb1 = se_raw[jcols[0]]; sbeta0 = beta0*slb0
    print("\n  Elasticity B(z2) = beta0*exp(beta1*SLOPE_FLAT) with beta0 = exp(log_beta0):")
    print(f"    beta0 = exp(log_beta0) = {beta0:.4f}  (SE {sbeta0:.4f})")
    for zv in np.unique(df["SLOPE_FLAT"].values.astype(float)):
        Bst = beta0*np.exp(b1*zv)
        sB = np.sqrt((np.exp(b1*zv)*sbeta0)**2 + (beta0*np.exp(b1*zv)*zv*sb1)**2)
        print(f"    SLOPE_FLAT={zv:.0f}: B = {Bst:.4f}  (SE {sB:.4f})")
    print(f"  NB theta = {np.exp(raw[-1]):.3f}")
    print(f"  AIC = {2*len(names)-2*ll:.2f}   BIC = {len(names)*np.log(N)-2*ll:.2f}")

    # ── Reporting-only: sample-average fitted mean (honest check, std-space) ──
    Xs = B.astype(float); Ls = L.astype(float)
    a0c = pz[0]; avs = pz[cols]
    lbc = pz[ib]; bvs = pz[jcols]
    logA_all = a0c + Xs @ avs
    B_all = np.exp(lbc + Ls @ bvs)
    log_mu_all = np.clip(logA_all + B_all*np.log(AADTc), -30, 30)
    mu_all = np.exp(log_mu_all)
    print("\n  Reporting (sample-average, centered fit):")
    print(f"    mean fitted mu = {mu_all.mean():.2f}   (observed mean = {y.mean():.2f})")
    print(f"    mean elasticity B = {B_all.mean():.4f}  "
          f"(range {B_all.min():.4f}–{B_all.max():.4f})")
    print(f"    mean log A(z1) = {logA_all.mean():.4f}")
    return rows

print("="*76)
print("EXPONENTIAL-ELASTICITY FORM: Np = A(z1)*AADT^B(z2),  B(z2)=beta0*exp(beta1*z2)")
print("="*76)

print("\n[1] FIXED-EFFECTS exp form:")
names, pz, ll, H, ev = run_fit([False]*len(baseline_vars), [False]*len(local_vars))
rows = report(names, pz, ll, H, ev, [False]*len(baseline_vars), [False]*len(local_vars))
out = r"C:\Users\ahernz\source\MetaCount\metacountregressor\Re_ CMF\washington_nb_exp_fixed_fit.csv"
pd.DataFrame(rows, columns=["Parameter","Estimate","Std.Err","z","p-value"]).to_csv(out, index=False)
print("saved:", out)

print("\n[2] HIERARCHICAL exp form (random effect on B exponent SLOPE_FLAT):")
names, pz, ll, H, ev = run_fit([False]*len(baseline_vars), [True]*len(local_vars))
rows = report(names, pz, ll, H, ev, [False]*len(baseline_vars), [True]*len(local_vars))
out = r"C:\Users\ahernz\source\MetaCount\metacountregressor\Re_ CMF\washington_nb_exp_randB_fit.csv"
pd.DataFrame(rows, columns=["Parameter","Estimate","Std.Err","z","p-value"]).to_csv(out, index=False)
print("saved:", out)

print("\n[3] HIERARCHICAL exp form (random on all A + B):")
names, pz, ll, H, ev = run_fit([True]*len(baseline_vars), [True]*len(local_vars))
rows = report(names, pz, ll, H, ev, [True]*len(baseline_vars), [True]*len(local_vars))
out = r"C:\Users\ahernz\source\MetaCount\metacountregressor\Re_ CMF\washington_nb_exp_randAB_fit.csv"
pd.DataFrame(rows, columns=["Parameter","Estimate","Std.Err","z","p-value"]).to_csv(out, index=False)
print("saved:", out)
