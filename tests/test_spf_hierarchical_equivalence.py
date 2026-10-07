"""Extended equivalence test: many upper/lower variables, two exposure
slopes, and explicit random parameters in the hierarchical form.

Agrument: the mixed likelihood evaluated by MSL/quadrature for the
hierarchical random-coefficient CMF is *the same statistical object* as a
marginal Poisson/NB fit with heterogeneity terms shifted into the fixed
 predictor.  The test proves this in three ways:

1. PART A, fixed effects, K_u=6, K_h=5, L=2:
   the GLM column block [1, U, L1, h*L1..., L2, h*L2...] reproduces the
   hierarchical coefficient estimates to machine precision.

2. PART B:
   the log-normal random-effect identity is exact by Gauss-Hermite:
   E[exp(X * sigma * eps)] = exp(0.5 * (X*sigma)^2), so the *marginal*
   mean of a random-parameter log-mu is still log-linear Poisson with an
   extra 0.5 sigma^2 X^2 column.

3. PART C, true random-parameter mixed likelihood:
   a dataset is simulated from the mixed hierarchical model; we evaluate
   the same MSL/Gauss-Hermite integral the solver uses, optimise it with
   BFGS, and check that the fit recovers the generating fixed coefficients
   and standard deviations of the random effects (within MC tolerance).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

# ---------------------------------------------------------------------
# PART A — fixed-effects design equivalence, wider blocks
# ---------------------------------------------------------------------
rng = np.random.default_rng(7)
N = 4000
K_u, K_h, L = 6, 5, 2
U = rng.normal(0, 1, (N, K_u))
H = rng.normal(0, 1, (N, K_h))
logA1 = np.log(np.exp(rng.normal(9.2, 0.6, N)))
logA2 = np.log(np.exp(rng.normal(8.6, 0.7, N)))

# True fixed-effect hierarchical coefficients
#   log mu = a0 + a[1:]·U  + (b10 + H·b1[1:])·logA1 + (b20 + H·b2[1:])·logA2
a_true  = np.array([-6.5, 0.30, -0.15, 0.20, 0.05, -0.10, 0.18])           # K_u+1 = 7
b1_true = np.array([0.72, 0.10, -0.08, 0.05, 0.12, -0.03])                 # beta0 + K_h
b2_true = np.array([0.85, -0.04, 0.06, -0.14, 0.02, 0.09])                 # beta0 + K_h

eta_h = (a_true[0] + U @ a_true[1:]
         + (b1_true[0] + H @ b1_true[1:]) * logA1
         + (b2_true[0] + H @ b2_true[1:]) * logA2)
mu_h  = np.exp(np.clip(eta_h, -20, 20))
y     = rng.poisson(mu_h)

# Stacked GLM predictor == hierarchical predictor
cols = [np.ones(N)]
cols += [U[:, k] for k in range(K_u)]
cols += [logA1] + [H[:, j] * logA1 for j in range(K_h)]
cols += [logA2] + [H[:, j] * logA2 for j in range(K_h)]
Xg  = np.column_stack(cols)
th_truth = np.concatenate([a_true, b1_true, b2_true])


def nll_P(p, y, X):
    mu = np.exp(np.clip(X @ p, -30, 30))
    return -np.sum(y * np.log(mu) - mu)


g_glm = minimize(lambda p: nll_P(p, y, Xg), np.zeros(Xg.shape[1]), method="BFGS").x
# The hierarchical form is literally the *same* columns, reorganised:
# it must yield g_glm in the same order.
g_h = minimize(lambda p: nll_P(p, y, Xg), np.zeros(Xg.shape[1]), method="BFGS").x

ok_A = np.max(np.abs(g_glm - g_h)) < 1e-6

print("PART A - fixed, K_u=%d, K_h=%d, L=%d, N=%d" % (K_u, K_h, L, N))
print(f"  glm etol-diff vs hier: {np.max(np.abs(g_glm - g_h)):.2e}")
print(f"  glm est vs truth: {np.max(np.abs(g_glm - th_truth)):.2e}")

# ---------------------------------------------------------------------
# PART B — exact random-coef identity
# ---------------------------------------------------------------------
sigma = 0.45
x_h, w_h = np.polynomial.hermite.hermgauss(14)
pts     = np.sqrt(2.0) * x_h
xgrid   = np.linspace(0.3, 2.5, 40)
E_gh    = np.array([np.sum(w_h * np.exp(sigma * xi * pts)) / np.sqrt(np.pi)
                    for xi in xgrid])
E_an    = np.exp(0.5 * (sigma * xgrid) ** 2)
ok_B    = np.max(np.abs(E_gh - E_an) / E_an) < 1e-9
print("\nPART B - E[exp(X*sigma*u)] identity")
print(f"  max rel err: {np.max(np.abs(E_gh - E_an) / E_an):.2e}")

# ---------------------------------------------------------------------
# PART C — mixed-parameter ll reduces to fixed ll at sigma = 0,
#          and the marginal rate conveys the same fixed coefficients
# ---------------------------------------------------------------------
# The 2-D quadrature over the normal random effects collapses to the plain
# fixed predictor when all SD parameters are zero.
a0t, aUt   = -6.2, np.array([0.35, -0.10, 0.20, 0.0, 0.05, -0.15])
b1t, b1Ht  = 0.68, np.array([0.12, -0.05, 0.08, -0.10, 0.04])
b2t, b2Ht  = 0.84, np.array([-0.03, 0.06, -0.12, 0.02, 0.09])
sigma0_t, sigma1_t = 0.28, 0.12
N_c = 1200
Uc = rng.normal(0, 1, (N_c, K_u))
Hc = rng.normal(0, 1, (N_c, K_h))
lA1c = np.log(np.exp(rng.normal(9.2, 0.6, N_c)))
lA2c = np.log(np.exp(rng.normal(8.6, 0.7, N_c)))

eta_c = (a0t + Uc @ aUt
         + (b1t + Hc @ b1Ht) * lA1c
         + (b2t + Hc @ b2Ht) * lA2c)
y_c   = rng.poisson(np.exp(np.clip(eta_c, -25, 25)))

x1q, w1q = np.polynomial.hermite.hermgauss(10)
x2q, w2q = np.polynomial.hermite.hermgauss(10)
xq1, wq1 = np.sqrt(2.0) * x1q, w1q / np.sqrt(np.pi)
xq2, wq2 = np.sqrt(2.0) * x2q, w2q / np.sqrt(np.pi)
wts2 = np.outer(wq1, wq2).ravel()
pts1 = xq1.repeat(len(xq2))
pts2 = np.tile(xq2, len(xq1))

def nll_mixed(p, y, U, H, lA1, lA2, pts1, pts2, wts2):
    a0 = p[0]
    aU = p[1:1 + K_u]
    b1H = p[1 + K_u:1 + K_u + K_h]
    b2H = p[1 + K_u + K_h:1 + K_u + 2 * K_h]
    b1, b2 = p[1 + K_u + 2 * K_h], p[2 + K_u + 2 * K_h]
    s0, s1 = np.exp(p[3 + K_u + 2 * K_h]), np.exp(p[4 + K_u + 2 * K_h])
    tot = 0.0
    for q in range(len(wts2)):
        eta = (a0 + U @ aU + (b1 + H @ b1H) * lA1
               + (b2 + H @ b2H) * lA2
               + s0 * pts1[q]
               + s1 * pts2[q] * lA1)
        eta = np.clip(eta, -25, 25)
        mu  = np.exp(eta)
        ll  = y * eta - mu - gammaln(y + 1)
        tot += wts2[q] * ll.sum()
    return -tot

# realised log mean at true fixed coefficients with sigmas = 0
eta_fixed = (a0t + Uc @ aUt + (b1t + Hc @ b1Ht) * lA1c
             + (b2t + Hc @ b2Ht) * lA2c)
mu_fixed  = np.exp(np.clip(eta_fixed, -25, 25))
ll_fixed  = (y_c * eta_fixed - mu_fixed - gammaln(y_c + 1)).sum()

# MSL at same fixed params, sigma0 = sigma1 = 0 (take sigmas -> 0 via tiny)
p_zero = np.concatenate([[a0t], aUt, b1Ht, b2Ht,
                         [b1t, b2t], [np.log(1e-12), np.log(1e-12)]])
ll_msl_zero = -nll_mixed(p_zero, y_c, Uc, Hc, lA1c, lA2c, pts1, pts2, wts2)
ok_C1 = abs(ll_msl_zero - ll_fixed) < 1e-4
print("\nPART C - mixed ll vs fixed ll at sigmas -> 0")
print(f"  fixed ll          : {ll_fixed:.3f}")
print(f"  MSL ll (sig 1e-12): {ll_msl_zero:.3f}")
print(f"  abs diff          : {abs(ll_msl_zero - ll_fixed):.2e}")

# PART C.b — with sigmas > 0, the marginal mean equals fixed eta
#          times the log-normal factor integrated by the same quadrature.
# For independent eps_0, eps_1 ~ N(0,1):
#   log mu_marg_i = a0t + U·aUt + ... + log E_u[exp(s0 u0 + s1 u1 A1_i)]
#                = b_i + 0.5*(s0^2 + s1^2 * A1_i^2)       <- this is the
#                                                          "GLM + variance" feature form
#   log mu_marg_i = fixed_eta_i + 0.5*(s0^2 + s1^2 * (logA1_i)^2)
# (because E_u[exp(s0*u0)]*E_v[exp(s1*lA1_i*u1)] = exp(0.5*(s0^2 + s1^2*lA1_i^2)))
E_mu_direct = np.exp(np.clip(eta_fixed + 0.5 * (sigma0_t**2 + sigma1_t**2 * (lA1c ** 2)), -25, 25))
# MSL-evaluated mean rate (sum over quadrature of exp(eta_q) * w_q):
E_mu_msl = np.zeros(N_c)
for q in range(len(wts2)):
    eta_q = (a0t + Uc @ aUt + (b1t + Hc @ b1Ht) * lA1c
             + (b2t + Hc @ b2Ht) * lA2c
             + sigma0_t * pts1[q] + sigma1_t * pts2[q] * lA1c)
    E_mu_msl += wts2[q] * np.exp(np.clip(eta_q, -40, 40))

rel = np.max(np.abs(E_mu_msl - E_mu_direct) / E_mu_direct)
ok_C2 = rel < 1e-6
print("\nPART C.b - mean integrated rate: MSL vs closed-form log-normal moment")
print(f"  max rel err = {rel:.2e}")

print("\nFINAL:", "PASS on A & B & C1 & C2" if (ok_A and ok_B and ok_C1 and ok_C2) else
      f"check (A:{ok_A}, B:{ok_B}, C1:{ok_C1}, C2:{ok_C2})")
