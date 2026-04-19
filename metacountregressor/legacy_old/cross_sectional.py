

from scipy.stats import norm
from scipy.special import gammaln, logsumexp
import numpy as np
import pandas as pd

from scipy.special import gammaln


# Define the hierarchical model structure
def two_level_spf(X, alpha0, alpha1, alpha2, beta0, beta1, gamma0, gamma1):
    AADTmaj, AADTmin, LT, RT, local = X

    # Sub-level CM-functions
    A = alpha0 * np.exp(alpha1 * LT + alpha2 * RT)
    B = beta0 * np.exp(beta1 * local)
    C = gamma0 * np.exp(gamma1 * local)

    # First-level SPF
    Nmv = A * (AADTmaj ** B) * (AADTmin ** C)
    return Nmv


def two_level_mu(params, AADTmaj, AADTmin, LT, RT, local):
    alpha0, alpha1, alpha2, beta0, beta1, gamma0, gamma1 = params

    A = alpha0 * np.exp(alpha1 * LT + alpha2 * RT)
    B = beta0 * np.exp(beta1 * local)
    C = gamma0 * np.exp(gamma1 * local)

    mu = A * (AADTmaj ** B) * (AADTmin ** C)
    return mu


def neg_loglike(params, y, AADTmaj, AADTmin, LT, RT, local):
    mu = two_level_mu(params, AADTmaj, AADTmin, LT, RT, local)
    # prevent underflows
    mu = np.clip(mu, 1e-10, 1e10)

    # Poisson log-likelihood
    loglike = np.sum(y * np.log(mu) - mu - gammaln(y + 1))
    return -loglike  # minimize negative log-likelihood





'''

# ---------------------------------------------
# 1. Example data
# ---------------------------------------------
data = pd.DataFrame({
    "crashes": [4,7,5,12,9,6,3,8,10,11],
    "AADTmaj": [10000,20000,15000,40000,30000,25000,8000,12000,35000,50000],
    "AADTmin": [5000,8000,7000,12000,9000,7500,4000,6500,10000,13000],
    "LT": [1,0,1,0,1,1,0,1,0,1],
    "RT": [1,1,0,1,1,0,1,0,1,0],
    # hypothetical "local" modifiers (each is binary or continuous)
    "urban": [1,0,0,1,1,0,0,1,0,1],
    "median": [0,1,1,0,0,1,1,0,1,0],
    "terrain": [0.3,0.1,0.2,0.5,0.4,0.2,0.1,0.4,0.5,0.6],
})

'''

# Matrix of multiple local variables
#locals_mat = data[["urban", "median", "terrain"]].values



def two_level_mu(params, AADTmaj, AADTmin, LT, RT, locals_mat):
    """
    params = [alpha0, alpha1, alpha2,
              beta0, β1..βk, gamma0, γ1..γk]
    locals_mat shape: (n, k)
    """
    k = locals_mat.shape[1]
    alpha0, alpha1, alpha2 = params[0:3]
    beta0 = params[3]
    beta = params[4:4+k]
    gamma0 = params[4+k]
    gamma = params[5+k:5+2*k]

    # Sub‑level components
    A = alpha0 * np.exp(alpha1 * LT + alpha2 * RT)

    # Vectors of local multipliers
    exp_beta_local = np.exp(locals_mat @ beta)
    exp_gamma_local = np.exp(locals_mat @ gamma)

    B = beta0 * exp_beta_local
    C = gamma0 * exp_gamma_local

    mu = A * (AADTmaj ** B) * (AADTmin ** C)
    return mu

def neg_loglike(params, y, AADTmaj, AADTmin, LT, RT, locals_mat):
    mu = two_level_mu(params, AADTmaj, AADTmin, LT, RT, locals_mat)
    mu = np.clip(mu, 1e-12, 1e12)
    loglike = np.sum(y * np.log(mu) - mu - gammaln(y + 1))
    return -loglike






def poisson_random_twolevel(params,
                            y,
                            AADTmaj, AADTmin, LT, RT, locals_mat,
                            R=200,
                            seed=42):
    """
    Poisson regression with random parameters built on
    the two-level SPF + CM-functions structure.

    Parameters
    ----------
    params : array-like
        Parameter vector:
        [alpha0, alpha1, alpha2,
         beta0, betas..., gamma0, gammas...,
         sA, sB, sC]
    y : (n,) array
        Crash counts (observed)
    AADTmaj, AADTmin : (n,) arrays
        Major/minor AADT volumes
    LT, RT : (n,) arrays
        Left/right-turn lane dummies
    locals_mat : (n, k) array
        Local factors for B and C CM-functions
    R : int
        Number of simulation draws for random parameters
    seed : int
        Random seed
    """

    np.random.seed(seed)
    n = len(y)
    k = locals_mat.shape[1]  # number of local factors

    # Unpack parameters ---------------------------------------------------------
    idx = 0
    alpha0, alpha1, alpha2 = params[idx:idx + 3];
    idx += 3

    beta0 = params[idx];
    idx += 1
    beta = params[idx:idx + k];
    idx += k

    gamma0 = params[idx];
    idx += 1
    gamma = params[idx:idx + k];
    idx += k

    sA, sB, sC = params[idx:idx + 3]  # std dev of random parameters

    # Halton/Normal random draws for random effects -----------------------------
    uA = norm.ppf(np.random.rand(R, n)) * sA  # (R,n)
    uB = norm.ppf(np.random.rand(R, n)) * sB
    uC = norm.ppf(np.random.rand(R, n)) * sC

    # Fixed portions ------------------------------------------------------------
    A_fixed = np.exp(alpha0 + alpha1 * LT + alpha2 * RT)
    B_fixed = beta0 + locals_mat @ beta
    C_fixed = gamma0 + locals_mat @ gamma

    # Simulation integration ----------------------------------------------------
    sim_loglike = np.zeros((R, n))
    for r in range(R):
        A = A_fixed * np.exp(uA[r])
        B = B_fixed + uB[r]
        C = C_fixed + uC[r]

        mu = A * (AADTmaj ** B) * (AADTmin ** C)
        mu = np.clip(mu, 1e-12, 1e12)

        # Log Poisson PMF
        sim_loglike[r] = y * np.log(mu) - mu - gammaln(y + 1)

    # Monte Carlo log-likelihood: log(mean over draws)
    loglik_i = logsumexp(sim_loglike, axis=0) - np.log(R)
    total_loglik = np.sum(loglik_i)

    return -total_loglik  # negative log-likelihood for minimization


from scipy.optimize import minimize





def poisson_random_twolevel_simple(params, y, AADT, LT, R=200, seed=42):
    """
    Poisson regression with the two-level SPF structure,
    one AADT variable, one LT variable, random A and B parameters.

    Parameters
    ----------
    params : array-like
        [alpha0, alpha1, beta0, beta1, sA, sB]
    y : (n,)
        Crash counts
    AADT : (n,)
        AADT volumes
    LT : (n,)
        Local term (e.g. left-turn lane dummy)
    R : int
        Number of simulation draws
    seed : int
        Random seed for reproducibility
    """

    np.random.seed(seed)
    n = len(y)

    # Unpack parameters ---------------------------------------------------------
    alpha0, alpha1, beta0, beta1, sA, sB = params

    # Random draws --------------------------------------------------------------
    uA = norm.ppf(np.random.rand(R, n)) * sA   # shape (R,n)
    uB = norm.ppf(np.random.rand(R, n)) * sB

    # Fixed portions ------------------------------------------------------------
    A_fixed = np.exp(alpha0 + alpha1 * LT)
    B_fixed = beta0 + beta1 * LT

    # Monte Carlo integration ---------------------------------------------------
    vectorise = True
    if not vectorise:
        sim_loglike = np.zeros((R, n))
        for r in range(R):
            A = A_fixed * np.exp(uA[r])
            B = B_fixed + uB[r]
            mu = A * (AADT ** B)
            mu = np.clip(mu, 1e-12, 1e12)

            sim_loglike[r] = y * np.log(mu) - mu - gammaln(y + 1)
    else:
        # Vectorized Monte Carlo simulation using broadcasting
        A = A_fixed * np.exp(uA)  # (R, n)
        B = B_fixed + uB  # (R, n)
        mu = A * (AADT[np.newaxis, :] ** B)  # (R, n)
        mu = np.clip(mu, 1e-12, 1e12)

    # Log-Poisson PMF over all draws at once
    sim_loglike = (y[np.newaxis, :] * np.log(mu)) - mu - gammaln(y[np.newaxis, :] + 1)

    # Average over random draws
    loglik_i = logsumexp(sim_loglike, axis=0) - np.log(R)
    total_loglik = np.sum(loglik_i)

    return -total_loglik  # for optimizer



def poisson_twolevel_AADT_LT(
    params,
    y,
    AADT,
    LT,
    Xf_A, Xr_A,   # covariates associated with A
    Xf_B, Xr_B,   # covariates associated with B
    draws_A, draws_B  # Halton draws for A & B random terms
):
    """
    Two-level Poisson SPF with random parameters driven by Halton draws.
    μ = exp(XfA βfA + XrA βrA*) * (AADT)^(XfB βfB + XrB βrB*)

    Each * means random coefficients (β_r + σ_r * z_r)
    """

    N = y.shape[0]
    Ra, Na, Ka = draws_A.shape
    Rb, Nb, Kb = draws_B.shape
    assert Ra == Rb and Na == Nb == N, "Draws must align (same N,R)"

    R = Ra

    # ---- Split parameter vector
    KfA = Xf_A.shape[1] if Xf_A is not None else 0
    KrA = Xr_A.shape[1] if Xr_A is not None else 0
    KfB = Xf_B.shape[1] if Xf_B is not None else 0
    KrB = Xr_B.shape[1] if Xr_B is not None else 0

    # Order: [βfA | βrA | σA | βfB | βrB | σB]
    βfA = params[0:KfA]
    βrA = params[KfA:KfA+KrA]
    σA  = params[KfA+KrA:KfA+2*KrA]

    βfB = params[KfA+2*KrA : KfA+2*KrA+KfB]
    βrB = params[KfA+2*KrA+KfB : KfA+2*KrA+KfB+KrB]
    σB  = params[KfA+2*KrA+KfB+KrB : KfA+2*KrA+KfB+KrB+KrB]

    # ---- Random coefficients for A and B
    βrA_draws = βrA[None, :] + σA[None, :] * draws_A[0:R, :, :]  # (R,N,KrA)
    βrB_draws = βrB[None, :] + σB[None, :] * draws_B[0:R, :, :]  # (R,N,KrB)

    # ---- Compute log(A) and B per draw
    logA_mean = (0 if KfA==0 else Xf_A @ βfA)               # (N,)
    logA_rand = (0 if KrA==0 else np.einsum("nk,rnk->rn", Xr_A, βrA_draws))
    logA_total = logA_mean[None,:] + logA_rand              # (R,N)

    B_mean = (0 if KfB==0 else Xf_B @ βfB)
    B_rand = (0 if KrB==0 else np.einsum("nk,rnk->rn", Xr_B, βrB_draws))
    B_total = B_mean[None,:] + B_rand                       # (R,N)

    # ---- Expected value μ
    μ = np.exp(logA_total) * (AADT[None, :] ** B_total)
    μ = np.clip(μ, 1e-12, 1e12)

    # ---- Log-likelihood per draw
    logl = y[None, :] * np.log(μ) - μ - gammaln(y[None, :] + 1)
    loglik_i = logsumexp(logl, axis=0) - np.log(R)
    total_loglik = np.sum(loglik_i)

    return -total_loglik

if __name__ == '__main__':
    # Example dataset
    data = pd.DataFrame({
        "crashes": [4, 7, 5, 12, 9, 6, 3, 8, 10, 11],
        "AADTmaj": [10000, 20000, 15000, 40000, 30000, 25000, 8000, 12000, 35000, 50000],
        "AADTmin": [5000, 8000, 7000, 12000, 9000, 7500, 4000, 6500, 10000, 13000],
        "LT": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],  # left-turn lanes
        "RT": [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],  # right-turn lanes
        "local_factor": [0.2, 0.5, 0.3, 0.8, 0.4, 0.3, 0.1, 0.5, 0.7, 0.6],
    })
    
    
    data =    pd.read_csv('data/Ex-16-3.csv')

    y = data["FREQ"].values
    AADTmaj = data["AADT"].values
    AADTmin = data["ADTLANE"].values
    LT = data["URB"].values
    RT = data["HISNOW"].values
    locals_mat = data['GBRPM', 'EXPOSE', 'INTPM'].values
    k = locals_mat.shape[1]  # number of local variables

    # Starting guesses (alpha0, alpha1, alpha2, beta0, β1..βk, gamma0, γ1..γk)
    p0 = np.concatenate(([1e-6, 0.1, 0.1, 0.3],
                         np.repeat(0.05, k),
                         [0.2],
                         np.repeat(0.05, k)))
    p0 = np.concatenate([
        [-6, 0, 0],  # α0, α1, α2
        [0.4],  # β0
        np.zeros(k),  # β locals
        [0.3],  # γ0
        np.zeros(k),  # γ locals
        [0.1, 0.1, 0.1]  # sA, sB, sC
    ])
    result = minimize(
        neg_loglike,
        x0=p0,
        args=(y, AADTmaj, AADTmin, LT, RT, locals_mat),
        method="L-BFGS-B",
        bounds=[(1e-10, None)] * len(p0)
    )
    print(result)







    
    # Example artificial dataset
    n = 100
    np.random.seed(1)
    AADTmaj = np.random.uniform(5000, 40000, n)
    AADTmin = np.random.uniform(2000, 15000, n)
    LT = np.random.binomial(1, 0.5, n)
    RT = np.random.binomial(1, 0.5, n)
    locals_mat = np.column_stack([
        np.random.binomial(1, 0.4, n),  # urban
        np.random.binomial(1, 0.3, n),  # median
        np.random.rand(n)  # terrain
    ])
    # Synthetic "true" parameters
    true_params = [-7, 0.2, 0.1, 0.4, 0.05, -0.02, 0.02, 0.06, 0.1, 0.15, 0.2, 0.05, 0.05, 0.05]
    mu_true = np.exp(-7 + 0.2 * LT + 0.1 * RT) * (AADTmaj ** 0.4) * (AADTmin ** 0.25)
    y = np.random.poisson(mu_true)

    # Fit
    p0 = np.array([-6, 0, 0, 0.3, 0, 0, 0, 0.3, 0, 0, 0.1, 0.1, 0.1])
    k = locals_mat.shape[1]
    p0 = np.concatenate([
        [-6, 0, 0],  # α0, α1, α2
        [0.4],  # β0
        np.zeros(k),  # β locals
        [0.3],  # γ0
        np.zeros(k),  # γ locals
        [0.1, 0.1, 0.1]  # sA, sB, sC
    ])
    res = minimize(poisson_random_twolevel, p0,
                   args=(y, AADTmaj, AADTmin, LT, RT, locals_mat),
                   method='L-BFGS-B',
                   options={'maxiter': 200, 'disp': 1})

    print(res)



    ## POISSON TWO LEVEL
    p0 = np.array([-6, 0, 0, 0.3, 0, 0.3, 0, 0, 0.1, 0.1, 0.1])
    res = minimize(poisson_random_twolevel_simple, p0,
                   args=(y, AADTmaj, LT, locals_mat),
                   method='L-BFGS-B',
                   options={'maxiter': 200, 'disp': 1})

    print(res)

    ## DONE


    data =    pd.read_csv('data/Ex-16-3variables.csv')