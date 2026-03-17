import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pandas as pd
import os
import itertools
from functools import partial
from typing import NamedTuple as TypingNamedTuple
from jaxopt import LBFGS
import pandas as pd
import scipy.stats as stats
from Solvers_METAJAX import *
from scipy.stats import qmc
from scipy.stats import norm
import argparse
import time
import psutil
import json
from datetime import datetime
import sys
from contextlib import redirect_stdout

print("Physical cores:", psutil.cpu_count(logical=False))
print("Logical cores:", psutil.cpu_count(logical=True))

DIST_MAP = {
    "normal": 0,
    "lognormal": 1,
    "triangular": 2,
    'uniform':3,
    
}

def generate_halton_normal(N, K, R, seed=42):

    sampler = qmc.Halton(d=K, scramble=True, seed=seed)

    u = sampler.random(R)          # (R, K)
    z = norm.ppf(u)                # (R, K)

    z = z.T                        # (K, R)
    z = np.tile(z[None, :, :], (N, 1, 1))  # (N, K, R)

    return z

def decode_distribution(dist_code, allowed_list):

    if len(allowed_list) == 1:
        return allowed_list[0]

    idx = dist_code % len(allowed_list)
    return allowed_list[idx]



def random_decision(D, rng):

    # 0 to 5 inclusive
    return rng.integers(0, 6, size=D)

def pareto_front(solutions):

    pareto = []

    for s in solutions:
        dominated = False

        for t in solutions:
            if (
                t.train_ll >= s.train_ll and
                t.test_rmse <= s.test_rmse and
                (t.train_ll > s.train_ll or
                 t.test_rmse < s.test_rmse)
            ):
                dominated = True
                break

        if not dominated:
            pareto.append(s)

    return pareto

class ModelStructureOptimizer:

    def __init__(self, df, mode="single", algorithm="de"):
        self.df = df
        self.mode = mode
        self.algorithm = algorithm
        self.solutions = []

    def evaluate(self, decision_vector):

        if self.mode == "single":
            score = single_objective(decision_vector, self.df)
            return score

        elif self.mode == "multi":
            train_ll, test_rmse = multi_objective(decision_vector, self.df)
            return train_ll, test_rmse

    def optimize(self, n_iter=20):

        for i in range(n_iter):

            decision = self.random_decision()

            result = self.evaluate(decision)

            self.solutions.append((decision, result))

        if self.mode == "multi":
            return pareto_front(self.solutions)
        else:
            return max(self.solutions, key=lambda x: x[1])


#===================================================


# =========================================================
# UTILITIES
# =========================================================

def build_cholesky(params, K):
    L = jnp.zeros((K, K))
    idx = 0
    for i in range(K):
        for j in range(i + 1):
            L = L.at[i, j].set(params[idx])
            idx += 1
    return L





def random_correlated(mean, chol_params, draws, K, dist_codes):

    L = jnp.zeros((K, K))
    tril = jnp.tril_indices(K)
    L = L.at[tril].set(chol_params)

    diag = jnp.diag_indices(K)
    L = L.at[diag].set(jnp.exp(L[diag]))

    z_corr = jnp.einsum("ij,njr->nir", L, draws)

    scale_dummy = jnp.zeros(K)  # not used, embedded in chol

    beta = transform_draws(
        z_corr,
        mean,
        scale_dummy,
        dist_codes
    )

    return beta


''' works with non hetrogeneity 
def random_correlated(mean, chol_params, draws, K):
    L = build_cholesky(chol_params, K)
    return mean[None, :, None] + jnp.einsum("ij,njr->nir", L, draws)
'''
def random_independent(mean, sd, draws, dist_codes):

    return transform_draws(
        draws,
        mean,
        sd,
        dist_codes
    )
'''
def random_independent(mean, sd, draws):
    return mean[None, :, None] + sd[None, :, None] * draws
'''

def poisson_loglik(y, mu):
    mu = jnp.clip(mu, 1e-12, 1e12)
    return y * jnp.log(mu) - mu - jsp.special.gammaln(y + 1)


def nb_loglik(y, mu, alpha):
    r = 1.0 / jnp.exp(alpha)
    return (
        jsp.special.gammaln(y + r)
        - jsp.special.gammaln(r)
        - jsp.special.gammaln(y + 1)
        + r * jnp.log(r / (r + mu))
        + y * jnp.log(mu / (r + mu))
    )

def ensure_3d(x):
    if x.ndim == 2:
        return x[..., None]
    return x


def transform_draws(draws, mean, scale, dist_codes):
    """
    draws: (N,K,R) standard normal or uniform
    mean:  (K,) or (N,K)
    scale: (K,)
    dist_codes: integer array (K,)
        0 = normal
        1 = lognormal
        2 = triangular
    """

    scale = jnp.exp(scale)

    if mean.ndim == 1:
        mean = mean[None, :, None]
    else:
        mean = mean[:, :, None]

    scale = scale[None, :, None]

    z = draws

    # Normal
    beta_normal = mean + scale * z

    # Lognormal
    beta_lognormal = jnp.exp(mean + scale * z)

    # Triangular (using inverse CDF of symmetric triangular)
    # Assume draws are uniform in (-1,1)
    u = jsp.stats.norm.cdf(z)
    beta_tri = mean + scale * (2*u - 1)
    
        # Uniform (mean + scale * u) where u in (-1,1)
    u = jsp.stats.norm.cdf(z) * 2 - 1
    beta_uniform = mean + scale * u

    # Stack and select
    betas = jnp.stack([beta_normal, beta_lognormal, beta_tri, beta_uniform], axis=-1)

    # dist_codes shape (K,)
    selector = jax.nn.one_hot(dist_codes, 4)  # (K,3)

    selector = selector[None, :, None, :]  # (1,K,1,3)

    beta = jnp.sum(betas * selector, axis=-1)

    return beta  # (N,K,R)
# LOG LIKELIHOOD
# =========================================================


@jax.jit
def mixed_model_loglik(params, data, masks):

    # ============================================================
    # Fixed architecture sizes (STATIC — safe for JIT slicing)
    # ============================================================

    MAX_RANDOM = masks["ind_mask"].shape[0]
    MAX_FIXED  = data["Xf"].shape[-1]        # includes intercept

    N_CHOL  = MAX_RANDOM * (MAX_RANDOM + 1) // 2
    N_GAMMA = MAX_RANDOM * MAX_RANDOM

    # ============================================================
    # Unpack parameters (STATIC SLICING — JAX SAFE)
    # ============================================================

    i = 0

    # Fixed effects
    beta_f = params[i:i+MAX_FIXED]
    i += MAX_FIXED

    # Independent random
    mean_ind = params[i:i+MAX_RANDOM]
    i += MAX_RANDOM

    sd_ind = params[i:i+MAX_RANDOM]
    i += MAX_RANDOM

    # Correlated random
    mean_cor = params[i:i+MAX_RANDOM]
    i += MAX_RANDOM

    chol_vec = params[i:i+N_CHOL]
    i += N_CHOL

    # Grouped random
    mean_g = params[i:i+MAX_RANDOM]
    i += MAX_RANDOM

    sd_g = params[i:i+MAX_RANDOM]
    i += MAX_RANDOM

    # Heterogeneity
    gamma = params[i:i+N_GAMMA]
    i += N_GAMMA

    # Zero inflation coefficients
    MAX_ZI = data["Xzi"].shape[-1]
    beta_zi = params[i:i+MAX_ZI]
    i += MAX_ZI

    # Dispersion
    alpha = params[i]

    # ============================================================
    # Apply masks
    # ============================================================

    beta_f = beta_f * jnp.concatenate(
        [jnp.array([1.0]), masks["fixed_mask"]]
    )

    mean_ind *= masks["ind_mask"]
    sd_ind   *= masks["ind_mask"]

    mean_cor *= masks["cor_mask"]

    mean_g *= masks["group_mask"]
    sd_g   *= masks["group_mask"]
    beta_zi = beta_zi * masks["zi_mask"]
    #beta_zi = beta_zi * jnp.concatenate(
    #[jnp.array([1.0]), masks["zi_mask"]]
    #)

    # ============================================================
    # Build correlated Cholesky matrix
    # ============================================================

    L = jnp.zeros((MAX_RANDOM, MAX_RANDOM))
    tril_idx = jnp.tril_indices(MAX_RANDOM)
    L = L.at[tril_idx].set(chol_vec)

    cor_mask_matrix = (
        masks["cor_mask"][:, None] *
        masks["cor_mask"][None, :]
    )
    L = L * cor_mask_matrix

    # ============================================================
    # Random draws
    # ============================================================

    draws = data["draws"]  # (N, MAX_RANDOM, R)

    # Independent random
    beta_ind = (
        mean_ind[None,:,None] +
        jnp.exp(sd_ind)[None,:,None] * draws
    )
    beta_ind *= masks["ind_mask"][None,:,None]

    # Correlated random
    beta_cor = (
        mean_cor[None,:,None] +
        jnp.einsum("ij,njr->nir", L, draws)
    )
    beta_cor *= masks["cor_mask"][None,:,None]

    # Group random (correct group-level indexing)
    group_ids   = data["group_ids"]      # (N,)
    group_draws = data["group_draws"]    # (G, MAX_RANDOM, R)

    z_group = group_draws[group_ids]     # (N, MAX_RANDOM, R)

    beta_g = (
        mean_g[None,:,None] +
        jnp.exp(sd_g)[None,:,None] * z_group
    )
    beta_g *= masks["group_mask"][None,:,None]

    # ============================================================
    # Heterogeneity in means
    # ============================================================

    gamma_matrix = gamma.reshape(MAX_RANDOM, MAX_RANDOM)

    

    het_effect = jnp.einsum("npk,kr->np", data["Xh"], gamma_matrix)
    het_effect = het_effect[..., None]  # (N, MAX_RANDOM, 1)
    
    beta_ind += het_effect * masks["ind_mask"][None,:,None]
    beta_cor += het_effect * masks["cor_mask"][None,:,None]

    # ============================================================
    # Linear predictor
    # ============================================================

    eta = jnp.einsum("npk,k->np", data["Xf"], beta_f)[..., None]

    eta += jnp.einsum("npk,nkr->npr", data["Xr"], beta_ind)
    eta += jnp.einsum("npk,nkr->npr", data["Xr"], beta_cor)
    eta += jnp.einsum("npk,nkr->npr", data["Xg"], beta_g)
    eta = jnp.clip(eta, -20, 20)
    mu = jnp.exp(eta)
    # Compute zero-inflation probability (always compute; mask later)
    eta_zi = jnp.einsum("npk,k->np", data["Xzi"], beta_zi)[..., None]
    pi = jax.nn.sigmoid(eta_zi)

    # Turn off zero inflation if mask == 0
    pi = pi * masks["zero_inflated"]
    # ============================================================
    # Likelihood
    # ============================================================

    y = data["y"]
    mask_obs = data["mask"]

    # ---- Count model (Poisson vs NB2) ----

    ll_poisson = poisson_loglik(y, mu)
    ll_nb      = nb2_loglik(y, eta, alpha)

    # model_flag: 0 = Poisson, 1 = NB2
    ll_count = (
        (1 - masks["model_flag"]) * ll_poisson +
        masks["model_flag"] * ll_nb
    )

    # ---- Zero inflation mixture ----
    eps = 1e-12
    ll_zi = jnp.where(
        y == 0,
        jnp.log(jnp.clip(pi + (1 - pi) * jnp.exp(ll_count), eps)),
        jnp.log(1.0 - pi) + ll_count
    )

    # zero_inflated: 0 = standard model, 1 = ZI model
    ll = (
        masks["zero_inflated"] * ll_zi +
        (1 - masks["zero_inflated"]) * ll_count
    )

    # ---- Apply observation mask ----

    ll = ll * mask_obs

    # ---- Panel likelihood integration ----

    ll_panel = jnp.sum(ll, axis=1)
    ll_ind = jsp.special.logsumexp(ll_panel, axis=-1) - jnp.log(draws.shape[-1])

    return -jnp.sum(ll_ind)
    # ============================================================
    # Zero inflation
    # ============================================================

    
mixed_model_loglik_jit = jax.jit(mixed_model_loglik)

def nb2_loglik(y, eta, alpha):

    alpha = jnp.exp(alpha)
    inv_alpha = 1.0 / alpha

    log_mu = eta
    log_inv_alpha = jnp.log(inv_alpha)

    # ✅ broadcast scalar to (N,P,R)
    log_inv_alpha = jnp.broadcast_to(log_inv_alpha, log_mu.shape)

    log_denom = jsp.special.logsumexp(
        jnp.stack([log_inv_alpha, log_mu], axis=0),
        axis=0
    )

    term1 = jsp.special.gammaln(y + inv_alpha)
    term2 = -jsp.special.gammaln(inv_alpha)
    term3 = -jsp.special.gammaln(y + 1)

    term4 = inv_alpha * (log_inv_alpha - log_denom)
    term5 = y * (log_mu - log_denom)

    return term1 + term2 + term3 + term4 + term5



def balance_panel_dataframe(df, id_col, y_col, feature_cols):

    df = df.sort_values(id_col)

    ids = df[id_col].unique()
    N = len(ids)

    counts = df.groupby(id_col).size().values
    P = counts.max()

    K = len(feature_cols)

    X = np.zeros((N, P, K))
    y = np.zeros((N, P, 1))
    mask = np.zeros((N, P))

    for n, pid in enumerate(ids):
        sub = df[df[id_col] == pid]
        T = len(sub)

        X[n, :T, :] = sub[feature_cols].values
        y[n, :T, 0] = sub[y_col].values
        mask[n, :T] = 1.0

    return X, y, mask

def extract_offset(df, id_col, offset_col):
    df = df.sort_values(id_col)
    ids = df[id_col].unique()
    N = len(ids)
    counts = df.groupby(id_col).size().values
    P = counts.max()

    offset = np.zeros((N, P, 1))

    for n, pid in enumerate(ids):
        sub = df[df[id_col] == pid]
        T = len(sub)
        offset[n, :T, 0] = sub[offset_col].values

    return offset

# 
def build_datasets(df, id_col, y_col):
    panel_df = prepare_panel_summary(df, id_col, y_col)
    panel_df = create_strata(panel_df)

    train_panels, test_panels, val_panels = stratified_panel_split(panel_df)

    train_ids = train_panels[id_col]
    test_ids = test_panels[id_col]
    val_ids = val_panels[id_col]

    df_train = df[df[id_col].isin(train_ids)]
    df_test  = df[df[id_col].isin(test_ids)]
    df_val   = df[df[id_col].isin(val_ids)]

    return df_train, df_test, df_val

#=======================================
# PARAM INDEX



    
def compute_standard_errors(params, objective):
    hess = jax.hessian(objective)(params)
    cov = jnp.linalg.pinv(hess)
    se = jnp.sqrt(jnp.diag(cov))
    return se

def build_cholesky_from_params(chol_params, K):
    L = np.zeros((K, K))
    idx = 0
    for i in range(K):
        for j in range(i + 1):
            L[i, j] = chol_params[idx]
            idx += 1
    return L
'''
def print_summary(result, objective, data, spec, param_index):

    params = result.params
    se = compute_standard_errors(params, objective)

    z_vals = params / se
    p_vals = 2 * (1 - stats.norm.cdf(jnp.abs(z_vals)))

    # Log-likelihood
    final_ll = -objective(params)
    k = len(params)
    n = data["y"].shape[0]

    aic = 2*k - 2*final_ll
    bic = k * jnp.log(n) - 2*final_ll

    # Build parameter names
    names = []

    # Fixed
    for name in spec.fixed_names:
        names.append(name)

    # Correlated
    for name in spec.random_cor_names:
        names.append(f"cor_mean({name})")

    if spec.Kr_cor > 0:
        corr_cols = spec.correlated_cols
        K = spec.Kr_cor

        for i in range(K):
            for j in range(i + 1):
                names.append(f"chol({corr_cols[i]},{corr_cols[j]})")

    # Independent
    for name in spec.random_ind_names:
        names.append(f"mean({name})")
    for name in spec.random_ind_names:
        names.append(f"sd({name})")

    # Grouped
    if spec.Kg > 0:
        for name in spec.grouped_names:
            names.append(f"group_mean({name})")
        for name in spec.grouped_names:
            names.append(f"group_sd({name})")

    # Heterogeneity
    if spec.Kh > 0:
        for rnd in spec.random_cor_names + spec.random_ind_names:
            for z in spec.hetro_names:
                names.append(f"hetro({rnd}|{z})")

    # NB dispersion
    if spec.model == "nb":
        names.append("dispersion")

    summary_df = pd.DataFrame({
        "Parameter": names,
        "Estimate": np.array(params),
        "Std.Err": np.array(se),
        "z-value": np.array(z_vals),
        "p-value": np.array(p_vals)
    })

    print("\n================ MODEL SUMMARY ================")
    print(summary_df.to_string(index=False))
    print("------------------------------------------------")
    print(f"Log-Likelihood: {float(final_ll):.4f}")
    print(f"AIC: {float(aic):.4f}")
    print(f"BIC: {float(bic):.4f}")
    print("================================================\n")
    
'''

from sklearn.model_selection import train_test_split


def prepare_panel_summary(df, id_col, y_col):

    panel_df = (
        df.groupby(id_col)[y_col]
        .agg(["mean", "sum"])
        .reset_index()
    )

    return panel_df

def prepare_panel_summary(df, id_col, y_col):

    panel_df = (
        df.groupby(id_col)[y_col]
        .agg(["mean", "sum"])
        .reset_index()
    )

    return panel_df

from sklearn.model_selection import train_test_split

def stratified_panel_split(panel_df, train_size=0.6, test_size=0.2, val_size=0.2, seed=42):

    assert abs(train_size + test_size + val_size - 1) < 1e-8

    # First split train vs temp
    train_panels, temp_panels = train_test_split(
        panel_df,
        test_size=(1 - train_size),
        stratify=panel_df["strata"],
        random_state=seed
    )

    # Split temp into test and val
    relative_test = test_size / (test_size + val_size)

    test_panels, val_panels = train_test_split(
        temp_panels,
        test_size=(1 - relative_test),
        stratify=temp_panels["strata"],
        random_state=seed
    )

    return train_panels, test_panels, val_panels


def create_strata(panel_df, col="mean", n_bins=5):

    panel_df["strata"] = pd.qcut(
        panel_df[col],
        q=n_bins,
        duplicates="drop"
    )

    return panel_df




def build_datasets(df, id_col, y_col):

    panel_df = prepare_panel_summary(df, id_col, y_col)
    panel_df = create_strata(panel_df)

    train_panels, test_panels, val_panels = stratified_panel_split(panel_df)

    train_ids = train_panels[id_col]
    test_ids = test_panels[id_col]
    val_ids = val_panels[id_col]

    df_train = df[df[id_col].isin(train_ids)]
    df_test = df[df[id_col].isin(test_ids)]
    df_val = df[df[id_col].isin(val_ids)]

    return df_train, df_test, df_val


def create_strata(panel_df, col="mean", n_bins=5):

    panel_df["strata"] = pd.qcut(
        panel_df[col],
        q=n_bins,
        duplicates="drop"
    )

    return panel_df

def split_data(df, train_size=0.6, test_size=0.2, val_size=0.2, seed=42):

    assert abs(train_size + test_size + val_size - 1.0) < 1e-8

    # First split: train vs temp
    df_train, df_temp = train_test_split(
        df,
        test_size=(1 - train_size),
        random_state=seed
    )

    # Second split: test vs validation
    relative_test_size = test_size / (test_size + val_size)

    df_test, df_val = train_test_split(
        df_temp,
        test_size=(1 - relative_test_size),
        random_state=seed
    )

    return df_train, df_test, df_val


'''
def get_operator(algo, config_id, evaluator, dimension):

    config = ALL_CONFIGS[algo][config_id]

    if algo == "de":

        operator = AdaptiveDE(
            evaluator=evaluator,
            dimension=dimension,
            **config
        )

    elif algo == "hs":

        operator = DynamicHarmony(
            evaluator=evaluator,
            dimension=dimension,
            **config
        )

    elif algo == "sa":

        operator = MultiStartSA(
            evaluator=evaluator,
            dimension=dimension,
            n_starts=config["n_starts"],
            mutation_rate=config["mutation_rate"],
            step_size=config["step_size"],
            alpha=config["alpha"],
        )

    return operator, config
'''


def generate_configs(param_grid):

    keys = param_grid.keys()
    values = param_grid.values()

    configs = []

    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)

    return configs





def generate_master_halton(N, K, R, seed=42):
    sampler = qmc.Halton(d=K, scramble=True, seed=seed)
    u = sampler.random(R)
    z = norm.ppf(u)
    z = z.T
    z = np.tile(z[None, :, :], (N, 1, 1))
    return jnp.array(z)

class CountModel:

    def __init__(self, data, masks, max_vars):
    
        self.data = data
        self.masks = masks
        self.max_vars = max_vars
        #self.param_index = self.build_param_index(spec)
        self.layout = self.build_param_layout()
        self.params = None
        
        self._objective = self._build_objective()
        
    def _build_objective(self):
        layout = self.layout
        data = self.data
        masks = self.masks

        def objective(params):
            return mixed_model_loglik(params, data, masks)

        return objective
    
    def build_param_layout(self):

        MAX_VARS = self.max_vars

        MAX_FIXED = MAX_VARS + 1
        MAX_RANDOM = MAX_VARS
        MAX_GROUP = MAX_VARS
        MAX_HETERO = MAX_VARS
        MAX_ZI = MAX_VARS
        MAX_CHOL = MAX_RANDOM * (MAX_RANDOM + 1) // 2

        layout = {}
        idx = 0

        layout["beta_f"] = (idx, idx + MAX_FIXED)
        idx += MAX_FIXED

        layout["mean_ind"] = (idx, idx + MAX_RANDOM)
        idx += MAX_RANDOM

        layout["sd_ind"] = (idx, idx + MAX_RANDOM)
        idx += MAX_RANDOM

        layout["mean_cor"] = (idx, idx + MAX_RANDOM)
        idx += MAX_RANDOM

        layout["chol_cor"] = (idx, idx + MAX_CHOL)
        idx += MAX_CHOL

        layout["mean_g"] = (idx, idx + MAX_GROUP)
        idx += MAX_GROUP

        layout["sd_g"] = (idx, idx + MAX_GROUP)
        idx += MAX_GROUP

        layout["gamma"] = (idx, idx + MAX_HETERO * MAX_RANDOM)
        idx += MAX_HETERO * MAX_RANDOM

        layout["beta_zi"] = (idx, idx + MAX_ZI)
        idx += MAX_ZI

        layout["dispersion"] = idx
        idx += 1

        layout["total"] = idx

        return layout
    

    
    def objective(self, params):
        return self._objective(params)

    def fit(self):
        init = jnp.zeros(self.layout["total"])
        solver = LBFGS(fun=self.objective)
        result = solver.run(init)
        self.params = result.params
        return result

    def loglik(self):
        return -self.objective(self.params)

    def bic(self):
        n = self.data["y"].shape[0]
        k = self.layout["total"]
        return -2 * self.loglik() + k * np.log(n)

    def predict(self):
        return compute_mean_prediction_fixed(
            self.params,
            self.data,
            self.masks,
            self.layout
        )

def build_jax_data_fixed(df, id_col, y_col, all_variables, R, group_id_col=None):

    X_all, y, mask = balance_panel_dataframe(
        df, id_col, y_col, all_variables
    )

    N, P, K = X_all.shape

    Xf = np.zeros((N, P, K + 1))
    Xf[..., 0] = 1.0
    Xf[..., 1:] = X_all
        # ---------------------------------
    # GROUP IDS
    # ---------------------------------

    if group_id_col is not None:

        df_sorted = df.sort_values(id_col)
        panel_ids = df_sorted[id_col].unique()

        # Map each panel to group
        panel_group = (
            df_sorted
            .groupby(id_col)[group_id_col]
            .first()
            .loc[panel_ids]
            .values
        )

        # Convert group labels → 0..G-1 integers
        unique_groups, group_ids = np.unique(panel_group, return_inverse=True)

        G = len(unique_groups)

        group_draws = generate_master_halton(G, K, R)

    else:
        group_ids = np.zeros(N, dtype=int)
        group_draws = generate_master_halton(1, K, R)

    data = {
        "Xf": jnp.array(Xf),
        "Xr": jnp.array(X_all),
        "Xg": jnp.array(X_all),
        "Xh": jnp.array(X_all),
        "Xzi": jnp.array(X_all),
        "y": jnp.array(y),
        "mask": jnp.array(mask),
        "draws": generate_master_halton(N, K, R),
        # ✅ ADD THESE
        "group_ids": jnp.array(group_ids),
        "group_draws": jnp.array(group_draws)
    }

    return data

def build_eta_fixed(params, data, masks, layout):

    beta_f = params[layout["beta_f"][0]:layout["beta_f"][1]]
    mean_ind = params[layout["mean_ind"][0]:layout["mean_ind"][1]]
    sd_ind   = params[layout["sd_ind"][0]:layout["sd_ind"][1]]
    mean_cor = params[layout["mean_cor"][0]:layout["mean_cor"][1]]
    chol_vec = params[layout["chol_cor"][0]:layout["chol_cor"][1]]
    mean_g = params[layout["mean_g"][0]:layout["mean_g"][1]]
    sd_g   = params[layout["sd_g"][0]:layout["sd_g"][1]]
    gamma  = params[layout["gamma"][0]:layout["gamma"][1]]
    beta_zi = params[layout["beta_zi"][0]:layout["beta_zi"][1]]

    MAX_RANDOM = masks["ind_mask"].shape[0]
    R = data["draws"].shape[-1]

    beta_f = beta_f * jnp.concatenate([jnp.array([1.0]), masks["fixed_mask"]])

    mean_ind *= masks["ind_mask"]
    sd_ind   *= masks["ind_mask"]
    mean_cor *= masks["cor_mask"]
    mean_g   *= masks["group_mask"]
    sd_g     *= masks["group_mask"]

    L = jnp.zeros((MAX_RANDOM, MAX_RANDOM))
    tril_idx = jnp.tril_indices(MAX_RANDOM)
    L = L.at[tril_idx].set(chol_vec)

    cor_mask_matrix = masks["cor_mask"][:, None] * masks["cor_mask"][None, :]
    L = L * cor_mask_matrix

    draws = data["draws"]

    beta_ind = mean_ind[None,:,None] + jnp.exp(sd_ind)[None,:,None] * draws
    beta_ind *= masks["ind_mask"][None,:,None]

    beta_cor = mean_cor[None,:,None] + jnp.einsum("ij,njr->nir", L, draws)
    beta_cor *= masks["cor_mask"][None,:,None]

    beta_g = mean_g[None,:,None] + jnp.exp(sd_g)[None,:,None] * draws
    beta_g *= masks["group_mask"][None,:,None]

    gamma_matrix = gamma.reshape(MAX_RANDOM, MAX_RANDOM)
    het_effect = jnp.einsum("npk,kr->npr", data["Xh"], gamma_matrix)

    beta_ind += het_effect * masks["ind_mask"][None,:,None]
    beta_cor += het_effect * masks["cor_mask"][None,:,None]

    eta = jnp.einsum("npk,k->np", data["Xf"], beta_f)[..., None]
    eta += jnp.einsum("npk,nkr->npr", data["Xr"], beta_ind)
    eta += jnp.einsum("npk,nkr->npr", data["Xr"], beta_cor)
    eta += jnp.einsum("npk,nkr->npr", data["Xg"], beta_g)

    return eta


class StructureEvaluator:

    def __init__(
        self,
        df,
        id_col,
        y_col,
        all_variables,
        allowed_roles,
        allowed_distributions,
        mode="single",
        group_id_col=None,
        offset_col=None,
        R=200
    ):

        self.id_col = id_col
        self.y_col = y_col
        self.offset_col=offset_col
        self.vars = all_variables
        self.allowed_roles = allowed_roles
        self.allowed_distributions = allowed_distributions
        self.mode = mode
        self.group_id_col = group_id_col
        self.R = R

        
        if mode == "multi":
            self.df_train, self.df_test, self.df_val = build_datasets(
                df, id_col, y_col
            )
        else:
            self.df_train = df
            self.df_test = None
            self.df_val = None


        # ✅ Build FIXED-SHAPE DATA ONCE PER SPLIT

        self.data_train = build_jax_data_fixed(
            df=self.df_train,
            id_col=self.id_col,
            y_col=self.y_col,
            all_variables=self.vars,
            R=self.R,
            group_id_col=self.group_id_col
        )

        if mode == "multi":

            self.data_test = build_jax_data_fixed(
                df=self.df_test,
                id_col=self.id_col,
                y_col=self.y_col,
                all_variables=self.vars,
                R=self.R,
                group_id_col=self.group_id_col
            )

            self.data_val = build_jax_data_fixed(
                df=self.df_val,
                id_col=self.id_col,
                y_col=self.y_col,
                all_variables=self.vars,
                R=self.R,
                group_id_col=self.group_id_col
            )
        else:
            self.data_test = None
            self.data_val = None
        
        


        # ✅ simple cache
        self.cache = {}

    # ----------------------------------------
    # Build Spec
    # ----------------------------------------

  

    # ----------------------------------------
    # Fitness
    # ----------------------------------------
    def fitness(self, decision):

        masks = self.build_masks(decision)

        def objective(params):
            return mixed_model_loglik_jit(params, self.data_train, masks)

        solver = LBFGS(fun=objective)
        result = solver.run(jnp.zeros(self.param_dim))

        return float(result.state.value)

    
    
    def evaluate_validation(self, decision):

        masks = self.build_masks(decision)

        model = CountModel(
            data=self.data_train,
            masks=masks,
            max_vars=len(self.vars)
        )

        model.fit()

        model_val = CountModel(
            data=self.data_val,
            masks=masks,
            max_vars=len(self.vars)
        )

        model_val.params = model.params

        preds = model_val.predict()
        y_val = np.array(self.data_val["y"]).squeeze()

        rmse = np.sqrt(np.mean((preds - y_val) ** 2))

        return rmse
    
    def build_masks(self, decision):

        D = len(self.vars)

        roles = decision[:D]
        dists = decision[D:-1]
        dispersion_bit = decision[-1]

        fixed_mask = (roles == 1).astype(np.float32)
        ind_mask   = (roles == 2).astype(np.float32)
        cor_mask   = (roles == 3).astype(np.float32)
        group_mask = (roles == 4).astype(np.float32)
        het_mask   = (roles == 5).astype(np.float32)
        zi_mask    = (roles == 6).astype(np.float32)
        zero_inflated = 1 if np.any(roles == 6) else 0
        
        dist_codes = dists.astype(np.int32)

        model_flag = dispersion_bit % 2  # 0=Poisson, 1=NB

        return {
            "fixed_mask": fixed_mask,
            "ind_mask": ind_mask,
            "cor_mask": cor_mask,
            "group_mask": group_mask,
            "het_mask": het_mask,
            "zi_mask": zi_mask,
            "dist_codes": dist_codes,
            "model_flag": model_flag,
            "zero_inflated": zero_inflated
        }

def compute_mean_prediction_fixed(params, data, masks, layout):

    eta = build_eta_fixed(params, data, masks, layout)
    eta = jnp.clip(eta, -20, 20)
    mu = jnp.exp(eta)

    return jnp.mean(mu, axis=-1)


def generate_draws(dist_name, shape):

    if dist_name == "normal":
        return np.random.normal(size=shape)

    elif dist_name == "lognormal":
        return np.random.lognormal(mean=0, sigma=1, size=shape)

    elif dist_name == "triangular":
        return np.random.triangular(-1, 0, 1, size=shape)

    elif dist_name == "uniform":
        return np.random.uniform(-1, 1, size=shape)

    else:
        raise ValueError("Unknown distribution")

   
def decode_best_solution(best_solution, evaluator):

    D = len(evaluator.vars)

    roles = best_solution[:D]
    dists = best_solution[D:]

    print("\n================ BEST MODEL STRUCTURE ================\n")

    for i, var in enumerate(evaluator.vars):

        role = roles[i]

        role_map = {
            0: "Excluded",
            1: "Fixed",
            2: "Random Independent",
            3: "Random Correlated",
            4: "Grouped",
            5: "Heterogeneity",
            6: "Zero Inflation"
        }

        role_name = role_map.get(role, "Unknown")

        dist = None
        if var in evaluator.allowed_distributions:
            allowed = evaluator.allowed_distributions[var]
            dist = allowed[dists[i] % len(allowed)]

        print(f"{var:12} → {role_name:22} | Dist: {dist}")

    print("\n======================================================\n")


def populate_allowed_roles(all_variables,
                           allowed_roles_partial,
                           default_roles=None):

    if default_roles is None:
        default_roles = [0,1,2,3,4,5,6]

    full_roles = {}

    for var in all_variables:
        full_roles[var] = allowed_roles_partial.get(
            var,
            default_roles
        )

    return full_roles


def populate_allowed_distributions(all_variables,
                                   allowed_dist_partial,
                                   default_dist=None):

    if default_dist is None:
        default_dist = ["normal", 'uniform']

    full = {}

    for var in all_variables:
        full[var] = allowed_dist_partial.get(
            var,
            default_dist
        )

    return full


def print_from_nsga_result(nsga_result, evaluator):

    solutions = nsga_result["solutions"]
    scores = nsga_result["scores"]

    # SINGLE OBJECTIVE
    if not isinstance(scores, np.ndarray) or scores.ndim == 0:

        best_solution = solutions

        print("\n================ BEST NSGA STRUCTURE ================\n")
        decode_best_solution(best_solution, evaluator)
        refit_and_print(evaluator, best_solution)
        return

    # MULTI OBJECTIVE
    solutions = np.array(solutions)
    scores = np.array(scores)

    best_idx = np.argmin(scores[:, 0])   # e.g., best BIC
    best_solution = solutions[best_idx]

    print("\n================ BEST NSGA STRUCTURE ================\n")
    decode_best_solution(best_solution, evaluator)
    refit_and_print(evaluator, best_solution)


def run_stepwise(evaluator, seed):

    np.random.seed(seed)

    solver = StepwiseStructureSolver(
        fitness_function=evaluator.fitness,
        all_variables=evaluator.vars,
        allowed_roles=evaluator.allowed_roles,
        allowed_distributions=evaluator.allowed_distributions,
        max_iter=50
    )

    solution, score = solver.optimize()

    return {
        "algorithm": "Stepwise",
        "seed": seed,
        "solution": solution,
        "score": score
    }

#def run_nsga(evaluator, operator_class, seed, pop_size=30, max_iter=40):
def run_nsga(evaluator, operator, seed, pop_size=30, max_iter=40, n_jobs=1):

    np.random.seed(seed)

    engine = NSGA2Engine(
    evaluator=evaluator,
    operator=operator,
    dimension=2 * len(evaluator.vars)+1,
    pop_size=pop_size,
    max_iter=max_iter,
    n_jobs=n_jobs,
    save_history=True
    )

    solutions, scores = engine.optimize()

    return {
        "algorithm": operator.__class__.__name__,
        "seed": seed,
        "solutions": solutions,
        "scores": scores,
        "fitness_history": engine.fitness_history,
        "hypervolume_history": engine.hypervolume_history,
        "pareto_history": engine.pareto_history
    }


def run_stepwise(evaluator, seed):

    np.random.seed(seed)

    solver = StepwiseStructureSolver(
        fitness_function=evaluator.fitness,
        all_variables=evaluator.vars,
        allowed_roles=evaluator.allowed_roles,
        allowed_distributions=evaluator.allowed_distributions,
        max_iter=50
    )

    solution, score = solver.optimize()

    return {
        "algorithm": "Stepwise",
        "seed": seed,
        "solution": solution,
        "score": score
    }



def refit_and_print(evaluator, decision):

    masks = evaluator.build_masks(decision)

    model = CountModel(
        data=evaluator.data_train,
        masks=masks,
        max_vars=len(evaluator.vars)
    )

    model.fit()

    print("\nLogLik:", float(model.loglik()))
    print("BIC:", float(model.bic()))
    
def print_from_sa_result(sa_result, evaluator):

    best_solution = sa_result["best_solution"]

    print("\n================ BEST SA STRUCTURE ================\n")
    decode_best_solution(best_solution, evaluator)

    # Optional refit like NSGA
    refit_and_print(evaluator, best_solution)
    
    
def summarize_experiment(result, config, runtime, config_id):

    summary = {}

    summary["timestamp"] = datetime.now().isoformat()
    summary["algorithm"] = result["algorithm"]
    summary["seed"] = result["seed"]
    summary["config_id"] = config_id
    summary["config"] = config
    summary["runtime_seconds"] = runtime

    # -------------------------
    # SINGLE OBJECTIVE
    # -------------------------
    if "best_score" in result:

        summary["best_score"] = float(result["best_score"])
        summary["num_solutions"] = len(result["solutions"])

    # -------------------------
    # MULTI OBJECTIVE (NSGA)
    # -------------------------
    elif "scores" in result:

        scores = np.array(result["scores"])
        if scores.size ==1:
            summary["best_score"] = result["scores"]
            summary['num_solutions'] =1 
            return summary
        summary["pareto_size"] = scores.size
        summary["best_score"] = result["scores"]
        summary["best_bic"] = float(np.min(scores[:, 0]))
        summary["best_rmse"] = float(np.min(scores[:, 1]))

        if "hypervolume_history" in result:
            summary["final_hypervolume"] = (
                float(result["hypervolume_history"][-1])
                if len(result["hypervolume_history"]) > 0 else None
            )

    return summary

def append_experiment_to_csv(summary, filename="results/experiment_log.csv"):

    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame([summary])

    if os.path.exists(filename):
        df.to_csv(filename, mode="a", header=False, index=False)
    else:
        df.to_csv(filename, index=False)

    print("✅ Logged experiment to CSV")




def run_glmulti_experiment(df_train, df_test, vars):
    import subprocess
    import tempfile
    import os

    r_exe = r"C:\Users\ahernz\AppData\Local\Programs\R\R-4.4.1\bin\x64\Rscript.exe"
    r_script = os.path.abspath("glm_multi.R")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f_train, \
         tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f_test, \
         tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f_out:

        df_train.to_csv(f_train.name, index=False)
        df_test.to_csv(f_test.name, index=False)
        vars_string = ",".join(vars)
        cmd = [
            r_exe,
            r_script,
            f_train.name,
            f_test.name,
            f_out.name,
            vars_string,
            "OFFSET"  
        ]

        print("Running:", cmd)  # optional debug

        subprocess.run(cmd, check=True)

        result = pd.read_csv(f_out.name)

    return result["train_bic"].iloc[0], result["test_rmse"].iloc[0]

def get_best_index(scores):
    if scores.ndim == 1:
        return np.argmin(scores)
    return np.argmin(scores[:, 0])

def main(seed, algo, pop_size, max_iter, n_jobs, config_id):
    start_time = time.time()
    print("Running experiment:")
    print("Algo:", algo)
    print("Config:", config_id)
    print("Seed:", seed)

    # ----------------------------
    # Generate grids
    # ----------------------------

    HS_GRID = {
        "population_size": [10, 30, 40, 20],
        "hmcr": [0.75, 0.9, 0.98, 0.6],
        "par_min": [0.05, 0.1],
        "par_max": [0.5, 0.8],
        "bw_min": [1],
        "bw_max": [2, 3, 4, 5],
    }

    DE_GRID = {
        "population_size": [20, 30, 10],
        "F": [0.3, 0.5, 0.8],
        "CR": [0.5, 0.7, 0.9],
    }

    SA_GRID = {
        "max_iter": [max_iter],
        "mutation_rate": [0.1, 0.3, 0.6],
        "step_size": [1, 2, 3],
        "alpha": [0.99, 0.99, 0.9],
        "n_starts": [1, 2]
    }

    HC_GRID = {
        "mutation_rate": [0.5],
        "step_size": [1],
        "min_changes": [1],
        "max_changes": [1],
        "n_starts": [1]
    }

    ALL_CONFIGS = {
    "de": generate_configs(DE_GRID),
    "hs": generate_configs(HS_GRID),
    "sa": generate_configs(SA_GRID),
    "hc": generate_configs(HC_GRID)
    }

    
    # =====================================================
    # SELECT CONFIG
    # =====================================================
    if algo in ALL_CONFIGS:
        configs = ALL_CONFIGS[algo]

        if config_id >= len(configs):
            raise ValueError("config_id out of range")

        config = configs[config_id]
        print(f"\nUsing config {config_id}: {config}\n" )
    else:
        configs = None
    
    # ----------------------------
    # Build evaluator
    # ----------------------------

    df = pd.read_csv('./data/Ex-16-3.csv')
    df.rename(columns={"FREQ": "Y"}, inplace=True)
    all_variables = [
    "INCLANES",
    "DECLANES",
    "WIDTH",
    "SPEED",
    "URB",
    "FC",
    "SINGLE",
    "DOUBLE",
    "TRAIN",
    "PEAKHR",
    "GRADEBR",
    "TANGENT",
    "CURVES",
    "MINRAD",
    "ACCESS",
    "MEDWIDTH",
    "SLOPE",
    "INTECHAG",
    "AVEPRE",
    "AVESNOW",
    "LOWPRE",
    "GBRPM",
    "EXPOSE",
    "INTPM",
    "CPM",
    "HISNOW",
    "FRICTION"
    ]
    
    
    
    

    allowed_distributions = populate_allowed_distributions(
        all_variables,
        {"CURVES": ["normal", "lognormal"],
         "FRICTION": ["normal", "triangular"]}
    )

    allowed_roles = populate_allowed_roles(all_variables, {})

    evaluator = StructureEvaluator(
        df=df,
        id_col="ID",
        y_col="Y",
        offset_col='OFFSET',
        all_variables=all_variables,
        allowed_roles=allowed_roles,
        allowed_distributions=allowed_distributions,
        group_id_col="FC",
        mode="single",
        R=200
    )

    # ----------------------------
    # Select config
    # ----------------------------

    # =====================================================
    # ROUTING
    # =====================================================

    # ---------------------------
    # MULTI-OBJECTIVE (NSGA2)
    # ---------------------------
    if algo in ["de", "hs"]:

        if algo == "de":
            op_config = {k: v for k, v in config.items() if k != "population_size"}
            operator = AdaptiveDE(**op_config)

        elif algo == "hs":
            op_config = {k: v for k, v in config.items() if k != "population_size"}
            operator = DynamicHarmony(**op_config)

        result = run_nsga(
            evaluator=evaluator,
            operator=operator,
            seed=seed,
            pop_size=config['population_size'],
            max_iter=max_iter,
            n_jobs=n_jobs
        )

        print_from_nsga_result(result, evaluator)
        # pick best solution
        
        
        #return result

    # ---------------------------
    # SINGLE-OBJECTIVE (SA / HC)
    # ---------------------------
    elif algo in ["sa", 'hc']:

        solver = MultiStartSA(
            evaluator=evaluator,
            dimension=2 * len(evaluator.vars)+1,
            **config
        )
      
        all_solutions, all_scores= solver.optimize()
        best_idx = np.argmin(all_scores)
        best_solution = all_solutions[best_idx]
        best_score = all_scores[best_idx]

        print("\n================ BEST SA STRUCTURE ================\n")
        decode_best_solution(best_solution, evaluator)
        print("Best Score:", best_score)
        refit_and_print(evaluator, best_solution)

        save_run_summary_to_txt(
            evaluator=evaluator,
            decision=best_solution,
            algo=algo,
            seed=seed,
            config_id=config_id
        )

        runtime = time.time() - start_time

        summary = summarize_experiment(
            result={
                "solutions": all_solutions,
                "scores": all_scores,
                "algorithm": algo,
                'seed':seed
            },
            config=config,
            runtime=runtime,
            config_id=config_id
        )

        append_experiment_to_csv(summary)

        return {
            "algorithm": "SA",
            "seed": seed,
            "solutions": all_solutions,
            "scores": all_scores,
            "best_solution": best_solution,
            "best_score": best_score,
            "convergence": solver.results}
    elif algo == 'glmulti':
        df_train, df_test, df_val = build_datasets(df, "ID", "Y")

        train_bic, test_rmse = run_glmulti_experiment(df_train, df_test, all_variables)

        runtime = time.time() - start_time

        summary = {
            "algorithm": "glmulti",
            "seed": seed,
            "train_bic": float(train_bic),
            "test_rmse": float(test_rmse),
            "runtime_seconds": runtime
        }

        append_experiment_to_csv(summary)

        print("\n================ GLMULTI RESULTS ================\n")
        print("Train BIC:", train_bic)
        print("Test RMSE:", test_rmse)

        return summary
        

    # ----------------------------
    # Run NSGA
    # ----------------------------
    runtime = time.time() - start_time
    result = run_nsga(
        evaluator=evaluator,
        operator=operator,
        seed=seed,
        pop_size=pop_size,
        max_iter=max_iter,
        n_jobs=n_jobs
    )

    print_from_nsga_result(result, evaluator)
    runtime = time.time() - start_time
    
    summary = summarize_experiment(
    result=result,
    config=config,
    runtime=runtime,
    config_id=config_id
    )

    append_experiment_to_csv(summary)

    save_experiment(result, config)
    
    return result




def save_run_summary_to_txt(evaluator, decision, algo, seed, config_id, folder="results"):

    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{algo}_seed{seed}_config{config_id}_{timestamp}.txt"
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        with redirect_stdout(f):

            print("======================================================")
            print("EXPERIMENT SUMMARY")
            print("======================================================")
            print("Algorithm:", algo)
            print("Seed:", seed)
            print("Config ID:", config_id)
            print("Timestamp:", timestamp)
            print("======================================================\n")

            # Print structure
            decode_best_solution(decision, evaluator)

            # Refit and print full model summary
            refit_and_print(evaluator, decision)

    print(f"✅ Summary saved to {filepath}")

def save_experiment(result, config, folder="results"):

    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    algo = result["algorithm"]
    seed = result["seed"]

    base_name = f"{algo}_seed{seed}_{timestamp}"

    # ✅ Save full result (pickle)
    with open(os.path.join(folder, base_name + ".pkl"), "wb") as f:
        pickle.dump(result, f)

    # ✅ Save config (json)
    with open(os.path.join(folder, base_name + "_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n✅ Saved experiment to {folder}/{base_name}")


def log_nsga_result(result, filename):

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "algorithm": result["algorithm"],
        "seed": result["seed"],
        "num_solutions": len(result["solutions"]),
        "scores": np.array(result["scores"]).tolist(),
        "solutions": np.array(result["solutions"]).tolist()
    }

    with open(filename, "w") as f:
        json.dump(log_data, f, indent=4)

    print(f"\n✅ NSGA results logged to {filename}")

if __name__ == "__main__":
    #
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config_id", type=int, default=0)
    parser.add_argument("--algo", type=str, default="hs") #glmulti
    parser.add_argument("--pop_size", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--n_jobs", type=int, default=1)

    args = parser.parse_args()

    main(args.seed, args.algo, args.pop_size, args.max_iter,args.n_jobs, args.config_id)
    
    
    
    
    '''
    
    D = len(all_variables) * 2  # roles + dists

    engine = NSGA2Engine(
    evaluator=evaluator,
    operator=DynamicHarmony(),
    dimension=D,
    pop_size=30,
    max_iter=40,
    n_jobs=1
)

    solutions, scores = engine.optimize()
    engine.plot_convergence()
    engine.plot_pareto_front()
    engine.plot_pareto_evolution()

    engine.save_history("experiment1.pkl")
    #best_solution, best_score = hs.optimize()
    
    '''
    
   











