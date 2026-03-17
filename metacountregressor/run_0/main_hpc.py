import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pandas as pd
import os
import itertools
from functools import partial
from typing import NamedTuple
from jaxopt import LBFGS
import pandas as pd
import scipy.stats as stats
from Solvers_METAJAX import *
from scipy.stats import qmc
from scipy.stats import norm
import argparse
import psutil

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

def build_spec_from_decision(
    df,
    decision_roles,
    decision_dists,
    all_variables,
    allowed_roles,
    allowed_distributions
):

    fixed_cols = []
    random_ind_cols = []
    random_cor_cols = []
    grouped_cols = []
    hetero_cols = []

    rdm_terms = []
    rdm_cor_terms = []
    grouped_terms = []

    for i, var in enumerate(all_variables):

        role = decision_roles[i]

        # Enforce allowed roles
        if role not in allowed_roles.get(var, [0]):
            return None  # invalid solution

        if role == 0:
            continue

        elif role == 1:
            fixed_cols.append(var)


        elif role == 2:

            random_ind_cols.append(var)

            allowed = allowed_distributions.get(var)

            if allowed is None or len(allowed) == 0:
                dist_name = "normal"
            else:
                dist_name = allowed[decision_dists[i] % len(allowed)]

            rdm_terms.append(f"{var}:{dist_name}")


        elif role == 3:

            random_cor_cols.append(var)

            allowed = allowed_distributions.get(var)

            if allowed is None or len(allowed) == 0:
                dist_name = "normal"
            else:
                dist_name = allowed[decision_dists[i] % len(allowed)]

            rdm_cor_terms.append(f"{var}:{dist_name}")


        elif role == 4:

            grouped_cols.append(var)

            allowed = allowed_distributions.get(var)

            if allowed is None or len(allowed) == 0:
                dist_name = "normal"
            else:
                dist_name = allowed[decision_dists[i] % len(allowed)]

            grouped_terms.append(f"{var}:{dist_name}")

    # Enforce logical constraints

    if len(random_cor_cols) == 1:
        return None  # correlated requires ≥ 2

    if len(random_ind_cols) + len(random_cor_cols) == 0:
        hetero_cols = []  # no heterogeneity if no random

    manual_spec = {
        'fixed_terms': fixed_cols,
        'rdm_terms': rdm_terms,
        'rdm_cor_terms': rdm_cor_terms,
        'grouped_terms': grouped_terms,
        'hetro_in_means': [f"{z}:normal" for z in hetero_cols],
        'dispersion': 1
    }

    return manual_spec





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


class ModelSolution:

    def __init__(self, result, spec, param_index):

        self.params = result.params
        self.spec = spec
        self.param_index = param_index

        # These will be filled later
        self.train_metrics = {}
        self.test_metrics = {}
        self.validation_metrics = {}

        self.train_ll = None
        self.test_ll = None
        self.validation_ll = None

        self.aic = None
        self.bic = None

    # -------------------------------------
    # Evaluate on a dataset
    # -------------------------------------

    def evaluate(self, data, name="train"):

        objective = partial(mixed_model_loglik, data=data, spec=self.spec)
        ll = -objective(self.params)

        metrics = evaluate_metrics(self.params, data, self.spec, name.upper())

        if name == "train":
            self.train_ll = float(ll)
            self.train_metrics = metrics

            k = len(self.params)
            n = data["y"].shape[0]

            self.aic = float(2*k - 2*ll)
            self.bic = float(k * np.log(n) - 2*ll)

        elif name == "test":
            self.test_ll = float(ll)
            self.test_metrics = metrics

        elif name == "validation":
            self.validation_ll = float(ll)
            self.validation_metrics = metrics

    # -------------------------------------
    # Print summary
    # -------------------------------------

    def print_performance(self):

        print("\n================ MODEL PERFORMANCE ================")

        print("\n--- TRAIN ---")
        print(f"LogLik: {self.train_ll:.4f}")
        print(self.train_metrics)

        print("\n--- TEST ---")
        print(f"LogLik: {self.test_ll:.4f}")
        print(self.test_metrics)

        print("\n--- VALIDATION ---")
        print(f"LogLik: {self.validation_ll:.4f}")
        print(self.validation_metrics)

        print("\nAIC:", self.aic)
        print("BIC:", self.bic)

        print("===================================================")


# =========================================================
# MODEL SPEC (HASHABLE FOR JAX)
# =========================================================

class ModelSpec(NamedTuple):
    Kf: int
    Kr_ind: int
    Kr_cor: int
    Kg: int
    Kh: int
    Kzi: int 
    model: str
    zero_inflated: bool  # ✅ NEW
    fixed_names: tuple
    zi_names: tuple
    random_ind_names: tuple
    random_cor_names: tuple
    grouped_names: tuple
    hetro_names: tuple
    random_ind_dists: tuple
    random_cor_dists: tuple
    grouped_dists: tuple
    @property
    def K_random_total(self):
        return self.Kr_cor + self.Kr_ind


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

def build_eta(params, data, spec: ModelSpec):

    blocks = unpack_params(params, spec)

    # Start with zero eta (N,P,1) inferred later
    eta = 0.0

    # =====================================================
    # FIXED EFFECTS
    # =====================================================
    if spec.Kf > 0:
        eta = jnp.einsum("npk,k->np", data["Xf"], blocks["beta_f"])[..., None]

    # =====================================================
    # HETEROGENEITY SHIFT
    # =====================================================
    shift = None
    if spec.Kh > 0 and spec.K_random_total > 0:

        Z = data["Xh"]  # (N,P,Kh)

        shift_full = jnp.einsum(
            "npk,km->npm",
            Z,
            blocks["gamma"]
        )

        # average across panel dimension
        shift = jnp.mean(shift_full, axis=1)  # (N,K_random_total)

    # =====================================================
    # CORRELATED RANDOM EFFECTS
    # =====================================================
    if spec.Kr_cor > 0:

        mean_cor = blocks["mean_cor"]

        if shift is not None:
            mean_cor = mean_cor[None, :] + shift[:, :spec.Kr_cor]

        beta_cor = random_correlated(
            mean_cor,
            blocks["chol"],
            data["draws_cor"],
            spec.Kr_cor,
            jnp.array([DIST_MAP[d] for d in spec.random_cor_dists])
        )

        eta += jnp.einsum("npk,nkr->npr", data["Xr_cor"], beta_cor)

    # =====================================================
    # INDEPENDENT RANDOM EFFECTS
    # =====================================================
    if spec.Kr_ind > 0:

        mean_ind = blocks["mean_ind"]

        if shift is not None:
            mean_ind = mean_ind[None, :] + shift[:, spec.Kr_cor:]

        beta_ind = random_independent(
            mean_ind,
            blocks["sd_ind"],
            data["draws_ind"],
            jnp.array([DIST_MAP[d] for d in spec.random_ind_dists])
        )

        eta += jnp.einsum("npk,nkr->npr", data["Xr_ind"], beta_ind)

    # =====================================================
    # GROUPED RANDOM EFFECTS
    # =====================================================
    if spec.Kg > 0:

        beta_g_all = transform_draws(
            data["draws_g"],
            blocks["mean_g"],
            blocks["sd_g"],
            jnp.array([DIST_MAP[d] for d in spec.grouped_dists])
        )

        beta_g = beta_g_all[data["group_ids"]]

        eta += jnp.einsum("npk,nkr->npr", data["Xg"], beta_g)

    # =====================================================
    # OFFSET
    # =====================================================
    offset = data["offset"]

    if offset.ndim == 2:
        offset = offset[..., None]

    eta += offset

    return eta

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

@partial(jax.jit, static_argnames=("spec",))
def mixed_model_lofglik(params, data, spec: ModelSpec):

    blocks = unpack_params(params, spec)

    eta = build_eta(params, data, spec)
    mu = jnp.exp(jnp.clip(eta, -15, 15))
    #eta = jnp.clip(eta, -20, 20)

    if mu.ndim == 2:
        mu = mu[..., None]

    y = ensure_3d(data["y"])
    mask = ensure_3d(data["mask"])

    R = mu.shape[-1]

    if spec.model == "poisson":
        ll = poisson_loglik(y, mu)

    elif spec.model == "nb":
        alpha = blocks["alpha"]
        ll = nb_loglik(y, mu, alpha)

    else:
        raise ValueError("Unknown model")

    ll = ll * mask
    ll_panel = jnp.sum(ll, axis=1)

    if R > 1:
        ll_ind = jsp.special.logsumexp(ll_panel, axis=-1) - jnp.log(R)
    else:
        ll_ind = ll_panel.squeeze(-1)

    return -jnp.sum(ll_ind)



@partial(jax.jit, static_argnames=("spec",))
def mixed_model_loglik(params, data, spec: ModelSpec):

    blocks = unpack_params(params, spec)

    eta = build_eta(params, data, spec)

    # Clip for numerical stability
    eta = jnp.clip(eta, -15, 15)

    if eta.ndim == 2:
        eta = eta[..., None]

    y = ensure_3d(data["y"])
    mask = ensure_3d(data["mask"])

    R = eta.shape[-1]

    # =========================
    # Likelihood
    # =========================
    # =========================
    # BASE COUNT LIKELIHOOD
    # =========================
    if spec.model == "poisson":
        mu = jnp.exp(eta)
        ll_count = poisson_loglik(y, mu)

    elif spec.model == "nb":
        alpha = blocks["alpha"]
        ll_count = nb2_loglik(y, eta, alpha)

    else:
        raise ValueError("Unknown model")

    # =========================
    # APPLY ZERO INFLATION
    # =========================
    if spec.zero_inflated:

        if spec.Kzi > 0:
            eta_zi = jnp.einsum(
                    "npk,k->np",
                    data["Xzi"],
                    blocks["beta_zi"]
            )[..., None]   # make (N,P,1)
        else:
            # intercept-only inflation
            eta_zi = jnp.zeros_like(eta[..., :1])

        pi = jax.nn.sigmoid(eta_zi)
        
        mu = jnp.exp(eta)

        # Probability of zero from count model
        if spec.model == "poisson":
            f0 = jnp.exp(-mu)
        else:
            # NB zero probability
            alpha_exp = jnp.exp(blocks["alpha"])
            inv_alpha = 1.0 / alpha_exp
            log_f0 = inv_alpha * (
                jnp.log(inv_alpha) - jnp.log(inv_alpha + mu)
            )
            f0 = jnp.exp(log_f0)
            #f0 = (inv_alpha / (inv_alpha + mu)) ** inv_alpha

        zero_mask = (y == 0)

        ll_zero = jnp.log(pi + (1 - pi) * f0 + 1e-12)
        ll_pos = jnp.log(1 - pi + 1e-12) + ll_count

        ll = jnp.where(zero_mask, ll_zero, ll_pos)

    else:
        ll = ll_count

    ll = ll * mask
    ll_panel = jnp.sum(ll, axis=1)

    # ✅ Log-sum-exp simulation averaging
    if R > 1:
        ll_ind = jsp.special.logsumexp(ll_panel, axis=-1) - jnp.log(R)
    else:
        ll_ind = ll_panel.squeeze(-1)

    return -jnp.sum(ll_ind)


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

def parse_manual_spec(manual_spec):

    fixed_cols = manual_spec.get("fixed_terms", [])

    rdm_terms = manual_spec.get("rdm_terms", [])
    random_ind = [term.split(":")[0] for term in rdm_terms]

    rdm_cor_terms = manual_spec.get("rdm_cor_terms", [])
    random_cor = [term.split(":")[0] for term in rdm_cor_terms]
    

    grouped_terms = manual_spec.get("grouped_terms", [])
    grouped_cols = [term.split(":")[0] for term in grouped_terms]
    hetro_terms = manual_spec.get("hetro_in_means", [])
    hetro_cols = [term.split(":")[0].strip() for term in hetro_terms]
    random_ind_dists = [term.split(":")[1] for term in rdm_terms]
    random_cor_dists = [term.split(":")[1] for term in rdm_cor_terms]
    grouped_dists = [term.split(":")[1] for term in grouped_terms]
    zi_cols = manual_spec.get("zi_terms", [])

    return fixed_cols, random_ind, random_cor, grouped_cols, hetro_cols, random_ind_dists, random_cor_dists, grouped_dists, zi_cols


def build_model_from_manual_spec(
    df,
    manual_spec,
    id_col,
    y_col,
    offset_col=None,
    draws_ind=None,
    draws_cor=None,
    draws_g=None,
    R=200
):

    (fixed_cols, random_ind, random_cor, grouped_cols, hetro_cols, random_ind_dists, random_cor_dists,
    grouped_dists, zi_cols) = parse_manual_spec(manual_spec)

    data, spec = build_jax_data(
        df=df,
        id_col=id_col,
        y_col=y_col,
        group_id_col=manual_spec.get("group_id_col", None),
        fixed_cols=fixed_cols,
        random_ind_cols=random_ind,
        random_cor_cols=random_cor,
        grouped_cols=grouped_cols,
        hetro_cols=hetro_cols,
        zi_cols=zi_cols,
        offset_col=offset_col,
        draws_ind=draws_ind,
        draws_cor=draws_cor,
        draws_g=draws_g,
        random_ind_dists=random_ind_dists,
        random_cor_dists=random_cor_dists,
        grouped_dists=grouped_dists,
        R=R
    )

    model_type = "nb" if manual_spec.get("dispersion", 0) else "poisson"
    spec = spec._replace(model=model_type)

    return data, spec


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

# =========================================================
# BUILD JAX DATA
# =========================================================
def build_jax_data(
    df,
    id_col,
    y_col,
    group_id_col=None,
    fixed_cols=None,
    random_ind_cols=None,
    random_cor_cols=None,
    grouped_cols=None,
    hetro_cols=None,
    offset_col=None,
    draws_ind=None,
    draws_cor=None,
    draws_g=None,
    random_ind_dists=None,
    random_cor_dists=None,
    grouped_dists=None,
    zi_cols = None,
    R=200
):

    # -----------------------------
    # Safe defaults
    # -----------------------------
    fixed_cols = fixed_cols or []
    random_ind_cols = random_ind_cols or []
    random_cor_cols = random_cor_cols or []
    grouped_cols = grouped_cols or []
    hetro_cols = hetro_cols or []
    zi_col = zi_cols or  []

    random_ind_dists = random_ind_dists or []
    random_cor_dists = random_cor_dists or []
    grouped_dists = grouped_dists or []

    # -----------------------------
    # Build balanced panel
    # -----------------------------
    all_features = list(set(
        fixed_cols + random_ind_cols + random_cor_cols + grouped_cols + hetro_cols
    ))

    X_all, y, mask = balance_panel_dataframe(
        df, id_col, y_col, all_features
    )

    # -----------------------------
    # Group IDs
    # -----------------------------
    if group_id_col is not None and len(grouped_cols) > 0:

        df_sorted = df.sort_values(id_col)
        group_codes = df_sorted[group_id_col].astype("category").cat.codes.values
        unique_groups = np.unique(group_codes)
        G = len(unique_groups)

    else:
        group_codes = None
        G = 0

    # -----------------------------
    # Extract feature blocks
    # -----------------------------
    col_map = {col: i for i, col in enumerate(all_features)}

    def extract(cols):
        if len(cols) == 0:
            return np.zeros((X_all.shape[0], X_all.shape[1], 0))
        idx = [col_map[c] for c in cols if c in col_map]
        return X_all[:, :, idx]

    Xf = extract(fixed_cols)
    Xr_ind = extract(random_ind_cols)
    Xr_cor = extract(random_cor_cols)
    Xg = extract(grouped_cols)
    Xh = extract(hetro_cols)
    Xzi = extract(zi_cols)

    N, P = y.shape[0], y.shape[1]

    # -----------------------------
    # Offset
    # -----------------------------
    if offset_col:
        offset = extract_offset(df, id_col, offset_col)
    else:
        offset = np.zeros((N, P, 1))
    N = X_all.shape[0]
    # -----------------------------
    # Build data dict
    # -----------------------------
    data = {
        "Xf": jnp.array(Xf),
        "Xr_ind": jnp.array(Xr_ind),
        "Xr_cor": jnp.array(Xr_cor),
        "Xg": jnp.array(Xg),
        "Xh": jnp.array(Xh),
        "Xzi": jnp.array(Xzi),
        "y": jnp.array(y),
        "mask": jnp.array(mask),
        "offset": jnp.array(offset),
        "draws_ind": jnp.zeros((N, 0, R)) if draws_ind is None else jnp.array(draws_ind),
        "draws_cor": jnp.zeros((N, 0, R)) if draws_cor is None else jnp.array(draws_cor),
        "draws_g": jnp.zeros((N, 0, R)) if draws_g is None else jnp.array(draws_g),

        "group_ids": jnp.array(group_codes) if group_codes is not None else jnp.zeros(N, dtype=int),
    }

    # -----------------------------
    # Build ModelSpec (FULLY WIRED)
    # -----------------------------
    spec = ModelSpec(
        Kf=Xf.shape[2],
        Kr_ind=Xr_ind.shape[2],
        Kr_cor=Xr_cor.shape[2],
        Kg=Xg.shape[2],
        Kh=Xh.shape[2],
        zi_names=tuple(zi_cols),
        Kzi=Xzi.shape[2],
        zero_inflated=(Xzi.shape[2] > 0),
        model="poisson",  # overwritten later if NB
        fixed_names=tuple(fixed_cols),
        random_ind_names=tuple(random_ind_cols),
        random_cor_names=tuple(random_cor_cols),
        grouped_names=tuple(grouped_cols),
        hetro_names=tuple(hetro_cols),
        random_ind_dists=tuple(random_ind_dists),
        random_cor_dists=tuple(random_cor_dists),
        grouped_dists=tuple(grouped_dists),
    )

    return data, spec




#=======================================
# PARAM INDEX
# =========================================================

def build_param_index(spec: ModelSpec):

    idx = 0
    index = {}

    index["fixed"] = (idx, idx + spec.Kf)
    idx += spec.Kf
    
    #Zero Inflate
    if spec.Kzi > 0:
        index["zi_beta"] = (idx, idx + spec.Kzi)
        idx += spec.Kzi

    # =====================
    # CORRELATED RANDOM
    # =====================
    if spec.Kr_cor > 0:
        index["cor_mean"] = (idx, idx + spec.Kr_cor)
        idx += spec.Kr_cor

        Kchol = spec.Kr_cor * (spec.Kr_cor + 1) // 2
        index["chol"] = (idx, idx + Kchol)
        idx += Kchol

    # =====================
    # INDEPENDENT RANDOM
    # =====================
    if spec.Kr_ind > 0:
        index["ind_mean"] = (idx, idx + spec.Kr_ind)
        idx += spec.Kr_ind

        index["ind_sd"] = (idx, idx + spec.Kr_ind)
        idx += spec.Kr_ind

    if spec.Kg > 0:
        index["group_mean"] = (idx, idx + spec.Kg)
        idx += spec.Kg

        index["group_sd"] = (idx, idx + spec.Kg)
        idx += spec.Kg

    if spec.Kh > 0 and spec.K_random_total > 0:
        Khet = spec.Kh * spec.K_random_total
        index["hetro"] = (idx, idx + Khet)
        idx += Khet

    if spec.model == "nb":
        index["dispersion"] = idx
        idx += 1

    index["total_params"] = idx

    return index

def compute_predictions(params, data, spec):
        eta, _ = build_eta(params, data, spec)
        if eta.ndim == 2:
            eta = eta[..., None]
        mu = jnp.exp(eta)
        return mu

    
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


def evaluate_predictions(params, data, spec, name):

    mu = compute_predictions(params, data, spec)
    mu_mean = np.array(mu.mean(axis=-1))  # average over draws

    y_true = np.array(data["y"]).squeeze()

    rmse = np.sqrt(np.mean((mu_mean - y_true) ** 2))
    mae = np.mean(np.abs(mu_mean - y_true))

    print(f"\n{name} RMSE: {rmse:.4f}")
    print(f"{name} MAE: {mae:.4f}")

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

def print_summary(result, objective, data, spec, param_index):

    import numpy as np
    import pandas as pd
    from scipy import stats

    params = result.params
    se = compute_standard_errors(params, objective)

    z_vals = params / se
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_vals)))

    # Log-likelihood
    final_ll = -objective(params)
    k = len(params)
    n = data["y"].shape[0]

    aic = 2*k - 2*final_ll
    bic = k * np.log(n) - 2*final_ll

    names = []

    # ==============================
    # BUILD PARAMETER NAME LIST
    # ==============================

    # Fixed
    for name in spec.fixed_names:
        names.append(name)

    # Zero Inflation
    if spec.Kzi > 0:
        for name in spec.zi_names:
            names.append(f"zi({name})")
    
    # Correlated means
    for name in spec.random_cor_names:
        names.append(f"cor_mean({name})")

    # Correlated Cholesky (store but we won't print later)
    chol_names = []
    if spec.Kr_cor > 0:
        corr_cols = spec.random_cor_names
        K = spec.Kr_cor
        for i in range(K):
            for j in range(i + 1):
                nm = f"chol({corr_cols[i]},{corr_cols[j]})"
                names.append(nm)
                chol_names.append(nm)

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

    # Dispersion
    if spec.model == "nb":
        names.append("dispersion")

    summary_df = pd.DataFrame({
        "Parameter": names,
        "Estimate": np.array(params),
        "Std.Err": np.array(se),
        "z-value": np.array(z_vals),
        "p-value": np.array(p_vals)
    })

    # ==========================================================
    # PRINT CLEAN SECTIONS
    # ==========================================================

    def print_section(df, title):
        if len(df) == 0:
            return
        print(f"\n================ {title} ================\n")
        print(df.to_string(index=False))

    # Remove raw chol from main table
    main_df = summary_df[~summary_df["Parameter"].isin(chol_names)]

    # Split sections
    fixed_df = main_df[
        main_df["Parameter"].isin(spec.fixed_names)
    ]
    
    zi_df = main_df[
    main_df["Parameter"].str.contains("zi\\(")
]

    cor_mean_df = main_df[
        main_df["Parameter"].str.contains("cor_mean")
    ]

    ind_df = main_df[
        main_df["Parameter"].str.contains("mean\\(") &
        ~main_df["Parameter"].str.contains("cor_mean") &
        ~main_df["Parameter"].str.contains("group")
    ]

    sd_df = main_df[
        main_df["Parameter"].str.contains("sd\\(") &
        ~main_df["Parameter"].str.contains("group")
    ]

    group_df = main_df[
        main_df["Parameter"].str.contains("group_")
    ]

    het_df = main_df[
        main_df["Parameter"].str.contains("hetro")
    ]

    disp_df = main_df[
        main_df["Parameter"] == "dispersion"
    ]

    print("\n================ MODEL SUMMARY ================")

    print_section(fixed_df, "FIXED EFFECTS")
    print_section(zi_df, "ZERO INFLATION")
    print_section(cor_mean_df, "CORRELATED RANDOM MEANS")
    print_section(ind_df, "INDEPENDENT RANDOM MEANS")
    print_section(sd_df, "INDEPENDENT RANDOM SDs")
    print_section(group_df, "GROUPED RANDOM EFFECTS")
    print_section(het_df, "HETEROGENEITY IN MEANS")
    print_section(disp_df, "DISPERSION")

    # ==========================================================
    # COMPUTE & PRINT VAR-COV MATRIX FROM CHOLESKY
    # ==========================================================

    if spec.Kr_cor > 0:

        # Extract chol parameters
        chol_params = summary_df[
            summary_df["Parameter"].isin(chol_names)
        ]["Estimate"].values

        # Rebuild L
        K = spec.Kr_cor
        L = np.zeros((K, K))
        idx = 0
        for i in range(K):
            for j in range(i + 1):
                L[i, j] = chol_params[idx]
                idx += 1

        Sigma = L @ L.T
        std = np.sqrt(np.diag(Sigma))
        Corr = Sigma / np.outer(std, std)

        cols = spec.random_cor_names

        print("\n================ CORRELATED RANDOM VAR-COV ================\n")

        for i in range(K):
            print(f"Var({cols[i]}) = {Sigma[i,i]:.6f}")

        for i in range(K):
            for j in range(i):
                print(f"Cov({cols[i]},{cols[j]}) = {Sigma[i,j]:.6f}")

        print("\n================ CORRELATION MATRIX ================\n")

        for i in range(K):
            for j in range(i):
                print(f"Corr({cols[i]},{cols[j]}) = {Corr[i,j]:.6f}")

    print("\n------------------------------------------------")
    print(f"Log-Likelihood: {float(final_ll):.4f}")
    print(f"AIC: {float(aic):.4f}")
    print(f"BIC: {float(bic):.4f}")
    print("================================================\n")



def compute_mean_prediction(params, data, spec):

    eta, _ = build_eta(params, data, spec)

    if eta.ndim == 2:
        eta = eta[..., None]

    mu = jnp.exp(eta)  # (N,P,R)

    # Average over simulation draws
    mu_mean = jnp.mean(mu, axis=-1)  # (N,P)

    return mu_mean


def evaluate_metrics(params, data, spec, name="DATA"):

    import numpy as np

    mu_hat = compute_mean_prediction(params, data, spec)

    y_true = np.array(data["y"])
    mask = np.array(data["mask"])

    mu_hat = np.array(mu_hat)

    # Ensure correct dimensions
    if y_true.ndim == 3:
        y_true = y_true.squeeze(-1)

    # Flatten everything
    y_true_flat = y_true.reshape(-1)
    mu_hat_flat = mu_hat.reshape(-1)
    mask_flat = mask.reshape(-1)

    # Keep only valid panel entries
    valid = mask_flat == 1

    y_true_flat = y_true_flat[valid]
    mu_hat_flat = mu_hat_flat[valid]

    # ===============================
    # METRICS
    # ===============================

    mse = np.mean((y_true_flat - mu_hat_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_flat - mu_hat_flat))

    # Avoid division by zero for MAPE
    nonzero = y_true_flat != 0
    mape = (
        np.mean(
            np.abs((y_true_flat[nonzero] - mu_hat_flat[nonzero])
                   / y_true_flat[nonzero])
        ) * 100
        if np.any(nonzero) else np.nan
    )

    # Poisson deviance
    poisson_dev = 2 * np.mean(
        y_true_flat * np.log((y_true_flat + 1e-12) / (mu_hat_flat + 1e-12))
        - (y_true_flat - mu_hat_flat)
    )

    print(f"\n================ {name} METRICS ================")
    print(f"MSE:   {mse:.6f}")
    print(f"RMSE:  {rmse:.6f}")
    print(f"MAE:   {mae:.6f}")
    print(f"MAPE:  {mape:.6f}")
    print(f"Poisson Deviance: {poisson_dev:.6f}")
    print("===============================================")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "poisson_deviance": poisson_dev
    }


def generate_master_halton(N, K, R, seed=42):
    sampler = qmc.Halton(d=K, scramble=True, seed=seed)
    u = sampler.random(R)
    z = norm.ppf(u)
    z = z.T
    z = np.tile(z[None, :, :], (N, 1, 1))
    return jnp.array(z)

class CountModel:

    def __init__(self, spec, data):
        self.spec = spec
        self.data = data
        self.param_index = build_param_index(spec)
        self.params = None

    def objective(self, params):
        return mixed_model_loglik(params, self.data, self.spec)

    def fit(self):
        init = jnp.zeros(self.param_index["total_params"])
        solver = LBFGS(fun=self.objective)
        result = solver.run(init)
        self.params = result.params
        return result

    def loglik(self):
        return -self.objective(self.params)

    def bic(self):
        n = self.data["y"].shape[0]
        k = len(self.params)
        return -2*self.loglik() + k*np.log(n)

    def predict(self):
        eta = build_eta(self.params, self.data, self.spec)
        mu = jnp.exp(eta)
        return mu.mean(axis=-1)






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
            self.df_train, self.df_test, self.df_val = build_datasets(df, id_col, y_col)
        else:
            self.df_train = df
            self.df_test = None

        # ✅ MASTER HALTON (max N)
        self.N_train = self.df_train[id_col].nunique()
        self.master_halton_train = generate_master_halton(
            self.N_train,
            len(all_variables),
            R
        )

        if mode == "multi":
            self.N_test = self.df_test[id_col].nunique()
            self.master_halton_test = generate_master_halton(
                self.N_test,
                len(all_variables),
                R,
                seed=123
            )

        # ✅ simple cache
        self.cache = {}

    # ----------------------------------------
    # Build Spec
    # ----------------------------------------

    def build_spec(self, decision):

        D = len(self.vars)
        roles = decision[:D]
        dists = decision[D:]
        dispersion_bit = decision[-1]
        use_nb = dispersion_bit % 2 == 1
        fixed, rdm_ind, rdm_cor, grouped = [], [], [], []
        hetero = []
        zi = []
        for i, var in enumerate(self.vars):

            role = roles[i]
            if role not in self.allowed_roles.get(var, [0]):
                return None

            if role == 1:
                fixed.append(var)
            elif role == 2:
                dist = decode_distribution(
                    dists[i],
                    self.allowed_distributions.get(var, ["normal"])
                )
                rdm_ind.append(f"{var}:{dist}")
            elif role == 3:
                dist = decode_distribution(
                    dists[i],
                    self.allowed_distributions.get(var, ["normal"])
                )
                rdm_cor.append(f"{var}:{dist}")
            elif role == 4:
                dist = decode_distribution(
                    dists[i],
                    self.allowed_distributions.get(var, ["normal"])
                )
                grouped.append(f"{var}:{dist}")
            elif role == 5:
                # Heterogeneity in means (no distribution)
                hetero.append(var)

            elif role == 6:
                # zero inflation variable
                zi.append(var)

        if len(rdm_cor) == 1:
            rdm_ind.extend(rdm_cor)
            rdm_cor = []

        return {
            "fixed_terms": fixed,
            "rdm_terms": rdm_ind,
            "rdm_cor_terms": rdm_cor,
            "grouped_terms": grouped,
            "hetro_in_means": hetero,
            "zi_terms": zi,
            "dispersion": 1 if use_nb else 0
        }

    # ----------------------------------------
    # Build Data Fast (Slicing Halton)
    # ----------------------------------------

    def build_data(self, df, spec_dict, master_halton):

        # First build spec only (no draws yet)
        data_tmp, spec = build_model_from_manual_spec(
            df=df,
            manual_spec=spec_dict,
            id_col=self.id_col,
            y_col=self.y_col,
            offset_col=self.offset_col,
            draws_ind=None,
            draws_cor=None,
            draws_g=None,
            R=self.R
        )

        var_index = {v: i for i, v in enumerate(self.vars)}

        ind_idx = [var_index[v] for v in spec.random_ind_names]
        cor_idx = [var_index[v] for v in spec.random_cor_names]
        g_idx   = [var_index[v] for v in spec.grouped_names]

        # ✅ SAFETY CHECKS
        if spec.Kr_ind != len(ind_idx):
            raise ValueError("Mismatch between spec.Kr_ind and ind_idx length")

        if spec.Kr_cor != len(cor_idx):
            raise ValueError("Mismatch between spec.Kr_cor and cor_idx length")

        if spec.Kg != len(g_idx):
            raise ValueError("Mismatch between spec.Kg and g_idx length")

        # ✅ Slice master halton
        draws_ind = master_halton[:, ind_idx, :] if spec.Kr_ind > 0 else None
        draws_cor = master_halton[:, cor_idx, :] if spec.Kr_cor > 0 else None

        # ✅ Grouped draws need their own N (groups, not individuals)
        if spec.Kg > 0:
            if self.group_id_col is None:
                raise ValueError("Grouped effects require group_id_col")

            G = df[self.group_id_col].nunique()

            # Generate separate master halton for groups
            master_halton_g = generate_master_halton(
                G,
                len(self.vars),
                self.R,
                seed=999
            )

            draws_g = master_halton_g[:, g_idx, :]
        else:
            draws_g = None

        # ✅ Rebuild full data with correct draws
        data, spec = build_model_from_manual_spec(
            df=df,
            manual_spec=spec_dict,
            id_col=self.id_col,
            y_col=self.y_col,
            offset_col=self.offset_col,
            draws_ind=draws_ind,
            draws_cor=draws_cor,
            draws_g=draws_g,
            R=self.R
        )

        return data, spec

    # ----------------------------------------
    # Fitness
    # ----------------------------------------

    def fitness(self, decision):

        key = tuple(decision.tolist())

        if key in self.cache:
            return self.cache[key]

        spec_dict = self.build_spec(decision)
        if spec_dict is None:
            return np.array([1e12, 1e12]) if self.mode=="multi" else 1e12

        try:

            # ✅ TRAIN
            data_train, spec = self.build_data(
                self.df_train,
                spec_dict,
                self.master_halton_train
            )

            model = CountModel(spec, data_train)
            model.fit()

            bic = model.bic()

            if self.mode == "single":
                value = float(bic)
                self.cache[key] = value
                return value

            # ✅ TEST
            data_test, _ = self.build_data(
                self.df_test,
                spec_dict,
                self.master_halton_test
            )

            model_test = CountModel(spec, data_test)
            model_test.params = model.params

            preds = model_test.predict()
            y_true = np.array(data_test["y"]).squeeze()
            rmse = np.sqrt(np.mean((preds - y_true)**2))

            value = np.array([float(bic), float(rmse)])

            self.cache[key] = value
            return value

        except Exception as e:
            print("Fitness error:", e)
            return np.array([1e12, 1e12]) if self.mode=="multi" else 1e12




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

def unpack_params(params, spec: ModelSpec):

        idx = 0
        out = {}

        # FIXED
        if spec.Kf > 0:
            out["beta_f"] = params[idx:idx + spec.Kf]
            idx += spec.Kf
        else:
            out["beta_f"] = None
        # Zero INflated
        if spec.Kzi > 0:
            out["beta_zi"] = params[idx:idx + spec.Kzi]
            idx += spec.Kzi
        else:
            out["beta_zi"] = None

        # CORRELATED RANDOM
        if spec.Kr_cor > 0:
            out["mean_cor"] = params[idx:idx + spec.Kr_cor]
            idx += spec.Kr_cor

            Kchol = spec.Kr_cor * (spec.Kr_cor + 1) // 2
            out["chol"] = params[idx:idx + Kchol]
            idx += Kchol
        else:
            out["mean_cor"] = None
            out["chol"] = None

        # INDEPENDENT RANDOM
        if spec.Kr_ind > 0:
            out["mean_ind"] = params[idx:idx + spec.Kr_ind]
            idx += spec.Kr_ind

            out["sd_ind"] = params[idx:idx + spec.Kr_ind]
            idx += spec.Kr_ind
        else:
            out["mean_ind"] = None
            out["sd_ind"] = None

        # GROUPED RANDOM
        if spec.Kg > 0:
            out["mean_g"] = params[idx:idx + spec.Kg]
            idx += spec.Kg

            out["sd_g"] = params[idx:idx + spec.Kg]
            idx += spec.Kg
        else:
            out["mean_g"] = None
            out["sd_g"] = None

        # HETEROGENEITY
        if spec.Kh > 0 and spec.K_random_total > 0:
            Khet = spec.Kh * spec.K_random_total
            gamma = params[idx:idx + Khet]
            idx += Khet

            out["gamma"] = gamma.reshape(spec.Kh, spec.K_random_total)
        else:
            out["gamma"] = None

        # NB DISPERSION
        if spec.model == "nb":
            out["alpha"] = params[idx]
            idx += 1
        else:
            out["alpha"] = None

        out["final_index"] = idx
        return out
    
    
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
            5: "Heterogeneity"
        }

        role_name = role_map.get(role, "Unknown")

        dist = None
        if var in evaluator.allowed_distributions:
            allowed = evaluator.allowed_distributions[var]
            dist = allowed[dists[i] % len(allowed)]

        print(f"{var:12} → {role_name:22} | Dist: {dist}")

    print("\n======================================================\n")

def debug_model_pipeline(df, id_col, y_col, group_id_col=None, R=200):

    print("\n================ DEBUGGING MODEL PIPELINE ================\n")

    N = df[id_col].nunique()
    G = df[group_id_col].nunique() if group_id_col else 0

    # -------------------------------------------------------
    # Helper to run one spec
    # -------------------------------------------------------
    def run_spec(name, manual_spec, draws_ind=None, draws_cor=None, draws_g=None):
        print(f"\n--- Testing: {name} ---")

        try:
            data, spec = build_model_from_manual_spec(
                df=df,
                manual_spec=manual_spec,
                id_col=id_col,
                y_col=y_col,
                draws_ind=draws_ind,
                draws_cor=draws_cor,
                draws_g=draws_g,
                R=R
            )

            param_index = build_param_index(spec)
            init = jnp.zeros(param_index["total_params"])

            objective = partial(mixed_model_loglik, data=data, spec=spec)
            init = jnp.zeros(param_index["total_params"])

            val = objective(init)

            print("Objective at init:", val)

            if jnp.isnan(val):
                print("❌ Objective already NaN at initialization")
                return

            solver = LBFGS(fun=objective)
            result = solver.run(init)

            ll = -objective(result.params)

            print("✅ SUCCESS")
            print("LogLik:", float(ll))

        except Exception as e:
            print("❌ FAILED")
            print("Error:", e)

    # =======================================================
    # 1️⃣ Fixed Only
    # =======================================================
    spec_fixed = {
        "fixed_terms": ["CURVES"],
        "rdm_terms": [],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": [],
        "dispersion": 1
    }

    run_spec("Fixed Only", spec_fixed)

    # =======================================================
    # 2️⃣ Independent Random
    # =======================================================
    draws_ind = generate_halton_normal(N=N, K=1, R=R, seed=42)

    spec_ind = {
        "fixed_terms": [],
        "rdm_terms": ["CURVES:normal"],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": [],
        "dispersion": 1
    }

    run_spec("Independent Random", spec_ind, draws_ind=draws_ind)

    # =======================================================
    # 3️⃣ Correlated Random (2 vars)
    # =======================================================
    draws_cor = generate_halton_normal(N=N, K=2, R=R, seed=42)

    spec_cor = {
        "fixed_terms": [],
        "rdm_terms": [],
        "rdm_cor_terms": [
            "CURVES:normal",
            "FRICTION:normal"
        ],
        "grouped_terms": [],
        "hetro_in_means": [],
        "dispersion": 1
    }

    run_spec("Correlated Random", spec_cor, draws_cor=draws_cor)

    # =======================================================
    # 4️⃣ Grouped Random
    # =======================================================
    if group_id_col:

        draws_g = generate_halton_normal(N=G, K=1, R=R, seed=123)

        spec_group = {
            "fixed_terms": ['TRAIN'],
            "rdm_terms": [],
            "rdm_cor_terms": [],
            "grouped_terms": ["TRAIN:normal"],
            "hetro_in_means": [],
            "dispersion": 1,
            "group_id_col": group_id_col
        }

        run_spec("Grouped Random", spec_group, draws_g=draws_g)

    # =======================================================
    # 5️⃣ Heterogeneity (requires random term)
    # =======================================================
    draws_ind = generate_halton_normal(N=N, K=1, R=R, seed=42)

    spec_het = {
        "fixed_terms": [],
        "rdm_terms": ["CURVES:normal"],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": ["INCLANES:normal", 'SPEED:normal'],
        "dispersion": 1
    }

    run_spec("Random + Heterogeneity", spec_het, draws_ind=draws_ind)

    print("\n================ DEBUG COMPLETE ================\n")
    
    
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

    solutions = np.array(nsga_result["solutions"])
    scores = np.array(nsga_result["scores"])

    # ✅ For single-objective (maximizing -BIC)
    best_idx = np.argmax(scores)
    best_solution = solutions[best_idx]

    print("\n================ BEST NSGA STRUCTURE ================\n")
    decode_best_solution(best_solution, evaluator)

    # 🔁 Refit and print coefficients
    refit_and_print(evaluator, best_solution)

def report_from_result(result, data, spec):
    """
    Print full model summary from an already-fitted result.
    No refitting.
    """

    from functools import partial

    objective = partial(mixed_model_loglik, data=data, spec=spec)

    param_index = build_param_index(spec)

    print_summary(
        result=result,
        objective=objective,
        data=data,
        spec=spec,
        param_index=param_index
    )

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

    # Build spec dictionary
    spec_dict = evaluator.build_spec(decision)
    if spec_dict is None:
        print("Invalid structure.")
        return

    # Build training data
    data_train, spec = evaluator.build_data(
        evaluator.df_train,
        spec_dict,
        evaluator.master_halton_train
    )

    # Fit model
    model = CountModel(spec, data_train)
    result = model.fit()

    # Print summary
    objective = partial(mixed_model_loglik, data=data_train, spec=spec)
    param_index = build_param_index(spec)

    print_summary(
        result=result,
        objective=objective,
        data=data_train,
        spec=spec,
        param_index=param_index
    )
    
def print_from_sa_result(sa_result, evaluator):

    best_solution = sa_result["best_solution"]

    print("\n================ BEST SA STRUCTURE ================\n")
    decode_best_solution(best_solution, evaluator)

    # Optional refit like NSGA
    refit_and_print(evaluator, best_solution)

def main(seed, algo, pop_size, max_iter, n_jobs, config_id):
    
    print("Running experiment:")
    print("Algo:", algo)
    print("Config:", config_id)
    print("Seed:", seed)

    # ----------------------------
    # Generate grids
    # ----------------------------

    HS_GRID = {
        "population_size": [30, 50, 40, 20],
        "hmcr": [0.75, 0.9, 0.98, 0.6],
        "par_min": [0.05, 0.1],
        "par_max": [0.5, 0.8],
        "bw_min": [1],
        "bw_max": [2, 3, 4, 5],
    }

    DE_GRID = {
        "population_size": [20, 50, 100],
        "F": [0.3, 0.5, 0.8],
        "CR": [0.5, 0.7, 0.9],
    }

    SA_GRID = {
        "mutation_rate": [0.1, 0.3, 0.6],
        "step_size": [1, 2, 3],
        "alpha": [0.99, 0.99, 0.9],
        "n_starts": [1, 10, 20]
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

    configs = ALL_CONFIGS[algo]

    if config_id >= len(configs):
        raise ValueError("config_id out of range")

    config = configs[config_id]
    print(f"\nUsing config {config_id}: {config}\n")
    
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
    "AADT",
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
    "FRICTIO",
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
    "FRICTION",
    "MIMEDSH"
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
        offset_col='offset',
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
        return result

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
        best_idx = np.argmax(all_scores)

        best_solution = all_solutions[best_idx]
        best_score = all_scores[best_idx]

        print("\n================ BEST SA STRUCTURE ================\n")
        decode_best_solution(best_solution, evaluator)
        print("Best Score:", best_score)
        refit_and_print(evaluator, best_solution)

        return {
            "algorithm": "SA",
            "seed": seed,
            "solutions": all_solutions,
            "scores": all_scores,
            "best_solution": best_solution,
            "best_score": best_score,
            "convergence": solver.results
        }

    # ----------------------------
    # Run NSGA
    # ----------------------------

    result = run_nsga(
        evaluator=evaluator,
        operator=operator,
        seed=seed,
        pop_size=pop_size,
        max_iter=max_iter,
        n_jobs=n_jobs
    )

    print_from_nsga_result(result, evaluator)

    return result


import json
from datetime import datetime


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
    parser.add_argument("--algo", type=str, default="sa")
    parser.add_argument("--pop_size", type=int, default=30)
    parser.add_argument("--max_iter", type=int, default=40)
    parser.add_argument("--n_jobs", type=int, default=8)

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
    
   











