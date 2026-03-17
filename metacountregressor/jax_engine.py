import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pandas as pd

from functools import partial
from typing import NamedTuple
from jaxopt import LBFGS
import pandas as pd
import scipy.stats as stats
from Solvers_METAJAX import BinaryHarmonySearch
from scipy.stats import qmc

def generate_halton_normal(n_obs, n_dim, R, seed=123):

    sampler = qmc.Halton(d=n_dim, scramble=True, seed=seed)

    # generate R draws
    u = sampler.random(R)

    # convert to normal via inverse CDF
    z = norm.ppf(u)

    # expand to panel dimension
    draws = np.tile(z, (n_obs, 1, 1))

    return draws

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

            dist_name = decode_distribution(
                decision_dists[i],
                allowed_distributions.get(var, ["normal"])
            )

            rdm_terms.append(f"{var}:{dist_name}")

        elif role == 3:
            random_cor_cols.append(var)

            dist_name = decode_distribution(
                decision_dists[i],
                allowed_distributions.get(var, ["normal"])
            )

            rdm_cor_terms.append(f"{var}:{dist_name}")

        elif role == 4:
            grouped_cols.append(var)

            dist_name = decode_distribution(
                decision_dists[i],
                allowed_distributions.get(var, ["normal"])
            )

            grouped_terms.append(f"{var}:{dist_name}")

        elif role == 5:
            hetero_cols.append(var)

        else:
            return None

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
    model: str
    fixed_names: tuple
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


def random_correlated(mean, chol_params, draws, K):
    """
    mean: (N, K)
    chol_params: flat lower triangle
    draws: (N, K, R)
    """

    L = build_cholesky(chol_params, K)  # (K,K)

    # Apply Cholesky to draws
    random_part = jnp.einsum(
        "kl,nlr->nkr",
        L,
        draws
    )  # (N,K,R)

    return mean[:, :, None] + random_part


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

    idx = 0
    eta = 0.0

    N = data["Xf"].shape[0]
    P = data["Xf"].shape[1]

    # =========================
    # FIXED
    # =========================
    if spec.Kf > 0:
        beta_f = params[idx:idx + spec.Kf]
        idx += spec.Kf
        eta = jnp.einsum("npk,k->np", data["Xf"], beta_f)[..., None]

    # =====================================================
    # PREPARE HETEROGENEITY (IF EXISTS)
    # =====================================================
    gamma_all = None
    if spec.Kh > 0 and spec.K_random_total > 0:

        Khet = spec.Kh * spec.K_random_total
        gamma_all = params[idx:idx + Khet]
        idx += Khet

        gamma_all = gamma_all.reshape(
            spec.Kh, spec.K_random_total
        )

        Z = data["Xh"]  # (N,P,Kh)

        # (N,P,Kh) x (Kh,Krandom) -> (N,P,Krandom)
        shift = jnp.einsum("npk,km->npm", Z, gamma_all)

        # Collapse panel dimension
        shift = jnp.mean(shift, axis=1)  # (N,Krandom)

    # =====================================================
    # CORRELATED RANDOM
    # =====================================================
    if spec.Kr_cor > 0:

        mean_cor = params[idx:idx + spec.Kr_cor]
        idx += spec.Kr_cor

        Kchol = spec.Kr_cor * (spec.Kr_cor + 1) // 2
        chol = params[idx:idx + Kchol]
        idx += Kchol

        if gamma_all is not None:
            shift_cor = shift[:, :spec.Kr_cor]
            mean_cor = mean_cor[None, :] + shift_cor
        else:
            mean_cor = mean_cor

        beta_cor = random_correlated(
            mean_cor,
            chol,
            data["draws_cor"],
            spec.Kr_cor,
            jnp.array(spec.random_cor_dists)
        )

        eta += jnp.einsum("npk,nkr->npr", data["Xr_cor"], beta_cor)

    # =====================================================
    # INDEPENDENT RANDOM
    # =====================================================
    if spec.Kr_ind > 0:

        mean_ind = params[idx:idx + spec.Kr_ind]
        idx += spec.Kr_ind

        sd_ind = params[idx:idx + spec.Kr_ind]
        idx += spec.Kr_ind

        if gamma_all is not None:
            shift_ind = shift[:, spec.Kr_cor:]
            mean_ind = mean_ind[None, :] + shift_ind
        else:
            mean_ind = mean_ind

        beta_ind = random_independent(
            mean_ind,
            sd_ind,
            data["draws_ind"],
            jnp.array(spec.random_ind_dists)
        )

        eta += jnp.einsum("npk,nkr->npr", data["Xr_ind"], beta_ind)

    # =====================================================
    # GROUPED RANDOM
    # =====================================================
    if spec.Kg > 0:

        mean_g = params[idx:idx + spec.Kg]
        idx += spec.Kg

        sd_g = params[idx:idx + spec.Kg]
        idx += spec.Kg

        beta_g_all = transform_draws(
            data["draws_g"],
            mean_g,
            sd_g,
            jnp.array(spec.grouped_dists)
        )

        beta_g = beta_g_all[data["group_ids"]]

        eta += jnp.einsum("npk,nkr->npr", data["Xg"], beta_g)

    # Offset
    offset = data["offset"]
    if offset.ndim == 2:
        offset = offset[..., None]

    eta += offset

    return eta, idx
# =========================================================
# ETA BUILDER
# =========================================================

def build_eta_old(params, data, spec: ModelSpec):
    idx = 0
    eta = 0.0

    # FIXED
    if spec.Kf > 0:
        beta_f = params[idx:idx + spec.Kf]
        idx += spec.Kf
        eta = jnp.einsum("npk,k->np", data["Xf"], beta_f)[..., None]

    # =========================
    # CORRELATED RANDOM BLOCK
    # =========================
    if spec.Kr_cor > 0:

        mean_cor = params[idx:idx + spec.Kr_cor]
        idx += spec.Kr_cor

        Kchol = spec.Kr_cor * (spec.Kr_cor + 1) // 2
        chol = params[idx:idx + Kchol]
        idx += Kchol

        # ----- HETEROGENEITY SHIFT -----
        if spec.Kh > 0:
            Khet = spec.Kh * spec.K_random_total
            gamma_all = params[idx:idx + Khet]
            idx += Khet

            gamma_all = gamma_all.reshape(
                spec.Kh, spec.K_random_total
            )

            # correlated slice
            gamma_cor = gamma_all[:, :spec.Kr_cor]

            # Z shape (N,P,Kh)
            Z = data["Xh"]

            # compute individual-specific mean shift
            shift_cor = jnp.einsum(
                "npk,km->npm",
                Z,
                gamma_cor
            )  # (N,P,Kr_cor)

            # collapse P since random coeff is individual-level
            shift_cor = jnp.mean(shift_cor, axis=1)

            mean_cor = mean_cor + shift_cor

        beta_cor = random_correlated(
            mean_cor,
            chol,
            data["draws_cor"],
            spec.Kr_cor
        )

        eta += jnp.einsum("npk,nkr->npr", data["Xr_cor"], beta_cor)
    
    
    
    # =========================
    # INDEPENDENT RANDOM BLOCK
    # =========================
    if spec.Kr_ind > 0:

        mean_ind = params[idx:idx + spec.Kr_ind]
        idx += spec.Kr_ind

        sd_ind = params[idx:idx + spec.Kr_ind]
        idx += spec.Kr_ind

        if spec.Kh > 0:

            gamma_all = params[param_index["hetro"][0]:param_index["hetro"][1]]
            gamma_all = gamma_all.reshape(
                spec.Kh, spec.K_random_total
            )

            gamma_ind = gamma_all[:, spec.Kr_cor:]

            Z = data["Xh"]

            shift_ind = jnp.einsum(
                "npk,km->npm",
                Z,
                gamma_ind
            )

            shift_ind = jnp.mean(shift_ind, axis=1)

            mean_ind = mean_ind + shift_ind

        beta_ind = random_independent(
            mean_ind,
            sd_ind,
            data["draws_ind"]
        )

        eta += jnp.einsum("npk,nkr->npr", data["Xr_ind"], beta_ind)
        
    
    

        mean_ind = params[idx:idx + spec.Kr_ind]
        idx += spec.Kr_ind

        sd_ind = params[idx:idx + spec.Kr_ind]
        idx += spec.Kr_ind

        beta_ind = random_independent(
            mean_ind,
            sd_ind,
            data["draws_ind"]
        )

        eta += jnp.einsum("npk,nkr->npr", data["Xr_ind"], beta_ind)

    # GROUPED
    if spec.Kg > 0:

        mean_g = params[idx:idx + spec.Kg]
        idx += spec.Kg

        sd_g = params[idx:idx + spec.Kg]
        idx += spec.Kg

        group_ids = data["group_ids"]      # (N,)
        draws_g = data["draws_g"]          # (G, Kg, R)

        # Generate group-level random effects
        beta_g_all = mean_g[None, :, None] + \
                    sd_g[None, :, None] * draws_g
        # (G, Kg, R)

        # Map to individuals
        beta_g = beta_g_all[group_ids]     # (N, Kg, R)

        eta += jnp.einsum(
            "npk,nkr->npr",
            data["Xg"],
            beta_g
        )

    
        #eta += jnp.einsum("npk,k->np", data["Xh"], gamma)[..., None]

    offset = data["offset"]
    if offset.ndim == 2:
        offset = offset[..., None]

    eta += offset
    return eta, idx

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
    beta_tri = mean + scale * z

    # Stack and select
    betas = jnp.stack([beta_normal, beta_lognormal, beta_tri], axis=-1)

    # dist_codes shape (K,)
    selector = jax.nn.one_hot(dist_codes, 3)  # (K,3)

    selector = selector[None, :, None, :]  # (1,K,1,3)

    beta = jnp.sum(betas * selector, axis=-1)

    return beta  # (N,K,R)
# LOG LIKELIHOOD
# =========================================================

@partial(jax.jit, static_argnames=("spec",))
def mixed_model_loglik(params, data, spec: ModelSpec, draws):

    eta, idx = build_eta(params, data, spec)
    mu = jnp.exp(eta)
    print("mu mean:", jnp.mean(mu))
    if mu.ndim == 2:
        mu = mu[..., None]

    y = ensure_3d(data["y"])
    mask = ensure_3d(data["mask"])

    R = mu.shape[-1] if mu.ndim == 3 else 1

    if spec.model == "poisson":
        ll = poisson_loglik(y, mu)

    elif spec.model == "nb":
        alpha = params[idx]
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

    return fixed_cols, random_ind, random_cor, grouped_cols, hetro_cols


def build_model_from_manual_spec(df, manual_spec, id_col, y_col, offset_col=None, R=200):

    fixed_cols, random_ind, random_cor, grouped_cols, hetro_cols = parse_manual_spec(manual_spec)

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
        offset_col=offset_col,
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
    group_id_col = None,
    fixed_cols=None,
    random_ind_cols=None,
    random_cor_cols=None,
    grouped_cols=None,
    hetro_cols=None,
    offset_col=None,
    R=200
):

    fixed_cols = fixed_cols or []
    random_ind_cols = random_ind_cols or []
    random_cor_cols = random_cor_cols or []
    grouped_cols = grouped_cols or []
    hetro_cols = hetro_cols or []

    all_features = list(set(
        fixed_cols + random_ind_cols + random_cor_cols + grouped_cols + hetro_cols
    ))

    X_all, y, mask = balance_panel_dataframe(
        df, id_col, y_col, all_features
    )
    # =========================
    # GROUP IDS
    # =========================
    if group_id_col is not None and len(grouped_cols) > 0:

        df_sorted = df.sort_values(id_col)

        # Convert group column to integer codes
        group_codes = df_sorted[group_id_col].astype("category").cat.codes.values

        # Unique groups
        unique_groups = np.unique(group_codes)
        G = len(unique_groups)

    else:
        group_codes = None
        G = 0
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

    N, P = y.shape[0], y.shape[1]

    if offset_col:
        offset = extract_offset(df, id_col, offset_col)
    else:
        offset = np.zeros((N, P, 1))

    draws_ind = (
        np.random.normal(size=(N, len(random_ind_cols), R))
        if len(random_ind_cols) > 0 else None
    )

    draws_cor = (
        np.random.normal(size=(N, len(random_cor_cols), R))
        if len(random_cor_cols) > 0 else None
    )

    draws_g = (
    np.random.normal(size=(G, len(grouped_cols), R))
    if len(grouped_cols) > 0 and G > 0 else None
    )

    data = {
        "Xf": jnp.array(Xf),
        "Xr_ind": jnp.array(Xr_ind),
        "Xr_cor": jnp.array(Xr_cor),
        "Xg": jnp.array(Xg),
        "Xh": jnp.array(Xh),
        "y": jnp.array(y),
        "mask": jnp.array(mask),
        "offset": jnp.array(offset),
        "draws_ind": jnp.array(draws_ind) if draws_ind is not None else None,
        "draws_cor": jnp.array(draws_cor) if draws_cor is not None else None,
        "draws_g": jnp.array(draws_g) if draws_g is not None else None,
        "group_ids": jnp.array(group_codes) if group_codes is not None else None,
    }

    spec = ModelSpec(
        Kf=Xf.shape[2],
        Kr_ind=Xr_ind.shape[2],
        Kr_cor=Xr_cor.shape[2],
        Kg=Xg.shape[2],
        Kh=Xh.shape[2],
        model="poisson",
        fixed_names=tuple(fixed_cols),
        random_ind_names=tuple(random_ind_cols),
        random_cor_names=tuple(random_cor_cols),
        grouped_names=tuple(grouped_cols),
        hetro_names=tuple(hetro_cols)
    )

    return data, spec
# =========================================================
# PARAM INDEX
# =========================================================

def build_param_index(spec: ModelSpec):

    idx = 0
    index = {}

    index["fixed"] = (idx, idx + spec.Kf)
    idx += spec.Kf

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
        R=100
    ):

        self.df = df
        self.id_col = id_col
        self.y_col = y_col
        self.vars = all_variables
        self.allowed_roles = allowed_roles
        self.allowed_distributions = allowed_distributions
        self.mode = mode
        self.R = R
        self.halton_draws = generate_halton_normal(
            n_obs=len(df),
            n_dim=len(all_variables),
            R=R,
            seed=42
        )
        if mode == "multi":
            self.df_train, self.df_test, self.df_val = build_datasets(
                df, id_col, y_col
            )
        else:
            self.df_train = df

    
    
    def generate_halton_draws(self):

        sampler = qmc.Halton(
            d=self.max_K,
            scramble=True,
            seed=self.seed
        )

        u = sampler.random(self.R)              # (R, K)
        z = norm.ppf(u)                         # normal transform

        # Expand across observations
        z = np.tile(z, (self.N, 1, 1))          # (N, R, K)

        return z
    
    # ----------------------------------------
    # Convert decision vector to manual spec
    # ----------------------------------------
    def build_spec(self, decision):

        D = len(self.vars)

        roles = decision[:D]
        dists = decision[D:]

        fixed_cols = []
        rdm_terms = []
        rdm_cor_terms = []
        grouped_terms = []
        hetero_cols = []

        for i, var in enumerate(self.vars):

            role = roles[i]

            if role not in self.allowed_roles.get(var, [0]):
                return None

            if role == 0:
                continue

            elif role == 1:
                fixed_cols.append(var)

            elif role == 2:
                dist = decode_distribution(
                    dists[i],
                    self.allowed_distributions.get(var, ["normal"])
                )
                rdm_terms.append(f"{var}:{dist}")

            elif role == 3:
                dist = decode_distribution(
                    dists[i],
                    self.allowed_distributions.get(var, ["normal"])
                )
                rdm_cor_terms.append(f"{var}:{dist}")

            elif role == 4:
                dist = decode_distribution(
                    dists[i],
                    self.allowed_distributions.get(var, ["normal"])
                )
                grouped_terms.append(f"{var}:{dist}")

            elif role == 5:
                hetero_cols.append(var)

        # Logical constraints
        if len(rdm_cor_terms) == 1:
            return None

        if len(rdm_terms) + len(rdm_cor_terms) == 0:
            hetero_cols = []

        return {
            "fixed_terms": fixed_cols,
            "rdm_terms": rdm_terms,
            "rdm_cor_terms": rdm_cor_terms,
            "grouped_terms": grouped_terms,
            "hetro_in_means": [f"{z}:normal" for z in hetero_cols],
            "dispersion": 1
        }

    # ----------------------------------------
    # Fitness Function
    # ----------------------------------------
    def fitness(self, decision):

        spec_dict = self.build_spec(decision)

        if spec_dict is None:
            return -1e12

        try:
            # TRAIN MODEL
            data_train, spec = build_model_from_manual_spec(
                df=self.df_train,
                manual_spec=spec_dict,
                id_col=self.id_col,
                y_col=self.y_col,
                R=self.R
            )

            param_index = build_param_index(spec)
            init = jnp.zeros(param_index["total_params"])

            objective = partial(
                mixed_model_loglik,
                data=data_train,
                spec=spec,
                draws = draws
            )

            solver = LBFGS(fun=objective)
            result = solver.run(init)

            train_ll = -objective(result.params)

            if self.mode == "single":
                return float(train_ll)

            # MULTI OBJECTIVE
            data_test, _ = build_model_from_manual_spec(
                df=self.df_test,
                manual_spec=spec_dict,
                id_col=self.id_col,
                y_col=self.y_col,
                R=self.R
            )

            metrics = evaluate_metrics(
                result.params,
                data_test,
                spec,
                name="TEST"
            )

            test_rmse = metrics["rmse"]

            # maximize LL, minimize RMSE
            return float(train_ll) - test_rmse

        except Exception:
            return -1e12


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


if __name__ == "__main__":
    
    allowed_distributions = {
    "CURVES": ["normal", "lognormal"],
    "FRICTION": ["normal", "triangular"],
    "MIMEDSH": ["normal"],
    "TRAIN": ["normal"],
    }
    all_variable = all_variables = [
    "CURVES",
    "FRICTION",
    "MIMEDSH",
    "TRAIN",
    "INCLANES"
    ]
    allowed_roles = {
    "CURVES": [0,1,2,3],
    "TRAIN": [0,4],
    "INCLANES": [0,5],
    }   

    df = pd.read_csv('./data/ex163.csv')
    df.rename(columns={"FREQ": "Y"}, inplace=True)
    evaluator = StructureEvaluator(
        df=df,
        id_col="id",
        y_col="Y",
        all_variables=all_variables,
        allowed_roles=allowed_roles,
        allowed_distributions=allowed_distributions,
        mode="single",   # or "multi"
        R=100
    )

    D = len(all_variables) * 2  # roles + dists

    hs = BinaryHarmonySearch(
        fitness_function=evaluator.fitness,
        dimension=D,
        max_iter=10
    )

    best_solution, best_score = hs.optimize()

    print("Best Decision:", best_solution)
    print("Best Score:", best_score)












"""



if __name__ == "__maindd__":

    print("Stage 5A Data Loading...")

    x_df = pd.read_csv('./data/ex163.csv')

    drop_these = ['Id', 'ID', 'old', 'G_N']
    for i in drop_these:
        x_df.drop(x_df.filter(regex=i).columns, axis=1, inplace=True)

    y_df = x_df[['FREQ']].copy()
    y_df.rename(columns={"FREQ": "Y"}, inplace=True)

    x_df['Offset'] = 0
    x_df["Constant"] = 1.0

    # Panel id (if no real panel, keep row id)
    x_df["id"] = np.arange(len(x_df))

    df = x_df.copy()
    df["Y"] = y_df["Y"]

    # =====================================================
    # ✅ PANEL-SAFE STRATIFIED SPLIT
    # =====================================================

    df_train, df_test, df_val = build_datasets(
        df,
        id_col="id",
        y_col="Y"
    )

    print("Train size:", len(df_train))
    print("Test size:", len(df_test))
    print("Validation size:", len(df_val))

    # =====================================================
    # MODEL SPEC
    # =====================================================

    manual_fit_spec = {
        'fixed_terms': ['Constant', 'INTECHAG', 'CURVES','FRICTION', 'MIMEDSH'],
        'rdm_terms': ['MIMEDSH:normal'],
        'rdm_cor_terms': ['CURVES:normal', 'FRICTION:normal'],
        'grouped_terms': ['TRAIN:normal'],
        'group_id_col': 'URB',
        'hetro_in_means': ['INCLANES:normal', 'FC:normal', 'SPEED:normal'],
        'dispersion': 1
    }

    # =====================================================
    # ✅ TRAIN MODEL ON TRAIN SET ONLY
    # =====================================================

    data_train, spec = build_model_from_manual_spec(
        df=df_train,
        manual_spec=manual_fit_spec,
        id_col="id",
        y_col="Y",
        offset_col=None,
        R=400
    )

    param_index = build_param_index(spec)
    init = jnp.zeros(param_index["total_params"])

    objective_train = partial(
        mixed_model_loglik,
        data=data_train,
        spec=spec
    )

    print("Starting estimation on TRAIN set...")

    solver = LBFGS(fun=objective_train)
    result = solver.run(init)

    print("\nEstimated parameters:")
    print(result.params)

    print_summary(result, objective_train, data_train, spec, param_index)

    # =====================================================
    # ✅ EVALUATE ON TEST SET
    # =====================================================

    data_test, _ = build_model_from_manual_spec(
        df=df_test,
        manual_spec=manual_fit_spec,
        id_col="id",
        y_col="Y",
        offset_col=None,
        R=400
    )

    objective_test = partial(
        mixed_model_loglik,
        data=data_test,
        spec=spec
    )

    test_ll = -objective_test(result.params)

    print("\n================ TEST PERFORMANCE ================")
    print(f"Test Log-Likelihood: {float(test_ll):.4f}")

    # =====================================================
    # ✅ FINAL VALIDATION PERFORMANCE
    # =====================================================

    data_val, _ = build_model_from_manual_spec(
        df=df_val,
        manual_spec=manual_fit_spec,
        id_col="id",
        y_col="Y",
        offset_col=None,
        R=400
    )

    objective_val = partial(
        mixed_model_loglik,
        data=data_val,
        spec=spec
    )

    val_ll = -objective_val(result.params)

    print("\n================ VALIDATION PERFORMANCE ================")
    print(f"Validation Log-Likelihood: {float(val_ll):.4f}")

    print("\n========================================================")
    
    
    print_summary(result, objective_train, data_train, spec, param_index)

    # Evaluate on TRAIN
    evaluate_metrics(result.params, data_train, spec, "TRAIN")

    # Evaluate on TEST
    evaluate_metrics(result.params, data_test, spec, "TEST")

    # Evaluate on VALIDATION
    evaluate_metrics(result.params, data_val, spec, "VALIDATION")




"""
'''

# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    print("Stage 5A Data Loading...")

    x_df = pd.read_csv('./data/ex163.csv')

    drop_these = ['Id', 'ID', 'old', 'G_N']
    for i in drop_these:
        x_df.drop(x_df.filter(regex=i).columns, axis=1, inplace=True)

    y_df = x_df[['FREQ']].copy()
    y_df.rename(columns={"FREQ": "Y"}, inplace=True)

    x_df['Offset'] = 0
    #x_df = x_df.drop(columns=['Headon', 'LEN_YR'])

    #drop_these_too = ['LEN', 'VS_Curve', 'FW_RS', 'RD', 'M', 'SP', 'FW']
   # for i in drop_these_too:
    #    x_df.drop(x_df.filter(regex=i).columns, axis=1, inplace=True)

    # Add constant
    x_df["Constant"] = 1.0

    # Fake panel id (cross-sectional)
    x_df["id"] = np.arange(len(x_df))

    df = x_df.copy()
    df["Y"] = y_df["Y"]

    manual_fit_spec = {
        'fixed_terms': ['const', 'INTECHAG', 'CURVES','FRICTION', 'MIMEDSH'],
        'rdm_terms': ['MIMEDSH:normal'],
        'rdm_cor_terms': ['CURVES:normal', 'FRICTION:normal'],
        'grouped_terms': ['TRAIN:normal'],
        'group_id_col': 'URB',   # <-- REA
        'hetro_in_means': ['INCLANES:normal ' , 'FC:normal' , 'SPEED:normal'],
        'dispersion': 1
    }
    
    
    data, spec = build_model_from_manual_spec(
        df=df,
        manual_spec=manual_fit_spec,
        id_col="id",
        y_col="Y",
        offset_col=None,
        #group_id_col=manual_fit_spec.get("group_id_col", None)
        R=400
    )

    param_index = build_param_index(spec)
    init = jnp.zeros(param_index["total_params"])

    objective = partial(mixed_model_loglik, data=data, spec=spec)

    print("Starting estimation...")
    
    
    solver = LBFGS(fun=objective)
    
   
    result = solver.run(init)

    print("\nEstimated parameters:")
    print(result.params)
    print_summary(result, objective, data, spec, param_index)
    
    '''