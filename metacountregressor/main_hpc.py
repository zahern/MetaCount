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
from jaxopt import LBFGS, GradientDescent, NonlinearCG, BFGS
import pandas as pd
import scipy.stats as stats
try:
    from .Solvers_METAJAX import *  # type: ignore[attr-defined]
except ImportError:
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
from scipy.optimize import minimize
import traceback

DIST_MAP = {
    "normal": 0,
    "lognormal": 1,
    "triangular": 2,
    'uniform':3,
    
}

def lognormal_loglik(y, eta, sigma):

    sigma = jax.nn.softplus(sigma)

    log_y = jnp.log(jnp.clip(y, 1e-12, 1e12))

    ll = (
        -0.5 * jnp.log(2 * jnp.pi * sigma**2)
        - (log_y - eta)**2 / (2 * sigma**2)
        - log_y
    )

    return ll


def gaussian_loglik(y, eta, sigma):

    sigma = jax.nn.softplus(sigma)

    ll = (
        -0.5 * jnp.log(2 * jnp.pi * sigma**2)
        - (y - eta) ** 2 / (2 * sigma**2)
    )

    return ll



def check_structure_recovery(
    pareto_solutions,
    evaluator,
    true_spec_dict,
):

    true_clean = strip_distribution(true_spec_dict)

    for i, decision in enumerate(pareto_solutions):

        spec_dict = evaluator.build_spec(decision)
        if spec_dict is None:
            continue

        spec_clean = strip_distribution(spec_dict)

        if spec_clean == true_clean:
            return True, i

    return False, None


def save_structure_recovery_file(
    recovered,
    algo,
    seed,
    config_id,
    folder="results"
):

    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{algo}_recovery_seed{seed}_config{config_id}_{timestamp}.txt"
    filepath = os.path.join(folder, filename)

    with open(filepath, "w") as f:

        if recovered:
            f.write("✅ TRUE STRUCTURE RECOVERED\n")
        #else:
        #    f.write("❌ TRUE STRUCTURE NOT RECOVERED\n")

    print(f"✅ Structure recovery saved to {filepath}")
    
    
def refit_and_save_true_model(
    evaluator,
    decision,
    algo,
    seed,
    config_id,
    folder="results"
):

    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{algo}_TRUE_MODEL_seed{seed}_config{config_id}_{timestamp}.txt"
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        with redirect_stdout(f):

            print("================================================")
            print("TRUE STRUCTURE RE-ESTIMATION")
            print("================================================")
            print("Algorithm:", algo)
            print("Seed:", seed)
            print("Config ID:", config_id)
            print("Timestamp:", timestamp)
            print("================================================\n")

            # Build spec
            spec_dict = evaluator.build_spec(decision)

            data_train, spec = evaluator.build_data(
                evaluator.df_train,
                spec_dict,
                evaluator.master_halton_train
            )

            model = CountModel(spec, data_train)
            result = model.fit()

            objective = partial(mixed_model_loglik,
                                data=data_train,
                                spec=spec)

            param_index = build_param_index(spec)

            print_summary(
                result=result,
                objective=objective,
                data=data_train,
                spec=spec,
                param_index=param_index
            )

    print(f"✅ True model summary saved to {filepath}")

def save_pareto_front(
    solutions,
    scores,
    evaluator,
    algo,
    seed,
    config_id,
    folder="results"
):

    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"{algo}_pareto_seed{seed}_config{config_id}_{timestamp}.csv"
    )

    filepath = os.path.join(folder, filename)

    rows = []

    for i in range(len(solutions)):

        decision = solutions[i]
        score = scores[i]

        spec_dict = evaluator.build_spec(decision)

        rows.append({
            "solution_id": i,
            "bic": float(score[0]) if score.ndim > 0 else float(score),
            "rmse": float(score[1]) if score.ndim > 0 else None,
            "structure": json.dumps(spec_dict)
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)

    print(f"\n✅ Saved Pareto front to {filepath}")



def extract_pareto_front(solutions, scores):
    """
    Generic Pareto front extractor.
    Minimization assumed.
    """

    solutions = np.array(solutions)
    scores = np.array(scores)

    # Single objective
    if scores.ndim == 1:
        best_idx = np.argmin(scores)
        return solutions[[best_idx]], scores[[best_idx]]

    n = len(scores)
    keep = []

    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue

            if np.all(scores[j] <= scores[i]) and np.any(scores[j] < scores[i]):
                dominated = True
                break

        if not dominated:
            keep.append(i)

    return solutions[keep], scores[keep]

def save_pareto_front(
    solutions,
    scores,
    evaluator,
    algo,
    seed,
    config_id,
    folder="results"
):

    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"{algo}_pareto_seed{seed}_config{config_id}_{timestamp}.csv"
    )

    filepath = os.path.join(folder, filename)

    rows = []

    for i in range(len(solutions)):

        decision = solutions[i]
        score = scores[i]

        spec_dict = evaluator.build_spec(decision)

        rows.append({
            "solution_id": i,
            "bic": float(score[0]) if score.ndim > 0 else float(score),
            "rmse": float(score[1]) if score.ndim > 0 else None,
            "structure": json.dumps(spec_dict)
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)

    print(f"\n✅ Saved Pareto front to {filepath}")


def generate_halton_normal(N, K, R, seed=42):

    sampler = qmc.Halton(d=K, scramble=False, seed=seed)
    sampler.fast_forward(50)
    u = sampler.random(N*R)          # (R, K)
    z = norm.ppf(u)                # (R, K)

    #z = z.T                        # (K, R)
   # z = np.tile(z[None, :, :], (N, 1, 1))  # (N, K, R)
    z = z.reshape(N, R, K).swapaxes(1, 2)
    return z


def decode_distribution(dist_code, allowed_list):

    if len(allowed_list) == 1:
        return allowed_list[0]

    idx = dist_code % len(allowed_list)
    return allowed_list[idx]







def random_decision(D, rng):

    # 0 to 5 inclusive
    return rng.integers(0, 7, size=D)

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
            self.bic = float(k * jnp.log(n) - 2*ll)

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
from dataclasses import dataclass, replace
@dataclass(frozen=True)
class ModelSpec:
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
    latent_classes: int =1
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
    #scale = jnp.abs(scale)
    scale = jax.nn.softplus(scale)

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





@partial(jax.jit, static_argnames=("spec", "indivi"))
def mixed_model_loglik(params, data, spec: ModelSpec, indivi = False):
    
    # ==========================================================
    # ✅ LATENT CLASS WRAPPER (NEW)
    # ==========================================================
    if spec.latent_classes > 1:

        C = spec.latent_classes
        base_spec = replace(spec, latent_classes=1)

        # Build base index once
        base_index = build_param_index(base_spec)
        K_base = base_index["total_params"]

        # Split parameters
        theta_all = params[:C * K_base].reshape(C, K_base)
        logits = params[C * K_base:]

        logits_full = jnp.concatenate([jnp.array([0.0]), logits])
        pi = jax.nn.softmax(logits_full)

        ll_classes = []

        for c in range(C):

            ll_c = -mixed_model_loglik(
                theta_all[c],
                data,
                base_spec,
                indivi = True           
            )

            ll_classes.append(ll_c + jnp.log(pi[c]))

        ll_stack = jnp.stack(ll_classes, axis = 1)
        ll_ind = jsp.special.logsumexp(ll_stack, axis=1)

        if indivi:
            return ll_ind

        return -jnp.sum(ll_ind)

        #return -jsp.special.logsumexp(ll_stack)

    blocks = unpack_params(params, spec)

    eta = build_eta(params, data, spec)

    # Clip for numerical stability
    eta = jnp.clip(eta, -25, 25)

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
    elif spec.model == "lognormal":
        sigma = blocks["sigma"]
        ll_count = lognormal_loglik(y, eta, sigma)
    elif spec.model == "gaussian":
        sigma = blocks["sigma"]
        ll_count = gaussian_loglik(y, eta, sigma)

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

    if indivi:
        return ll_ind
    return -jnp.sum(ll_ind)



@partial(jax.jit, static_argnames=("spec",))
def mixed_model_loglik_individual(params, data, spec: ModelSpec):
    return mixed_model_loglik(params, data, spec, indivi = True)

def nb1_loglik(y, mu, alpha):

    alpha = jnp.exp(alpha)

    r = mu / alpha
    p = 1.0 / (1.0 + alpha)

    ll = (
        jsp.special.gammaln(y + r)
        - jsp.special.gammaln(r)
        - jsp.special.gammaln(y + 1)
        + r * jnp.log(p)
        + y * jnp.log(1 - p)
    )

    return ll

def nb2_loglik(y, eta, alpha):

    alpha = jax.nn.softplus(alpha)
   # alpha = jnp.abs(alpha)
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
    spec = replace(spec, model=model_type)

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
    zi_cols = zi_cols or  []

    random_ind_dists = random_ind_dists or []
    random_cor_dists = random_cor_dists or []
    grouped_dists = grouped_dists or []

    # -----------------------------
    # Build balanced panel
    # -----------------------------
    # Always include intercept
    intercept_name = "__INTERCEPT__"

    df = df.copy()
    df[intercept_name] = 1.0
    
    all_features = list(set(
        ["__INTERCEPT__"] + fixed_cols + random_ind_cols + random_cor_cols + grouped_cols + hetro_cols +zi_cols
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

    fixed_cols_with_intercept = ["__INTERCEPT__"] + fixed_cols
    Xf = extract(fixed_cols_with_intercept)
    X_fixed = extract(fixed_cols)
    X_intercept = extract(["__INTERCEPT__"])
    Xf = np.concatenate([X_intercept, X_fixed], axis=2)
    #Xf = extract(fixed_cols)
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
        zero_inflated = (len(zi_cols) > 0),
        model="poisson",  # overwritten later if NB
        fixed_names=tuple(fixed_cols_with_intercept),
        random_ind_names=tuple(random_ind_cols),
        random_cor_names=tuple(random_cor_cols),
        grouped_names=tuple(grouped_cols),
        hetro_names=tuple(hetro_cols),
        random_ind_dists=tuple(random_ind_dists),
        random_cor_dists=tuple(random_cor_dists),
        grouped_dists=tuple(grouped_dists),
    )

    return data, spec


def build_base_index(spec):
    

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
    
    if spec.model in {"lognormal", "gaussian"}:
        index["sigma"] = idx
        idx += 1

    index["total_params"] = idx

    return index


    # ← your current build_param_index body

#=======================================
# PARAM INDEX
# =========================================================
def build_param_index(spec: ModelSpec):

    if spec.latent_classes == 1:
        return build_base_index(spec)
    base_spec = replace(spec, latent_classes=1)
    base_index = build_base_index(base_spec)

    K_base = base_index["total_params"]
    C = spec.latent_classes

    idx = 0
    index = {}

    index["class_params"] = (0, C * K_base)
    idx = C * K_base

    index["class_logits"] = (idx, idx + C - 1)
    idx += C - 1

    index["K_base"] = K_base
    index["total_params"] = idx

    return index



def compute_predictions(params, data, spec):
        eta = build_eta(params, data, spec)
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


class DummyResult:
    def __init__(self, params):
        self.params = params

def print_summary(result, objective, data, spec, param_index):

    import numpy as np
    import pandas as pd
    from scipy import stats
    
    if spec.latent_classes > 1:

        C = spec.latent_classes

        base_spec = replace(spec, latent_classes=1)
        base_index = build_base_index(base_spec)
        K_base = base_index["total_params"]

        params = np.asarray(result.params)

        theta_all = params[:C * K_base].reshape(C, K_base)

        logits = params[C * K_base:]
        logits_full = np.concatenate(([0.0], logits))
        pi = np.exp(logits_full) / np.sum(np.exp(logits_full))

        print("\n====================================================")
        print("        LATENT CLASS MIXED MODEL SUMMARY")
        print("====================================================")

        # ---------------------------------------------
        # Print each class separately
        # ---------------------------------------------
        for c in range(C):

            print(f"\n################ CLASS {c+1} ################\n")

            class_result = DummyResult(theta_all[c])
            class_result.params = theta_all[c]

            print_summary(
                result=class_result,
                objective=None,
                data=data,
                spec=base_spec,
                param_index=base_index
            )

        # ---------------------------------------------
        # Print class probabilities
        # ---------------------------------------------
        print("\n================ CLASS PROBABILITIES ================\n")

        for c in range(C):
            print(f"pi_{c+1}: {pi[c]:.6f}")

        print("\n====================================================\n")

        return
    

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

    # Fixed (Intercept first)
    #names.append("Intercept")
    
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







def generate_panel_data(N_ids=1000, T=4, seed=0):

    import numpy as np
    import pandas as pd
    from scipy.stats import multivariate_normal

    np.random.seed(seed)

    # -----------------------------
    # TRUE PARAMETERS
    # -----------------------------

    beta_0 = -2.5
    beta_shoulder = -0.3
    beta_aadt = 0.4
    beta_length = 0.2

    mu_speed = 0.05
    mu_lanes = 0.2

    sd_speed = 0.03
    sd_lanes = 0.1
    rho = -0.5

    alpha_true = 1.0

    Sigma = np.array([
        [sd_speed**2, rho*sd_speed*sd_lanes],
        [rho*sd_speed*sd_lanes, sd_lanes**2]
    ])

    # -----------------------------
    # RANDOM COEFFICIENTS (ONE PER ID)
    # -----------------------------

    betas = multivariate_normal.rvs(
        mean=[mu_speed, mu_lanes],
        cov=Sigma,
        size=N_ids
    )

    beta_speed_i = betas[:,0]
    beta_lanes_i = betas[:,1]

    # -----------------------------
    # BUILD PANEL
    # -----------------------------

    rows = []

    for i in range(N_ids):
        speed = np.random.uniform(30, 80)
        lanes = np.random.choice([2,3,4])
        shoulder = np.random.uniform(0, 10)
        length = np.random.uniform(0.5, 5)
        aadt = np.random.lognormal(mean=9, sigma=0.6)
        rural = np.random.binomial(1, 0.6)
        curves = np.random.choice([0, 0, 1, 2])
        slope = np.random.choice([0, 0,0,0, 0, 1, 1, 1, 2, 2, 3])
        insig_3 = np.sqrt(np.random.uniform(0, 50))
        insig_4 = np.random.triangular(0, 4, 8)
        for t in range(T):

            
            
            # Dummy example
            
            eta = (
                beta_0
                + beta_shoulder * shoulder
                + beta_aadt * np.log(aadt)
                + beta_length * length
                + beta_speed_i[i] * speed
                + beta_lanes_i[i] * lanes
            )

            mu = np.exp(eta)
            p = mu / (mu + alpha_true)

            y = np.random.negative_binomial(alpha_true, 1 - p)

            rows.append([
                i, y, speed, lanes, shoulder,
                np.log(aadt), length, rural, curves, slope, insig_3, insig_4
            ])

    df = pd.DataFrame(rows, columns=[
        "ID","Y","speed","lanes","shoulder", 
        "log_aadt","length",'rural', 'curves', 'slope', 'insig_1', 'insig_2'
    ])
    true_params = {
        "__INTERCEPT__": beta_0,
        "shoulder": beta_shoulder,
        "log_aadt": beta_aadt,
        "length": beta_length,
        "cor_mean(speed)": mu_speed,
        "cor_mean(lanes)": mu_lanes,
        "sd(speed)": sd_speed,
        "sd(lanes)": sd_lanes,
        "corr(speed,lanes)": rho,
        "dispersion": alpha_true
    }
    

    return df, true_params



def multi_start_estimation(model, n_starts=10, seed=0):

    import jax
    import jax.numpy as jnp
    from jaxopt import LBFGS

    best_ll = jnp.inf
    best_result = None

    for s in range(n_starts):

        key = jax.random.PRNGKey(seed + s)

        init = 0.1 * jax.random.normal(
            key,
            (model.param_index["total_params"],)
        )

        solver = LBFGS(
            fun=model.objective,
            maxiter=2000
        )

        result = solver.run(init)

        ll = result.state.value

        print(f"Start {s}: LL = {ll:.4f}")

        if ll < best_ll:
            best_ll = ll
            best_result = result

    print("\n✅ Best LL:", best_ll)

    return best_result



def monte_carlo_recovery(
    n_rep=1,
    N_ids=1000,
    T=4,
    R=200,
    n_starts=1
):

    import numpy as np
    import pandas as pd
    from functools import partial

    all_results = []

    for r in range(n_rep):

        print(f"\n====== Replication {r} ======")

        # --------------------------------------------------
        # 1️⃣ Generate panel data + true parameters
        # --------------------------------------------------

        df, true_params = generate_panel_data(
            N_ids=N_ids,
            T=T,
            seed=r
        )

        # --------------------------------------------------
        # 2️⃣ Manual model specification
        # --------------------------------------------------

        manual_spec = {
            "fixed_terms": [
                "shoulder",
                "log_aadt",
                "length",
                "rural"
            ],
            "rdm_terms": [],
            "rdm_cor_terms": [
                "speed:normal",
                "lanes:normal"
            ],
            "grouped_terms": [],
            "hetro_in_means": [],
            "zi_terms": [],
            "dispersion": 1
        }

        # --------------------------------------------------
        # 3️⃣ Generate Halton draws (PER ID)
        # --------------------------------------------------

        draws_cor = generate_halton_normal(
            N=N_ids,
            K=2,
            R=R,
            seed=r
        )

        # --------------------------------------------------
        # 4️⃣ Build model
        # --------------------------------------------------

        data, spec = build_model_from_manual_spec(
            df=df,
            manual_spec=manual_spec,
            id_col="ID",
            y_col="Y",
            offset_col=None,
            draws_cor=draws_cor,
            R=R
        )

        # Force NB
        spec = replace(spec, model="nb")

        model = CountModel(spec, data)

        # --------------------------------------------------
        # 5️⃣ Multi-start estimation
        # --------------------------------------------------

        result = multi_start_estimation(
            model,
            n_starts=n_starts,
            seed=r
        )

        # --------------------------------------------------
        # 6️⃣ Extract estimates
        # --------------------------------------------------

        objective = partial(mixed_model_loglik,
                            data=data,
                            spec=spec)

        param_index = build_param_index(spec)

        summary_df = print_summary(
            result=result,
            objective=objective,
            data=data,
            spec=spec,
            param_index=param_index,
            return_df=True   # <-- modify your function to allow this
        )

        est = dict(zip(summary_df["Parameter"],
                       summary_df["Estimate"]))

        # --------------------------------------------------
        # 7️⃣ Store recovery results
        # --------------------------------------------------

        for k in true_params:

            if k in est:

                all_results.append({
                    "rep": r,
                    "param": k,
                    "true": true_params[k],
                    "est": est[k]
                })

    # ======================================================
    # 8️⃣ Compute Monte Carlo bias + RMSE
    # ======================================================

    results_df = pd.DataFrame(all_results)

    summary = (
        results_df
        .groupby("param")
        .apply(lambda x: pd.Series({
            "True": x["true"].iloc[0],
            "Mean_Est": x["est"].mean(),
            "Bias": x["est"].mean() - x["true"].iloc[0],
            "RMSE": np.sqrt(np.mean((x["est"] - x["true"])**2))
        }))
    )

    print("\n\n========= MONTE CARLO SUMMARY =========\n")
    print(summary)

    return summary



def run_full_synthetic_recovery_experiment(R=300, seed=42):

    print("\n==============================")
    print("RUNNING FULL SYNTHETIC RECOVERY TEST")
    print("==============================\n")

    
    import numpy as np
    import pandas as pd
    from scipy.stats import multivariate_normal

    np.random.seed(seed)

    print("\n==============================")
    print("MINIMAL CORRELATED RANDOM TEST")
    print("==============================\n")

    # -------------------------------------------------
    # TRUE PARAMETERS
    # -------------------------------------------------

    N = 3000

    beta_0 = -2.0
    beta_shoulder = -0.3

    mu_speed = 0.05
    mu_lanes = 0.2

    sd_speed = 0.03
    sd_lanes = 0.1
    rho = -0.5

    alpha_true = 1.0  # NB dispersion

    Sigma = np.array([
        [sd_speed**2, rho*sd_speed*sd_lanes],
        [rho*sd_speed*sd_lanes, sd_lanes**2]
    ])

    # -------------------------------------------------
    # DATA
    # -------------------------------------------------

    speed = np.random.uniform(30, 80, N)
    lanes = np.random.choice([2,3,4], N)
    shoulder = np.random.uniform(0, 10, N)
    shoulder = np.random.uniform(0, 10, N)

    betas = multivariate_normal.rvs(
        mean=[mu_speed, mu_lanes],
        cov=Sigma,
        size=N
    )

    beta_speed = betas[:,0]
    beta_lanes = betas[:,1]

    eta = (
        beta_0 +
        beta_shoulder * shoulder +
        beta_speed * speed +
        beta_lanes * lanes
    )

    mu = np.exp(eta)

    p = mu / (mu + alpha_true)
    y = np.random.negative_binomial(alpha_true, 1 - p)

    df = pd.DataFrame({
        "ID": np.arange(N),
        "Y": y,
        "speed": speed,
        "lanes": lanes,
        "shoulder": shoulder
    })
    df.to_csv('data/meta_synth_multi.csv')
    # -------------------------------------------------
    # SPEC (MINIMAL)
    # -------------------------------------------------

    manual_spec = {
        "fixed_terms": ["shoulder"],
        "rdm_terms": [],
        "rdm_cor_terms": [
            "speed:normal",
            "lanes:normal"
        ],
        "grouped_terms": [],
        "hetro_in_means": [],
        "zi_terms": [],
        "dispersion": 1
    }

    draws_cor = generate_halton_normal(
        N=N,
        K=2,
        R=R,
        seed=seed
    )

    data, spec = build_model_from_manual_spec(
        df=df,
        manual_spec=manual_spec,
        id_col="ID",
        y_col="Y",
        draws_cor=draws_cor,
        R=R
    )

    spec = replace(spec, model="nb")

    # -------------------------------------------------
    # FIT
    # -------------------------------------------------

    model = CountModel(spec, data)

    # IMPORTANT: random init, not zeros
    key = jax.random.PRNGKey(seed)
    init = 0.01 * jax.random.normal(
        key,
        (model.param_index["total_params"],)
    )

    solver = LBFGS(fun=model.objective, maxiter=2000)
    result = solver.run(init)

    objective = partial(mixed_model_loglik, data=data, spec=spec)
    param_index = build_param_index(spec)

    print_summary(
        result=result,
        objective=objective,
        data=data,
        spec=spec,
        param_index=param_index
    )

    print("\n================ TRUE VALUES ================\n")
    print("Intercept:", beta_0)
    print("Shoulder:", beta_shoulder)
    print("Mean Speed:", mu_speed)
    print("Mean Lanes:", mu_lanes)
    print("SD Speed:", sd_speed)
    print("SD Lanes:", sd_lanes)
    print("Correlation:", rho)
    print("Dispersion:", alpha_true)
    
    



def estimate_latent_class_mixed_example():

    from functools import partial
    from dataclasses import replace
    import numpy as np

    print("\n==============================================")
    print(" ESTIMATING 2-CLASS MIXED NEGATIVE BINOMIAL (EM)")
    print("==============================================\n")

    # --------------------------------------------------
    # 1️⃣ Generate synthetic panel data
    # --------------------------------------------------

    df, _ = generate_panel_data(
        N_ids=800,
        T=3,
        seed=0
    )

    # --------------------------------------------------
    # 2️⃣ Model specification
    # --------------------------------------------------

    manual_spec = {
        "fixed_terms": ["shoulder", "log_aadt", "length"],
        "rdm_terms": [],
        "rdm_cor_terms": [
            "speed:normal",
            "lanes:normal"
        ],
        "grouped_terms": [],
        "hetro_in_means": [],
        "zi_terms": [],
        "dispersion": 1
    }

    # --------------------------------------------------
    # 3️⃣ Halton draws
    # --------------------------------------------------

    draws_cor = generate_halton_normal(
        N=df["ID"].nunique(),
        K=2,
        R=200,
        seed=0
    )

    # --------------------------------------------------
    # 4️⃣ Build data + spec
    # --------------------------------------------------

    data, spec = build_model_from_manual_spec(
        df=df,
        manual_spec=manual_spec,
        id_col="ID",
        y_col="Y",
        draws_cor=draws_cor,
        R=200
    )

    spec = replace(spec, model="nb", latent_classes=2)

    # --------------------------------------------------
    # 5️⃣ Build parameter structure
    # --------------------------------------------------

    base_spec = replace(spec, latent_classes=1)
    base_index = build_param_index(base_spec)
    K_base = base_index["total_params"]

    C = spec.latent_classes

    print("Parameters per class:", K_base)
    print("Total parameters (EM structure):", C * K_base + (C - 1))

    # --------------------------------------------------
    # 6️⃣ INITIALIZATION
    # --------------------------------------------------

    print("\n🔹 Estimating pooled single-class model for initialization...\n")

    # Estimate single class first
    model_single = CountModel(base_spec, data)
    result_single = model_single.fit()

    theta_single = result_single.params

    # Create two slightly perturbed classes
    theta_init = np.concatenate([
        theta_single,
        theta_single + np.random.normal(0, 0.05, size=K_base)
    ])

    logits_init = np.zeros(C - 1)  # equal class shares

    init_params = np.concatenate([theta_init, logits_init])

    # --------------------------------------------------
    # 7️⃣ EM ESTIMATION
    # --------------------------------------------------

    print("\n🚀 Running EM algorithm...\n")

    params_em = fit_em(
        init_params=init_params,
        data=data,
        spec=spec,
        max_iter=50,
        tol=1e-5,
        verbose=True
    )

    print("\n✅ EM Estimation complete.\n")

    # --------------------------------------------------
    # 8️⃣ OPTIONAL: Full MLE polish
    # --------------------------------------------------

    print("🔹 Polishing with full MLE...\n")

    from scipy.optimize import minimize

    result_final = minimize(
        lambda p: mixed_model_loglik(p, data, spec),
        params_em,
        method="L-BFGS-B"
    )

    print("Final log-likelihood:", -result_final.fun)

    # --------------------------------------------------
    # 9️⃣ Print summary
    # --------------------------------------------------

    objective = partial(mixed_model_loglik, data=data, spec=spec)

    print_summary(
        result=result_final,
        objective=objective,
        data=data,
        spec=spec,
        param_index=build_param_index(spec)
    )




def print_summary(result, objective, data, spec, param_index):

    import numpy as np
    import pandas as pd
    from scipy import stats

    def softplus(x):
        return np.log1p(np.exp(x))

    params = np.array(result.params)
    se = np.array(compute_standard_errors(result.params, objective))

    z_vals = params / se
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_vals)))

    final_ll = -objective(result.params)
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

    # Zero inflation
    if spec.Kzi > 0:
        for name in spec.zi_names:
            names.append(f"zi({name})")

    # Correlated means
    for name in spec.random_cor_names:
        names.append(f"cor_mean({name})")

    # Correlated Cholesky
    chol_names = []
    if spec.Kr_cor > 0:
        cols = spec.random_cor_names
        K = spec.Kr_cor
        for i in range(K):
            for j in range(i + 1):
                nm = f"chol({cols[i]},{cols[j]})"
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
        "Estimate": params,
        "Std.Err": se,
        "z-value": z_vals,
        "p-value": p_vals
    })

    # ==========================================================
    # ✅ TRANSFORM PARAMETERS TO TRUE SCALE
    # ==========================================================

    # Independent SDs
    sd_mask = summary_df["Parameter"].str.contains(r"^sd\(")
    summary_df.loc[sd_mask, "Estimate"] = softplus(
        summary_df.loc[sd_mask, "Estimate"]
    )

    # Grouped SDs
    gsd_mask = summary_df["Parameter"].str.contains("group_sd")
    summary_df.loc[gsd_mask, "Estimate"] = softplus(
        summary_df.loc[gsd_mask, "Estimate"]
    )

    # NB dispersion
    if spec.model == "nb":
        disp_mask = summary_df["Parameter"] == "dispersion"
        summary_df.loc[disp_mask, "Estimate"] = softplus(
            summary_df.loc[disp_mask, "Estimate"]
        )

    # ==========================================================
    # PRINT SECTIONS
    # ==========================================================

    def print_section(df, title):
        if len(df) == 0:
            return
        print(f"\n================ {title} ================\n")
        print(df.to_string(index=False))

    # Remove raw chol from main display
    main_df = summary_df[~summary_df["Parameter"].isin(chol_names)]

    fixed_df = main_df[
        main_df["Parameter"].isin(spec.fixed_names)
    ]

    zi_df = main_df[
        main_df["Parameter"].str.contains("zi\\(")
    ]

    cor_mean_df = main_df[
        main_df["Parameter"].str.contains("cor_mean")
    ]

    ind_mean_df = main_df[
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

    print("\n================ MODEL SUMMARY ================\n")

    print_section(fixed_df, "FIXED EFFECTS")
    print_section(zi_df, "ZERO INFLATION")
    print_section(cor_mean_df, "CORRELATED RANDOM MEANS")
    print_section(ind_mean_df, "INDEPENDENT RANDOM MEANS")
    print_section(sd_df, "INDEPENDENT RANDOM SDs")
    print_section(group_df, "GROUPED RANDOM EFFECTS")
    print_section(het_df, "HETEROGENEITY IN MEANS")
    print_section(disp_df, "DISPERSION")

    # ==========================================================
    # ✅ REBUILD TRUE VAR-COV FROM CHOLESKY
    # ==========================================================

    if spec.Kr_cor > 0:

        chol_params = params[
            summary_df["Parameter"].isin(chol_names)
        ]

        K = spec.Kr_cor
        L = np.zeros((K, K))

        idx = 0
        for i in range(K):
            for j in range(i + 1):
                val = chol_params[idx]

                # ✅ exponentiate diagonal
                if i == j:
                    val = np.exp(val)

                L[i, j] = val
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


def print_summary(result, objective, data, spec, param_index, se = None, return_df = None):

    import numpy as np
    import pandas as pd
    from scipy import stats
    
    import numpy as np
    import pandas as pd
    from scipy import stats
    
    if spec.latent_classes > 1:

        C = spec.latent_classes

        base_spec = replace(spec, latent_classes=1)
        base_index = build_base_index(base_spec)
        K_base = base_index["total_params"]

        params = np.asarray(result.x)
        if se is None:
            se = compute_standard_errors(params, objective)
        se_all = se[:C * K_base].reshape(C, K_base)
        logit_se = se[C * K_base:]

        theta_all = params[:C * K_base].reshape(C, K_base)

        logits = params[C * K_base:]
        logits_full = np.concatenate(([0.0], logits))
        pi = np.exp(logits_full) / np.sum(np.exp(logits_full))

        print("\n====================================================")
        print("        LATENT CLASS MIXED MODEL SUMMARY")
        print("====================================================")

        # ---------------------------------------------
        # Print each class separately
        # ---------------------------------------------
        for c in range(C):

            print(f"\n################ CLASS {c+1} ################\n")

            class_result = DummyResult(theta_all[c])
            class_result.params = theta_all[c]

            print_summary(
                result=class_result,
                objective=None,
                data=data,
                spec=base_spec,
                param_index=base_index,
                se=se_all[c] 
            )

        # ---------------------------------------------
        # Print class probabilities
        # ---------------------------------------------
        print("\n================ CLASS PROBABILITIES ================\n")

        for c in range(C):
            print(f"pi_{c+1}: {pi[c]:.6f}")

        print("\n====================================================\n")
        for c in range(C - 1):
            print(f"logit_{c+2}  SE: {logit_se[c]:.6f}")

        return
    

    def softplus(x):
        return np.log1p(np.exp(x))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    params_raw = np.asarray(result.params, dtype=np.float64)
    if se is None:
        se_raw = np.asarray(
            compute_standard_errors(result.params, objective),
            dtype=np.float64
        )
    else:
        se_raw = se

    names = []

    # ==============================
    # BUILD PARAMETER NAME LIST
    # ==============================

    for name in spec.fixed_names:
        names.append(name)

    if spec.Kzi > 0:
        for name in spec.zi_names:
            names.append(f"zi({name})")

    for name in spec.random_cor_names:
        names.append(f"cor_mean({name})")

    chol_names = []
    if spec.Kr_cor > 0:
        cols = spec.random_cor_names
        K = spec.Kr_cor
        for i in range(K):
            for j in range(i + 1):
                nm = f"chol({cols[i]},{cols[j]})"
                names.append(nm)
                chol_names.append(nm)

    for name in spec.random_ind_names:
        names.append(f"mean({name})")
    for name in spec.random_ind_names:
        names.append(f"sd({name})")

    if spec.Kg > 0:
        for name in spec.grouped_names:
            names.append(f"group_mean({name})")
        for name in spec.grouped_names:
            names.append(f"group_sd({name})")

    if spec.Kh > 0:
        for rnd in spec.random_cor_names + spec.random_ind_names:
            for z in spec.hetro_names:
                names.append(f"hetro({rnd}|{z})")

    if spec.model == "nb":
        names.append("dispersion")
    elif spec.model == 'lognormal':
        names.append('lognormal')

    df = pd.DataFrame({
        "Parameter": names,
        "Estimate_raw": params_raw,
        "StdErr_raw": se_raw
    })

    # ==========================================================
    # ✅ APPLY DELTA METHOD WHERE NEEDED
    # ==========================================================

    df["Estimate"] = df["Estimate_raw"]
    df["Std.Err"] = df["StdErr_raw"]

    for i, row in df.iterrows():

        name = row["Parameter"]
        theta = row["Estimate_raw"]
        se_theta = row["StdErr_raw"]

        # Independent SD
        if name.startswith("sd("):
            phi = softplus(theta)
            deriv = sigmoid(theta)

        # Grouped SD
        elif name.startswith("group_sd"):
            phi = softplus(theta)
            deriv = sigmoid(theta)

        # NB dispersion
        elif name == "dispersion":
            phi = softplus(theta)
            deriv = sigmoid(theta)

        # Cholesky diagonal
        elif name.startswith("chol("):
            # detect diagonal terms
            inside = name[5:-1]
            var1, var2 = inside.split(",")
            if var1 == var2:
                phi = np.exp(theta)
                deriv = np.exp(theta)
            else:
                phi = theta
                deriv = 1.0

        else:
            phi = theta
            deriv = 1.0

        df.at[i, "Estimate"] = float(phi)
        df.at[i, "Std.Err"] = float(se_theta * abs(deriv))

    # Recompute z and p
    df["z-value"] = df["Estimate"] / df["Std.Err"]
    df["p-value"] = 2 * (1 - stats.norm.cdf(np.abs(df["z-value"])))

    # ==========================================================
    # CLEAN DISPLAY (remove raw columns)
    # ==========================================================

    df_display = df[["Parameter", "Estimate", "Std.Err", "z-value", "p-value"]]

    # Remove raw chol from main table
    main_df = df_display[~df_display["Parameter"].isin(chol_names)]

    print("\n================ MODEL SUMMARY ================\n")
    print(main_df.to_string(index=False))

    # ==========================================================
    # ✅ TRUE VAR-COV MATRIX FROM TRANSFORMED CHOLESKY
    # ==========================================================

    if spec.Kr_cor > 0:

        K = spec.Kr_cor
        L = np.zeros((K, K))

        chol_rows = df[df["Parameter"].isin(chol_names)]

        idx = 0
        for i in range(K):
            for j in range(i + 1):
                row = chol_rows.iloc[idx]
                L[i, j] = row["Estimate"]   # already transformed
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

    # ==========================================================
    # Fit stats
    # ==========================================================
    if objective is not None:

        final_ll = -objective(result.params)
        k = len(result.params)
        n = data["y"].shape[0]

        aic = 2*k - 2*final_ll
        bic = k * np.log(n) - 2*final_ll

        print("\n------------------------------------------------")
        print(f"Log-Likelihood: {float(final_ll):.4f}")
        print(f"AIC: {float(aic):.4f}")
        print(f"BIC: {float(bic):.4f}")
        print("================================================\n")
        if return_df is not None and return_df:
            return df

def compute_mean_prediction(params, data, spec):

    eta = build_eta(params, data, spec)

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


def generate_master_halton(N, K, R, seed=42,  burn=50):
    sampler = qmc.Halton(d=K, scramble=False, seed=seed)
    sampler.fast_forward(burn)
    u = sampler.random(R)
    z = norm.ppf(u)
    z = z.T
    z = np.tile(z[None, :, :], (N, 1, 1))
    u = sampler.random(N * R)
    z = norm.ppf(u)
    z = z.reshape(N, R, K).swapaxes(1, 2)
    
    
    return jnp.array(z)

class CountModel:

    def __init__(self, spec, data):
        self.spec = spec
        self.data = data
        self.param_index = build_param_index(spec)
        self.params = None

    def objective(self, params):
        return mixed_model_loglik(params, self.data, self.spec)

    '''
    def fit(self):
        init = jnp.zeros(self.param_index["total_params"])
        
        key = jax.random.PRNGKey(0)

        n_params = self.param_index["total_params"]

        # Small random normal noise
        init = 0.001 * jax.random.normal(key, shape=(n_params,))
        solver = LBFGS(fun=self.objective, maxiter =2000, tol=1e-7,
        history_size=20)
        result = solver.run(init)
        self.params = result.params
        return result
    '''
    def fit(self, use_prefit=False):

        n_params = self.param_index["total_params"]

        # --------------------------------------------------
        # 1️⃣ INITIALIZATION
        # --------------------------------------------------
        key = jax.random.PRNGKey(0)
        init = 0.01 * jax.random.normal(key, (n_params,))

        if use_prefit:
            try:
                pre_beta = fit_simple_poisson_full(self.data, self.spec)

                cursor_pre = 0  # position inside pre_beta

                # ✅ Fixed
                if self.spec.Kf > 0:
                    start, end = self.param_index["fixed"]
                    k = self.spec.Kf
                    init = init.at[start:end].set(pre_beta[cursor_pre:cursor_pre+k])
                    cursor_pre += k

                # ✅ Correlated means
                if self.spec.Kr_cor > 0:
                    start, end = self.param_index["cor_mean"]
                    k = self.spec.Kr_cor
                    init = init.at[start:end].set(pre_beta[cursor_pre:cursor_pre+k])
                    cursor_pre += k

                # ✅ Independent means
                if self.spec.Kr_ind > 0:
                    start, end = self.param_index["ind_mean"]
                    k = self.spec.Kr_ind
                    init = init.at[start:end].set(pre_beta[cursor_pre:cursor_pre+k])
                    cursor_pre += k

                # ✅ Grouped means
                if self.spec.Kg > 0:
                    start, end = self.param_index["group_mean"]
                    k = self.spec.Kg
                    init = init.at[start:end].set(pre_beta[cursor_pre:cursor_pre+k])
                    cursor_pre += k

                # Everything else stays zero:
                # sd's
                # chol params
                # heterogeneity
                # dispersion

            except Exception as e:
                print("Prefit failed:", e)
                key = jax.random.PRNGKey(0)
                init = 0.001 * jax.random.normal(key, (n_params,))

        else:
            key = jax.random.PRNGKey(0)
            init = 0.001 * jax.random.normal(key, (n_params,))

        # --------------------------------------------------
        # 2️⃣ OPTIMIZE
        # --------------------------------------------------
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
        if self.spec.model == "gaussian":
            return eta.mean(axis=-1)
        mu = jnp.exp(eta)
        return mu.mean(axis=-1)

    def scipy_trust(self, seed=0):

        key = jax.random.PRNGKey(seed)
        init = 0.01 * jax.random.normal(key, (self.param_index["total_params"],))
        init = np.array(init)

        def obj_np(p):
            return np.array(self.objective(jnp.array(p)))

        def grad_np(p):
            return np.array(jax.grad(self.objective)(jnp.array(p)))

        result = minimize(
            obj_np,
            init,
            method="trust-constr",
            jac=grad_np,
            options={"maxiter": 3000}
        )

        ll = -obj_np(result.x)

        print("\nSCIPY TRUST-CONSTR LL:", ll)
        print("Iterations:", result.niter)

        return result
    
    def scipy_newton_cg(self, seed=0):

        key = jax.random.PRNGKey(seed)
        init = 0.01 * jax.random.normal(key, (self.param_index["total_params"],))
        init = np.array(init)

        def obj_np(p):
            return np.array(self.objective(jnp.array(p)))

        def grad_np(p):
            return np.array(jax.grad(self.objective)(jnp.array(p)))

        def hessp(p, v):
            return np.array(
                jax.jvp(jax.grad(self.objective),
                        (jnp.array(p),),
                        (jnp.array(v),))[1]
            )

        result = minimize(
            obj_np,
            init,
            method="Newton-CG",
            jac=grad_np,
            hessp=hessp,
            options={"maxiter": 3000}
        )

        ll = -obj_np(result.x)

        print("\nSCIPY NEWTON-CG LL:", ll)

        return result

  
    
    

    def test_solvers(self, seed=0):

        key = jax.random.PRNGKey(seed)
        n_params = self.param_index["total_params"]
        init = 0.00 * jax.random.normal(key, (n_params,))

        obj = self.objective
        grad = jax.grad(obj)

        results = {}

        solvers = {
            "LBFGS": LBFGS(fun=obj, maxiter=2000, tol=1e-7),
            "BFGS": BFGS(fun=obj, maxiter=2000, tol=1e-7),
            "NonlinearCG": NonlinearCG(fun=obj, maxiter=2000, tol=1e-7),
            "GradientDescent": GradientDescent(fun=obj, maxiter=5000, stepsize=1e-3),
        }

        for name, solver in solvers.items():
            print(f"\nRunning {name}...")
            res = solver.run(init)

            params = res.params
            ll = obj(params)
            gnorm = jnp.linalg.norm(grad(params))

            results[name] = {
                "loglike": float(ll),
                "grad_norm": float(gnorm),
                "iterations": res.state.iter_num,
            }

            print(f"{name} LL: {ll:.6f}")
            print(f"{name} Grad norm: {gnorm:.6e}")
            print(f"{name} Iterations: {res.state.iter_num}")

        return results





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
        self.structure_cache = set()

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
        self.structure_cache = set()

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
    def structural_signature(self, spec_dict):

        total_random = (
            len(spec_dict["rdm_terms"]) +
            len(spec_dict["rdm_cor_terms"]) +
            len(spec_dict["grouped_terms"])
        )

        # Heterogeneity only matters if random parameters exist
        if total_random > 0:
            hetero_effective = tuple(sorted(spec_dict["hetro_in_means"]))
        else:
            hetero_effective = tuple()  # collapse heterogeneity

        return (
            tuple(sorted(spec_dict["fixed_terms"])),
            tuple(sorted(spec_dict["rdm_terms"])),
            tuple(sorted(spec_dict["rdm_cor_terms"])),
            tuple(sorted(spec_dict["grouped_terms"])),
            hetero_effective,
            tuple(sorted(spec_dict["zi_terms"])),
            spec_dict["dispersion"]
        )

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
        sig = self.structural_signature(spec_dict)

        if sig in self.structure_cache:
            return np.array([1e12, 1e12]) if self.mode=="multi" else 1e12
        
        self.structure_cache.clear()
        self.structure_cache.add(sig)
        try:

            # ✅ TRAIN
            data_train, spec = self.build_data(
                self.df_train,
                spec_dict,
                self.master_halton_train
            )

            model = CountModel(spec, data_train)
            model.fit()
           # print("Expected total params:", model.param_index["total_params"])
           # print("Actual total params:", len(model.params))

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
        elif spec.model in {"lognormal", "gaussian"}:
            out["sigma"] = params[idx]
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
            5: "Heterogeneity",
            6: "Zero Inflated"
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
            print("Expected total params:", param_index["total_params"])
            print("Actual total params:", len(result.params))
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
                           default_roles=None, grouped =  True):

    if default_roles is None:
        if grouped:
            default_roles = [0,1,2,3,4,5,6]
        else:
            default_roles = [0,1,2,3,5,6]
                

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
        default_dist = ["normal", 'uniform', 'lognormal', 'triangular']

    full = {}

    for var in all_variables:
        if allowed_dist_partial is None:
            full[var] = default_dist
        else:    
        
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
    engine.save_search_stats_txt(
    algo=operator.__class__.__name__,
    seed=seed,
    config_id=seed
    )
    engine.finalize_plots(
    algo=operator.__class__.__name__,
    seed=seed
    )
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

        scores = np.atleast_1d(np.array(result["scores"]))

        # Single objective case
        if scores.ndim == 1:
            summary["num_solutions"] = scores.shape[0]
            summary["best_score"] = float(np.min(scores))
            return summary

        # Multi-objective case
        summary["num_solutions"] = scores.shape[0]
        summary["pareto_size"] = scores.shape[0]
        summary["best_bic"] = float(np.min(scores[:, 0]))
        summary["best_rmse"] = float(np.min(scores[:, 1]))

        if "hypervolume_history" in result:
            hv = result["hypervolume_history"]
            summary["final_hypervolume"] = (
                float(hv[-1]) if len(hv) > 0 else None
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

    return result["bic"].iloc[0], result["rmse"].iloc[0]

def run_r_experiment(df_train, df_test, vars, r_path ='step_aic_r'):
    import subprocess
    import tempfile
    import os
    #step_aic_r, dredge_test, glmnet_poisson
    r_exe = r"C:\Users\ahernz\AppData\Local\Programs\R\R-4.4.1\bin\x64\Rscript.exe"
    r_script = os.path.abspath(f"{r_path}.R")

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
    return result['bic'].iloc[0], result['rmse'].iloc[0]
    return result["train_bic"].iloc[0], result["test_rmse"].iloc[0]

def get_best_index(scores):
    if scores.ndim == 1:
        return np.argmin(scores)
    return np.argmin(scores[:, 0])



def experiment_washington():
    df = pd.read_csv('./data/Ex-16-3.csv')
    print("Unique IDs:", df["ID"].nunique())
    print("Total rows:", len(df))
    df['EXPOSE'] = df['LENGTH']*df['AADT']*365/100000000
    df['OFFSET'] = 0
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
        all_variables, None
      
    )
    #  {"CURVES": ["normal", "lognormal"],
       #  "FRICTION": ["normal", "triangular"]}
    allowed_roles = populate_allowed_roles(all_variables, {"EXPOSE": [1]},
                                                           default_roles=[0,1,2,3,5])

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
    return evaluator, df, all_variables

def experiment_synthetic(seed = 0):
    N_ids = 1000
    T = 4 
    r= seed
    df, true_params = generate_panel_data(
            N_ids=N_ids,
            T=T,
            seed=r
        )
    all_var = df.columns.to_list()
    all_variables = [col for col in all_var if col not in ['ID', 'Y']]
    allowed_distributions = populate_allowed_distributions(
        all_variables, None
    )

    allowed_roles = populate_allowed_roles(all_variables, dict(), default_roles=[0,1,2,3,5])
    evaluator = StructureEvaluator(
        df=df,
        id_col="ID",
        y_col="Y",
        offset_col=None,
        all_variables=all_variables,
        allowed_roles=allowed_roles,
        allowed_distributions=allowed_distributions,
        group_id_col=None,
        mode="multi",
        R=200
    )
    return evaluator, df, all_variables
    

def strip_distribution(spec_dict):

    def strip(terms):
        return sorted([t.split(":")[0] for t in terms])

    return {
        "fixed_terms": sorted(spec_dict["fixed_terms"]),
        "rdm_terms": strip(spec_dict["rdm_terms"]),
        "rdm_cor_terms": strip(spec_dict["rdm_cor_terms"]),
        "grouped_terms": strip(spec_dict["grouped_terms"]),
        "hetro_in_means": sorted(spec_dict["hetro_in_means"]),
        "zi_terms": sorted(spec_dict["zi_terms"]),
        "dispersion": spec_dict["dispersion"]
    }


def main(seed, algo, config, max_iter, n_jobs, config_id, experiment):
    #monte_carlo_recovery()
    start_time = time.time()

    print("\n==============================")
    print("Running experiment:")
    print("Algo:", algo)
    print("Seed:", seed)
    print("Config ID:", config_id)
    print("Config:", config)
    print("Experiment:", experiment)
    print("==============================\n")

    # ------------------------------------------------
    # Build evaluator
    # ------------------------------------------------
    if experiment == "ex_wash":
        evaluator, df, all_variables = experiment_washington()

    elif experiment == "ex_synth":
        evaluator, df, all_variables = experiment_synthetic(seed=seed)

    else:
        raise ValueError("Unknown experiment")

    # ==========================================================
    # 🔵 MULTI-OBJECTIVE (NSGA2)
    # ==========================================================
    if algo in ["de", "hs"]:

        if algo == "de":
            operator = AdaptiveDE(
                F=config["F"],
                CR=config["CR"]
            )

        elif algo == "hs":
            operator = DynamicHarmony(
                hmcr=config["hmcr"],
                par_min=config["par_min"],
                par_max=config["par_max"],
                bw_min=config["bw_min"],
                bw_max=config["bw_max"]
            )

        result = run_nsga(
            evaluator=evaluator,
            operator=operator,
            seed=seed,
            pop_size=config["population_size"],
            max_iter=max_iter,
            n_jobs=n_jobs
        )
        print_from_nsga_result(result, evaluator)
        if evaluator.mode == "multi":
            solutions = np.array(result["solutions"])
            scores = np.array(result["scores"])

            # Recompute Pareto just to be safe
            pareto_solutions, pareto_scores = extract_pareto_front(
                solutions,
                scores
            )

            print("\n================ NSGA FINAL PARETO FRONT ================\n")

            for i in range(len(pareto_solutions)):
                print(
                    f"{i} | BIC={pareto_scores[i][0]:.4f} "
                    f"| RMSE={pareto_scores[i][1]:.4f}"
                )

            save_pareto_front(
                pareto_solutions,
                pareto_scores,
                evaluator,
                algo,
                seed,
                config_id
            )

            result = {
                "algorithm": algo,
                "seed": seed,
                "solutions": pareto_solutions,
                "scores": pareto_scores
            }
            # --------------------------------------------
            # ✅ TRUE STRUCTURE CHECK (only synthetic)
            # --------------------------------------------
            if experiment == "ex_synth":

                # define the TRUE synthetic structure
                true_spec_dict = {
                    "fixed_terms": ["shoulder", "log_aadt", "length", "rural"],
                    "rdm_terms": [],
                    "rdm_cor_terms": [
                        "speed:normal",
                        "lanes:normal"
                    ],
                    "grouped_terms": [],
                    "hetro_in_means": [],
                    "zi_terms": [],
                    "dispersion": 1
                }

                recovered, idx = check_structure_recovery(
                    pareto_solutions,
                    evaluator,
                    true_spec_dict
                )

                save_structure_recovery_file(
                    recovered,
                    algo,
                    seed,
                    config_id
                )

                if recovered:
                    print("\n✅ TRUE STRUCTURE FOUND IN PARETO FRONT")

                    true_decision = pareto_solutions[idx]

                    refit_and_save_true_model(
                        evaluator,
                        true_decision,
                        algo,
                        seed,
                        config_id
                    )

    # ==========================================================
    # 🔴 SINGLE OBJECTIVE (MultiStartSA)
    # ==========================================================
    elif algo in ["sa", "hc"]:

        solver = MultiStartSA(
            evaluator=evaluator,
            dimension=2 * len(evaluator.vars) + 1,
            **config
        )

        solutions, scores = solver.optimize()

        solutions = np.array(solutions)
        scores = np.array(scores)

        # --------------------------------------------
        # MULTI OBJECTIVE
        # --------------------------------------------
        if evaluator.mode == "multi":

            pareto_solutions, pareto_scores = extract_pareto_front(
                solutions,
                scores
            )

            print("\n================ SA FINAL PARETO FRONT ================\n")

            for i in range(len(pareto_solutions)):
                print(
                    f"{i} | BIC={pareto_scores[i][0]:.4f} "
                    f"| RMSE={pareto_scores[i][1]:.4f}"
                )

            save_pareto_front(
                pareto_solutions,
                pareto_scores,
                evaluator,
                algo,
                seed,
                config_id
            )

            result = {
                "algorithm": algo,
                "seed": seed,
                "solutions": pareto_solutions,
                "scores": pareto_scores
            }

        # --------------------------------------------
        # SINGLE OBJECTIVE
        # --------------------------------------------
        else:

            best_idx = np.argmin(scores)
            best_solution = solutions[best_idx]
            best_score = scores[best_idx]

            print("\n================ BEST SA STRUCTURE ================\n")
            decode_best_solution(best_solution, evaluator)
            print("Best Score:", best_score)

            refit_and_print(evaluator, best_solution)

            result = {
                "algorithm": algo,
                "seed": seed,
                "solutions": solutions,
                "scores": scores,
                "best_solution": best_solution,
                "best_score": best_score
            }

    else:
        raise ValueError("Unknown algorithm")

    # ------------------------------------------------
    # Save + log
    # ------------------------------------------------
    runtime = time.time() - start_time

    summary = summarize_experiment(
        result=result,
        config=config,
        runtime=runtime,
        config_id=config_id
    )

    append_experiment_to_csv(summary)

    return result



def main_old(seed, algo, pop_size, max_iter, n_jobs, config_id, experiment):
#def main(seed, algo, config, max_iter, n_jobs, config_id, experiment):
    
    start_time = time.time()
    print("Running experiment:")
    print("Algo:", algo)
    print("Config:", config_id)
    print("Seed:", seed)

    # ----------------------------
    # Generate grids
    # ----------------------------

    HS_GRID = {
        "population_size": [30, 40, 20],
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
    if experiment == 'ex_wash':
        evaluator, df, all_variables = experiment_washington()
    elif experiment == 'ex_synth':
        evaluator, df, all_variables = experiment_synthetic()
    elif experiment == 'testing':
        monte_carlo_recovery()
        run_full_synthetic_recovery_experiment()
    '''
    df = pd.read_csv('./data/Ex-16-3.csv')
    print("Unique IDs:", df["ID"].nunique())
    print("Total rows:", len(df))
    df['EXPOSE'] = df['LENGTH']*df['AADT']*365/100000000
    df['OFFSET'] = 0
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

    allowed_roles = populate_allowed_roles(all_variables, {"EXPOSE": [1],
                                                           "CPM":[6], 'SPEED':[6]
                                                           })

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
        R=1000
    )
    '''

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
    if algo == "nlogit_replicate":
        run_nlogit_ex16_model(df, R=200)
        return
        

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


def benchmarks(seed = 0, case = 1):
    
    start_time = time.time()
    evaluator, df, all_variables = experiment_washington()
    df_train, df_test, df_val = build_datasets(df, "ID", "Y")
    if case == 0:
        train_bic, test_rmse = run_glmulti_experiment(df, df_test, all_variables)
    else:
        
        ## STEP AIC
        train_bic, test_rmse = run_r_experiment(df, df_test, all_variables)
    
        runtime = time.time() - start_time

        summary = {
                "algorithm": "step_aic",
                "seed": seed,
                "train_bic": float(train_bic),
                "test_rmse": float(test_rmse),
                "runtime_seconds": runtime
            }

        append_experiment_to_csv(summary)
        print("\n================ step bic ================\n")
        print("BIC:", train_bic)
        ###  DREDDGE
        train_bic, test_rmse = run_r_experiment(df, df_test, all_variables, 'glmnet_poisson')
    
        runtime = time.time() - start_time

        summary = {
                "algorithm": "step_aic",
                "seed": seed,
                "train_bic": float(train_bic),
                "test_rmse": float(test_rmse),
                "runtime_seconds": runtime
            }

        append_experiment_to_csv(summary)

        print("\n================ s DREDDGE ================\n")
        print("BIC:", train_bic)
        
        
        ###  glmnet
        train_bic, test_rmse = run_glmulti_experiment(df, df_test, all_variables)
        #train_bic, test_rmse = run_r_experiment(df, df_test, all_variables, 'dlmnet_poisson')
    
        runtime = time.time() - start_time

        summary = {
                "algorithm": "step_aic",
                "seed": seed,
                "train_bic": float(train_bic),
                "test_rmse": float(test_rmse),
                "runtime_seconds": runtime
            }

        append_experiment_to_csv(summary)

        print("\n================ s DREDDGE ================\n")
        print("BIC:", train_bic)
        

    

def run_nlogit_ex16_model(df, R=800, seed=42):
    """
    Reproduces the NLOGIT Ex16-2 random-parameter NB model:

    Fixed:
        LOWPRE, GBRPM, FRICTION

    Random (independent normal):
        EXPOSE, INTPM, CPM, HISNOW

    Negative binomial
    200 Halton draws
    """

    print("\n================ RUNNING NLOGIT EQUIVALENT MODEL ================\n")

    # ----------------------------------------------------
    # 1️⃣ Define manual spec exactly matching NLOGIT
    # ----------------------------------------------------

    manual_spec = {
        "fixed_terms": [
            "LOWPRE",
            "GBRPM",
             "HISNOW",
            "FRICTION",
        ],
        "rdm_terms": [
            "EXPOSE:normal",
            "INTPM:normal",
            "CPM:normal",
        ],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": [],
        "zi_terms": [],
        "dispersion": 1  # Negative Binomial
    }

    # ----------------------------------------------------
    # 2️⃣ Generate Halton draws
    # ----------------------------------------------------

    N = df["ID"].nunique()
    K_random = 3

    draws_ind = generate_halton_normal(
        N=N,
        K=K_random,
        R=R,
        seed=seed
    )

    # ----------------------------------------------------
    # 3️⃣ Build JAX data + spec
    # ----------------------------------------------------

    data, spec = build_model_from_manual_spec(
        df=df,
        manual_spec=manual_spec,
        id_col="ID",
        y_col="Y",           # already renamed in your main
        offset_col=None,     # EXPOSE is a regressor, not offset
        draws_ind=draws_ind,
        draws_cor=None,
        draws_g=None,
        R=R
    )

    # ----------------------------------------------------
    # 4️⃣ Fit model
    # ----------------------------------------------------

    model = CountModel(spec, data)
    result = model.fit()
    #solver_results = model.test_solvers(seed=seed)
    
    

# Pick best solver automatically
    #best_solver = min(solver_results, key=lambda k: solver_results[k]["loglike"])

    #print(f"\n✅ Best solver: {best_solver}")
    #print(f"Best LogLik: {solver_results[best_solver]['loglike']:.6f}")
    #result = model.fit()

    #print("\n✅ Model estimation complete.")

    # ----------------------------------------------------
    # 5️⃣ Print full summary
    # ----------------------------------------------------

    objective = partial(mixed_model_loglik, data=data, spec=spec)
    param_index = build_param_index(spec)

    print_summary(
        result=result,
        objective=objective,
        data=data,
        spec=spec,
        param_index=param_index
    )

    # ----------------------------------------------------
    # 6️⃣ Return everything
    # ----------------------------------------------------

    return result, spec, data


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


def fit_simple_poisson_full(data, spec):
    """
    Prefit Poisson treating ALL regressors
    (fixed + random + grouped) as fixed.
    """

    blocks = []

    if spec.Kf > 0:
        blocks.append(data["Xf"])

    if spec.Kr_cor > 0:
        blocks.append(data["Xr_cor"])

    if spec.Kr_ind > 0:
        blocks.append(data["Xr_ind"])

    if spec.Kg > 0:
        blocks.append(data["Xg"])

    if len(blocks) == 0:
        return None

    X_all = jnp.concatenate(blocks, axis=2)

    y = data["y"]
    mask = data["mask"]

    K_total = X_all.shape[2]

    # Flatten
    X_flat = X_all.reshape(-1, K_total)
    y_flat = y.reshape(-1)
    mask_flat = mask.reshape(-1)

    valid = mask_flat == 1
    X_flat = X_flat[valid]
    y_flat = y_flat[valid]

    def poisson_objective(beta):
        eta = X_flat @ beta
        mu = jnp.exp(jnp.clip(eta, -15, 15))
        ll = y_flat * jnp.log(mu + 1e-12) - mu - jsp.special.gammaln(y_flat + 1)
        return -jnp.sum(ll)

    init = jnp.zeros(K_total)

    solver = LBFGS(fun=poisson_objective, maxiter=500)
    result = solver.run(init)

    return result.params

def fit_simple_poisson(data, spec):
    """
    Prefit Poisson using only fixed effects (Xf).
    Does NOT rebuild spec.
    """

    Xf = data["Xf"]            # (N, P, Kf)
    y = data["y"]              # (N, P, 1)
    mask = data["mask"]        # (N, P)

    Kf = spec.Kf

    # Flatten panel
    Xf_flat = Xf.reshape(-1, Kf)
    y_flat = y.reshape(-1)
    mask_flat = mask.reshape(-1)

    valid = mask_flat == 1
    Xf_flat = Xf_flat[valid]
    y_flat = y_flat[valid]

    def poisson_objective(beta):
        eta = Xf_flat @ beta
        mu = jnp.exp(jnp.clip(eta, -15, 15))
        ll = y_flat * jnp.log(mu + 1e-12) - mu - jsp.special.gammaln(y_flat + 1)
        return -jnp.sum(ll)

    init = jnp.zeros(Kf)

    solver = LBFGS(fun=poisson_objective, maxiter=500)
    result = solver.run(init)

    return result.params


def build_master_grid(seeds):

    HS_GRID = {
        "population_size": [30, 40, 20],
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
        "max_iter": [7000],
        "mutation_rate": [0.5, 0.3, 0.6],
        "step_size": [5, 2, 3],
        "alpha": [0.99, 0.9],
        "n_starts": [1, 2]
    }

    HC_GRID = {
        "mutation_rate": [0.5],
        "step_size": [1],
        "min_changes": [1],
        "max_changes": [1],
        "n_starts": [1]
    }

    ALL_GRIDS = {
        "sa": SA_GRID,
        "de": DE_GRID,
        "hs": HS_GRID,
        "hc": HC_GRID
    }

    experiments = ["ex_wash", "ex_synth"]

    master = []

    for exp in experiments:
        for algo, grid in ALL_GRIDS.items():
            configs = generate_configs(grid)
            for config_id, config in enumerate(configs):
                for seed in seeds:
                    master.append({
                        "experiment": exp,
                        "algorithm": algo,
                        "config_id": config_id,
                        "config": config,
                        "seed": seed
                    })

    return master


def e_step(params, data, spec):

    C = spec.latent_classes
    base_spec = replace(spec, latent_classes=1)

    base_index = build_base_index(base_spec)
    K = base_index["total_params"]

    params = np.asarray(params)

    theta_all = params[:C*K].reshape(C, K)

    logits = params[C*K:]
    logits_full = np.concatenate(([0.0], logits))
    pi = np.exp(logits_full) / np.sum(np.exp(logits_full))

    N = data["N_ids"]

    logL = np.zeros((N, C))

    for c in range(C):

        logL[:, c] = mixed_model_loglik_individual(
            theta_all[c],
            data,
            base_spec
        )

    # log posterior numerator
    log_num = logL + np.log(pi)

    # stabilize
    max_log = log_num.max(axis=1, keepdims=True)
    w = np.exp(log_num - max_log)
    w /= w.sum(axis=1, keepdims=True)

    return w, pi

def weighted_objective(theta, data, spec, weights):

    logL_i = mixed_model_loglik_individual(theta, data, spec)

    return -np.sum(weights * logL_i)

import numpy as np
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from dataclasses import replace


def fit_em(init_params, data, spec, max_iter=100, tol=1e-6, verbose=True):

    assert spec.latent_classes > 1, "EM only needed for latent classes"

    C = spec.latent_classes
    base_spec = replace(spec, latent_classes=1)

    # Build param index once
    base_index = build_param_index(base_spec)
    K_base = base_index["total_params"]

    params = np.array(init_params)
    test_ll = mixed_model_loglik(
    params[:K_base],
    data,
    base_spec,
    indivi=True
    )
    
    N = test_ll.shape[0]

    for iteration in range(max_iter):

        params_old = params.copy()

        # ==========================================================
        # E-STEP
        # ==========================================================

        theta_all = params[:C * K_base].reshape(C, K_base)
        logits = params[C * K_base:]

        logits_full = np.concatenate(([0.0], logits))
        pi = np.exp(logits_full)
        pi /= pi.sum()

        logL = np.zeros((N, C))

        for c in range(C):

            ll_ind = mixed_model_loglik(
                theta_all[c],
                data,
                base_spec,
                indivi=True
            )

            logL[:, c] = np.array(ll_ind)
        #todo bug here, in FIT_EM
        log_num = logL + np.log(pi)

        # stabilize
        max_log = log_num.max(axis=1, keepdims=True)
        w = np.exp(log_num - max_log)
        w /= w.sum(axis=1, keepdims=True)

        # ==========================================================
        # M-STEP
        # ==========================================================

        # ✅ Update class probabilities
        pi_new = w.mean(axis=0)

        logits_new = np.log(pi_new[1:] / pi_new[0])

        # ✅ Update class-specific parameters
        theta_new = []

        for c in range(C):

            weights_c = w[:, c]

            def weighted_objective(theta_c):

                ll_ind = mixed_model_loglik(
                    theta_c,
                    data,
                    base_spec,
                    indivi=True
                )

                return -np.sum(weights_c * np.array(ll_ind))

            result = minimize(
                weighted_objective,
                theta_all[c],
                method="L-BFGS-B"
            )

            theta_new.append(result.x)

        theta_new = np.concatenate(theta_new)

        params = np.concatenate([theta_new, logits_new])

        # ==========================================================
        # Convergence Check
        # ==========================================================

        diff = np.max(np.abs(params - params_old))

        if verbose:
            total_ll = mixed_model_loglik(params, data, spec)
            print(f"EM iter {iteration:3d} | max Δ = {diff:.3e} | LL = {-total_ll:.6f}")

        if diff < tol:
            if verbose:
                print(f"\n✅ EM converged in {iteration} iterations\n")
            break

    return params

def evaluate_true_spec_synthetic(
    N_ids=1000,
    T=4,
    R=200,
    seed=0
):
    """
    1) Generate synthetic data from true DGP
    2) Perform panel-level train/test split
    3) Estimate TRUE model specification on train
    4) Compute:
        - Train BIC
        - Test RMSE
    """

    from functools import partial
    from dataclasses import replace
    import numpy as np

    print("\n===================================================")
    print(" TRUE SPEC EVALUATION (SYNTHETIC DATA)")
    print("===================================================\n")

    # ==========================================================
    # 1️⃣ Generate synthetic panel data
    # ==========================================================

    df, true_params = generate_panel_data(
        N_ids=N_ids,
        T=T,
        seed=seed
    )

    print("Generated data shape:", df.shape)
    print("Unique individuals:", df["ID"].nunique())

    # ==========================================================
    # 2️⃣ Panel train/test split (stratified)
    # ==========================================================

    df_train, df_test, df_val = build_datasets(
        df,
        id_col="ID",
        y_col="Y"
    )

    print("\nSplit sizes:")
    print("Train IDs:", df_train["ID"].nunique())
    print("Test IDs :", df_test["ID"].nunique())
    print("Val IDs  :", df_val["ID"].nunique())

    # ==========================================================
    # 3️⃣ TRUE SPECIFICATION
    # ==========================================================

    manual_spec = {
        "fixed_terms": [
            "shoulder",
            "log_aadt",
            "length",
            "rural"
        ],
        "rdm_terms": [],
        "rdm_cor_terms": [
            "speed:normal",
            "lanes:normal"
        ],
        "grouped_terms": [],
        "hetro_in_means": [],
        "zi_terms": [],
        "dispersion": 1
    }

    # ==========================================================
    # 4️⃣ Generate Halton draws
    # ==========================================================

    N_train = df_train["ID"].nunique()
    N_test  = df_test["ID"].nunique()

    draws_train = generate_halton_normal(
        N=N_train,
        K=2,
        R=R,
        seed=seed
    )

    draws_test = generate_halton_normal(
        N=N_test,
        K=2,
        R=R,
        seed=seed
    )

    # ==========================================================
    # 5️⃣ Build training model
    # ==========================================================

    data_train, spec = build_model_from_manual_spec(
        df=df_train,
        manual_spec=manual_spec,
        id_col="ID",
        y_col="Y",
        draws_cor=draws_train,
        R=R
    )

    spec = replace(spec, model="nb")

    model = CountModel(spec, data_train)

    print("\nEstimating TRUE model on training data...\n")

    result = model.fit()
    param_index = build_param_index(spec)
    
   
    # ==========================================================
    # 6️⃣ Compute Train BIC
    # ==========================================================

    objective = partial(mixed_model_loglik, data=data_train, spec=spec)
    print_summary(
    result=result,
    objective=objective,
    data=data_train,
    spec=spec,
    param_index=param_index
    )  
    
    
    train_ll = -objective(result.params)
    k = len(result.params)
    n_train = data_train["y"].shape[0]

    train_bic = float(k * np.log(n_train) - 2 * train_ll)

    print("Train LogLik:", float(train_ll))
    print("Train BIC:", train_bic)

    # ==========================================================
    # 7️⃣ Test Evaluation
    # ==========================================================

    data_test, _ = build_model_from_manual_spec(
        df=df_test,
        manual_spec=manual_spec,
        id_col="ID",
        y_col="Y",
        draws_cor=draws_test,
        R=R
    )
     

    model_test = CountModel(spec, data_test)
    model_test.params = result.params

    preds = model_test.predict()
    y_true = np.array(data_test["y"]).squeeze()

    test_rmse = float(np.sqrt(np.mean((preds - y_true)**2)))

    print("Test RMSE:", test_rmse)

    print("\n===================================================")
    print(" FINAL RESULTS")
    print("===================================================")
    print("Train BIC :", train_bic)
    print("Test RMSE :", test_rmse)
    print("===================================================\n")
    print("\n===================================================")
    print(" FINAL RESULTS")
    print("===================================================")
    print("Train BIC :", train_bic)
    print("Test RMSE :", test_rmse)
    print("===================================================\n")

    # ==========================================================
    # 9️⃣ Print TRUE parameter values for comparison
    # ==========================================================

    print("\n================ TRUE PARAMETERS ================\n")
    for k, v in true_params.items():
        print(f"{k:20s}: {v}")

    print("\n=================================================\n")
    
    
    return {
        "train_bic": train_bic,
        "test_rmse": test_rmse,
        "train_loglik": float(train_ll)
    }


def run_lognormal_duration_full_demo(
    N_ids=800,
    T=3,
    R=200,
    seed=0
):
    """
    FULLY INTEGRATED PIPELINE

    1) Generate synthetic duration panel data
    2) Build mixed lognormal model
    3) Estimate model
    4) Compute predictions
    5) Print fit statistics
    6) Plot Actual vs Predicted
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import jax
    import jax.numpy as jnp
    from functools import partial
    from dataclasses import replace

    np.random.seed(seed)

    print("\n===================================================")
    print("   MIXED LOGNORMAL DURATION MODEL - FULL DEMO")
    print("===================================================\n")

    # ==========================================================
    # 1️⃣ DATA GENERATION
    # ==========================================================

    def generate_data():

        beta_0 = 1.0
        beta_income = 0.03
        beta_distance = -0.15

        mu_speed = 0.02
        sd_speed = 0.05

        sigma_true = 0.4

        rows = []

        for i in range(N_ids):

            beta_speed_i = np.random.normal(mu_speed, sd_speed)

            income = np.random.normal(50, 10)
            distance = np.random.uniform(1, 20)

            for t in range(T):

                speed = np.random.uniform(30, 80)

                eta = (
                    beta_0
                    + beta_income * income
                    + beta_distance * distance
                    + beta_speed_i * speed
                )

                duration = np.exp(
                    np.random.normal(eta, sigma_true)
                )

                rows.append([
                    i,
                    duration,
                    income,
                    distance,
                    speed
                ])

        df = pd.DataFrame(rows, columns=[
            "ID",
            "DURATION",
            "income",
            "distance",
            "speed"
        ])

        true_params = {
            "Intercept": beta_0,
            "Income": beta_income,
            "Distance": beta_distance,
            "Mean Speed": mu_speed,
            "SD Speed": sd_speed,
            "Sigma": sigma_true
        }

        return df, true_params

    df, true_params = generate_data()

    print("Generated data shape:", df.shape)

    # ==========================================================
    # 2️⃣ MODEL SPECIFICATION
    # ==========================================================

    manual_spec = {
        "fixed_terms": ["income", "distance"],
        "rdm_terms": ["speed:normal"],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": [],
        "zi_terms": [],
        "dispersion": 0
    }

    draws_ind = generate_halton_normal(
        N=N_ids,
        K=1,
        R=R,
        seed=seed
    )

    data, spec = build_model_from_manual_spec(
        df=df,
        manual_spec=manual_spec,
        id_col="ID",
        y_col="DURATION",
        draws_ind=draws_ind,
        R=R
    )

    spec = replace(spec, model="lognormal")

    # ==========================================================
    # 3️⃣ ESTIMATE MODEL
    # ==========================================================

    model = CountModel(spec, data)
    result = model.fit()

    objective = partial(mixed_model_loglik, data=data, spec=spec)
    param_index = build_param_index(spec)

    print_summary(
        result=result,
        objective=objective,
        data=data,
        spec=spec,
        param_index=param_index
    )

    # ==========================================================
    # 4️⃣ PREDICTION FUNCTION
    # ==========================================================

    def compute_predictions(result, data, spec):

        params = result.params
        blocks = unpack_params(params, spec)

        beta = blocks["beta_f"]

        eta = data["Xf"] @ beta

        # Add mean random component
        if blocks["mean_ind"] is not None:
            eta += data["Xr_ind"] @ blocks["mean_ind"]

        if blocks.get("mean_cor") is not None and data.get("Xr_cor") is not None:
            eta += data["Xr_cor"] @ blocks["ean_cor"]
        
        sigma = jax.nn.softplus(blocks["sigma"])

        # ✅ Proper mean of lognormal
        pred_mean = jnp.exp(eta + 0.5 * sigma**2)

        return np.array(pred_mean)

    # ==========================================================
    # 5️⃣ EVALUATION
    # ==========================================================

    y_actual = np.array(data["y"])
    y_pred = compute_predictions(result, data, spec)[:,:, None]

    rmse = np.sqrt(np.mean((y_actual - y_pred)**2))
    mae = np.mean(np.abs(y_actual - y_pred))
    corr = np.corrcoef(y_actual, y_pred)[0, 1]

    print("\n================ MODEL FIT =================\n")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"Corr : {corr:.4f}")

    comparison_df = pd.DataFrame({
        "Actual": y_actual[:10],
        "Predicted": y_pred[:10]
    })

    print("\nFirst 10 Observations:")
    print(comparison_df)

    # ==========================================================
    # 6️⃣ PLOT
    # ==========================================================

    plt.figure(figsize=(6,6))
    plt.scatter(y_actual, y_pred, alpha=0.4)

    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())

    plt.plot([min_val, max_val],
             [min_val, max_val],
             'r--')

    plt.xlabel("Actual Duration")
    plt.ylabel("Predicted Duration")
    plt.title("Actual vs Predicted (Mixed Lognormal)")
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # 7️⃣ TRUE PARAMS PRINT
    # ==========================================================

    print("\n================ TRUE PARAMETERS ================\n")
    for k, v in true_params.items():
        print(f"{k:15s}: {v}")

    print("\n===================================================\n")

    return result, spec, data, y_pred




def run_lognormal_duration_demo(N_ids=800, T=3, R=200, seed=0):
    """
    FULL PIPELINE:
    1) Generate synthetic duration panel data
    2) Build mixed lognormal duration model
    3) Estimate
    4) Print results
    """

    import numpy as np
    import pandas as pd
    from functools import partial
    from dataclasses import replace

    np.random.seed(seed)

    print("\n==============================================")
    print("RUNNING LOGNORMAL DURATION MODEL DEMO")
    print("==============================================\n")

    # ==========================================================
    # TRUE PARAMETERS
    # ==========================================================

    beta_0 = 1.0
    beta_income = 0.03
    beta_distance = -0.15

    mu_speed = 0.02
    sd_speed = 0.05

    sigma_true = 0.4   # lognormal variance

    # ==========================================================
    # GENERATE PANEL DATA
    # ==========================================================

    rows = []

    for i in range(N_ids):

        # random coefficient per individual
        beta_speed_i = np.random.normal(mu_speed, sd_speed)

        income = np.random.normal(50, 10)
        distance = np.random.uniform(1, 20)

        for t in range(T):

            speed = np.random.uniform(30, 80)

            eta = (
                beta_0
                + beta_income * income
                + beta_distance * distance
                + beta_speed_i * speed
            )

            # lognormal duration
            duration = np.exp(
                np.random.normal(eta, sigma_true)
            )

            rows.append([
                i,
                duration,
                income,
                distance,
                speed
            ])

    df = pd.DataFrame(rows, columns=[
        "ID",
        "DURATION",
        "income",
        "distance",
        "speed"
    ])

    print("Generated data shape:", df.shape)

    # ==========================================================
    # MODEL SPECIFICATION
    # ==========================================================

    manual_spec = {
        "fixed_terms": ["income", "distance"],
        "rdm_terms": ["speed:normal"],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": [],
        "zi_terms": [],
        "dispersion": 0
    }

    # ==========================================================
    # HALTON DRAWS
    # ==========================================================

    draws_ind = generate_halton_normal(
        N=N_ids,
        K=1,
        R=R,
        seed=seed
    )

    # ==========================================================
    # BUILD MODEL
    # ==========================================================

    data, spec = build_model_from_manual_spec(
        df=df,
        manual_spec=manual_spec,
        id_col="ID",
        y_col="DURATION",
        draws_ind=draws_ind,
        R=R
    )

    # 🔥 IMPORTANT: Set model to lognormal
    spec = replace(spec, model="lognormal")

    # ==========================================================
    # FIT MODEL
    # ==========================================================

    model = CountModel(spec, data)
    result = model.fit()

    # ==========================================================
    # PRINT SUMMARY
    # ==========================================================

    objective = partial(mixed_model_loglik, data=data, spec=spec)
    param_index = build_param_index(spec)

    print_summary(
        result=result,
        objective=objective,
        data=data,
        spec=spec,
        param_index=param_index
    )

    # ==========================================================
    # PRINT TRUE VALUES
    # ==========================================================

    print("\n================ TRUE PARAMETERS ================\n")
    print("Intercept:", beta_0)
    print("Income:", beta_income)
    print("Distance:", beta_distance)
    print("Mean Speed (random):", mu_speed)
    print("SD Speed:", sd_speed)
    print("Sigma (lognormal):", sigma_true)

    print("\n==============================================\n")

    return result, spec, data

if __name__ == '__main__':
    estimate_latent_class_mixed_example()
    #evaluate_true_spec_synthetic()
    #run_lognormal_duration_full_demo()
    #benchmarks()
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_index", type=int, default=0)
    args = parser.parse_args()

    SEEDS = [0]

    master_grid = build_master_grid(SEEDS)
    print("Total jobs:", len(master_grid))
    job = master_grid[args.job_index]

    print("\n================================================")
    print("Running job:", args.job_index)
    print(job)
    print("================================================\n")
    main(
    seed=job["seed"],
    algo=job["algorithm"],
    config=job["config"],
    max_iter=7000,
    n_jobs=8,
    config_id=job["config_id"],
    experiment=job["experiment"]
    
    )



'''
    main(
        seed=job["seed"],
        algo=job["algorithm"],
        pop_size=job["config"].get("population_size", 20),
        max_iter=6000,
        n_jobs=8,
        config_id=job["config_id"],
        experiment=job["experiment"])
        '''
    
    
    
'''

if __name__ == "__main__":
    #
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config_id", type=int, default=8)
    parser.add_argument("--algo", type=str, default="hs") #glmulti, nlogit_replicate
    parser.add_argument("--pop_size", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--experiment", type=str, default='ex_wash') #ex_wash ex_synth

    args = parser.parse_args()

    main(args.seed, args.algo, args.pop_size, args.max_iter,args.n_jobs, args.config_id, args.experiment)
    

    
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
    
   











