# =======================================================================
# main_hpc_lc_patch.py
# =======================================================================
#
# Drop this file next to main_hpc.py and import it INSTEAD of importing
# the patched pieces from main_hpc directly.  The module monkey-patches
# the live objects so everything downstream (StructureEvaluatorLC, etc.)
# picks up the changes automatically.
#
# What is added
# ─────────────
# Membership variables: covariates that explain which latent class an
# individual belongs to.  Instead of fixed class-share constants pi_c,
# the class probability for individual n becomes:
#
#   pi_c(n) = softmax( gamma_c · [1, z_n1, …, z_nK] )
#
# where z_nk are the membership covariates for individual n.
#
# Role encoding (extends the existing role scheme)
# ─────────────────────────────────────────────────
#   Role 7 – Membership only
#             Variable enters the class-probability equation (gamma).
#             It has NO effect in the outcome equation.
#             When latent_classes = 1 → treated as excluded (role 0).
#
#   Role 8 – Membership + class-specific outcome
#             Variable enters BOTH the class-probability equation (gamma)
#             AND the outcome equation as a fixed covariate.
#             Because the model has C classes, each class gets its own
#             fixed coefficient for this variable automatically.
#             When latent_classes = 1 → treated as fixed (role 1).
#
# Parameter layout for a C-class model with K_mem membership variables
# ─────────────────────────────────────────────────────────────────────
#   params = [
#       theta_1  (K_base params for class 1's outcome model)
#       theta_2
#       ...
#       theta_C
#       gamma_flat  ((C-1) * (K_mem + 1) params)
#                   gamma_flat.reshape(C-1, K_mem+1)
#                   columns: [intercept, z1, z2, …, zK_mem]
#                   row c: log-odds coefficients for class c+1 vs class 1
#   ]
#
# Backward compatibility
# ─────────────────────
# When K_mem = 0, gamma_flat has shape (C-1, 1) — only an intercept per
# class — which is identical to the previous constant logits vector.
# All existing code that does not specify membership_terms continues to
# work without modification.
#
# HOW TO APPLY
# ─────────────
# At the top of experiment_package.py (or your run script) add:
#
#   import main_hpc_lc_patch   # applies patches and exports new symbols
#
# Then import from this module instead of main_hpc where needed:
#
#   from main_hpc_lc_patch import (
#       ModelSpec, build_param_index, mixed_model_loglik,
#       build_model_from_manual_spec, print_summary_lc
#   )
# =======================================================================

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pandas as pd
from dataclasses import dataclass, replace
from functools import partial
from scipy import stats as scipy_stats
from jaxopt import LBFGS

try:
    from .regularization import compute_regularized_estimates_jax
except ImportError:
    from regularization import compute_regularized_estimates_jax

# ── Import the rest of main_hpc unchanged ──────────────────────────────
try:
    from . import main_hpc as _hpc  # type: ignore[attr-defined]
    from .main_hpc import (
        build_jax_data,          # extended below
        build_model_from_manual_spec as _orig_build_model,
        parse_manual_spec,
        balance_panel_dataframe,
        extract_offset,
        generate_halton_normal,
        generate_sobol_normal,
        build_base_index,
        CountModel,
        compute_standard_errors,
        decode_distribution,
        poisson_loglik,
        nb2_loglik,
        gaussian_loglik,
        lognormal_loglik,
        tobit_loglik,
        build_eta,
        ensure_3d,
        unpack_params,
        DIST_MAP,
    )
except ImportError:
    import main_hpc as _hpc
    from main_hpc import (
        build_jax_data,          # extended below
        build_model_from_manual_spec as _orig_build_model,
        parse_manual_spec,
        balance_panel_dataframe,
        extract_offset,
        generate_halton_normal,
        generate_sobol_normal,
        build_base_index,
        CountModel,
        compute_standard_errors,
        decode_distribution,
        poisson_loglik,
        nb2_loglik,
        gaussian_loglik,
        lognormal_loglik,
        tobit_loglik,
        build_eta,
        ensure_3d,
        unpack_params,
        DIST_MAP,
    )

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# 0.  Tobit OLS initialiser
#     Computes OLS starting values from the non-censored observations.
#     Used by fit_manual_model to bypass the Poisson-style prefit for
#     Tobit models (which would give completely wrong starting values).
# ═══════════════════════════════════════════════════════════════════════

def _tobit_ols_init(data: dict, K_base: int) -> np.ndarray:
    """
    Return an initial parameter vector of length K_base for a Tobit model.

    Strategy
    --------
    1. Average the fixed-effect design matrix Xf and outcomes y over
       the panel dimension to get one row per individual.
    2. Fit OLS on the non-censored rows (y > 0).
    3. Estimate sigma from the OLS residuals.
    4. Pack [beta_ols, sigma_raw] where sigma_raw = log(exp(sigma)-1)
       (inverse of softplus so that softplus(sigma_raw) = sigma).
    """
    y_raw = np.array(data["y"])          # (N, P, 1) or (N, P)
    Xf    = np.array(data["Xf"])         # (N, P, Kf)

    if y_raw.ndim == 3:
        y_flat = y_raw[:, :, 0].mean(axis=1)
    else:
        y_flat = y_raw.mean(axis=1)

    X_flat = Xf.mean(axis=1)             # (N, Kf)
    Kf     = X_flat.shape[1]

    nz = y_flat > 0
    if nz.sum() < Kf + 2:
        nz = np.ones(len(y_flat), dtype=bool)

    y_nz = y_flat[nz]
    X_nz = X_flat[nz]

    try:
        beta_ols = np.linalg.lstsq(X_nz, y_nz, rcond=None)[0]
    except Exception:
        beta_ols = np.zeros(Kf)

    resid     = y_nz - X_nz @ beta_ols
    sigma_hat = max(float(resid.std()), 0.1)
    # inverse-softplus so that softplus(sigma_raw) ≈ sigma_hat
    sigma_raw = float(np.log(np.exp(sigma_hat) - 1.0 + 1e-8))

    params = np.zeros(K_base)
    params[:Kf]        = beta_ols
    params[K_base - 1] = sigma_raw       # sigma is always the last param

    return params


# ═══════════════════════════════════════════════════════════════════════
# 1.  EXTENDED ModelSpec
#     Adds membership_names and K_membership.
#     Replaces the dataclass in main_hpc.
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelSpec:
    Kf:                  int
    Kr_ind:              int
    Kr_cor:              int
    Kg:                  int
    Kh:                  int
    Kzi:                 int
    model:               str
    zero_inflated:       bool
    fixed_names:         tuple
    zi_names:            tuple
    random_ind_names:    tuple
    random_cor_names:    tuple
    grouped_names:       tuple
    hetro_names:         tuple
    random_ind_dists:    tuple
    random_cor_dists:    tuple
    grouped_dists:       tuple
    latent_classes:      int   = 1
    # ── MEMBERSHIP ────────────────────────────────────────────────────
    membership_names:    tuple = ()   # pooled set of all membership variables (union over classes)
    K_membership:        int   = 0    # len(membership_names) — total unique membership vars
    class_membership_idx: tuple = ()  # per-class-column indices into the pooled membership list
                                      #   tuple of tuples, one per non-reference class (length C-1)
                                      #   e.g. ( (0,), (0,1) ) means class 2 uses var 0; class 3 uses vars 0,1
    class_models:        tuple = ()   # per-class model strings (eg ("poisson","nb"))
    min_class_proportion: float = 0.15  # minimum posterior-mean proportion per class (will be scaled by C internally)
    # ── PER-CLASS COVARIATE SELECTION ─────────────────────────────────
    class_fixed_idx:     tuple = ()   # per-class column indices into Xf (tuple of tuples)
    class_rdm_ind_idx:   tuple = ()   # per-class column indices into Xr_ind
    class_rdm_cor_idx:   tuple = ()   # per-class column indices into Xr_cor
    class_variable_masks: tuple = ()  # per-class variable sets (frozensets) [NEW: alternative to indices]
    l2_penalty:          float = 0.0  # naive L2 ridge strength on non-intercept params (0 = off; opt-in only)
    # ── DEFAULT IN-LOOP REGULARISATION ─────────────────────────────────
    # Every M-step outcome-parameter update is shrunk with PARTE (Alghamdi
    # et al. 2026) immediately after the LBFGS solve, using (k, d) selected
    # adaptively from the per-class Fisher information — there is no fixed
    # "default lambda" the way there would be for naive L2; the paper's
    # whole premise is that (k, d) are data-adaptive rather than preset.
    parte_shrinkage_in_loop: bool = True
    parte_variant:        str  = "k3d3"  # "k1d1" | "k2d2" | "k3d3"

    @property
    def K_random_total(self):
        return self.Kr_cor + self.Kr_ind

    @property
    def models(self) -> tuple:
        """Per-class model strings.  Broadcasts `model` when class_models is empty."""
        if self.class_models and self.latent_classes > 1:
            return tuple(self.class_models) + tuple(
                self.model for _ in range(self.latent_classes - len(self.class_models))
            )
        return tuple(self.model for _ in range(max(1, self.latent_classes)))


# Monkey-patch so the rest of main_hpc sees the new class
_hpc.ModelSpec = ModelSpec


# ═══════════════════════════════════════════════════════════════════════
# 2.  build_param_index
#     LC tail is now (C-1)*(K_mem+1) instead of (C-1).
# ═══════════════════════════════════════════════════════════════════════

def build_param_index(spec: ModelSpec) -> dict:

    if spec.latent_classes == 1:
        return build_base_index(spec)

    C     = spec.latent_classes
    K_mem = spec.K_membership
    models = spec.models  # per-class model strings

    # ── Per-class membership indices (new) ─────────────────────────────
    # class_membership_idx: tuple of tuples, length C-1, one per non-reference class
    #   class 2 (first non-ref) → class_membership_idx[0] = (0, 2) etc.
    _cm_idx = spec.class_membership_idx
    if _cm_idx and len(_cm_idx) == C - 1 and K_mem > 0:
        per_class_K_mem = tuple(len(idx_tup) for idx_tup in _cm_idx)
    else:
        per_class_K_mem = tuple(K_mem for _ in range(C - 1))
        # Build default indices: all vars for all classes
        _cm_idx = tuple(tuple(range(K_mem)) for _ in range(C - 1))

    # Compute per-class K_base (may differ by model type AND per-class variables)
    base_spec_no_lc = replace(spec, latent_classes=1)
    class_offsets = []   # start index of each class's theta in flat param vector
    class_K_base  = []   # number of parameters per class
    offset = 0
    for c in range(C):
        _model_c = models[c]
        # Build per-class spec with correct Kf and Kr* from class_fixed_idx
        _spec_c = base_spec_no_lc
        if spec.class_fixed_idx and c < len(spec.class_fixed_idx):
            cfix = spec.class_fixed_idx[c]
            if len(cfix) < spec.Kf:
                _spec_c = replace(_spec_c, Kf=len(cfix))
        _base_idx_c = build_base_index(_spec_c, model=_model_c)
        _Kc = _base_idx_c["total_params"]
        class_offsets.append(offset)
        class_K_base.append(_Kc)
        offset += _Kc

    total_theta = offset

    index = {}
    index["class_params"]   = (0, total_theta)
    index["class_offsets"]  = tuple(class_offsets)
    index["class_K_base"]   = tuple(class_K_base)
    index["class_models"]   = tuple(models)

    # ── Gamma: jagged per-class arrays (one per non-reference class) ───
    # gamma_c has shape (K_c+1,) where K_c = number of membership vars for that class
    gamma_offsets = []
    gamma_sizes = []
    idx = total_theta
    for c in range(C - 1):
        Kc = per_class_K_mem[c]
        gamma_offsets.append(idx)
        gamma_sizes.append(Kc + 1)  # +1 for intercept
        idx += (Kc + 1)

    index["class_gamma_offsets"] = tuple(gamma_offsets)
    index["class_gamma_sizes"]   = tuple(gamma_sizes)
    index["class_gamma"]         = (total_theta, idx)
    index["membership_class_idx"] = _cm_idx  # which pooled cols each class uses

    gamma_size = idx - total_theta
    index["gamma_size"] = gamma_size

    index["K_base"]         = class_K_base[0] if class_K_base else 0  # legacy compat
    index["K_mem"]          = K_mem
    index["total_params"]   = idx

    return index


_hpc.build_param_index = build_param_index


# ═══════════════════════════════════════════════════════════════════════
# 3.  Extended build_jax_data
#     Adds membership_cols parameter; appends Xmem to the data dict.
# ═══════════════════════════════════════════════════════════════════════

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
    zi_cols=None,
    membership_cols=None,
    class_membership_cols=None,  # NEW: per-class membership variable lists (list of lists, length C-1)
    class_fixed_cols=None,       # per-class fixed variable lists (list of lists)
    class_rdm_ind_cols=None,     # per-class random indep lists
    class_rdm_cor_cols=None,     # per-class random corr lists
    draw_method='sobol',        # 'halton' or 'sobol' (Sobol faster, more stable)
    R=200,
):
    fixed_cols        = fixed_cols        or []
    random_ind_cols   = random_ind_cols   or []
    random_cor_cols   = random_cor_cols   or []
    grouped_cols      = grouped_cols      or []
    hetro_cols        = hetro_cols        or []
    zi_cols           = zi_cols           or []
    membership_cols   = membership_cols   or []          # NEW
    random_ind_dists  = random_ind_dists  or []
    random_cor_dists  = random_cor_dists  or []
    grouped_dists     = grouped_dists     or []

    intercept_name = "__INTERCEPT__"
    df = df.copy()
    df[intercept_name] = 1.0

    # ── Standardise continuous predictors (membership cols too) ──────────
    _predictor_cols = list(set(
        fixed_cols + random_ind_cols + random_cor_cols
        + grouped_cols + hetro_cols + zi_cols + membership_cols
    ))
    _scaler = _hpc.compute_scaler(df, _predictor_cols)
    if _scaler:
        df = _hpc.apply_scaler(df, _scaler)

    all_features = list(set(
        [intercept_name]
        + fixed_cols + random_ind_cols + random_cor_cols
        + grouped_cols + hetro_cols + zi_cols + membership_cols   # NEW
    ))

    X_all, y, mask = balance_panel_dataframe(df, id_col, y_col, all_features)

    # Group IDs
    if group_id_col is not None and len(grouped_cols) > 0:
        df_sorted   = df.sort_values(id_col)
        group_codes = df_sorted[group_id_col].astype("category").cat.codes.values
        G           = len(np.unique(group_codes))
    else:
        group_codes = None
        G           = 0

    col_map = {col: i for i, col in enumerate(all_features)}

    def extract(cols):
        if len(cols) == 0:
            return np.zeros((X_all.shape[0], X_all.shape[1], 0))
        idx = [col_map[c] for c in cols if c in col_map]
        return X_all[:, :, idx]

    fixed_cols_with_intercept = [intercept_name] + fixed_cols
    Xf   = np.concatenate([extract([intercept_name]), extract(fixed_cols)], axis=2)
    Xr_ind = extract(random_ind_cols)
    Xr_cor = extract(random_cor_cols)
    Xg     = extract(grouped_cols)
    Xh     = extract(hetro_cols)
    Xzi    = extract(zi_cols)
    Xmem   = extract(membership_cols)    # NEW  shape (N, P, K_mem)

    # ── Per-class column indices (for class-specific variable sets) ────
    _fixed_names = [intercept_name] + fixed_cols
    _fixed_map = {name: i for i, name in enumerate(_fixed_names)}
    _class_fixed_idx = []
    if class_fixed_cols and len(class_fixed_cols) > 0:
        for cfix in class_fixed_cols:
            cfix_with_int = [intercept_name] + list(cfix)
            _class_fixed_idx.append(tuple(
                _fixed_map[n] for n in cfix_with_int if n in _fixed_map
            ))
    else:
        _class_fixed_idx = ()
    class_fixed_idx = tuple(_class_fixed_idx)

    # ── Per-class membership column indices ────────────────────────────
    _mem_names = list(membership_cols)
    _mem_map = {name: i for i, name in enumerate(_mem_names)}
    _class_membership_idx = []  # one tuple per non-reference class
    if class_membership_cols and len(class_membership_cols) > 0 and len(_mem_names) > 0:
        for cm_list in class_membership_cols:
            _idx_c = tuple(_mem_map[n] for n in cm_list if n in _mem_map)
            _class_membership_idx.append(_idx_c)
    class_membership_idx = tuple(_class_membership_idx) if _class_membership_idx else ()

    N, P = y.shape[0], y.shape[1]

    if offset_col:
        offset = extract_offset(df, id_col, offset_col)
    else:
        offset = np.zeros((N, P, 1))

    # Auto-generate draws when not provided but random cols are present.
    _draw_fn = generate_sobol_normal if draw_method == 'sobol' else generate_halton_normal
    if draws_ind is None and random_ind_cols:
        draws_ind = _draw_fn(N, len(random_ind_cols), R, seed=42)
    if draws_cor is None and random_cor_cols:
        draws_cor = _draw_fn(N, len(random_cor_cols), R, seed=43)
    if draws_g is None and grouped_cols:
        draws_g   = _draw_fn(N, len(grouped_cols),    R, seed=44)

    data = {
        "Xf":       jnp.array(Xf),
        "Xr_ind":   jnp.array(Xr_ind),
        "Xr_cor":   jnp.array(Xr_cor),
        "Xg":       jnp.array(Xg),
        "Xh":       jnp.array(Xh),
        "Xzi":      jnp.array(Xzi),
        "Xmem":     jnp.array(Xmem),    # NEW
        "y":        jnp.array(y),
        "mask":     jnp.array(mask),
        "offset":   jnp.array(offset),
        "draws_ind":jnp.zeros((N, 0, R)) if draws_ind is None else jnp.array(draws_ind),
        "draws_cor":jnp.zeros((N, 0, R)) if draws_cor is None else jnp.array(draws_cor),
        "draws_g":  jnp.zeros((N, 0, R)) if draws_g   is None else jnp.array(draws_g),
        "group_ids":jnp.array(group_codes) if group_codes is not None
                    else jnp.zeros(N, dtype=int),
        # Plain Python dict — used by print_summary for back-transformation.
        "scaler": _scaler,
    }

    spec = ModelSpec(
        Kf=Xf.shape[2],
        Kr_ind=Xr_ind.shape[2],
        Kr_cor=Xr_cor.shape[2],
        Kg=Xg.shape[2],
        Kh=Xh.shape[2],
        zi_names=tuple(zi_cols),
        Kzi=Xzi.shape[2],
        zero_inflated=(len(zi_cols) > 0),
        model="poisson",
        fixed_names=tuple(fixed_cols_with_intercept),
        random_ind_names=tuple(random_ind_cols),
        random_cor_names=tuple(random_cor_cols),
        grouped_names=tuple(grouped_cols),
        hetro_names=tuple(hetro_cols),
        random_ind_dists=tuple(random_ind_dists),
        random_cor_dists=tuple(random_cor_dists),
        grouped_dists=tuple(grouped_dists),
        membership_names=tuple(membership_cols),    # NEW
        K_membership=Xmem.shape[2],                # NEW
        class_membership_idx=class_membership_idx, # NEW: per-class membership column indices
        class_models=(),                           # set by caller via replace()
        class_fixed_idx=class_fixed_idx,           # per-class Xf column indices
    )

    return data, spec


_hpc.build_jax_data = build_jax_data


# ═══════════════════════════════════════════════════════════════════════
# 4.  Extended parse_manual_spec
#     Handles "membership_terms" key in the spec dict.
# ═══════════════════════════════════════════════════════════════════════

def parse_manual_spec(manual_spec: dict):
    fixed_cols        = manual_spec.get("fixed_terms", [])
    rdm_terms         = manual_spec.get("rdm_terms", [])
    rdm_cor_terms     = manual_spec.get("rdm_cor_terms", [])
    grouped_terms     = manual_spec.get("grouped_terms", [])
    hetro_terms       = manual_spec.get("hetro_in_means", [])
    zi_cols           = manual_spec.get("zi_terms", [])
    membership_cols   = manual_spec.get("membership_terms", [])
    class_membership  = manual_spec.get("class_membership", None)  # per-class membership lists
    class_fixed       = manual_spec.get("class_fixed", None)   # per-class fixed lists
    class_rdm_ind     = manual_spec.get("class_rdm_ind", None)
    class_rdm_cor     = manual_spec.get("class_rdm_cor", None)

    random_ind       = [t.split(":")[0] for t in rdm_terms]
    random_cor       = [t.split(":")[0] for t in rdm_cor_terms]
    grouped_cols     = [t.split(":")[0] for t in grouped_terms]
    hetro_cols       = [t.split(":")[0].strip() for t in hetro_terms]

    random_ind_dists  = [t.split(":")[1] for t in rdm_terms]
    random_cor_dists  = [t.split(":")[1] for t in rdm_cor_terms]
    grouped_dists     = [t.split(":")[1] for t in grouped_terms]

    return (
        fixed_cols, random_ind, random_cor, grouped_cols, hetro_cols,
        random_ind_dists, random_cor_dists, grouped_dists,
        zi_cols, membership_cols,
        class_fixed, class_rdm_ind, class_rdm_cor,
        class_membership,
    )


_hpc.parse_manual_spec = parse_manual_spec


# ═══════════════════════════════════════════════════════════════════════
# 5.  Extended build_model_from_manual_spec
#     Passes membership_cols to build_jax_data.
# ═══════════════════════════════════════════════════════════════════════

def build_model_from_manual_spec(
    df, manual_spec, id_col, y_col,
    offset_col=None, draws_ind=None, draws_cor=None, draws_g=None,
    draw_method='sobol', R=200
):
    (
        fixed_cols, random_ind, random_cor, grouped_cols, hetro_cols,
        random_ind_dists, random_cor_dists, grouped_dists,
        zi_cols, membership_cols,
        class_fixed, class_rdm_ind, class_rdm_cor,
        class_membership,
    ) = parse_manual_spec(manual_spec)

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
        membership_cols=membership_cols,
        class_membership_cols=class_membership,   # NEW: per-class membership lists
        class_fixed_cols=class_fixed,
        offset_col=offset_col,
        draws_ind=draws_ind,
        draws_cor=draws_cor,
        draws_g=draws_g,
        random_ind_dists=random_ind_dists,
        random_cor_dists=random_cor_dists,
        grouped_dists=grouped_dists,
        draw_method=draw_method,
        R=R,
    )

    model_type = "nb" if manual_spec.get("dispersion", 0) else "poisson"
    lc         = int(manual_spec.get("latent_classes", 1))
    class_models = tuple(manual_spec.get("class_models", ()))
    min_class_prop = float(manual_spec.get("min_class_proportion", 0.15))
    spec       = replace(spec, model=model_type, latent_classes=lc,
                         class_models=class_models,
                         min_class_proportion=min_class_prop)

    return data, spec


_hpc.build_model_from_manual_spec = build_model_from_manual_spec


# ═══════════════════════════════════════════════════════════════════════
# 6.  Mixed-model log-likelihood with membership covariates
#
#     The LC branch is the only section changed.  The single-class path
#     is identical to main_hpc so we keep it intact.
#
#     Class-probability model (NEW)
#     ─────────────────────────────
#     gamma  : (C-1, K_mem+1)   row c = log-odds coefficients for class c+1
#     Z_full : (N, K_mem+1)     col 0 = 1 (intercept), cols 1..K = members
#
#     log pi_i = log_softmax( Z_full @ gamma.T , axis=1 )   shape (N, C)
#
#     When K_mem=0, Z_full = ones(N,1) and gamma is just the (C-1) scalar
#     logits — backward-compatible with the old constant-pi behaviour.
# ═══════════════════════════════════════════════════════════════════════


def _l2_penalty(params, spec: ModelSpec):
    """
    L2 ridge penalty on non-intercept parameters.
    Only penalises outcome-model parameters (not membership gamma).
    Returns 0.0 when spec.l2_penalty <= 0.
    """
    if spec.l2_penalty <= 0.0:
        return 0.0
    lam = float(spec.l2_penalty)
    if spec.latent_classes > 1:
        C = spec.latent_classes
        pindex = build_param_index(spec)
        class_offsets = list(pindex["class_offsets"])
        class_K_base = list(pindex["class_K_base"])
        # Penalise all per-class outcome params except the intercept (param[0] of each class)
        s = 0.0
        for c in range(C):
            oc = class_offsets[c]
            kc = class_K_base[c]
            if kc > 1:
                s += jnp.sum(params[oc + 1 : oc + kc] ** 2)
        return lam * s
    else:
        if len(params) > 1:
            return lam * jnp.sum(params[1:] ** 2)
        return 0.0


def mixed_model_loglik_reg(params, data, spec: ModelSpec, indivi: bool = False):
    """Regularised log-likelihood (L2 penalty added to unregularised objective)."""
    base = mixed_model_loglik(params, data, spec, indivi=indivi)
    penalty = _l2_penalty(params, spec)
    if indivi:
        N = data["y"].shape[0]
        return base + penalty / N
    return base + penalty


def _parte_shrink_theta(theta_c, objective, param_index_c: dict, variant: str = "k3d3"):
    """
    Apply PARTE (Alghamdi et al. 2026) shrinkage to a per-class outcome
    parameter vector immediately after its M-step MLE solve.

    ``objective`` must be the same weighted negative log-likelihood the
    M-step just minimised over ``theta_c`` — its Hessian at the solution
    is the (weighted) Fisher information PARTE shrinks against.  (k, d)
    are selected adaptively (eqs 37-41); there is no free hyperparameter.

    Falls back to the unshrunk ``theta_c`` when the class has fewer than
    two fixed-effect parameters (PARTE needs a >=2-D block to be meaningful).
    """
    fixed_key = "fixed" if "fixed" in param_index_c else "beta_f"
    if fixed_key not in param_index_c:
        return theta_c
    i0, i1 = int(param_index_c[fixed_key][0]), int(param_index_c[fixed_key][1])
    if (i1 - i0) < 2:
        return theta_c

    hess = jax.hessian(objective)(theta_c)
    theta_parte, _se_full, _beta_parte, _k, _d = compute_regularized_estimates_jax(
        theta_c, hess, i0, i1, variant
    )
    return theta_parte


@partial(jax.jit, static_argnames=("spec", "indivi"))
def mixed_model_loglik(params, data, spec: ModelSpec, indivi: bool = False):

    # ── LATENT CLASS BRANCH ─────────────────────────────────────────
    if spec.latent_classes > 1:

        C         = spec.latent_classes
        K_mem     = spec.K_membership
        models    = spec.models  # tuple of per-class model strings
        base_spec_nolc = replace(spec, latent_classes=1)

        # Compute per-class param sizes (account for per-class variable sets)
        class_K_base = []
        class_offsets = []
        offset = 0
        for c in range(C):
            _spec_c = base_spec_nolc
            if spec.class_fixed_idx and c < len(spec.class_fixed_idx):
                cfix = spec.class_fixed_idx[c]
                if len(cfix) < spec.Kf:
                    _spec_c = replace(_spec_c, Kf=len(cfix))
            _kc = build_base_index(_spec_c, model=models[c])["total_params"]
            class_K_base.append(_kc)
            class_offsets.append(offset)
            offset += _kc
        total_theta = offset

        # Class-specific outcome parameters (jagged — NOT rectangular)
        theta_all = []  # list of jnp arrays with different lengths
        for c in range(C):
            kc = class_K_base[c]
            oc = class_offsets[c]
            theta_all.append(params[oc:oc + kc])

        # ── Per-class membership gamma and Z matrices ──────────────────
        cm_idx = getattr(spec, 'class_membership_idx', ())
        if not cm_idx or len(cm_idx) != C - 1 or K_mem == 0:
            # Fallback: all membership vars for all classes (backward compat)
            cm_idx = tuple(tuple(range(K_mem)) for _ in range(C - 1))
        N = data["y"].shape[0]

        gamma_list = []   # per-class gamma arrays (list of jnp 1-d arrays)
        g_offset = total_theta
        logits_cols = []
        for _c in range(C - 1):  # _c indexes non-reference classes (1...C-1)
            idx_tup = cm_idx[_c]
            Kc = len(idx_tup)
            gamma_c = params[g_offset : g_offset + Kc + 1]  # intercept + Kc membership vars
            gamma_list.append(gamma_c)
            # Build Z_c for this class transition
            if Kc > 0:
                Z_sub = jnp.mean(data["Xmem"][:, :, list(idx_tup)], axis=1)  # (N, Kc)
                Z_c = jnp.concatenate([jnp.ones((N, 1)), Z_sub], axis=1)      # (N, Kc+1)
            else:
                Z_c = jnp.ones((N, 1))  # intercept only
            logits_cols.append(Z_c @ gamma_c)   # (N,)
            g_offset += (Kc + 1)

        # Individual-specific log class probabilities
        logits_i = jnp.stack(logits_cols, axis=1)            # (N, C-1)
        logits_full = jnp.concatenate(
            [jnp.zeros((N, 1)), logits_i], axis=1
        )                                                     # (N, C)
        log_pi = jax.nn.log_softmax(logits_full, axis=1)     # (N, C)

        # Per-class individual log-likelihoods (each with its own model)
        ll_classes = []
        for c in range(C):
            base_spec_c = replace(base_spec_nolc, model=models[c])
            # ── Per-class data slicing ─────────────────────────────────
            _data_c = data
            _spec_c = base_spec_c
            cfix = spec.class_fixed_idx[c] if spec.class_fixed_idx and c < len(spec.class_fixed_idx) else None
            if cfix is not None and len(cfix) < data["Xf"].shape[2]:
                # Slice Xf to only this class's columns
                Xf_c = data["Xf"][:, :, list(cfix)]
                _data_c = dict(data)
                _data_c["Xf"] = Xf_c
                _spec_c = replace(base_spec_c, Kf=len(cfix),
                                  fixed_names=tuple(
                                      data.get("_fixed_names_all", ())[i]
                                      for i in cfix if hasattr(data, '_fixed_names_all')
                                  ) or tuple(f"f{i}" for i in range(len(cfix))))
            ll_c = mixed_model_loglik(
                theta_all[c], _data_c, _spec_c, indivi=True
            )                                               # (N,) log-likelihoods (negative)
            ll_classes.append(ll_c + log_pi[:, c])

        ll_stack = jnp.stack(ll_classes, axis=1)            # (N, C)
        ll_ind   = jsp.special.logsumexp(ll_stack, axis=1)  # (N,)

        if indivi:
            return ll_ind
        return -jnp.sum(ll_ind)

    # ── SINGLE-CLASS BRANCH (unchanged from main_hpc) ───────────────
    blocks = unpack_params(params, spec)
    eta    = build_eta(params, data, spec)
    # Linear-predictor models work in data scale — wider clip to preserve gradients.
    # Survival AFT families also use log(t) scale so their etas are unbounded.
    if spec.model in {"gaussian", "tobit", "weibull", "loglogistic"}:
        eta = jnp.clip(eta, -500.0, 500.0)
    else:
        eta = jnp.clip(eta, -25.0, 25.0)

    if eta.ndim == 2:
        eta = eta[..., None]

    y    = ensure_3d(data["y"])
    mask = ensure_3d(data["mask"])
    R    = eta.shape[-1]

    if spec.model == "poisson":
        mu       = jnp.exp(eta)
        ll_count = poisson_loglik(y, mu)

    elif spec.model == "nb":
        alpha    = blocks["alpha"]
        ll_count = nb2_loglik(y, eta, alpha)

    elif spec.model == "lognormal":
        sigma    = blocks["sigma"]
        ll_count = lognormal_loglik(y, eta, sigma)
    elif spec.model == "gaussian":
        sigma    = blocks["sigma"]
        ll_count = gaussian_loglik(y, eta, sigma)
    elif spec.model == "tobit":
        # Left-censored at 0; eta is the linear predictor for the latent Y*
        sigma    = blocks["sigma"]
        ll_count = tobit_loglik(y, eta, sigma)
    else:
        raise ValueError(f"Unknown model: {spec.model}")

    if spec.zero_inflated:
        if spec.Kzi > 0:
            eta_zi = jnp.einsum(
                "npk,k->np", data["Xzi"], blocks["beta_zi"]
            )[..., None]
        else:
            eta_zi = jnp.zeros_like(eta[..., :1])

        pi_zi = jax.nn.sigmoid(eta_zi)
        mu = jnp.exp(eta)

        if spec.model == "poisson":
            f0 = jnp.exp(-mu)
        elif spec.model == "nb":
            alpha_e  = jnp.exp(blocks["alpha"])
            inv_a    = 1.0 / alpha_e
            f0       = jnp.exp(inv_a * (jnp.log(inv_a) - jnp.log(inv_a + mu)))
        elif spec.model == "lognormal":
            f0 = jnp.zeros_like(mu)
        elif spec.model == "gaussian":
            sigma = jax.nn.softplus(blocks["sigma"])
            f0 = jnp.exp(-0.5 * jnp.log(2 * jnp.pi * sigma**2) - (eta**2) / (2 * sigma**2))
        else:
            raise ValueError(f"Unknown zero-inflated model: {spec.model}")

        zero_mask = (y == 0)
        ll_zero   = jnp.log(pi_zi + (1 - pi_zi) * f0 + 1e-12)
        ll_pos    = jnp.log(1 - pi_zi + 1e-12) + ll_count
        ll        = jnp.where(zero_mask, ll_zero, ll_pos)
    else:
        ll = ll_count

    ll       = ll * mask
    ll_panel = jnp.sum(ll, axis=1)

    if R > 1:
        ll_ind = jsp.special.logsumexp(ll_panel, axis=-1) - jnp.log(R)
    else:
        ll_ind = ll_panel.squeeze(-1)

    if indivi:
        return ll_ind
    return -jnp.sum(ll_ind)


_hpc.mixed_model_loglik = mixed_model_loglik


# ═══════════════════════════════════════════════════════════════════════
# 7.  fit_em — EM algorithm aware of membership covariates
#
#     The original fit_em in main_hpc.py treated params[C*K_base:] as a
#     flat (C-1,) logits vector.  With membership variables the gamma
#     section has shape (C-1, K_mem+1) — one intercept + K_mem slopes
#     per class-pair.  This replacement:
#       • E-step: computes individual-specific log_pi via per-class Z_c @ gamma_c
#       • M-step (gamma): minimises the weighted MNL cross-entropy per class
#       • M-step (theta): unchanged (weighted outcome log-lik per class)
#     When K_mem=0 the behaviour is identical to the original.
# ═══════════════════════════════════════════════════════════════════════

def fit_em(init_params, data, spec: ModelSpec,
           max_iter=100, tol=1e-6, verbose=True, return_trace=False):
    """
    EM algorithm for latent-class mixed count models.

    Improvements over baseline:
    - Progressive M-step budget: LBFGS maxiter starts at 50 and grows to 300
      as EM converges.  Early iterations only need rough M-step solutions.
    - Temperature annealing on E-step posteriors: T > 1 in early iterations
      softens the posteriors and prevents premature class collapse.
    - LL-based convergence: tracked via the full joint likelihood, not the
      max parameter change, which is scale-dependent and easily fooled.
    - Best-params tracking: the params with the highest observed LL are
      returned, guarding against the last iterate being slightly worse.
    - Deferred collapse guard: collapse check is skipped for the first 3
      iterations so classes have time to separate before being penalised.
    - Pure-JAX gamma objective: numpy arrays inside the objective function
      break autodiff gradients; all operations now use jnp.
    - Per-class model support: each class can have its own distributional
      form (poisson, nb, etc.) via spec.class_models.
    """
    from jax.nn import log_softmax as _log_softmax

    assert spec.latent_classes > 1, "EM only needed for latent classes"

    C          = spec.latent_classes
    K_mem      = spec.K_membership
    models     = spec.models  # per-class model strings
    base_spec_nolc = replace(spec, latent_classes=1)
    # ── Per-class membership indices ───────────────────────────────────
    cm_idx = getattr(spec, 'class_membership_idx', ())
    if not cm_idx or len(cm_idx) != C - 1 or K_mem == 0:
        cm_idx = tuple(tuple(range(K_mem)) for _ in range(C - 1))
    per_class_K_mem = tuple(len(idx_tup) for idx_tup in cm_idx)
    gamma_total_size = sum(Kc + 1 for Kc in per_class_K_mem)  # per-class gamma sizes including intercept

    # Compute per-class param sizes and specs (account for per-class vars)
    class_K_base = []
    class_offsets = []
    class_base_specs = []
    class_param_index = []
    offset = 0
    for c in range(C):
        _spec_c = base_spec_nolc
        if spec.class_fixed_idx and c < len(spec.class_fixed_idx):
            cfix = spec.class_fixed_idx[c]
            if len(cfix) < spec.Kf:
                _spec_c = replace(_spec_c, Kf=len(cfix))
        _pindex_c = build_base_index(_spec_c, model=models[c])
        class_K_base.append(_pindex_c["total_params"])
        class_offsets.append(offset)
        class_base_specs.append(replace(_spec_c, model=models[c]))
        class_param_index.append(_pindex_c)
        offset += _pindex_c["total_params"]
    total_theta = offset

    params = np.array(init_params)
    N = int(data["y"].shape[0])

    # Per-class data: pre-slice Xf to each class's own fixed-effect columns
    # (matching class_base_specs[c].Kf) once, up front — every per-class
    # mixed_model_loglik call below must use this, not the unsliced `data`,
    # whenever class_fixed_idx has reduced that class's Kf below the global.
    class_data = []
    for c in range(C):
        cfix = spec.class_fixed_idx[c] if spec.class_fixed_idx and c < len(spec.class_fixed_idx) else None
        if cfix is not None and len(cfix) < data["Xf"].shape[2]:
            _dc = dict(data)
            _dc["Xf"] = data["Xf"][:, :, list(cfix)]
            class_data.append(_dc)
        else:
            class_data.append(data)

    # Build per-class Z matrices (one per non-reference class) — fixed for all iters
    Xmem_np = np.array(data["Xmem"])  # (N, P, K_mem)
    Z_mats = []   # list of (N, Kc+1) arrays, one per non-reference class
    for _c in range(C - 1):
        idx_tup = cm_idx[_c]
        Kc = len(idx_tup)
        if Kc > 0:
            Z_sub = np.mean(Xmem_np[:, :, list(idx_tup)], axis=1)  # (N, Kc)
            Z_c = np.concatenate([np.ones((N, 1)), Z_sub], axis=1)  # (N, Kc+1)
        else:
            Z_c = np.ones((N, 1))  # intercept only
        Z_mats.append(Z_c)
    # Pre-convert to JAX once (used inside gamma objective)
    Z_mats_jnp = tuple(jnp.array(Z_c) for Z_c in Z_mats)

    # Track best params seen across all iterations
    best_params = params.copy()
    try:
        best_ll = -float(mixed_model_loglik(jnp.array(params), data, spec))
    except Exception:
        best_ll = -np.inf

    prev_ll = best_ll
    prev_params = params.copy()
    # A per-class M-step LBFGS call has no bound on how far theta can move
    # and occasionally wanders to a numerically extreme point (e.g. an
    # overflowing linear predictor), producing a params update that is
    # drastically worse than the previous iterate. Since that divergent
    # update otherwise becomes the *next* iteration's warm start, it
    # propagates forward and derails several subsequent iterations --
    # visible as large spikes in the objective trace. DIVERGENCE_TOL bounds
    # how much the LL is allowed to drop in one iteration before that
    # iterate is discarded in favour of the last good one.
    DIVERGENCE_TOL = 50.0
    trace = []  # (iteration, T, m_iters, LL, delta_LL, class_shares)

    for iteration in range(max_iter):

        # ==============================================================
        # Temperature schedule for E-step
        # ==============================================================
        warmup_frac = min(1.0, iteration / max(1, max_iter * 0.4))
        T = max(1.0, 2.0 - warmup_frac)              # 2.0 → 1.0 over first 40%

        # ==============================================================
        # Progressive M-step budget
        # ==============================================================
        m_iters = min(50 + 25 * (iteration // 3), 300)

        # ==========================================================
        # E-STEP
        # ==========================================================
        # Extract per-class thetas (jagged)
        theta_all = []
        for c in range(C):
            oc = class_offsets[c]
            kc = class_K_base[c]
            theta_all.append(params[oc:oc + kc])

        # Extract per-class gamma arrays (jagged)
        gamma_list = []
        g_offset = total_theta
        for _c in range(C - 1):
            Kc = per_class_K_mem[_c]
            gamma_list.append(params[g_offset : g_offset + Kc + 1])
            g_offset += (Kc + 1)

        # Individual-specific log class probabilities (N, C)
        logits_cols = []
        for _c in range(C - 1):
            logits_cols.append(Z_mats[_c] @ gamma_list[_c])  # (N,)
        logits_i    = np.column_stack(logits_cols)            # (N, C-1)
        logits_full = np.concatenate(
            [np.zeros((N, 1)), logits_i], axis=1
        )                                                      # (N, C)
        log_pi = np.array(_log_softmax(jnp.array(logits_full), axis=1))

        # Per-class individual log-likelihoods  (N, C)
        logL = np.zeros((N, C))
        for c in range(C):
            ll_ind = mixed_model_loglik(
                jnp.array(theta_all[c]), class_data[c], class_base_specs[c], indivi=True
            )
            logL[:, c] = np.array(ll_ind)

        # Tempered posteriors: divide log-joint by temperature T
        log_num = (logL + log_pi) / T                           # (N, C)

        # Posterior class membership weights
        max_log = log_num.max(axis=1, keepdims=True)
        w = np.exp(log_num - max_log)
        w /= w.sum(axis=1, keepdims=True)

        # Collapse guard: deferred past iter 3 so classes can separate first.
        mean_w = w.mean(axis=0)                                 # (C,)
        _raw_prop = getattr(spec, 'min_class_proportion', 0.15)
        # Scale min_prop by C: with 5 classes ~0.03 floor, 2 classes ~0.15
        min_prop = max(0.02, _raw_prop / max(1, C * 0.5))

        # Class balance Dirichlet-prior penalty: adds pseudocounts to
        # prevent extreme class imbalance (e.g. 85%/15% split).
        # When mean_w[c] < min_prop, the penalty ramps up.
        # prior_weight = 0 when balanced; grows as classes shrink below threshold.
        below_thresh = np.maximum(0.0, min_prop - mean_w)
        prior_weight = float(np.sum(below_thresh) * max(5.0, C * 2.0))

        _collapse_cutoff = max(0.005, 0.02 / C)
        if iteration >= 3 and np.any(mean_w < _collapse_cutoff):
            if verbose:
                print(f"  [EM] class collapse at iter {iteration} "
                      f"(min mean weight {mean_w.min():.4f} < {_collapse_cutoff:.4f}) — stopping early")
            break

        # ==========================================================
        # M-STEP
        # ==========================================================

        # Update class-specific outcome parameters (progressive LBFGS budget)
        theta_new = []
        for c in range(C):
            wc = w[:, c].copy()
            _base_c = class_base_specs[c]
            _data_c = class_data[c]

            def weighted_objective(theta_c, _wc=wc, _spec=_base_c, _data=_data_c):
                ll_ind = mixed_model_loglik(
                    theta_c, _data, _spec, indivi=True
                )
                loss = -jnp.sum(jnp.array(_wc) * jnp.array(ll_ind))
                if spec.l2_penalty > 0.0 and len(theta_c) > 1:
                    loss = loss + spec.l2_penalty * jnp.sum(theta_c[1:] ** 2)
                return loss

            solver_theta = LBFGS(fun=weighted_objective, maxiter=m_iters)
            result = solver_theta.run(jnp.array(theta_all[c]))
            theta_c_mle = jnp.array(result.params)
            if spec.parte_shrinkage_in_loop:
                theta_c_mle = _parte_shrink_theta(
                    theta_c_mle, weighted_objective,
                    class_param_index[c], spec.parte_variant
                )
            theta_new.append(np.array(theta_c_mle))

        theta_new_flat = np.concatenate(theta_new)

        # ── Update each per-class gamma independently ──────────────────
        _w_jnp = jnp.array(w)
        _pw = prior_weight  # capture for closure
        gamma_new_parts = []
        for _c in range(C - 1):
            Z_c_jnp = Z_mats_jnp[_c]
            Kc = per_class_K_mem[_c]
            _class_c = _c  # capture for closure

            def gamma_objective_c(gamma_c, _Zc=Z_c_jnp, _cc=_class_c):
                """Per-class gamma objective: optimises only class c+1's gamma."""
                # Build full logits including all other classes' contributions
                Ncur = _Zc.shape[0]
                logit_cols = []
                for _tc in range(C - 1):
                    if _tc == _cc:
                        logit_cols.append(_Zc @ gamma_c)  # optimised class (N,)
                    else:
                        # Use current (non-optimised) gamma for other classes
                        gc_other = gamma_list[_tc]
                        Z_other = Z_mats_jnp[_tc]
                        logit_cols.append(Z_other @ jnp.array(gc_other))
                li = jnp.stack(logit_cols, axis=1)                      # (N, C-1)
                zeros_col = jnp.zeros((Ncur, 1))
                lf = jnp.concatenate([zeros_col, li], axis=1)          # (N, C)
                lp = _log_softmax(lf, axis=1)                           # (N, C)
                ce = -jnp.sum(_w_jnp * lp)
                # Dirichlet balance prior
                log_pi_marg = jax.nn.log_softmax(
                    jnp.mean(lf, axis=0, keepdims=True), axis=1
                )
                balance_penalty = -_pw * jnp.sum(log_pi_marg)
                return ce + balance_penalty

            solver_gamma_c = LBFGS(fun=gamma_objective_c, maxiter=m_iters)
            gamma_c_prev = jnp.array(gamma_list[_c])
            result_gamma_c = solver_gamma_c.run(gamma_c_prev)
            gamma_c_new = np.array(result_gamma_c.params)
            gamma_list[_c] = gamma_c_new  # update for next class's optimisation
            gamma_new_parts.append(gamma_c_new)

        gamma_new = np.concatenate(gamma_new_parts)

        params = np.concatenate([theta_new_flat, gamma_new])

        # ==========================================================
        # LL-based convergence + best-params tracking
        # ==========================================================

        try:
            current_ll = -float(mixed_model_loglik(jnp.array(params), data, spec))
        except Exception:
            current_ll = -np.inf

        if np.isfinite(current_ll) and current_ll > best_ll:
            best_ll     = current_ll
            best_params = params.copy()

        # Divergence guard: discard this iteration's update if it is
        # non-finite or drops the LL by more than DIVERGENCE_TOL, reverting
        # to the last good iterate instead of carrying a diverged M-step
        # forward as the next E-step's warm start (see note above the loop).
        if not np.isfinite(current_ll) or current_ll < prev_ll - DIVERGENCE_TOL:
            params = prev_params.copy()
            current_ll = prev_ll
        else:
            prev_params = params.copy()

        ll_delta = abs(current_ll - prev_ll) if np.isfinite(current_ll) else np.inf
        prev_ll  = current_ll

        class_shares_tuple = tuple(float(mw) for mw in mean_w)
        if trace is not None:
            trace.append((iteration, float(T), m_iters, current_ll,
                          ll_delta, class_shares_tuple))

        if verbose:
            print(f"EM iter {iteration:3d} | T={T:.2f} | M-iters={m_iters:3d} | "
                  f"LL = {current_ll:.4f} | delta_LL = {ll_delta:.2e} | "
                  f"class_shares={' '.join(f'{mw:.2f}' for mw in mean_w)}")

        if iteration >= 3 and ll_delta < tol:
            if verbose:
                print(f"  [EM] converged at iter {iteration}  "
                      f"(delta_LL={ll_delta:.2e} < tol={tol:.0e})")
            break

    # Return best params seen (guards against last iterate being slightly worse)
    if return_trace:
        return best_params, trace
    return best_params


_hpc.fit_em = fit_em          # keep original accessible as fit_em


# ═══════════════════════════════════════════════════════════════════════
# 7a-sq. fit_em_squarem
#
#   Drop-in replacement for fit_em that applies the Squared Extrapolation
#   Method (SQUAREM) of Varadhan & Roland (2008) to accelerate convergence.
#
#   Algorithm (outer loop):
#     1. θ₁ = F(θ)      — one full E+M step
#     2. θ₂ = F(θ₁)     — second full E+M step
#     3. r = θ₁ − θ,  v = θ₂ − 2θ₁ + θ
#     4. α = min(−‖r‖/‖v‖, −1)      (step length ≤ −1)
#     5. θ_prop = θ − 2α·r + α²·v   (extrapolation)
#     6. Step-halve α toward −1 until LL(θ_prop) ≥ LL at step 1
#        or fall back to θ₂ if no gain.
#
#   All parameters (theta + gamma) are unconstrained reals so no projection
#   is needed after the extrapolation.
# ═══════════════════════════════════════════════════════════════════════

def fit_em_squarem(init_params, data, spec: ModelSpec,
                   max_iter=100, tol=1e-6, verbose=True, return_trace=False):
    """SQUAREM-accelerated EM for latent-class mixed count models.

    Each outer iteration executes exactly two full E+M steps (identical to
    the inner body of ``fit_em``) and then proposes a squared extrapolation.
    Convergence is reached in far fewer total E+M calls than standard EM for
    well-separated classes.

    Parameters
    ----------
    init_params, data, spec, max_iter, tol, verbose, return_trace
        Same semantics as :func:`fit_em`.

    Returns
    -------
    best_params : np.ndarray
        Parameter vector with the highest observed log-likelihood.
    trace : list of tuples, optional
        ``(outer_iter, em_calls, alpha, loglik, delta_ll, class_shares)``
        returned only when ``return_trace=True``.
    """
    from jax.nn import log_softmax as _log_softmax

    assert spec.latent_classes > 1, "SQUAREM-EM only needed for latent classes"

    C          = spec.latent_classes
    K_mem      = spec.K_membership
    models     = spec.models
    base_spec_nolc = replace(spec, latent_classes=1)
    # ── Per-class membership indices ───────────────────────────────────
    cm_idx = getattr(spec, 'class_membership_idx', ())
    if not cm_idx or len(cm_idx) != C - 1 or K_mem == 0:
        cm_idx = tuple(tuple(range(K_mem)) for _ in range(C - 1))
    per_class_K_mem = tuple(len(idx_tup) for idx_tup in cm_idx)
    gamma_total_size = sum(Kc + 1 for Kc in per_class_K_mem)

    class_K_base = []
    class_offsets = []
    class_base_specs = []
    class_param_index = []
    offset = 0
    for c in range(C):
        _spec_c = base_spec_nolc
        if spec.class_fixed_idx and c < len(spec.class_fixed_idx):
            cfix = spec.class_fixed_idx[c]
            if len(cfix) < spec.Kf:
                _spec_c = replace(_spec_c, Kf=len(cfix))
        _pindex_c = build_base_index(_spec_c, model=models[c])
        class_K_base.append(_pindex_c["total_params"])
        class_offsets.append(offset)
        class_base_specs.append(replace(_spec_c, model=models[c]))
        class_param_index.append(_pindex_c)
        offset += _pindex_c["total_params"]
    total_theta = offset

    params = np.array(init_params)
    N = int(data["y"].shape[0])

    # Per-class data: pre-slice Xf to each class's own fixed-effect columns
    # (matching class_base_specs[c].Kf) once, up front — every per-class
    # mixed_model_loglik call below must use this, not the unsliced `data`,
    # whenever class_fixed_idx has reduced that class's Kf below the global.
    class_data = []
    for c in range(C):
        cfix = spec.class_fixed_idx[c] if spec.class_fixed_idx and c < len(spec.class_fixed_idx) else None
        if cfix is not None and len(cfix) < data["Xf"].shape[2]:
            _dc = dict(data)
            _dc["Xf"] = data["Xf"][:, :, list(cfix)]
            class_data.append(_dc)
        else:
            class_data.append(data)

    # Build per-class Z matrices (fixed for all SQUAREM iters)
    Xmem_np = np.array(data["Xmem"])
    Z_mats = []
    for _c in range(C - 1):
        idx_tup = cm_idx[_c]
        Kc = len(idx_tup)
        if Kc > 0:
            Z_sub = np.mean(Xmem_np[:, :, list(idx_tup)], axis=1)
            Z_c = np.concatenate([np.ones((N, 1)), Z_sub], axis=1)
        else:
            Z_c = np.ones((N, 1))
        Z_mats.append(Z_c)
    Z_mats_jnp = tuple(jnp.array(Z_c) for Z_c in Z_mats)

    # ── Quick marginal loglik (no M-step) for step-halving checks ──────
    def _eval_loglik(p):
        try:
            return -float(mixed_model_loglik(jnp.array(p), data, spec))
        except Exception:
            return -np.inf

    best_params = params.copy()
    try:
        best_ll = _eval_loglik(params)
    except Exception:
        best_ll = -np.inf

    prev_ll = best_ll
    trace   = []
    em_calls = 0

    # ── Single E+M step (mirrors the body of fit_em's inner loop) ───────
    def _one_em_step(p, m_iters, T=1.0):
        theta_all = []
        for c in range(C):
            oc = class_offsets[c]; kc = class_K_base[c]
            theta_all.append(p[oc:oc + kc])
        # Extract per-class gamma
        gamma_list = []
        g_off = total_theta
        for _c in range(C - 1):
            Kc = per_class_K_mem[_c]
            gamma_list.append(p[g_off : g_off + Kc + 1])
            g_off += (Kc + 1)

        # Per-class logits
        logits_cols = []
        for _c in range(C - 1):
            logits_cols.append(Z_mats[_c] @ gamma_list[_c])
        logits_i    = np.column_stack(logits_cols)
        logits_full = np.concatenate([np.zeros((N, 1)), logits_i], axis=1)
        log_pi = np.array(_log_softmax(jnp.array(logits_full), axis=1))

        logL = np.zeros((N, C))
        for c in range(C):
            ll_ind = mixed_model_loglik(
                jnp.array(theta_all[c]), class_data[c], class_base_specs[c], indivi=True
            )
            logL[:, c] = np.array(ll_ind)

        log_num = (logL + log_pi) / T
        max_log = log_num.max(axis=1, keepdims=True)
        w = np.exp(log_num - max_log)
        w /= w.sum(axis=1, keepdims=True)
        mean_w = w.mean(axis=0)

        # M-step: outcome params
        theta_new = []
        for c in range(C):
            wc = w[:, c].copy()
            _base_c = class_base_specs[c]
            _data_c = class_data[c]

            def weighted_objective(theta_c, _wc=wc, _spec=_base_c, _data=_data_c):
                ll_ind = mixed_model_loglik(theta_c, _data, _spec, indivi=True)
                loss = -jnp.sum(jnp.array(_wc) * jnp.array(ll_ind))
                if spec.l2_penalty > 0.0 and len(theta_c) > 1:
                    loss = loss + spec.l2_penalty * jnp.sum(theta_c[1:] ** 2)
                return loss

            solver_theta = LBFGS(fun=weighted_objective, maxiter=m_iters)
            result = solver_theta.run(jnp.array(theta_all[c]))
            theta_c_mle = jnp.array(result.params)
            if spec.parte_shrinkage_in_loop:
                theta_c_mle = _parte_shrink_theta(
                    theta_c_mle, weighted_objective,
                    class_param_index[c], spec.parte_variant
                )
            theta_new.append(np.array(theta_c_mle))

        # M-step: per-class gamma (optimise each class independently)
        _raw_prop = getattr(spec, 'min_class_proportion', 0.15)
        min_prop = max(0.02, _raw_prop / max(1, C * 0.5))
        below_thresh = np.maximum(0.0, min_prop - mean_w)
        prior_weight = float(np.sum(below_thresh) * max(5.0, C * 2.0))
        _w_jnp = jnp.array(w)
        _pw    = prior_weight

        gamma_new_parts = []
        _glist = [np.copy(x) for x in gamma_list]  # mutable copy for sequential updates
        for _c in range(C - 1):
            Z_c_jnp = Z_mats_jnp[_c]
            _cc = _c

            def gamma_objective_c(gamma_c, _Zc=Z_c_jnp, _cc=_cc):
                Ncur = _Zc.shape[0]
                logit_cols = []
                for _tc in range(C - 1):
                    if _tc == _cc:
                        logit_cols.append(_Zc @ gamma_c)
                    else:
                        gc_other = _glist[_tc]
                        Z_other = Z_mats_jnp[_tc]
                        logit_cols.append(Z_other @ jnp.array(gc_other))
                li = jnp.stack(logit_cols, axis=1)
                zeros_col = jnp.zeros((Ncur, 1))
                lf = jnp.concatenate([zeros_col, li], axis=1)
                lp = _log_softmax(lf, axis=1)
                ce = -jnp.sum(_w_jnp * lp)
                log_pi_marg = jax.nn.log_softmax(
                    jnp.mean(lf, axis=0, keepdims=True), axis=1
                )
                balance_penalty = -_pw * jnp.sum(log_pi_marg)
                return ce + balance_penalty

            solver_gamma_c = LBFGS(fun=gamma_objective_c, maxiter=m_iters)
            result_gamma_c = solver_gamma_c.run(jnp.array(_glist[_c]))
            gc_new = np.array(result_gamma_c.params)
            _glist[_c] = gc_new
            gamma_new_parts.append(gc_new)
        gamma_new = np.concatenate(gamma_new_parts)

        new_p = np.concatenate([np.concatenate(theta_new), gamma_new])
        return new_p, mean_w

    for outer_iter in range(max_iter):
        warmup_frac = min(1.0, outer_iter / max(1, max_iter * 0.4))
        T      = max(1.0, 2.0 - warmup_frac)
        m_iters = min(50 + 25 * (outer_iter // 3), 300)

        # Two full E+M steps
        params1, mean_w1 = _one_em_step(params,  m_iters, T=T)
        em_calls += 1
        params2, mean_w2 = _one_em_step(params1, m_iters, T=T)
        em_calls += 1

        # Collapse guard (after warmup) — threshold relaxes with more classes
        _collapse_cutoff = max(0.005, 0.02 / C)
        if outer_iter >= 3 and np.any(mean_w2 < _collapse_cutoff):
            if verbose:
                print(f"  [SQUAREM] class collapse at outer_iter {outer_iter} "
                      f"(min mean weight {mean_w2.min():.4f} < {_collapse_cutoff:.4f}) — stopping early")
            params = params2
            break

        r = params1 - params
        v = params2 - 2.0 * params1 + params
        norm_v = np.linalg.norm(v)

        if norm_v < 1e-14:
            params   = params2
            ll_cand  = _eval_loglik(params2)
            used_alpha = -1.0
        else:
            alpha  = min(-np.linalg.norm(r) / norm_v, -1.0)
            ll1    = _eval_loglik(params1)

            accepted  = False
            ll_cand   = _eval_loglik(params2)
            used_alpha = -1.0
            p_acc      = params2

            for _ in range(10):
                p_prop = params - 2.0 * alpha * r + alpha ** 2 * v
                ll_prop = _eval_loglik(p_prop)
                if np.isfinite(ll_prop) and ll_prop >= ll1:
                    p_acc      = p_prop
                    ll_cand    = ll_prop
                    used_alpha = alpha
                    accepted   = True
                    break
                alpha = (alpha + (-1.0)) / 2.0  # halve toward α = −1

            params = p_acc

        current_ll = _eval_loglik(params)
        if np.isfinite(current_ll) and current_ll > best_ll:
            best_ll     = current_ll
            best_params = params.copy()

        ll_delta = abs(current_ll - prev_ll) if np.isfinite(current_ll) else np.inf
        prev_ll  = current_ll

        class_shares = tuple(float(mw) for mw in mean_w2)
        if return_trace:
            trace.append((outer_iter, em_calls, float(used_alpha), current_ll,
                          ll_delta, class_shares))

        if verbose:
            print(f"SQUAREM iter {outer_iter:3d} | em_calls={em_calls:4d} | "
                  f"α={used_alpha:.3f} | LL={current_ll:.4f} | "
                  f"delta_LL={ll_delta:.2e} | "
                  f"shares={' '.join(f'{mw:.2f}' for mw in mean_w2)}")

        if outer_iter >= 3 and ll_delta < tol:
            if verbose:
                print(f"  [SQUAREM] converged at outer_iter {outer_iter}  "
                      f"(delta_LL={ll_delta:.2e} < tol={tol:.0e}, em_calls={em_calls})")
            break

    if return_trace:
        return best_params, trace
    return best_params


# Make SQUAREM the default EM for all callers (experiment_package, etc.)
# The original standard EM remains available as fit_em for direct use.
_hpc.fit_em = fit_em_squarem


# ═══════════════════════════════════════════════════════════════════════
# 7b. _seed_classes_from_clusters
#
#     Given a single-class warm-start theta_1, cluster observations in
#     the space of (fixed covariates, per-obs LL) and fit a per-cluster
#     weighted model.  Returns C genuinely differentiated theta vectors,
#     which prevents EM from collapsing all classes onto the same solution.
# ═══════════════════════════════════════════════════════════════════════

def _seed_classes_from_clusters(
    theta_1: np.ndarray,
    data: dict,
    base_spec: ModelSpec,
    C: int,
    K_base: int,      # default K_base (for the first class); may differ per class
    rng: np.random.Generator,
    class_K_base: list = None,   # per-class K_base (optional; if provided, overrides K_base per class)
) -> list:
    """
    Returns a list of C numpy arrays (each with its own shape) to use as
    per-class starting parameters.

    Strategy
    --------
    1. Compute per-observation log-likelihoods under the single-class fit.
    2. Build a feature matrix from the mean fixed-effect covariates Xf
       and the individual LLs.
    3. K-means cluster observations into C groups.
    4. For each cluster, shift only the intercept (params[0]) by the
       log-ratio of cluster mean outcome to overall mean outcome.
       All other parameters stay at theta_1 + small noise.
       This is numerically stable and avoids per-cluster LBFGS divergence.
    """
    from sklearn.cluster import KMeans

    ll_ind = np.array(
        mixed_model_loglik(theta_1, data, base_spec, indivi=True)
    )                                                       # (N,)

    Xf = np.array(data["Xf"])                              # (N, P, Kf)
    if Xf.ndim == 3:
        Xf_mean = Xf.mean(axis=1)                          # (N, Kf)
    else:
        Xf_mean = Xf

    features = np.concatenate([Xf_mean, ll_ind[:, None]], axis=1)
    col_std  = features.std(0) + 1e-8
    features_scaled = (features - features.mean(0)) / col_std

    try:
        km = KMeans(
            n_clusters=C,
            n_init=10,
            max_iter=300,
            random_state=int(rng.integers(2**31)),
        )
        labels = km.fit_predict(features_scaled)
    except Exception:
        labels = np.arange(features_scaled.shape[0]) % C

    # Overall mean outcome (collapse panel dimension first)
    y_all = np.array(data["y"])                            # (N, P, 1) or (N,)
    if y_all.ndim == 3:
        y_all = y_all.mean(axis=1).squeeze(-1)             # (N,)
    elif y_all.ndim == 2:
        y_all = y_all.mean(axis=1)
    y_global_mean = float(np.maximum(y_all.mean(), 1e-3))

    # Normalise class_K_base if not provided
    if class_K_base is None:
        class_K_base = [K_base] * C

    thetas = []
    for c in range(C):
        in_cluster = labels == c
        n_c = int(in_cluster.sum())
        kc = class_K_base[c]

        # Start from theta_1, truncate or pad to match this class's K_base
        if len(theta_1) >= kc:
            theta_c = theta_1[:kc].copy()
        else:
            theta_c = np.zeros(kc)
            theta_c[:len(theta_1)] = theta_1

        if n_c >= 3:
            y_c_mean = float(np.maximum(y_all[in_cluster].mean(), 1e-3))
            # Shift intercept so the class predicts y_c_mean at the
            # global covariate average — prevents EM collapse.
            delta_intercept = np.log(y_c_mean) - np.log(y_global_mean)
            theta_c[0] = theta_c[0] + delta_intercept
        else:
            theta_c[0] = theta_c[0] + rng.normal(0, 0.3)

        # Small noise on all other structural parameters (not dispersion)
        n_struct = min(kc - 1, base_spec.Kf - 1)
        if n_struct > 0:
            theta_c[1:1 + n_struct] += rng.normal(0, 0.05, n_struct)

        # Warm weighted-LBFGS per cluster (only if same model and sufficient data)
        if n_c >= 10 and len(theta_1) == kc:
            try:
                # Hard cluster weights: 1.0 inside cluster, 0.0 outside
                w_hard = np.zeros(ll_ind.shape[0], dtype=float)
                w_hard[in_cluster] = 1.0

                def _cluster_obj(theta_c_jax, _w=w_hard):
                    ll_c = mixed_model_loglik(
                        theta_c_jax, data, base_spec, indivi=True
                    )
                    return -jnp.sum(jnp.array(_w) * jnp.array(ll_c))

                from jaxopt import LBFGS as _LBFGS
                _sol = _LBFGS(fun=_cluster_obj, maxiter=30).run(jnp.array(theta_c))
                theta_c = np.array(_sol.params)
            except Exception:
                pass  # Keep intercept-shifted theta_c on failure

        thetas.append(theta_c)

    return thetas


# ═══════════════════════════════════════════════════════════════════════
# 8.  print_summary — extended with membership gamma section
# ═══════════════════════════════════════════════════════════════════════

def print_summary(result, objective, data, spec: ModelSpec,
                  param_index, se=None, return_df=None):
    """
    Full model summary.  For LC models with membership variables the
    gamma matrix is printed as a proper table (one column per membership
    variable, one row per class).
    """

    # ── LC DISPATCH ─────────────────────────────────────────────────
    if spec.latent_classes > 1:
        C         = spec.latent_classes
        K_mem     = spec.K_membership
        models    = spec.models  # per-class model strings
        base_spec_nolc = replace(spec, latent_classes=1)

        # Compute per-class param sizes
        class_K_base = list(param_index.get("class_K_base", [build_base_index(base_spec_nolc)["total_params"]] * C))
        class_offsets = list(param_index.get("class_offsets", [i * class_K_base[0] for i in range(C)]))
        class_models = list(param_index.get("class_models", models))

        params_np = np.asarray(result.params if hasattr(result, "params")
                               else result.x)

        if se is None:
            se_np = np.asarray(compute_standard_errors(params_np, objective))
        else:
            se_np = np.asarray(se)

        # Extract per-class thetas (jagged)
        theta_all = []
        se_all = []
        for c in range(C):
            oc = class_offsets[c]
            kc = class_K_base[c]
            theta_all.append(params_np[oc:oc + kc])
            se_all.append(se_np[oc:oc + kc])

        total_theta = class_offsets[-1] + class_K_base[-1] if C > 0 else 0
        # ── Per-class gamma extraction (jagged) ─────────────────────────
        cm_idx = getattr(spec, 'class_membership_idx', ())
        if not cm_idx or len(cm_idx) != C - 1 or K_mem == 0:
            cm_idx = tuple(tuple(range(K_mem)) for _ in range(C - 1))
        gamma_list = []
        se_gamma_list = []
        g_off = total_theta
        for _c in range(C - 1):
            Kc = len(cm_idx[_c])
            gamma_list.append(params_np[g_off : g_off + Kc + 1])
            se_gamma_list.append(se_np[g_off : g_off + Kc + 1])
            g_off += (Kc + 1)

        # Compute marginal class probabilities with per-class Z mats
        Xmem_np = np.array(data["Xmem"])
        N = int(data["y"].shape[0])
        logits_cols = []
        for _c in range(C - 1):
            idx_tup = cm_idx[_c]
            if len(idx_tup) > 0:
                Z_sub = np.mean(Xmem_np[:, :, list(idx_tup)], axis=1)
                Z_c = np.concatenate([np.ones((N, 1)), Z_sub], axis=1)
            else:
                Z_c = np.ones((N, 1))
            logits_cols.append(Z_c @ gamma_list[_c])
        logits_i    = np.column_stack(logits_cols)
        logits_full = np.concatenate([np.zeros((N, 1)), logits_i], axis=1)
        pi = np.exp(logits_full) / np.exp(logits_full).sum(axis=1, keepdims=True)
        pi_mean = pi.mean(axis=0)  # marginal class probs

        print("\n" + "=" * 65)
        print("   LATENT CLASS MIXED MODEL SUMMARY")
        if K_mem > 0:
            print(f"   Membership covariates: {list(spec.membership_names)}")
            for _c in range(C - 1):
                idx_tup = cm_idx[_c]
                if idx_tup:
                    class_vars = [spec.membership_names[j] for j in idx_tup if j < len(spec.membership_names)]
                    print(f"     Class {_c+2} membership vars: {class_vars}")
        print(f"   Per-class distributions: {list(models)}")
        print("=" * 65)

        # ── Per-class outcome params ──────────────────────────────
        class DummyRes:
            pass

        for c in range(C):
            _model_c = models[c]
            _theta_c = theta_all[c]  # per-class outcome params only

            # Build per-class spec with correct Kf/fixed_names when
            # class_fixed_idx is used (each class may use different variables).
            _spec_c = base_spec_nolc
            _data_c = data
            cfix = spec.class_fixed_idx[c] if spec.class_fixed_idx and c < len(spec.class_fixed_idx) else None
            if cfix is not None and len(cfix) < data["Xf"].shape[2]:
                _fn_all = list(spec.fixed_names) if spec.fixed_names else []
                _fn_c = tuple(_fn_all[i] for i in cfix if i < len(_fn_all))
                _spec_c = replace(_spec_c, Kf=len(cfix), fixed_names=_fn_c)
                _data_c = dict(data)
                _data_c["Xf"] = data["Xf"][:, :, list(cfix)]
            _spec_c = replace(_spec_c, model=_model_c, latent_classes=1)
            _base_idx_c = build_base_index(_spec_c, model=_model_c)

            print(f"\n{'#' * 20}  CLASS {c+1}  (pi = {float(pi_mean[c]):.4f})  "
                  f"[{_model_c.upper()}]  {'#' * 20}\n")
            dummy       = DummyRes()
            dummy.params = _theta_c
            dummy.x      = _theta_c

            obj_c = partial(
                mixed_model_loglik, data=_data_c,
                spec=_spec_c
            )
            try:
                print_summary(
                    result=dummy, objective=obj_c, data=_data_c,
                    spec=_spec_c,
                    param_index=_base_idx_c, se=se_all[c]
                )
            except Exception as exc:
                print(f"\n  [fitness error] {exc}")
                # Fallback: print raw params without SEs
                for i, val in enumerate(_theta_c):
                    print(f"  param[{i}] = {float(val):+.6f}")
                print()

        # ── Membership gamma (per-class) ──────────────────────────
        # This whole block is display-only.  Wrap it so that any
        # (e.g. formatting) failure cannot abort the fit — the fitted
        # model and its quality metrics are reported regardless.
        try:
            print("\n" + "=" * 65)
            print("   CLASS-MEMBERSHIP EQUATION")
            print("   log[pi_c(n) / pi_1(n)] = g_c0  +  sum_k g_ck * z_nk")
            print("   (Class 1 is the reference; all coefficients vs class 1)")
            print("=" * 65)

            for _c in range(C - 1):
                class_num = _c + 2
                idx_tup   = cm_idx[_c]
                Kc        = len(idx_tup)
                if Kc > 0:
                    col_names = [spec.membership_names[j] for j in idx_tup if j < len(spec.membership_names)]
                else:
                    col_names = []
                col_names_full = ["(intercept)"] + col_names
                gamma_c = np.ravel(gamma_list[_c])
                se_c    = np.ravel(se_gamma_list[_c])

                print(f"\n  --- Class {class_num} vs Class 1 ---")
                print(f"  {'Parameter':>18}  {'Estimate':>10}  {'SE':>8}  {'z':>7}  {'p':>8}")
                print("  " + "-" * 55)
                for k, cn in enumerate(col_names_full):
                    g     = float(gamma_c[k]) if k < len(gamma_c) else float("nan")
                    sg    = float(se_c[k])  if k < len(se_c)     else float("nan")
                    z_val = float(g / sg) if sg > 1e-12 else 0.0
                    p_val = float(2 * (1 - scipy_stats.norm.cdf(abs(z_val))))
                    stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 \
                            else "*" if p_val < 0.10 else ""
                    print(f"  {cn:>18}  {g:>+10.4f}  {sg:>8.4f}  {z_val:>7.3f}  {p_val:>7.4f}{' '+stars}")

            print(f"\n  NOTE: g_c0 is the class-c log-odds intercept vs class 1.")
            if K_mem > 0:
                print("  g_ck > 0: higher value of z_k -> higher probability of class c+1.")

            # ── Class shares ─────────────────────────────────────────
            print("\n" + "-" * 65)
            print("  MARGINAL CLASS PROBABILITIES (at sample-mean covariates)\n")
            for c in range(C):
                print(f"  pi_{c+1} = {float(pi_mean[c]):.6f}")

            print("\n" + "=" * 65 + "\n")
        except Exception as exc:
            print(f"\n  [fitness error] membership summary: {exc}")
            print("  (model still fitted; continuing with raw results)")

        # ── Build summary dict for LC models ─────────────────────
        lc_ll = float(-objective(params_np))
        lc_k  = len(params_np)
        lc_n  = data["y"].shape[0]
        return {
            "loglik":    lc_ll,
            "num_parm":  lc_k,
            "n_obs":     lc_n,
            "aic":       2 * lc_k - 2 * lc_ll,
            "bic":       lc_k * np.log(lc_n) - 2 * lc_ll,
            "latent_classes": C,
            "class_probs":    pi.tolist(),
        }

    # ── SINGLE-CLASS SUMMARY (delegate to main_hpc's print_summary) ─
    _orig_print = _hpc.__dict__.get("_orig_print_summary")
    if _orig_print is None:
        # Fall back: minimal inline summary
        params_np = np.asarray(result.params if hasattr(result, "params")
                               else result.x)
        if se is None:
            se_np = np.asarray(compute_standard_errors(params_np, objective))
        else:
            se_np = np.asarray(se)

        z_vals = params_np / np.where(se_np > 1e-12, se_np, 1e-12)
        p_vals = 2 * (1 - scipy_stats.norm.cdf(np.abs(z_vals)))

        final_ll = float(-objective(params_np))
        k, n     = len(params_np), data["y"].shape[0]

        df_out = pd.DataFrame({
            "Estimate": params_np,
            "Std.Err":  se_np,
            "z-value":  z_vals,
            "p-value":  p_vals,
        })

        print("\n================ MODEL SUMMARY ================\n")
        print(df_out.to_string(float_format="%.4f"))
        print(f"\nLog-Likelihood: {final_ll:.4f}")
        print(f"AIC: {2*k - 2*final_ll:.4f}")
        print(f"BIC: {k*np.log(n) - 2*final_ll:.4f}\n")

        return {
            "loglik":   final_ll,
            "num_parm": k,
            "n_obs":    n,
            "aic":      2 * k - 2 * final_ll,
            "bic":      k * np.log(n) - 2 * final_ll,
        }
    else:
        _orig_print(result, objective, data, spec, param_index, se=se)
        params_np = np.asarray(result.params if hasattr(result, "params")
                               else result.x)
        final_ll = float(-objective(params_np))
        k, n = len(params_np), data["y"].shape[0]
        return {
            "loglik":   final_ll,
            "num_parm": k,
            "n_obs":    n,
            "aic":      2 * k - 2 * final_ll,
            "bic":      k * np.log(n) - 2 * final_ll,
        }


# Keep the original print_summary available under a private alias so the
# LC dispatcher can delegate back to it.
_hpc._orig_print_summary = _hpc.print_summary
_hpc.print_summary       = print_summary


# ═══════════════════════════════════════════════════════════════════════
# 9.  unpack_lc_params — helper for extracting per-class thetas + gamma
# ═══════════════════════════════════════════════════════════════════════

def unpack_lc_params(params, spec: ModelSpec):
    """
    Extract per-class theta arrays and gamma list from flat LC params.

    Returns
    -------
    theta_list : list of np.ndarray   (one per class, lengths may differ)
    gamma_list : list of np.ndarray   (one per non-reference class, lengths may differ)
    pindex     : dict                 param index with class_offsets/class_K_base
    """
    import numpy as _np
    C     = spec.latent_classes
    K_mem = spec.K_membership
    pindex = build_param_index(spec)
    class_offsets = list(pindex["class_offsets"])
    class_K_base  = list(pindex["class_K_base"])
    params_np = _np.asarray(params)

    theta_list = []
    for c in range(C):
        oc = class_offsets[c]
        kc = class_K_base[c]
        theta_list.append(params_np[oc:oc + kc])

    total_theta = class_offsets[-1] + class_K_base[-1] if C > 0 else 0
    gamma_list = []
    # Per-class gamma: use class_gamma_sizes from pindex
    g_offsets = pindex.get("class_gamma_offsets", ())
    g_sizes   = pindex.get("class_gamma_sizes", ())
    if g_offsets and g_sizes:
        for _c in range(C - 1):
            g_off = g_offsets[_c]
            g_sz = g_sizes[_c]
            gamma_list.append(params_np[g_off : g_off + g_sz])
    else:
        # Fallback: rectangular gamma
        gamma_flat = params_np[total_theta:]
        for _c in range(C - 1):
            gamma_c = gamma_flat[_c * (K_mem + 1) : (_c + 1) * (K_mem + 1)]
            gamma_list.append(gamma_c)

    return theta_list, gamma_list, pindex


def compute_lc_posteriors(params, data, spec: ModelSpec):
    """
    Compute posterior class probabilities for an LC model.

    Returns
    -------
    posterior  : np.ndarray  (N, C)  softmax(log_joint)
    log_pi     : np.ndarray  (N, C)  prior log class probabilities
    logL       : np.ndarray  (N, C)  per-class log-likelihoods
    """
    import numpy as _np

    C     = spec.latent_classes
    K_mem = spec.K_membership
    models  = spec.models
    base_spec_nolc = replace(spec, latent_classes=1)

    theta_list, gamma_list, pindex = unpack_lc_params(params, spec)

    N = int(data["y"].shape[0])

    # Prior log-probabilities using per-class Z matrices
    cm_idx = getattr(spec, 'class_membership_idx', ())
    if not cm_idx or len(cm_idx) != C - 1 or K_mem == 0:
        cm_idx = tuple(tuple(range(K_mem)) for _ in range(C - 1))
    Xmem = _np.array(data["Xmem"])
    logits_cols = []
    for _c in range(C - 1):
        idx_tup = cm_idx[_c]
        if len(idx_tup) > 0:
            Z_sub = _np.mean(Xmem[:, :, list(idx_tup)], axis=1)
            Z_c = _np.concatenate([_np.ones((N, 1)), Z_sub], axis=1)
        else:
            Z_c = _np.ones((N, 1))
        logits_cols.append(Z_c @ gamma_list[_c])
    logits_i    = _np.column_stack(logits_cols)
    logits_full = _np.concatenate([_np.zeros((N, 1)), logits_i], axis=1)
    log_pi = _np.array(jax.nn.log_softmax(jnp.array(logits_full), axis=1))

    # Per-class individual log-likelihoods
    logL = _np.zeros((N, C))
    for c in range(C):
        _data_c = data
        _base_spec_c = base_spec_nolc
        cfix = spec.class_fixed_idx[c] if spec.class_fixed_idx and c < len(spec.class_fixed_idx) else None
        if cfix is not None and len(cfix) < data["Xf"].shape[2]:
            _data_c = dict(data)
            _data_c["Xf"] = data["Xf"][:, :, list(cfix)]
            _base_spec_c = replace(_base_spec_c, Kf=len(cfix))
        base_spec_c = replace(_base_spec_c, model=models[c])
        ll_ind = mixed_model_loglik(
            jnp.array(theta_list[c]), _data_c, base_spec_c, indivi=True
        )
        logL[:, c] = _np.array(ll_ind)

    # Full posterior
    log_joint = logL + log_pi
    log_joint -= log_joint.max(axis=1, keepdims=True)
    posterior = _np.exp(log_joint)
    posterior /= posterior.sum(axis=1, keepdims=True)

    return posterior, log_pi, logL


# ═══════════════════════════════════════════════════════════════════════
# Public re-exports for experiment_package.py
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "ModelSpec",
    "build_param_index",
    "build_jax_data",
    "build_model_from_manual_spec",
    "parse_manual_spec",
    "mixed_model_loglik",
    "mixed_model_loglik_reg",
    "_l2_penalty",
    "fit_em",
    "fit_em_squarem",
    "print_summary",
    "unpack_lc_params",
    "compute_lc_posteriors",
]
