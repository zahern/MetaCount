"""
adaptive_search.py
==================
Adaptive model-search infrastructure for MetaCount:

  1. IdentifiabilityChecker
     Post-fit analysis that extracts per-variable t-statistics, flags near-zero
     or poorly-identified parameters, and returns structured diagnostics.

  2. AdaptiveRoleMemory
     Accumulates (variable, role) -> (alpha, beta) Beta-distribution counts
     across fitness evaluations using Thompson Sampling.  Feeds back into
     sample_allowed_role() so the SA/HS gradually avoids known-bad assignments.

  3. SpecificationBandit  (contextual LinUCB)
     A proper contextual multi-armed bandit where each arm is a
     (variable_idx, role_code) change-proposal.  Context features describe the
     current model's fit quality and per-variable diagnostics.  Arms are scored
     by UCB = x^T theta_hat + alpha * sqrt(x^T A^{-1} x), balancing exploration
     and exploitation.

  4. BanditGuidedSA
     Wraps ExperimentBuilder.run() and replaces the random SA proposal
     distribution with bandit-guided proposals.  The bandit is updated after
     every accepted or rejected fitness evaluation.

  5. PostFitPruner
     After any search, prunes the best solution by greedily removing variables
     that contribute < 1 standard error of effect (backward elimination guided
     by individual t-statistics).

Why contextual bandit (not full RL / DQN)?
------------------------------------------
- Reward IS immediately observable (BIC after each evaluation).  No credit
  assignment across time steps is needed, so RL's Q-function adds complexity
  without benefit over a well-designed bandit.
- The action space (variable × role) is small and structured — ideal for
  tabular Thompson Sampling or LinUCB.
- LinUCB generalises across variable sets via context features, whereas a
  Q-table would need one entry per (state, action) pair (exponential states).
- Thompson Sampling converges to optimal arm selection in O(sqrt(T log K))
  regret vs O(T) for random search — provably more efficient exploration.

Usage
-----
from adaptive_search import BanditGuidedSA, PostFitPruner

builder  = ExperimentBuilder(df, id_col="ID", y_col="CRASHES")
evaluator = builder.build_evaluator(max_latent_classes=2, R=200)

bandit_sa = BanditGuidedSA(builder=builder, evaluator=evaluator)
result    = bandit_sa.run(max_iter=3000, seed=0)

pruned = PostFitPruner(builder, evaluator).prune(result["best_solution"], model="nb")
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 0. Fast diagonal Hessian t-statistics  (O(p) JAX forward-over-reverse)
# ─────────────────────────────────────────────────────────────────────────────

def _fast_tstats(
    params_jnp,   # jnp.ndarray of shape (p,)
    objective,    # callable: params -> scalar NLL  (minimised)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-parameter t-statistics using the *diagonal* of the Hessian.

    Why diagonal instead of full Hessian?
    - Full Hessian costs O(p²) JVP passes — expensive for large models.
    - Diagonal requires O(p) passes: one jvp(grad, e_i) per parameter.
    - The diagonal H_ii is sufficient to compute SE_i = 1/sqrt(H_ii)
      and t_i = theta_i / SE_i for identifiability screening.

    How it works (forward-over-reverse):
      grad(L)(θ)  is the gradient vector.
      jvp(grad, θ, e_i)  pushes a basis vector e_i through the gradient,
      returning the i-th column of the Hessian.  We only keep element [i]
      of that column (the diagonal entry).

    Parameters
    ----------
    params_jnp : jnp.ndarray
        Current parameter vector at which to evaluate the Hessian.
    objective : callable
        Function ``f(params) -> scalar`` that computes the NLL.
        Must be JAX-differentiable (no numpy side-effects inside).

    Returns
    -------
    t_stats : np.ndarray, shape (p,)
        t_i = θ_i / SE_i.  NaN where SE is not computable.
    se : np.ndarray, shape (p,)
        Standard errors.  NaN where H_ii <= 0 (direction not identified).
    """
    import jax
    import jax.numpy as jnp

    p        = len(params_jnp)
    grad_fn  = jax.grad(objective)
    diag_h   = np.empty(p, dtype=float)

    for i in range(p):
        e_i       = jnp.zeros(p).at[i].set(1.0)
        _, col_i  = jax.jvp(grad_fn, (params_jnp,), (e_i,))
        diag_h[i] = float(col_i[i])

    # SE = 1/sqrt(H_ii); valid only when H_ii > 0 (PD direction at optimum)
    se = np.where(
        diag_h > 0,
        1.0 / np.sqrt(np.maximum(diag_h, 1e-12)),
        np.nan,
    )
    params_np = np.asarray(params_jnp, dtype=float)
    t_stats   = np.where(np.isfinite(se), params_np / se, np.nan)
    return t_stats, se


# ─────────────────────────────────────────────────────────────────────────────
# 1. IdentifiabilityChecker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParamDiagnostic:
    name: str
    estimate: float
    se: float
    t_stat: float
    is_weak: bool          # |t| < t_threshold
    is_extreme: bool       # |estimate| > extreme_threshold
    is_nonfinite: bool     # SE or estimate is nan/inf
    reason: str            # human-readable diagnosis


class IdentifiabilityChecker:
    """
    Post-fit diagnostics that extract per-variable t-statistics and flag
    identifiability problems.

    Flags raised
    ------------
    - ``near_zero``:  |t-stat| < t_threshold (default 1.0).
      The variable contributes negligibly — the model is
      effectively the same without it, wasting a parameter.

    - ``extreme``:  |estimate| > extreme_threshold (default 15).
      Likely a perfect predictor, collinear variable, or
      numerical blow-up in the optimizer.

    - ``nonfinite``:  SE or estimate is nan or inf.
      Flat likelihood direction — parameter is unidentified.

    - ``high_cond``:  Hessian condition number > cond_threshold (default 1e8).
      The model as a whole is on the verge of non-identification.

    The returned diagnostics include per-variable verdicts AND a global
    identifiability verdict that combines all signals.
    """

    def __init__(
        self,
        t_threshold:       float = 1.0,
        extreme_threshold: float = 15.0,
        cond_threshold:    float = 1e8,
    ):
        self.t_threshold       = t_threshold
        self.extreme_threshold = extreme_threshold
        self.cond_threshold    = cond_threshold

    def check(
        self,
        params:    np.ndarray,
        objective,          # callable: objective(params) -> scalar LL
        spec,               # ModelSpec with .fixed_names etc.
        silent:    bool = True,
    ) -> Dict[str, Any]:
        """
        Run all identifiability checks on a fitted model.

        Returns
        -------
        dict with keys:
          'params'       — raw param vector
          'diagnostics'  — list[ParamDiagnostic]
          'weak_names'   — set of variable names with |t| < threshold
          'extreme_names'— set of variable names with extreme coefficients
          'nonfinite_names'— set of unidentified variable names
          'hess_cond'    — Hessian condition number
          'globally_ok'  — bool: True if no critical issues found
        """
        try:
            import jax
            import jax.numpy as jnp
            from main_hpc import compute_standard_errors
        except ImportError:
            return {"globally_ok": True, "diagnostics": [], "weak_names": set(),
                    "extreme_names": set(), "nonfinite_names": set(), "hess_cond": 1.0}

        params_np = np.asarray(params, dtype=float)

        # ── Standard errors via Hessian ───────────────────────────────
        try:
            se_np = np.asarray(compute_standard_errors(params_np, objective))
        except Exception:
            se_np = np.full_like(params_np, np.nan)

        t_stats = np.where(
            np.isfinite(se_np) & (se_np > 0),
            params_np / se_np,
            np.nan,
        )

        # ── Hessian condition number ──────────────────────────────────
        hess_cond = 1.0
        try:
            hess_np = np.asarray(jax.hessian(objective)(jnp.asarray(params_np)))
            hess_np = np.where(np.isfinite(hess_np), hess_np, 0.0)
            eigvals = np.linalg.eigvalsh(hess_np)
            pos = eigvals[eigvals > 0]
            if len(pos) > 0:
                hess_cond = float(pos.max() / pos.min())
        except Exception:
            pass

        # ── Build per-parameter diagnostics ──────────────────────────
        # Collect parameter names from spec
        param_names: List[str] = []
        if hasattr(spec, "fixed_names"):
            param_names.extend(list(spec.fixed_names))
        if hasattr(spec, "random_ind_names"):
            for nm in spec.random_ind_names:
                param_names.append(f"{nm}:mean")
                param_names.append(f"{nm}:sd")
        if hasattr(spec, "random_cor_names"):
            for nm in spec.random_cor_names:
                param_names.append(f"{nm}:mean_cor")
        # Pad or trim to match param vector length
        while len(param_names) < len(params_np):
            param_names.append(f"param_{len(param_names)}")
        param_names = param_names[:len(params_np)]

        diagnostics: List[ParamDiagnostic] = []
        weak_names:      set = set()
        extreme_names:   set = set()
        nonfinite_names: set = set()

        for i, nm in enumerate(param_names):
            est = float(params_np[i])
            se  = float(se_np[i]) if i < len(se_np) else np.nan
            ts  = float(t_stats[i]) if i < len(t_stats) else np.nan

            is_nonfinite = not (np.isfinite(est) and np.isfinite(se))
            is_weak      = np.isfinite(ts) and abs(ts) < self.t_threshold
            is_extreme   = np.isfinite(est) and abs(est) > self.extreme_threshold

            reasons = []
            if is_nonfinite:
                reasons.append("non-finite SE (unidentified direction)")
            if is_weak:
                reasons.append(f"|t|={abs(ts):.2f} < {self.t_threshold} (near-zero effect)")
            if is_extreme:
                reasons.append(f"|coef|={abs(est):.1f} > {self.extreme_threshold} (extreme)")
            reason = "; ".join(reasons) if reasons else "OK"

            diag = ParamDiagnostic(
                name=nm, estimate=est, se=se, t_stat=ts,
                is_weak=is_weak, is_extreme=is_extreme,
                is_nonfinite=is_nonfinite, reason=reason,
            )
            diagnostics.append(diag)

            # Strip distribution suffix to get base variable name
            base_nm = nm.split(":")[0].replace("__INTERCEPT__", "__intercept__")
            if is_nonfinite:
                nonfinite_names.add(base_nm)
            if is_weak:
                weak_names.add(base_nm)
            if is_extreme:
                extreme_names.add(base_nm)

        globally_ok = (
            len(nonfinite_names) == 0
            and len(extreme_names) == 0
            and hess_cond < self.cond_threshold
        )

        result = {
            "params":          params_np,
            "diagnostics":     diagnostics,
            "weak_names":      weak_names,
            "extreme_names":   extreme_names,
            "nonfinite_names": nonfinite_names,
            "hess_cond":       hess_cond,
            "globally_ok":     globally_ok,
        }

        if not silent:
            self._print(result)

        return result

    def _print(self, result: dict) -> None:
        print("\n  IDENTIFIABILITY DIAGNOSTICS")
        print(f"  Hessian condition number : {result['hess_cond']:.2e}")
        if result["globally_ok"]:
            print("  Status: OK")
        else:
            print(f"  Status: ISSUES FOUND")
            if result["nonfinite_names"]:
                print(f"  Unidentified params : {sorted(result['nonfinite_names'])}")
            if result["extreme_names"]:
                print(f"  Extreme estimates   : {sorted(result['extreme_names'])}")

        print(f"\n  {'Parameter':<30}  {'Estimate':>10}  {'SE':>8}  {'t-stat':>8}  Status")
        print("  " + "-" * 70)
        for d in result["diagnostics"][:30]:  # cap at 30 rows
            flag = ""
            if d.is_nonfinite:  flag = " [UNIDENTIFIED]"
            elif d.is_extreme:  flag = " [EXTREME]"
            elif d.is_weak:     flag = " [WEAK]"
            ts_str = f"{d.t_stat:>+8.2f}" if np.isfinite(d.t_stat) else f"{'nan':>8}"
            se_str = f"{d.se:>8.4f}"      if np.isfinite(d.se)      else f"{'nan':>8}"
            print(f"  {d.name:<30}  {d.estimate:>+10.4f}  {se_str}  {ts_str}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. AdaptiveRoleMemory  (Thompson Sampling per variable × role)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveRoleMemory:
    """
    Maintains Beta(alpha, beta) priors over each (variable_idx, role_code) pair.

    Thompson Sampling update rule:
      success (good fit, |t| >= threshold) → alpha += 1
      failure (weak fit, |t| < threshold)  → beta  += 1

    Sampling:
      For each arm (var, role), sample theta ~ Beta(alpha, beta).
      Higher theta → arm more likely to be proposed.

    Integration:
      Call update() after each fitness evaluation.
      Call get_role_weights(var_idx) to get a probability vector over roles
      that can replace or blend with the existing ROLE_PROBS.
    """

    def __init__(
        self,
        n_vars:      int,
        n_roles:     int = 9,
        prior_alpha: float = 1.0,    # uniform Beta(1,1) = uniform prior
        prior_beta:  float = 1.0,
        blend:       float = 0.5,    # 0 = pure bandit, 1 = pure ROLE_PROBS base
    ):
        self.n_vars    = n_vars
        self.n_roles   = n_roles
        self.blend     = blend
        # Alpha and beta counts: shape (n_vars, n_roles)
        self.alpha = np.full((n_vars, n_roles), prior_alpha)
        self.beta  = np.full((n_vars, n_roles), prior_beta)
        # Visit / violation counts (compatible with existing Solvers_METAJAX hook)
        self._visit_counts    = defaultdict(int)
        self._violation_counts = defaultdict(int)

    def update(
        self,
        var_idx:  int,
        role:     int,
        success:  bool,
    ) -> None:
        """Record outcome for (var_idx, role)."""
        if role < 0 or role >= self.n_roles:
            return
        if success:
            self.alpha[var_idx, role] += 1.0
        else:
            self.beta[var_idx, role]  += 1.0
        # Also update legacy counters so Solvers_METAJAX hook works
        var_key = (var_idx, role)
        self._visit_counts[var_key] += 1
        if not success:
            self._violation_counts[var_key] += 1

    def update_from_diagnostics(
        self,
        decision:    np.ndarray,
        diagnostics: dict,
        var_names:   List[str],
    ) -> None:
        """
        Bulk update from a full IdentifiabilityChecker result.
        For each active variable in the decision vector, check if its name
        appears in weak/extreme/nonfinite sets and update accordingly.
        """
        n_vars = len(var_names)
        weak      = diagnostics.get("weak_names", set())
        extreme   = diagnostics.get("extreme_names", set())
        nonfinite = diagnostics.get("nonfinite_names", set())
        bad_names = weak | extreme | nonfinite

        for i in range(min(n_vars, len(decision))):
            role = int(decision[i])
            nm   = var_names[i]
            if role == 0:
                continue  # excluded — no signal
            success = nm not in bad_names
            self.update(i, role, success)

    def get_role_weights(
        self,
        var_idx:     int,
        allowed_roles: List[int],
        base_probs:    np.ndarray,       # ROLE_PROBS indexed by allowed roles
        n_samples:     int = 20,
    ) -> np.ndarray:
        """
        Thompson Sampling: draw n_samples from each arm's Beta prior,
        use the sample mean as the arm quality estimate.
        Blend with base_probs using self.blend.

        Returns a probability vector over allowed_roles (same length).
        """
        ts_scores = np.zeros(len(allowed_roles))
        for j, role in enumerate(allowed_roles):
            a = self.alpha[var_idx, role]
            b = self.beta[var_idx, role]
            # Expected value of Beta(a, b) = a / (a+b)
            # Add Thompson noise: draw from Beta
            draws = np.random.beta(a, b, n_samples)
            ts_scores[j] = draws.mean()

        # Normalise Thompson scores → probability vector
        ts_sum = ts_scores.sum()
        if ts_sum > 0:
            ts_probs = ts_scores / ts_sum
        else:
            ts_probs = np.ones(len(allowed_roles)) / len(allowed_roles)

        # Blend with base ROLE_PROBS
        base_sum = base_probs.sum()
        base_norm = base_probs / base_sum if base_sum > 0 else ts_probs.copy()

        blended = self.blend * base_norm + (1.0 - self.blend) * ts_probs
        blended = np.maximum(blended, 1e-8)
        return blended / blended.sum()

    def get_violation_rate(self, var_idx: int, role: int) -> float:
        """Return the empirical failure rate for (var_idx, role)."""
        a = self.alpha[var_idx, role]
        b = self.beta[var_idx, role]
        # Posterior mean of failure = b / (a+b) minus the prior (Beta(1,1) prior)
        # Effective counts (subtract prior)
        a_eff = max(a - 1.0, 0.0)
        b_eff = max(b - 1.0, 0.0)
        total = a_eff + b_eff
        return b_eff / total if total > 0 else 0.0

    def report(self, var_names: List[str]) -> pd.DataFrame:
        """Return a DataFrame of per-(variable, role) statistics."""
        rows = []
        for i, nm in enumerate(var_names):
            for role in range(self.n_roles):
                a = self.alpha[i, role]
                b = self.beta[i, role]
                n = (a - 1) + (b - 1)  # effective observations (minus prior)
                if n < 1:
                    continue
                rows.append({
                    "variable": nm,
                    "role":     role,
                    "n_evals":  int(n),
                    "success_rate": (a - 1) / n,
                    "failure_rate": (b - 1) / n,
                    "posterior_mean_quality": a / (a + b),
                })
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values("failure_rate", ascending=False)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. SpecificationBandit  (contextual LinUCB)
# ─────────────────────────────────────────────────────────────────────────────

class SpecificationBandit:
    """
    Contextual multi-armed bandit (LinUCB) for model specification search.

    Each arm represents a specific (variable_idx, role_code) proposal.
    The bandit learns which variable-role assignments improve BIC given the
    current model context.

    Context features for arm (var_idx, role)
    -----------------------------------------
    x = [
        bic_norm,           # current BIC / (n_obs * log(n_obs))  (bounded quality)
        ll_norm,            # current LL / n_obs
        n_active_norm,      # fraction of variables active
        var_mean_norm,      # variable's standardised mean
        var_std_norm,       # variable's standardised std
        var_type_binary,    # is the variable binary?
        var_type_count,     # is the variable a count?
        role_is_random,     # is the proposed role a random effect?
        role_is_membership, # is the proposed role membership (7 or 8)?
        bic_trend,          # recent BIC trend (positive = improving)
    ]

    Reward signal (updated in BanditGuidedSA.run())
    ------------------------------------------------
    reward = BIC_improvement - n_insig * INSIG_PENALTY

    BIC_improvement  = -(delta_BIC)          positive when BIC falls
    n_insig          = count of vars with |t| < t_min after the fit
    INSIG_PENALTY    = 5.0 per insignificant variable

    This means the bandit learns: "proposing a variable that ends up
    statistically insignificant is bad — even if BIC barely improves."

    LinUCB update rule (after each evaluation):
      A_arm += x @ x.T           (rank-1 covariance update)
      b_arm += reward * x        (reward-weighted context accumulation)
      theta_hat = A^{-1} b       (ridge-regularised least-squares estimate)

    UCB selection score:
      score = x^T theta_hat  +  alpha * sqrt(x^T A^{-1} x)
               ─────────────     ──────────────────────────
               Exploitation        Exploration bonus
               "predicted reward   "uncertainty: how little we know
               for this arm        about this arm in this context"
               given context x"

    x^T theta_hat  — linear prediction of reward for arm in context x.
                     Arms that historically gave high reward in similar
                     contexts score highly.

    alpha * sqrt(x^T A^{-1} x)  — the UCB bonus.
                     A^{-1} is the inverse of the ridge-regularised
                     covariance of context vectors seen so far.
                     x^T A^{-1} x is large when x lies in a direction
                     that hasn't been well-explored (few similar contexts
                     observed), giving those arms an exploration bonus.
                     alpha controls the exploration/exploitation trade-off:
                       alpha=0 → pure greedy (exploit only)
                       alpha >> 1 → mostly explore
                     LinUCB is UCB1 generalised to feature spaces.
    """

    N_FEATURES = 10

    def __init__(
        self,
        n_vars:       int,
        n_roles:      int = 9,
        alpha:        float = 1.0,     # exploration bonus coefficient
        lambda_reg:   float = 1.0,     # L2 regularisation on A
        var_stats:    Optional[Dict[str, Dict]] = None,
    ):
        self.n_vars    = n_vars
        self.n_roles   = n_roles
        self.alpha     = alpha
        self.var_stats = var_stats or {}
        n_arms = n_vars * n_roles

        d = self.N_FEATURES
        # A_arm: (n_arms, d, d), b_arm: (n_arms, d)
        self.A = np.stack([lambda_reg * np.eye(d)] * n_arms)   # (n_arms, d, d)
        self.b = np.zeros((n_arms, d))                          # (n_arms, d)
        self._theta = np.zeros((n_arms, d))                     # cached theta_hat

        self._bic_history: List[float] = []

    def _arm_idx(self, var_idx: int, role: int) -> int:
        return var_idx * self.n_roles + role

    def _context(
        self,
        var_idx:    int,
        role:       int,
        current_bic: float,
        current_ll:  float,
        n_active:   int,
        n_obs:      int,
    ) -> np.ndarray:
        """Build the context feature vector for arm (var_idx, role)."""
        bic_norm = current_bic / max(n_obs * np.log(max(n_obs, 2)), 1.0)
        ll_norm  = current_ll  / max(n_obs, 1.0)
        act_norm = n_active / max(self.n_vars, 1.0)

        vs = self.var_stats.get(str(var_idx), {})
        var_mean = vs.get("mean_norm", 0.0)
        var_std  = vs.get("std_norm",  1.0)
        var_bin  = float(vs.get("is_binary", False))
        var_cnt  = float(vs.get("is_count",  False))

        role_random     = float(role in (2, 3, 4))
        role_membership = float(role in (7, 8))

        if len(self._bic_history) >= 5:
            recent = self._bic_history[-5:]
            bic_trend = (recent[-1] - recent[0]) / (abs(recent[0]) + 1e-6)
        else:
            bic_trend = 0.0

        return np.array([
            bic_norm, ll_norm, act_norm,
            var_mean, var_std, var_bin, var_cnt,
            role_random, role_membership, bic_trend,
        ], dtype=float)

    def update(
        self,
        var_idx:     int,
        role:        int,
        reward:      float,
        context_vec: np.ndarray,
    ) -> None:
        """Rank-1 update of the LinUCB parameters."""
        arm = self._arm_idx(var_idx, role)
        x = context_vec
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        try:
            self._theta[arm] = np.linalg.solve(self.A[arm], self.b[arm])
        except np.linalg.LinAlgError:
            pass  # Keep old theta on singular A

    def select_arm(
        self,
        current_solution: np.ndarray,
        current_bic:      float,
        current_ll:       float,
        n_obs:            int,
        allowed_roles:    Dict[int, List[int]],
        n_active:         int,
        epsilon:          float = 0.1,
    ) -> Tuple[int, int, np.ndarray]:
        """
        Select the most promising (var_idx, role) change.

        Uses epsilon-greedy on top of UCB scoring:
          - With probability epsilon: random allowed change (exploration)
          - Otherwise: highest UCB score (exploitation)

        Returns (var_idx, new_role, context_vec).
        """
        if np.random.rand() < epsilon:
            # Pure random exploration
            var_idx = np.random.randint(self.n_vars)
            roles   = allowed_roles.get(var_idx, [0, 1])
            role    = int(np.random.choice(roles))
            ctx     = self._context(var_idx, role, current_bic, current_ll,
                                    n_active, n_obs)
            return var_idx, role, ctx

        best_score = -np.inf
        best_var, best_role, best_ctx = 0, 0, np.zeros(self.N_FEATURES)

        for var_idx in range(self.n_vars):
            for role in allowed_roles.get(var_idx, [0, 1]):
                arm = self._arm_idx(var_idx, role)
                ctx = self._context(var_idx, role, current_bic, current_ll,
                                    n_active, n_obs)
                theta = self._theta[arm]
                try:
                    A_inv_x = np.linalg.solve(self.A[arm], ctx)
                    bonus   = self.alpha * np.sqrt(max(ctx @ A_inv_x, 0.0))
                except np.linalg.LinAlgError:
                    bonus = self.alpha

                score = float(theta @ ctx) + bonus
                if score > best_score:
                    best_score = score
                    best_var, best_role, best_ctx = var_idx, role, ctx

        return best_var, best_role, best_ctx

    def record_bic(self, bic: float) -> None:
        self._bic_history.append(bic)

    def report_top_arms(self, var_names: List[str], top_n: int = 10) -> pd.DataFrame:
        """Return a DataFrame of the top-scoring arms by posterior theta mean."""
        rows = []
        for var_idx in range(self.n_vars):
            for role in range(self.n_roles):
                arm    = self._arm_idx(var_idx, role)
                theta  = self._theta[arm]
                # Posterior mean quality = ||theta||_2 (higher → arm expected to improve BIC)
                quality = float(np.linalg.norm(theta))
                rows.append({
                    "variable": var_names[var_idx] if var_idx < len(var_names) else str(var_idx),
                    "role":     role,
                    "quality":  quality,
                    "n_updates": int(np.trace(self.A[arm])) - self.N_FEATURES,
                })
        df = pd.DataFrame(rows).sort_values("quality", ascending=False)
        return df.head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BanditGuidedSA  (SA with adaptive bandit proposals + identifiability guard)
# ─────────────────────────────────────────────────────────────────────────────

class BanditGuidedSA:
    """
    Simulated Annealing guided by a contextual bandit.

    At each SA step the neighbour is generated by:
      1. Select (var_idx, role) from the SpecificationBandit (UCB or epsilon-greedy)
      2. Apply the change to the current solution
      3. Evaluate fitness
      4. Run IdentifiabilityChecker on the fitted params (lightweight: uses cached SEs)
      5. Update AdaptiveRoleMemory and SpecificationBandit with the outcome
      6. Accept/reject via Metropolis criterion

    This replaces the uniform-random neighbour of vanilla SA with a
    learned proposal distribution that progressively favours changes
    that have historically improved BIC.

    The identifiability guard removes variables with |t| < 1.0 from the
    proposed solution and records the (var, role) pair as a failure — so
    the bandit learns to avoid proposing those combinations in future.
    """

    def __init__(
        self,
        builder,                           # ExperimentBuilder
        evaluator,                         # StructureEvaluatorLC
        t_start:        float = 5.0,
        t_end:          float = 0.01,
        alpha:          float = 0.995,     # SA cooling rate
        bandit_alpha:   float = 1.5,       # LinUCB exploration bonus
        blend:          float = 0.4,       # bandit/base blend in role memory
        t_stat_min:     float = 1.0,       # identifiability t-stat threshold
        extreme_coef:   float = 15.0,
        use_checker:    bool  = True,      # run IdentifiabilityChecker post-fit
        model:          str   = "nb",
        R:              int   = 200,
    ):
        self.builder    = builder
        self.evaluator  = evaluator
        self.model      = model
        self.R          = R
        self.use_checker = use_checker

        self.T0         = t_start
        self.T_end      = t_end
        self.cooling    = alpha

        n_vars = len(evaluator.vars)
        self.role_memory = AdaptiveRoleMemory(n_vars=n_vars, blend=blend)
        self.checker     = IdentifiabilityChecker(
            t_threshold=t_stat_min, extreme_threshold=extreme_coef
        )

        # Build var_stats for bandit context
        var_stats: Dict[str, Dict] = {}
        for i, vn in enumerate(evaluator.vars):
            s = builder.df[vn] if vn in builder.df.columns else pd.Series([0.0])
            var_stats[str(i)] = {
                "mean_norm":  float(s.mean()),
                "std_norm":   float(s.std() + 1e-8),
                "is_binary":  int(s.nunique() == 2),
                "is_count":   int((s >= 0).all() and (s % 1 == 0).all() and s.nunique() > 2),
            }

        self.bandit = SpecificationBandit(
            n_vars=n_vars,
            n_roles=9,
            alpha=bandit_alpha,
            var_stats=var_stats,
        )

        # Patch the evaluator so Solvers_METAJAX sample_allowed_role can see our memory
        self.evaluator._sign_visit_counts     = self.role_memory._visit_counts
        self.evaluator._sign_violation_counts = self.role_memory._violation_counts

    def _initial_solution(self) -> np.ndarray:
        D   = len(self.evaluator.vars)
        has_lc = getattr(self.evaluator, "max_latent_classes", 1) > 1
        dim    = 3 * D + 2 if has_lc else 2 * D + 1
        for _ in range(200):
            roles = np.array([
                self.evaluator.allowed_roles.get(v, [0, 1])[
                    np.random.randint(len(self.evaluator.allowed_roles.get(v, [0, 1])))
                ]
                for v in self.evaluator.vars
            ])
            dists  = np.random.randint(0, 6, D)
            tail   = np.random.randint(0, 2, dim - 2 * D)
            sol    = np.concatenate([roles, dists, tail]).astype(int)
            if np.sum(roles != 0) >= 2:
                return sol
        return np.zeros(dim, dtype=int)

    def _apply_change(
        self,
        solution: np.ndarray,
        var_idx:  int,
        role:     int,
    ) -> np.ndarray:
        """Apply (var_idx, role) change to a copy of solution."""
        neighbor = solution.copy()
        neighbor[var_idx] = role
        return neighbor

    def _identifiability_penalty_from_cache(
        self,
        cache:    Optional[dict],
        decision: np.ndarray,
    ) -> Tuple[float, dict]:
        """
        Compute an identifiability penalty using the cached fit result.

        Reads ``evaluator._last_fit_cache`` (populated after every ``fitness()``
        call) instead of re-fitting the model.  Uses the diagonal Hessian
        t-statistics (_fast_tstats) to identify insignificant variables.

        Penalty formula
        ---------------
        penalty = n_weak * log(N) + n_bad * 2*log(N)
          n_weak  = variables with |t| < t_threshold (statistically non-significant)
          n_bad   = variables with non-finite SE (numerically unidentified)
          N       = number of observations

        Also updates ``self.role_memory`` with per-variable success/failure
        so that the Thompson Sampling prior learns to avoid proposing roles
        that consistently produce insignificant parameters.

        Returns (penalty: float, diagnostics: dict)
        """
        if cache is None or not self.use_checker:
            return 0.0, {}

        try:
            from functools import partial

            # ── Try relative import first (package mode), fall back to absolute
            try:
                from .main_hpc_lc_patch import mixed_model_loglik
            except ImportError:
                from main_hpc_lc_patch import mixed_model_loglik

            import jax.numpy as jnp

            params    = cache["params"]
            spec      = cache["spec"]
            data      = cache["data"]
            n_obs     = int(data["y"].shape[0])
            t_min     = self.checker.t_threshold

            # Build the NLL objective for this cached (spec, data)
            obj       = partial(mixed_model_loglik, data=data, spec=spec)
            params_j  = jnp.asarray(params, dtype=float)

            # O(p) diagonal Hessian t-stats
            t_stats, se = _fast_tstats(params_j, obj)

            # ── Per-variable t-stat lookup ───────────────────────────
            fixed_names  = list(getattr(spec, "fixed_names", []))
            name_to_pidx = {
                nm: i for i, nm in enumerate(fixed_names)
                if nm != "__INTERCEPT__"
            }

            var_names = list(self.evaluator.vars)
            n_vars    = min(len(var_names), len(decision))
            n_weak    = 0
            n_bad     = 0
            weak_names: set = set()

            for vi in range(n_vars):
                role = int(decision[vi])
                if role == 0:
                    continue
                nm   = var_names[vi]
                pidx = name_to_pidx.get(nm)
                if pidx is None or pidx >= len(t_stats):
                    continue

                ts      = float(t_stats[pidx])
                is_nan  = not np.isfinite(ts)
                is_weak = (not is_nan) and abs(ts) < t_min

                if is_nan:
                    n_bad += 1
                    weak_names.add(nm)
                elif is_weak:
                    n_weak += 1
                    weak_names.add(nm)

                # Update per-(var, role) Thompson prior
                success = (not is_nan) and (not is_weak)
                self.role_memory.update(vi, role, success)

            # ── Penalty: n_weak * log(N) + n_bad * 2*log(N) ─────────
            log_n   = np.log(max(n_obs, 2))
            penalty = n_weak * log_n + n_bad * 2.0 * log_n

            diag = {
                "weak_names":  weak_names,
                "n_weak":      n_weak,
                "n_bad":       n_bad,
                "t_stats":     t_stats,
                "se":          se,
                "fixed_names": fixed_names,
            }
            return float(penalty), diag

        except Exception:
            return 0.0, {}

    def run(
        self,
        max_iter:    int  = 3000,
        seed:        int  = 0,
        print_every: int  = 100,
    ) -> dict:
        """
        Run the bandit-guided SA.

        Returns
        -------
        dict with keys:
          best_solution, best_score, history, role_memory_report,
          bandit_top_arms
        """
        np.random.seed(seed)

        current_sol   = self._initial_solution()
        current_score = float(self.evaluator.fitness(current_sol))
        best_sol      = current_sol.copy()
        best_score    = current_score
        T             = self.T0

        n_vars    = len(self.evaluator.vars)
        n_obs     = len(self.builder.df)
        allowed_roles_map = {
            i: self.evaluator.allowed_roles.get(v, [0, 1])
            for i, v in enumerate(self.evaluator.vars)
        }

        history: List[float] = [current_score]
        self.bandit.record_bic(current_score)

        # Penalty weight per insignificant variable in the reward signal.
        # This is *separate* from the BIC penalty added to neighbor_score:
        # the BIC penalty steers SA acceptance; this steers the bandit's
        # reward regression so it learns "proposing insignificant variables
        # gives bad reward".
        INSIG_REWARD_PENALTY = 5.0

        print(f"\n  BanditGuidedSA | n_vars={n_vars} | max_iter={max_iter} | T0={self.T0}")
        print(f"  Significance guard: |t| >= {self.checker.t_threshold:.1f} required")
        print(f"  Insig reward penalty per variable: {INSIG_REWARD_PENALTY:.1f}\n")

        for it in range(max_iter):
            T = max(self.T_end, T * self.cooling)
            n_active = int(np.sum(current_sol[:n_vars] != 0))
            epsilon  = max(0.05, 0.3 - it / max_iter * 0.25)  # anneal exploration

            # ── Bandit proposal ───────────────────────────────────
            var_idx, new_role, ctx = self.bandit.select_arm(
                current_solution = current_sol,
                current_bic      = current_score,
                current_ll       = -current_score / 2.0,   # rough estimate
                n_obs            = n_obs,
                allowed_roles    = allowed_roles_map,
                n_active         = n_active,
                epsilon          = epsilon,
            )
            neighbor = self._apply_change(current_sol, var_idx, new_role)

            # ── Evaluate fitness (populates evaluator._last_fit_cache) ────
            neighbor_score = float(self.evaluator.fitness(neighbor))

            # ── Identifiability penalty from cached fit ───────────
            # Reads _last_fit_cache that fitness() just populated.
            # Uses O(p) diagonal Hessian t-stats — no second fit call.
            penalty = 0.0
            diag    = {}
            if np.isfinite(neighbor_score) and neighbor_score < 1e11:
                cache   = getattr(self.evaluator, "_last_fit_cache", None)
                penalty, diag = self._identifiability_penalty_from_cache(
                    cache, neighbor
                )
                # Add BIC-scale penalty so SA acceptance reflects significance
                neighbor_score = neighbor_score + penalty

            delta    = neighbor_score - current_score
            n_insig  = diag.get("n_weak", 0) + diag.get("n_bad", 0)

            # ── Metropolis accept/reject ───────────────────────────
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                current_sol   = neighbor
                current_score = neighbor_score
                # Reward = BIC improvement MINUS significance penalty.
                # This teaches the bandit that proposing specs with
                # insignificant variables is bad even when they get accepted.
                bic_reward    = max(-delta, -1000.0)
                insig_penalty = n_insig * INSIG_REWARD_PENALTY
                reward        = bic_reward - insig_penalty
            else:
                # Rejected: mild negative reward, still penalise insig vars
                insig_penalty = n_insig * INSIG_REWARD_PENALTY * 0.5
                reward        = min(delta, 0.0) / 100.0 - insig_penalty

            # ── Bandit update ──────────────────────────────────────
            self.bandit.update(var_idx, new_role, reward, ctx)
            self.bandit.record_bic(current_score)

            if current_score < best_score:
                best_score = current_score
                best_sol   = current_sol.copy()

            history.append(current_score)

            if print_every > 0 and it % print_every == 0:
                weak_count  = diag.get("n_weak", 0)
                bad_count   = diag.get("n_bad",  0)
                penalty_str = f"  pen={penalty:.0f}" if penalty > 0 else ""
                sig_str     = ""
                if weak_count or bad_count:
                    sig_str = f"  INSIG:{weak_count}  NONID:{bad_count}"
                print(f"  iter {it:5d} | T={T:.4f} | BIC={current_score:.1f} | "
                      f"best={best_score:.1f} | eps={epsilon:.2f}"
                      f"{penalty_str}{sig_str}")

        print(f"\n  Best BIC: {best_score:.2f}")
        print(f"  Role memory summary:")
        rpt = self.role_memory.report(list(self.evaluator.vars))
        if len(rpt) > 0:
            print(rpt.head(10).to_string(index=False))

        print(f"\n  Top bandit arms:")
        top = self.bandit.report_top_arms(list(self.evaluator.vars))
        print(top.to_string(index=False))

        return {
            "best_solution":       best_sol,
            "best_score":          best_score,
            "history":             history,
            "role_memory_report":  rpt,
            "bandit_top_arms":     top,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PostFitPruner  (backward-elimination identifiability repair)
# ─────────────────────────────────────────────────────────────────────────────

class PostFitPruner:
    """
    Post-search backward elimination that greedily removes variables whose
    t-statistic falls below a threshold.

    Algorithm
    ---------
    1. Fit the current best solution → extract per-variable t-statistics.
    2. Find the variable with the smallest |t-stat| below t_min.
    3. Remove it (set role = 0) and re-fit.
    4. If BIC improves, keep the removal; else restore.
    5. Repeat until no variable has |t| < t_min.

    This is a deterministic repair pass that runs after the search and
    ensures the final reported model is free of identifiability-violating
    variables.
    """

    def __init__(
        self,
        builder,
        evaluator,
        t_min:  float = 1.0,
        model:  str   = "nb",
        R:      int   = 200,
    ):
        self.builder   = builder
        self.evaluator = evaluator
        self.t_min     = t_min
        self.model     = model
        self.R         = R
        self.checker   = IdentifiabilityChecker(t_threshold=t_min)

    def prune(
        self,
        solution:     np.ndarray,
        print_steps:  bool = True,
    ) -> dict:
        """
        Prune the given solution via backward elimination.

        Returns
        -------
        dict:  pruned_solution, pruned_bic, removed_vars, final_fit
        """
        from functools import partial
        try:
            from main_hpc_lc_patch import mixed_model_loglik
        except ImportError:
            from main_hpc_lc_patch import mixed_model_loglik

        current_sol = solution.copy()
        n_vars      = len(self.evaluator.vars)
        removed: List[str] = []

        print("\n  POST-FIT PRUNER  (backward elimination | t_min={:.1f})".format(self.t_min))

        for pass_n in range(n_vars):
            spec_dict = self.evaluator.build_spec(current_sol)
            if spec_dict is None:
                break
            try:
                fit = self.builder.fit_manual_model(
                    manual_spec=spec_dict, model=self.model,
                    R=self.R, print_report=False,
                )
            except Exception as exc:
                print(f"  [pruner] fit failed: {exc}")
                break

            params = np.asarray(fit["result"].params)
            spec   = fit["spec"]
            data   = fit["data"]
            obj    = partial(mixed_model_loglik, data=data, spec=spec)
            diag   = self.checker.check(params, obj, spec, silent=True)

            current_bic = fit.get("summary", {}).get("bic", np.inf)

            # Find the weakest active variable
            worst_var  = None
            worst_ts   = np.inf
            for d in diag["diagnostics"]:
                nm = d.name.split(":")[0].replace("__INTERCEPT__", "")
                if nm in ("__intercept__", ""):
                    continue
                if d.is_weak or d.is_nonfinite:
                    ts_abs = abs(d.t_stat) if np.isfinite(d.t_stat) else 0.0
                    if ts_abs < worst_ts:
                        worst_ts  = ts_abs
                        worst_var = nm

            if worst_var is None:
                print(f"  Pass {pass_n+1}: no weak variables — stopping.")
                break

            # Try removing it
            var_idx = None
            for i, v in enumerate(self.evaluator.vars):
                if v == worst_var:
                    var_idx = i
                    break
            if var_idx is None:
                break

            old_role   = int(current_sol[var_idx])
            trial_sol  = current_sol.copy()
            trial_sol[var_idx] = 0

            trial_spec = self.evaluator.build_spec(trial_sol)
            if trial_spec is None:
                break

            try:
                trial_fit = self.builder.fit_manual_model(
                    manual_spec=trial_spec, model=self.model,
                    R=self.R, print_report=False,
                )
                trial_bic = trial_fit.get("summary", {}).get("bic", np.inf)
            except Exception:
                trial_bic = np.inf

            if print_steps:
                print(f"  Pass {pass_n+1}: '{worst_var}' |t|={worst_ts:.2f}  "
                      f"BIC {current_bic:.1f} -> {trial_bic:.1f}  "
                      + ("REMOVED" if trial_bic <= current_bic + 2 else "KEPT"))

            if trial_bic <= current_bic + 2:   # accept removal if BIC doesn't worsen by >2
                current_sol = trial_sol
                removed.append(f"{worst_var} (was role {old_role}, |t|={worst_ts:.2f})")
            else:
                break  # no further improvement possible via removal

        # Final re-fit
        final_spec = self.evaluator.build_spec(current_sol)
        final_fit  = None
        pruned_bic = np.inf
        if final_spec:
            try:
                final_fit = self.builder.fit_manual_model(
                    manual_spec=final_spec, model=self.model,
                    R=self.R, print_report=False,
                )
                pruned_bic = final_fit.get("summary", {}).get("bic", np.inf)
            except Exception:
                pass

        print(f"\n  Removed variables : {removed if removed else '(none)'}")
        print(f"  Final BIC         : {pruned_bic:.2f}")

        return {
            "pruned_solution": current_sol,
            "pruned_bic":      pruned_bic,
            "removed_vars":    removed,
            "final_fit":       final_fit,
        }
