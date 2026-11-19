"""Estimation environment for Delphos-style count-model specification."""

from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .state import (
    RANDOM_ROLES,
    CountModelState,
    SearchSpace,
    VariableSpec,
)


@dataclass(frozen=True)
class Action:
    """One feasible modelling action."""

    kind: str  # role | dist | transform | dispersion | latent | terminate
    var_index: int = -1
    value: str = ""
    index: int = -1


class ActionSpace:
    """Fixed action catalogue with state-dependent action masking."""

    def __init__(self, space: SearchSpace):
        self.space = space
        actions: List[Action] = []
        for i, spec in enumerate(space.variables):
            for role in spec.allowed_roles:
                actions.append(Action("role", i, role))
            for dist in spec.allowed_distributions:
                actions.append(Action("dist", i, dist))
            if len(spec.transform_columns) > 1:
                for transform, _ in spec.transform_columns:
                    actions.append(Action("transform", i, transform))
        if space.allow_dispersion:
            actions.append(Action("dispersion", -1, "toggle"))
        for k in range(1, max(1, space.max_latent_classes) + 1):
            actions.append(Action("latent", -1, str(k)))
        actions.append(Action("terminate", -1, "terminate"))

        # Stable indices
        prepared = [replace(a, index=i) for i, a in enumerate(actions)]
        self.actions: Tuple[Action, ...] = tuple(prepared)
        self.by_index: Tuple[Action, ...] = self.actions
        self._index = {(a.kind, a.var_index, a.value): a.index for a in self.actions}

    @property
    def size(self) -> int:
        return len(self.actions)

    def index_of(self, action: Action) -> int:
        return self._index[(action.kind, action.var_index, action.value)]

    def action(self, index: int) -> Action:
        return self.actions[index]

    def apply(self, state: CountModelState, action: Action) -> CountModelState:
        spec = self.space.variables
        if action.kind == "role":
            term = state.terms[action.var_index]
            v = spec[action.var_index]
            role = v.coerce_role(action.value)
            if v.forced_role is not None:
                role = v.forced_role
            dist = term.distribution
            if role in RANDOM_ROLES:
                dist = v.coerce_distribution(dist)
            return state.with_term(
                action.var_index,
                role=role,
                distribution=dist,
                active=role != "excluded",
                block=v.block,
            )
        if action.kind == "dist":
            v = spec[action.var_index]
            return state.with_term(
                action.var_index,
                distribution=v.coerce_distribution(action.value),
            )
        if action.kind == "transform":
            v = spec[action.var_index]
            transform = (
                action.value
                if action.value in dict(v.transform_columns)
                else v.default_transform
            )
            return state.with_term(
                action.var_index, transform=transform, block=v.block
            )
        if action.kind == "dispersion":
            return replace(state, dispersion=1 - int(bool(state.dispersion)))
        if action.kind == "latent":
            k = int(action.value)
            k = min(max(k, 1), max(1, self.space.max_latent_classes))
            return replace(state, latent_classes=k)
        if action.kind == "terminate":
            return state
        raise ValueError(f"unknown action kind {action.kind!r}")

    def mask(
        self,
        state: CountModelState,
        prev_state: Optional[CountModelState] = None,
        *,
        allow_terminate: bool = True,
    ) -> np.ndarray:
        """Boolean mask; True means the action is feasible."""
        mask = np.zeros(self.size, dtype=bool)
        for action in self.actions:
            if action.kind == "role":
                v = self.space.variables[action.var_index]
                term = state.terms[action.var_index]
                role = action.value
                if v.forced_role is not None and role != v.forced_role:
                    continue
                if role not in v.allowed_roles:
                    continue
                if role in ("membership", "membership_fixed") and state.latent_classes <= 1:
                    continue
                target_active = role != "excluded"
                if role == term.role and target_active == bool(term.active):
                    continue
            elif action.kind == "dist":
                term = state.terms[action.var_index]
                v = self.space.variables[action.var_index]
                if not (term.active and term.role in RANDOM_ROLES):
                    continue
                if action.value not in v.allowed_distributions:
                    continue
                if action.value == term.distribution:
                    continue
            elif action.kind == "transform":
                term = state.terms[action.var_index]
                if action.value == term.transform:
                    continue
            elif action.kind == "dispersion":
                if not self.space.allow_dispersion:
                    continue
            elif action.kind == "latent":
                if self.space.max_latent_classes <= 1:
                    continue
                if int(action.value) == int(state.latent_classes):
                    continue
            elif action.kind == "terminate":
                if not allow_terminate:
                    continue
            else:  # pragma: no cover - catalogue is built internally
                continue

            if prev_state is not None and action.kind != "terminate":
                nxt = self.apply(state, action)
                if nxt.signature() == prev_state.signature():
                    continue
            mask[action.index] = True
        return mask


@dataclass(frozen=True)
class EvalResult:
    """Outcome of estimating one complete specification."""

    bic: float
    loglik: float
    n_params: int
    n_obs: int
    n_active: int
    family: str
    reward: float
    success: bool
    from_cache: bool = False

    def as_dict(self) -> Dict[str, float]:
        return {
            "bic": self.bic,
            "loglik": self.loglik,
            "n_params": self.n_params,
            "n_obs": self.n_obs,
            "n_active": self.n_active,
            "family": self.family,
            "reward": self.reward,
            "success": bool(self.success),
            "from_cache": bool(self.from_cache),
        }


class DelphosCountEnv:
    """MDP environment that estimates a count-model candidate on terminate.

    The reward follows the Delphos formulation: a bounded improvement in
    log-likelihood per observation relative to a baseline (null) model.
    Failed / invalid specifications receive a fixed penalty of -1.
    """

    def __init__(
        self,
        evaluator,
        space: SearchSpace,
        *,
        max_steps: Optional[int] = None,
        baseline_state: Optional[CountModelState] = None,
        failure_penalty: float = -1.0,
        reward_mode: str = "tanh",
        start_mode: str = "linear",
        seed: int = 0,
        verbose: bool = False,
    ):
        self.evaluator = evaluator
        self.space = space
        self.actions = ActionSpace(space)
        self.max_steps = int(max_steps or (2 * space.n_variables + 4))
        self.failure_penalty = float(failure_penalty)
        self.reward_mode = str(reward_mode)
        if self.reward_mode not in ("tanh", "linear", "bic"):
            raise ValueError("reward_mode must be 'tanh', 'linear' or 'bic'")
        self.start_mode = str(start_mode)
        if self.start_mode not in ("linear", "null", "mixed"):
            raise ValueError("start_mode must be 'linear', 'null' or 'mixed'")
        self.rng = np.random.default_rng(seed)
        self.verbose = bool(verbose)

        self._eval_cache: Dict[Tuple, EvalResult] = {}
        self._baseline = baseline_state
        self._baseline_eval: Optional[EvalResult] = None

        self.state: Optional[CountModelState] = None
        self.prev_state: Optional[CountModelState] = None
        self.steps = 0
        self.history: List[EvalResult] = []
        self.best_state: Optional[CountModelState] = None
        self.best_eval: Optional[EvalResult] = None
        self.n_fits = 0
        self.n_episodes = 0

    # ------------------------------------------------------------------
    # Episode interface
    # ------------------------------------------------------------------
    def reset(self) -> CountModelState:
        if self.start_mode == "linear":
            self.state = self.space.default_state()
        elif self.start_mode == "null":
            self.state = self.space.null_state()
        else:  # mixed
            self.state = (
                self.space.default_state()
                if self.rng.random() < 0.5
                else self.space.null_state()
            )
        self.prev_state = None
        self.steps = 0
        self.n_episodes += 1
        return self.state

    def step(self, action: Action):
        assert self.state is not None, "call reset() first"
        self.steps += 1
        if action.kind == "terminate":
            result = self.evaluate_state(self.state)
            return self.state, result.reward, True, {"eval": result}

        self.prev_state = self.state
        self.state = self.space.normalize(self.actions.apply(self.state, action))

        # Hard episode-length cap: auto-submit rather than dropping the
        # candidate.  This mirrors a modeller who stops after a maximum number
        # of specification edits.
        if self.steps >= self.max_steps:
            result = self.evaluate_state(self.state)
            return self.state, result.reward, True, {
                "eval": result,
                "auto_terminated": True,
            }
        return self.state, 0.0, False, {}

    def action_mask(self, allow_terminate: bool = True) -> np.ndarray:
        assert self.state is not None
        return self.actions.mask(
            self.state, self.prev_state, allow_terminate=allow_terminate
        )

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    def _baseline_result(self) -> EvalResult:
        if self._baseline_eval is None:
            state = self._baseline or self.space.null_state()
            # Do not recurse through _reward: the baseline is what the
            # improvement reward is measured against.
            self._baseline_eval = self._evaluate_raw(
                state, compute_reward=False
            )
        return self._baseline_eval

    def evaluate_state(self, state: CountModelState) -> EvalResult:
        key = tuple(self.space.to_decision(state).tolist())
        if key in self._eval_cache:
            cached = self._eval_cache[key]
            return replace(cached, from_cache=True)

        result = self._evaluate_raw(state, key=key)
        self._eval_cache[key] = result
        self.history.append(result)
        if result.success and (
            self.best_eval is None or result.bic < self.best_eval.bic
        ):
            self.best_eval = result
            self.best_state = state
        return result

    def _evaluate_raw(
        self,
        state: CountModelState,
        key: Optional[Tuple] = None,
        *,
        compute_reward: bool = True,
    ) -> EvalResult:
        decision = self.space.to_decision(state)
        if key is None:
            key = tuple(decision.tolist())

        self.n_fits += 1
        if self.verbose:
            print(f"  [fit {self.n_fits}] {state.derived_family()}")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                value = self.evaluator.fitness(decision)
        except Exception as exc:  # estimation crash -> infeasible
            if self.verbose:
                print(f"  [fit failed] {exc}")
            return EvalResult(
                bic=1e12,
                loglik=float("-inf"),
                n_params=0,
                n_obs=0,
                n_active=state.n_active,
                family=state.derived_family(),
                reward=self.failure_penalty,
                success=False,
            )

        bic = float(np.asarray(value).reshape(-1)[0])
        success = math.isfinite(bic) and bic < 1e11
        if not success:
            return EvalResult(
                bic=1e12,
                loglik=float("-inf"),
                n_params=0,
                n_obs=0,
                n_active=state.n_active,
                family=state.derived_family(),
                reward=self.failure_penalty,
                success=False,
            )

        fc = getattr(self.evaluator, "_last_fit_cache", None)
        n_params = 0
        n_obs = 0
        loglik = float("nan")
        if fc is not None:
            try:
                n_params = int(len(fc["params"]))
                n_obs = int(np.asarray(fc["data"]["y"]).reshape(-1).shape[0])
                if abs(float(fc.get("bic", bic)) - bic) < 1e-4 * max(1.0, abs(bic)):
                    loglik = 0.5 * (n_params * math.log(n_obs) - bic)
            except Exception:
                n_params = 0
                n_obs = 0
                loglik = float("nan")

        if not math.isfinite(loglik):
            # The evaluator may have returned a value from its own cache, in
            # which case ``_last_fit_cache`` belongs to a different fit.  We
            # can still recover the parameter count and sample size without
            # refitting by building the spec and its parameter index.
            try:
                from metacountregressor.main_hpc import build_param_index

                spec_dict = self.evaluator.build_spec(decision)
                data, spec = self.evaluator.build_data(
                    self.evaluator.df_train,
                    spec_dict,
                    self.evaluator.master_halton_train,
                )
                n_params = int(build_param_index(spec)["total_params"])
                n_obs = int(np.asarray(data["y"]).reshape(-1).shape[0])
                loglik = 0.5 * (n_params * math.log(n_obs) - bic)
            except Exception:
                n_params = n_params or 0
                n_obs = n_obs or 0
                loglik = float("nan")

        reward = (
            self._reward(bic, loglik, n_obs, n_params)
            if compute_reward
            else 0.0
        )
        return EvalResult(
            bic=bic,
            loglik=loglik,
            n_params=n_params,
            n_obs=n_obs,
            n_active=state.n_active,
            family=state.derived_family(),
            reward=reward,
            success=True,
        )

    def _reward(self, bic: float, loglik: float, n_obs: int, n_params: int) -> float:
        base = self._baseline_result()
        if (
            n_obs > 0
            and math.isfinite(loglik)
            and math.isfinite(base.loglik)
            and base.loglik != float("-inf")
        ):
            delta = (loglik - base.loglik) / max(n_obs, 1)
        elif math.isfinite(base.bic):
            delta = (base.bic - bic) / max(n_obs or base.n_obs or 1, 1)
        else:
            delta = 0.0
        if self.reward_mode == "linear":
            return float(np.clip(delta, -5.0, 5.0))
        if self.reward_mode == "bic":
            if math.isfinite(base.bic):
                scale = max(100.0, 0.2 * abs(base.bic))
                return float(np.clip((base.bic - bic) / scale, -5.0, 5.0))
            return float(np.clip((base.bic - bic) / 1000.0, -5.0, 5.0))
        return float(np.tanh(delta))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def current_decision(self) -> np.ndarray:
        assert self.state is not None
        return self.space.to_decision(self.state)

    def best(self) -> Tuple[Optional[CountModelState], Optional[EvalResult]]:
        return self.best_state, self.best_eval
