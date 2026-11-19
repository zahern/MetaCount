"""DeepSet-Q agent for count-model specification.

The network follows the Delphos idea: encode the *set* of modelling terms
with a shared term encoder ``phi``, pool by summation, concatenate a context
vector, and output one Q-value per feasible action in the fixed action
catalogue.  No PyTorch dependency: the small network is implemented in JAX
with a hand-written Adam optimiser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .state import (
    DISTRIBUTIONS,
    ROLE_CODES,
    CountModelState,
    RANDOM_ROLES,
    SearchSpace,
)

Array = jax.Array
Params = Dict[str, Array]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def _init_params(key, term_dim: int, ctx_dim: int, hidden: int, emb: int, n_actions: int) -> Params:
    def he(shape, fan_in):
        return jax.random.normal(key, shape) * jnp.sqrt(2.0 / max(fan_in, 1))

    keys = jax.random.split(key, 6)
    return {
        "phi1": jax.random.normal(keys[0], (term_dim, hidden)) * jnp.sqrt(2.0 / term_dim),
        "phi1b": jnp.zeros((hidden,)),
        "phi2": jax.random.normal(keys[1], (hidden, emb)) * jnp.sqrt(2.0 / hidden),
        "phi2b": jnp.zeros((emb,)),
        "q1": jax.random.normal(keys[2], (emb + ctx_dim, hidden)) * jnp.sqrt(2.0 / (emb + ctx_dim)),
        "q1b": jnp.zeros((hidden,)),
        "q2": jax.random.normal(keys[3], (hidden, n_actions)) * jnp.sqrt(2.0 / hidden),
        "q2b": jnp.zeros((n_actions,)),
    }


def _q_values(params: Params, terms: Array, context: Array) -> Array:
    """Q-values for one state or a batch of states.

    terms   : (..., n_terms, term_dim)
    context : (..., ctx_dim)
    returns : (..., n_actions)
    """
    h = jax.nn.relu(jnp.matmul(terms, params["phi1"]) + params["phi1b"])
    h = jnp.matmul(h, params["phi2"]) + params["phi2b"]
    z = jnp.sum(h, axis=-2)
    q_in = jnp.concatenate([z, context], axis=-1)
    q = jax.nn.relu(jnp.matmul(q_in, params["q1"]) + params["q1b"])
    return jnp.matmul(q, params["q2"]) + params["q2b"]


def _loss(params: Params, target_params: Params, batch: Dict[str, Array], gamma: float) -> Array:
    """Monte-Carlo Q loss with Delphos-style delayed credit assignment.

    ``returns`` stores ``gamma ** (L - l) * R_terminal`` for each transition
    in an episode (Eq. 13 of the Delphos paper).  Training directly on that
    return propagates the terminal model-quality signal to the earlier
    specification decisions without waiting for a bootstrapped value function
    to converge.
    """
    terms = batch["terms"]
    ctx = batch["context"]
    actions = batch["actions"]
    returns = batch["returns"]

    q_all = _q_values(params, terms, ctx)
    q_sa = jnp.take_along_axis(q_all, actions[:, None], axis=1)[:, 0]
    return jnp.mean((q_sa - returns) ** 2)


def _adam_init(params: Params) -> Tuple[Params, Params, Array]:
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return m, v, jnp.asarray(0, dtype=jnp.int32)


def _adam_update(params, grads, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    t = t + 1
    m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1.0 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1.0 - b2) * (g * g), v, grads)
    bc1 = 1.0 - b1 ** t.astype(jnp.float32)
    bc2 = 1.0 - b2 ** t.astype(jnp.float32)

    def step(p, mm, vv):
        return p - lr * (mm / bc1) / (jnp.sqrt(vv / bc2) + eps)

    params = jax.tree_util.tree_map(step, params, m, v)
    return params, m, v, t


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, n_terms: int, term_dim: int, ctx_dim: int, n_actions: int):
        self.capacity = int(capacity)
        self.n_terms = int(n_terms)
        self.term_dim = int(term_dim)
        self.ctx_dim = int(ctx_dim)
        self.n_actions = int(n_actions)
        self.terms = np.zeros((capacity, n_terms, term_dim), dtype=np.float32)
        self.context = np.zeros((capacity, ctx_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.next_terms = np.zeros((capacity, n_terms, term_dim), dtype=np.float32)
        self.next_context = np.zeros((capacity, ctx_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.zeros((capacity, n_actions), dtype=bool)
        self.pos = 0
        self.size = 0

    def add(
        self,
        terms,
        context,
        action,
        reward,
        next_terms,
        next_context,
        done,
        next_mask,
        mc_return: float,
    ):
        i = self.pos
        self.terms[i] = terms
        self.context[i] = context
        self.actions[i] = action
        self.rewards[i] = reward
        self.returns[i] = mc_return
        self.next_terms[i] = next_terms
        self.next_context[i] = next_context
        self.done[i] = float(done)
        self.next_mask[i] = next_mask
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Dict[str, jnp.ndarray]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "terms": jnp.asarray(self.terms[idx]),
            "context": jnp.asarray(self.context[idx]),
            "actions": jnp.asarray(self.actions[idx]),
            "rewards": jnp.asarray(self.rewards[idx]),
            "returns": jnp.asarray(self.returns[idx]),
            "next_terms": jnp.asarray(self.next_terms[idx]),
            "next_context": jnp.asarray(self.next_context[idx]),
            "done": jnp.asarray(self.done[idx]),
            "next_mask": jnp.asarray(self.next_mask[idx]),
        }


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    hidden: int = 64
    embedding: int = 32
    learning_rate: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 64
    buffer_size: int = 5000
    warmup_episodes: int = 10
    updates_per_episode: int = 2
    target_sync: int = 10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_frac: float = 0.8


class DeepSetQAgent:
    def __init__(self, space: SearchSpace, action_space, config: Optional[AgentConfig] = None, seed: int = 0):
        self.space = space
        self.action_space = action_space
        self.config = config or AgentConfig()
        self.rng = np.random.default_rng(seed)

        self.n_vars = space.n_variables
        self.transform_names = tuple(
            dict.fromkeys(
                t
                for spec in space.variables
                for t, _ in spec.transform_columns
            )
        )
        self.role_names = tuple(ROLE_CODES)
        self.dist_names = tuple(DISTRIBUTIONS)
        self.block_names = ("outcome", "baseline", "local")

        self.term_dim = (
            self.n_vars
            + len(self.transform_names)
            + len(self.role_names)
            + len(self.dist_names)
            + len(self.block_names)
            + 1
        )
        # dispersion, lc fraction, has_random, has_zi, step frac, best reward,
        # n_active fraction, baseline bic scaled
        self.ctx_dim = 8

        key = jax.random.PRNGKey(seed)
        self.params = _init_params(
            key,
            self.term_dim,
            self.ctx_dim,
            self.config.hidden,
            self.config.embedding,
            action_space.size,
        )
        self.target_params = jax.tree_util.tree_map(lambda x: x, self.params)
        self.m, self.v, self.t = _adam_init(self.params)
        self._loss_grad = jax.jit(jax.value_and_grad(_loss))

        self.buffer = ReplayBuffer(
            self.config.buffer_size,
            self.n_vars,
            self.term_dim,
            self.ctx_dim,
            action_space.size,
        )
        self.train_steps = 0
        self.loss_history = []

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------
    def encode_state(
        self,
        state: CountModelState,
        *,
        step_frac: float = 0.0,
        best_reward: float = 0.0,
        baseline_bic: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        terms = np.zeros((self.n_vars, self.term_dim), dtype=np.float32)
        for i, term in enumerate(state.terms):
            offset = 0
            terms[i, offset + i] = 1.0
            offset += self.n_vars
            if term.transform in self.transform_names:
                terms[i, offset + self.transform_names.index(term.transform)] = 1.0
            offset += len(self.transform_names)
            terms[i, offset + self.role_names.index(term.role)] = 1.0
            offset += len(self.role_names)
            if term.distribution in self.dist_names:
                terms[i, offset + self.dist_names.index(term.distribution)] = 1.0
            offset += len(self.dist_names)
            if term.block in self.block_names:
                terms[i, offset + self.block_names.index(term.block)] = 1.0
            offset += len(self.block_names)
            terms[i, offset] = 1.0 if term.active else 0.0

        ctx = np.array(
            [
                float(bool(state.dispersion)),
                float(state.latent_classes) / max(1, self.space.max_latent_classes),
                1.0 if state.has_random else 0.0,
                1.0 if state.has_zi else 0.0,
                float(step_frac),
                float(np.clip(best_reward, -1.0, 1.0)),
                float(state.n_active) / max(1, self.n_vars),
                float(np.tanh(baseline_bic / 1e4)),
            ],
            dtype=np.float32,
        )
        return terms, ctx

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    def q_values(self, state: CountModelState, **feature_kwargs) -> np.ndarray:
        terms, ctx = self.encode_state(state, **feature_kwargs)
        q = _q_values(
            self.params, jnp.asarray(terms)[None, ...], jnp.asarray(ctx)[None, ...]
        )
        return np.asarray(q[0])

    def act(
        self,
        state: CountModelState,
        mask: np.ndarray,
        epsilon: float,
        *,
        feature_kwargs: Optional[dict] = None,
    ) -> int:
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            # The catalogue always contains terminate, but guard anyway.
            return self.action_space.size - 1
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        q = self.q_values(state, **(feature_kwargs or {}))
        q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def remember(
        self,
        state: CountModelState,
        context_kwargs: dict,
        action_index: int,
        reward: float,
        next_state: CountModelState,
        next_context_kwargs: dict,
        done: bool,
        next_mask: np.ndarray,
        mc_return: float = 0.0,
    ) -> None:
        terms, ctx = self.encode_state(state, **context_kwargs)
        next_terms, next_ctx = self.encode_state(next_state, **next_context_kwargs)
        self.buffer.add(
            terms,
            ctx,
            action_index,
            reward,
            next_terms,
            next_ctx,
            done,
            next_mask,
            mc_return,
        )

    def train(self, n_updates: int = 1) -> Optional[float]:
        if self.buffer.size < self.config.batch_size:
            return None
        loss_value = None
        for _ in range(max(1, int(n_updates))):
            batch = self.buffer.sample(self.config.batch_size, self.rng)
            loss_value, grads = self._loss_grad(
                self.params, self.target_params, batch, self.config.gamma
            )
            self.params, self.m, self.v, self.t = _adam_update(
                self.params,
                grads,
                self.m,
                self.v,
                self.t,
                lr=self.config.learning_rate,
            )
            self.train_steps += 1
            if self.train_steps % max(1, self.config.target_sync) == 0:
                self.target_params = jax.tree_util.tree_map(lambda x: x, self.params)
            self.loss_history.append(float(loss_value))
        return None if loss_value is None else float(loss_value)

    def epsilon(self, episode: int, n_episodes: int) -> float:
        c = self.config
        decay_episodes = max(1, int(n_episodes * c.epsilon_decay_frac))
        frac = min(1.0, episode / decay_episodes)
        return float(c.epsilon_start + frac * (c.epsilon_end - c.epsilon_start))
