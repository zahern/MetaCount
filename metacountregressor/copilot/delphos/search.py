"""Training / inference loop for the Delphos-style count-model agent."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .agent import AgentConfig, DeepSetQAgent
from .env import Action, DelphosCountEnv, EvalResult
from .state import CountModelState


@dataclass
class SearchConfig:
    n_episodes: int = 200
    seed: int = 0
    verbose: bool = True
    log_every: int = 20
    agent: AgentConfig = field(default_factory=AgentConfig)


@dataclass
class EpisodeRecord:
    episode: int
    length: int
    reward: float
    bic: float
    success: bool
    family: str
    n_active: int
    epsilon: float
    best_bic: float
    loss: Optional[float]
    elapsed: float


class DelphosSearch:
    def __init__(
        self,
        env: DelphosCountEnv,
        config: Optional[SearchConfig] = None,
        agent: Optional[DeepSetQAgent] = None,
    ):
        self.env = env
        self.config = config or SearchConfig()
        self.agent = agent or DeepSetQAgent(
            env.space,
            env.actions,
            self.config.agent,
            seed=self.config.seed,
        )
        if self.agent.n_vars != env.space.n_variables:
            raise ValueError(
                "pretrained agent term count does not match the target task"
            )
        if self.agent.action_space.size != env.actions.size:
            raise ValueError(
                "pretrained agent action space does not match the target task "
                "(the tasks must share the same variable catalogue)"
            )
        self.history: List[EpisodeRecord] = []
        self.start_time = 0.0

    # ------------------------------------------------------------------
    def _feature_kwargs(self, best_reward: float, step_frac: float) -> dict:
        baseline = self.env._baseline_result()
        return dict(
            step_frac=step_frac,
            best_reward=best_reward,
            baseline_bic=baseline.bic if np.isfinite(baseline.bic) else 0.0,
        )

    def run(self, n_episodes: Optional[int] = None, verbose: Optional[bool] = None) -> List[EpisodeRecord]:
        n_episodes = int(n_episodes or self.config.n_episodes)
        verbose = self.config.verbose if verbose is None else verbose
        self.start_time = time.time()
        best_bic = float("inf")
        best_reward = -1.0
        c = self.config.agent

        for episode in range(n_episodes):
            state = self.env.reset()
            eps = self.agent.epsilon(episode, n_episodes)
            done = False
            ep_reward = 0.0
            ep_length = 0
            ep_eval: Optional[EvalResult] = None
            transitions = []

            while not done:
                mask = self.env.action_mask()
                fkw = self._feature_kwargs(
                    best_reward, self.env.steps / max(1, self.env.max_steps)
                )
                action_idx = self.agent.act(
                    state, mask, eps, feature_kwargs=fkw
                )
                action = self.env.actions.action(action_idx)
                next_state, reward, done, info = self.env.step(action)

                if done:
                    ep_reward = float(reward)
                    ep_eval = info.get("eval")
                    next_mask = np.zeros(self.env.actions.size, dtype=bool)
                    next_fkw = self._feature_kwargs(
                        max(best_reward, ep_reward),
                        self.env.steps / max(1, self.env.max_steps),
                    )
                else:
                    next_mask = self.env.action_mask()
                    next_fkw = self._feature_kwargs(
                        best_reward, self.env.steps / max(1, self.env.max_steps)
                    )

                transitions.append(
                    (
                        state,
                        fkw,
                        action_idx,
                        reward,
                        next_state,
                        next_fkw,
                        done,
                        next_mask,
                    )
                )
                state = next_state
                ep_length += 1

            # Delphos Eq. 13: propagate the terminal model-quality reward to
            # every specification decision in the episode.  This is a
            # Monte-Carlo return, so earlier actions receive credit without
            # waiting for a bootstrapped Q-function.
            n_steps = len(transitions)
            gamma = c.gamma
            for l, tr in enumerate(transitions):
                mc_return = (gamma ** max(0, n_steps - 1 - l)) * ep_reward
                self.agent.remember(*tr, mc_return=float(mc_return))

            if episode + 1 >= c.warmup_episodes:
                loss = self.agent.train(c.updates_per_episode)
            else:
                loss = None

            if ep_eval is not None and ep_eval.success:
                best_bic = min(best_bic, ep_eval.bic)
            best_reward = max(best_reward, ep_reward)

            rec = EpisodeRecord(
                episode=episode + 1,
                length=ep_length,
                reward=ep_reward,
                bic=ep_eval.bic if ep_eval else float("inf"),
                success=bool(ep_eval.success) if ep_eval else False,
                family=ep_eval.family if ep_eval else "",
                n_active=ep_eval.n_active if ep_eval else 0,
                epsilon=eps,
                best_bic=best_bic,
                loss=loss,
                elapsed=time.time() - self.start_time,
            )
            self.history.append(rec)

            if verbose and (
                (episode + 1) % max(1, self.config.log_every) == 0
                or episode == 0
            ):
                print(
                    f"  [RL {episode + 1:4d}/{n_episodes}] "
                    f"reward={ep_reward:+.3f} bic={rec.bic:.2f} "
                    f"best={best_bic:.2f} eps={eps:.2f} len={ep_length} "
                    f"loss={loss if loss is None else round(loss, 4)} "
                    f"t={rec.elapsed:.1f}s",
                    flush=True,
                )

        return self.history

    # ------------------------------------------------------------------
    def propose(
        self,
        n_candidates: int = 20,
        *,
        epsilon: float = 0.05,
        verbose: bool = False,
    ) -> List[Tuple[CountModelState, EvalResult]]:
        """Roll out the learned policy and return unique estimated candidates."""
        seen = set()
        out: List[Tuple[CountModelState, EvalResult]] = []
        best_reward = -1.0
        for _ in range(int(n_candidates)):
            state = self.env.reset()
            done = False
            while not done:
                mask = self.env.action_mask()
                fkw = self._feature_kwargs(
                    best_reward, self.env.steps / max(1, self.env.max_steps)
                )
                action_idx = self.agent.act(state, mask, epsilon, feature_kwargs=fkw)
                state, reward, done, info = self.env.step(
                    self.env.actions.action(action_idx)
                )
                if done:
                    best_reward = max(best_reward, float(reward))
                    ev = info.get("eval")
                    sig = state.signature()
                    if ev is not None and sig not in seen:
                        seen.add(sig)
                        out.append((state, ev))
                    if verbose:
                        print(f"  candidate bic={ev.bic:.3f} {ev.family} {ev.n_active}")
        return out
