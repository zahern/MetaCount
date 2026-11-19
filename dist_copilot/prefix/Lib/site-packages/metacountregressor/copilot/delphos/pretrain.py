"""Multitask pretraining and warm-starting for the Delphos agent.

The single-task agent is sample-hungry because every task starts from a
random Q-network.  This module adds the missing ingredient from the Delphos
multitask paper: train one shared DeepSet-Q policy across several related
specification tasks, then fine-tune it on the target task.

The current implementation requires the tasks to share a variable catalogue
(the same columns, transforms and action space).  That covers the common
case of repeated crash datasets / bootstrap panels with a common schema.
For datasets with different schemas, map variables to shared concepts first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .agent import AgentConfig, DeepSetQAgent
from .env import DelphosCountEnv, EvalResult
from .search import DelphosSearch, SearchConfig
from .state import CountModelState


@dataclass
class PretrainConfig:
    n_episodes: int = 60
    updates_per_episode: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.20
    epsilon_decay_frac: float = 0.8
    verbose: bool = True
    log_every: int = 10


@dataclass
class TaskRecord:
    episode: int
    task: str
    length: int
    reward: float
    bic: float
    best_bic: float
    epsilon: float
    loss: Optional[float]
    elapsed: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode": self.episode,
            "task": self.task,
            "length": self.length,
            "reward": self.reward,
            "bic": self.bic,
            "best_bic": self.best_bic,
            "epsilon": self.epsilon,
            "loss": self.loss,
            "elapsed": self.elapsed,
        }


class MultiTaskTrainer:
    """Train one shared specification policy across related tasks."""

    def __init__(
        self,
        envs: Sequence[DelphosCountEnv],
        agent: Optional[DeepSetQAgent] = None,
        config: Optional[PretrainConfig] = None,
        agent_config: Optional[AgentConfig] = None,
        seed: int = 0,
    ):
        if not envs:
            raise ValueError("at least one task environment is required")
        self.envs = list(envs)
        self.config = config or PretrainConfig()
        self.agent = agent or DeepSetQAgent(
            self.envs[0].space,
            self.envs[0].actions,
            agent_config or AgentConfig(
                hidden=128,
                embedding=64,
                learning_rate=3e-3,
                batch_size=64,
                warmup_episodes=8,
                updates_per_episode=self.config.updates_per_episode,
                epsilon_start=self.config.epsilon_start,
                epsilon_end=self.config.epsilon_end,
                epsilon_decay_frac=self.config.epsilon_decay_frac,
            ),
            seed=seed,
        )
        self._validate()
        self.history: List[TaskRecord] = []
        self.best: Dict[str, Tuple[Optional[CountModelState], Optional[EvalResult]]] = {}
        self.start_time = 0.0

    def _validate(self) -> None:
        ref = self.envs[0]
        for env in self.envs:
            if env.space.n_variables != self.agent.n_vars:
                raise ValueError("all tasks must have the same number of variables")
            if env.actions.size != self.agent.action_space.size:
                raise ValueError("all tasks must have the same action catalogue")
            if [v.name for v in env.space.variables] != [
                v.name for v in ref.space.variables
            ]:
                raise ValueError(
                    "the current multitask trainer requires a shared variable "
                    "catalogue; map task-specific variables to shared concepts"
                )

    def _features(self, env: DelphosCountEnv, best_reward: float) -> dict:
        base = env._baseline_result()
        return dict(
            step_frac=env.steps / max(1, env.max_steps),
            best_reward=best_reward,
            baseline_bic=base.bic if np.isfinite(base.bic) else 0.0,
        )

    def fit(self, n_episodes: Optional[int] = None) -> List[TaskRecord]:
        n_episodes = int(n_episodes or self.config.n_episodes)
        self.start_time = time.time()
        best_reward = -1.0
        global_best = float("inf")

        for episode in range(n_episodes):
            env = self.envs[episode % len(self.envs)]
            task_name = env.space.task or f"task{episode % len(self.envs)}"
            state = env.reset()
            eps = self.agent.epsilon(episode, n_episodes)
            transitions = []
            done = False
            ep_reward = 0.0
            ep_eval: Optional[EvalResult] = None
            length = 0

            while not done:
                mask = env.action_mask()
                fkw = self._features(env, best_reward)
                action_idx = self.agent.act(
                    state, mask, eps, feature_kwargs=fkw
                )
                next_state, reward, done, info = env.step(
                    env.actions.action(action_idx)
                )
                if done:
                    ep_reward = float(reward)
                    ep_eval = info.get("eval")
                    next_mask = np.zeros(env.actions.size, dtype=bool)
                    next_fkw = self._features(
                        env, max(best_reward, ep_reward)
                    )
                else:
                    next_mask = env.action_mask()
                    next_fkw = self._features(env, best_reward)
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
                length += 1

            n_steps = len(transitions)
            gamma = self.agent.config.gamma
            for l, tr in enumerate(transitions):
                mc = (gamma ** max(0, n_steps - 1 - l)) * ep_reward
                self.agent.remember(*tr, mc_return=float(mc))

            if episode + 1 >= self.agent.config.warmup_episodes:
                loss = self.agent.train(self.config.updates_per_episode)
            else:
                loss = None

            if ep_eval is not None and ep_eval.success:
                global_best = min(global_best, ep_eval.bic)
                prev = self.best.get(task_name)
                if prev is None or ep_eval.bic < prev[1].bic:
                    self.best[task_name] = (env.best_state, ep_eval)
            best_reward = max(best_reward, ep_reward)

            rec = TaskRecord(
                episode=episode + 1,
                task=task_name,
                length=length,
                reward=ep_reward,
                bic=ep_eval.bic if ep_eval else float("inf"),
                best_bic=global_best,
                epsilon=eps,
                loss=loss,
                elapsed=time.time() - self.start_time,
            )
            self.history.append(rec)
            if self.config.verbose and (
                (episode + 1) % max(1, self.config.log_every) == 0
                or episode == 0
            ):
                print(
                    f"  [pretrain {episode + 1:4d}/{n_episodes}] "
                    f"task={task_name} reward={ep_reward:+.3f} "
                    f"best={global_best:.2f} eps={eps:.2f} "
                    f"len={length} loss={loss if loss is None else round(loss, 4)}"
                )
        return self.history

    def fine_tune(
        self,
        env: DelphosCountEnv,
        n_episodes: int = 20,
        *,
        verbose: bool = True,
    ) -> List[Any]:
        """Continue training the shared agent on a held-out target task."""
        cfg = SearchConfig(
            n_episodes=int(n_episodes),
            seed=0,
            verbose=verbose,
            agent=self.agent.config,
        )
        search = DelphosSearch(env, cfg, agent=self.agent)
        return search.run(n_episodes=n_episodes, verbose=verbose)

    def zero_shot(
        self,
        env: DelphosCountEnv,
        n_candidates: int = 10,
        *,
        epsilon: float = 0.05,
    ) -> List[Tuple[CountModelState, EvalResult]]:
        cfg = SearchConfig(
            n_episodes=int(n_candidates),
            seed=0,
            verbose=False,
            agent=self.agent.config,
        )
        search = DelphosSearch(env, cfg, agent=self.agent)
        return search.propose(n_candidates=n_candidates, epsilon=epsilon)


def pretrain_and_finetune(
    pretrain_envs: Sequence[DelphosCountEnv],
    target_env: DelphosCountEnv,
    *,
    pretrain_episodes: int = 60,
    finetune_episodes: int = 20,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[MultiTaskTrainer, List[Any]]:
    """Convenience wrapper for the common pretrain -> fine-tune workflow."""
    trainer = MultiTaskTrainer(
        pretrain_envs,
        config=PretrainConfig(n_episodes=pretrain_episodes, verbose=verbose),
        seed=seed,
    )
    trainer.fit()
    history = trainer.fine_tune(
        target_env, n_episodes=finetune_episodes, verbose=verbose
    )
    return trainer, history
