"""Expert-demonstration warm-start for the Delphos agent.

Multitask pretraining helps when several same-schema datasets are available.
When they are not, the cheapest fix for cold-start is to seed the replay
buffer with a small number of good specifications found by SA / random /
AGDS, expressed as action trajectories.  This is behaviour cloning on the
terminal return, followed by normal RL fine-tuning.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .agent import DeepSetQAgent
from .env import Action, DelphosCountEnv
from .state import RANDOM_ROLES, CountModelState, ModelTerm, SearchSpace


def state_from_mapping(
    space: SearchSpace,
    mapping: Dict[str, Dict[str, Any]],
    *,
    dispersion: int = 1,
    latent_classes: int = 1,
) -> CountModelState:
    """Build a state from ``{variable: {role, transform, distribution}}``.

    Variables not mentioned are excluded.  This is the easiest way to encode
    an SA/random best model as a demonstration.
    """
    terms = []
    for spec in space.variables:
        cfg = mapping.get(spec.name)
        if cfg is None:
            terms.append(
                ModelTerm(
                    variable=spec.name,
                    transform=spec.default_transform,
                    role="excluded",
                    distribution=spec.default_distribution,
                    block=spec.block,
                    active=False,
                )
            )
            continue
        role = str(cfg.get("role", "fixed"))
        transform = str(cfg.get("transform", spec.default_transform))
        dist = str(cfg.get("distribution", spec.default_distribution))
        terms.append(
            ModelTerm(
                variable=spec.name,
                transform=transform,
                role=role,
                distribution=dist,
                block=spec.block,
                active=role != "excluded",
            )
        )
    state = CountModelState(
        terms=tuple(terms),
        dispersion=int(dispersion),
        latent_classes=int(latent_classes),
        family=space.family,
        task=space.task,
    )
    return space.normalize(state)


def action_sequence(env: DelphosCountEnv, target: CountModelState) -> List[Action]:
    """Canonical action sequence from the default state to ``target``."""
    start = env.space.default_state()
    actions: List[Action] = []
    for i, (cur, tgt) in enumerate(zip(start.terms, target.terms)):
        spec = env.space.variables[i]
        if spec.forced_role is None:
            if tgt.transform != cur.transform:
                actions.append(Action("transform", i, tgt.transform))
            if (tgt.role != cur.role) or (tgt.active != cur.active):
                role = tgt.role if tgt.active else "excluded"
                actions.append(Action("role", i, role))
        if tgt.active and tgt.role in RANDOM_ROLES:
            if tgt.distribution != cur.distribution:
                actions.append(Action("dist", i, tgt.distribution))
    if env.space.allow_dispersion and int(target.dispersion) != int(start.dispersion):
        actions.append(Action("dispersion", -1, "toggle"))
    if (
        env.space.max_latent_classes > 1
        and int(target.latent_classes) != int(start.latent_classes)
    ):
        actions.append(Action("latent", -1, str(target.latent_classes)))
    actions.append(Action("terminate", -1, "terminate"))
    return actions


def add_demonstration(
    agent: DeepSetQAgent,
    env: DelphosCountEnv,
    target: CountModelState,
    *,
    repeats: int = 1,
    train_updates: int = 0,
) -> float:
    """Add one expert trajectory to the replay buffer.

    Returns the terminal reward of the target specification.  The trajectory
    gets the Delphos discounted terminal return on every transition, so the
    agent learns which earlier actions lead to the expert model.
    """
    # Build the trajectory first, without mutating the caller's env.
    probe = DelphosCountEnv(
        env.evaluator,
        env.space,
        max_steps=env.max_steps,
        baseline_state=env._baseline,
        failure_penalty=env.failure_penalty,
        reward_mode=env.reward_mode,
        start_mode="linear",
        seed=0,
        verbose=False,
    )
    state = probe.space.default_state()
    probe.state = state
    probe.prev_state = None
    probe.steps = 0
    best_reward = 0.0
    transitions = []
    terminal_reward = 0.0
    for action in action_sequence(probe, target):
        mask = probe.action_mask()
        idx = probe.actions.index_of(action)
        if not mask[idx] and action.kind != "terminate":
            # The canonical sequence should be feasible; if a mask blocks it
            # due to a task-specific constraint, skip that edit.
            continue
        fkw = dict(
            step_frac=probe.steps / max(1, probe.max_steps),
            best_reward=best_reward,
            baseline_bic=probe._baseline_result().bic,
        )
        next_state, reward, done, info = probe.step(action)
        if done:
            terminal_reward = float(reward)
            next_mask = np.zeros(probe.actions.size, dtype=bool)
            next_fkw = dict(
                step_frac=probe.steps / max(1, probe.max_steps),
                best_reward=max(best_reward, terminal_reward),
                baseline_bic=probe._baseline_result().bic,
            )
        else:
            next_mask = probe.action_mask()
            next_fkw = dict(
                step_frac=probe.steps / max(1, probe.max_steps),
                best_reward=best_reward,
                baseline_bic=probe._baseline_result().bic,
            )
        transitions.append(
            (state, fkw, idx, reward, next_state, next_fkw, done, next_mask)
        )
        state = next_state
        if done:
            break

    n_steps = len(transitions)
    gamma = agent.config.gamma
    for _ in range(max(1, int(repeats))):
        for l, tr in enumerate(transitions):
            mc = (gamma ** max(0, n_steps - 1 - l)) * terminal_reward
            agent.remember(*tr, mc_return=float(mc))
    if train_updates:
        agent.train(train_updates)
    return terminal_reward


def warm_start_with_experts(
    agent: DeepSetQAgent,
    env: DelphosCountEnv,
    targets: Sequence[CountModelState],
    *,
    repeats: int = 2,
    train_updates: int = 20,
) -> List[float]:
    """Seed the replay buffer with several expert specifications."""
    rewards = []
    for target in targets:
        rewards.append(
            add_demonstration(
                agent, env, target, repeats=repeats, train_updates=0
            )
        )
    if train_updates:
        agent.train(train_updates)
    return rewards
