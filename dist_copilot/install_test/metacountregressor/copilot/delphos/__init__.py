"""Delphos-style reinforcement-learning specification search for count models.

This package adapts the Delphos MDP / DeepSet-Q idea (Nova, Hess &
van Cranenburgh) to the ``metacountregressor`` model-specification problem.

The important difference from the metaheuristic drivers is that the search
state is explicit: each variable carries its transformation, heterogeneity
role (fixed / random / grouped / zero-inflation / latent-class membership),
random-parameter distribution, and CMF block.  The global state carries the
count family (Poisson vs NB), dispersion, and number of latent classes.
"""

from .state import (
    DISTRIBUTIONS,
    ROLE_CODES,
    ROLE_NAMES,
    CountModelState,
    ModelTerm,
    SearchSpace,
    VariableSpec,
)
from .env import Action, ActionSpace, DelphosCountEnv, EvalResult
from .agent import DeepSetQAgent, ReplayBuffer
from .search import DelphosSearch, SearchConfig
from .pretrain import MultiTaskTrainer, PretrainConfig, pretrain_and_finetune
from .demonstrations import (
    action_sequence,
    add_demonstration,
    state_from_mapping,
    warm_start_with_experts,
)

__all__ = [
    "DISTRIBUTIONS",
    "ROLE_CODES",
    "ROLE_NAMES",
    "CountModelState",
    "ModelTerm",
    "SearchSpace",
    "VariableSpec",
    "Action",
    "ActionSpace",
    "DelphosCountEnv",
    "EvalResult",
    "DeepSetQAgent",
    "ReplayBuffer",
    "DelphosSearch",
    "SearchConfig",
    "MultiTaskTrainer",
    "PretrainConfig",
    "pretrain_and_finetune",
    "action_sequence",
    "add_demonstration",
    "state_from_mapping",
    "warm_start_with_experts",
]
