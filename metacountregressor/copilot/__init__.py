"""LLM-orchestrated experiment setup for metacountregressor.

The copilot turns a natural-language modelling conversation into a validated
``ExperimentSpec``.  It is deliberately split into two layers:

* **Advisor** - an LLM (or a deterministic rule-based fallback) proposes
  interpretations, variable roles, constraint tiers and diagnoses.
* **Runner** - deterministic code validates every proposal against the data
  and calls the existing ``metacountregressor`` evaluators.

The LLM never executes code and never edits the specification directly; it
only emits structured proposals that the copilot validates and the user
confirms.
"""

from .schema import (
    ColumnProfile,
    ConstraintTier,
    DataProfile,
    Diagnosis,
    ExperimentSpec,
    Issue,
    Turn,
    VariableDecision,
    spec_from_dict,
)
from .profile import load_data, profile_dataframe
from .advisor import Advisor, OpenAICompatibleAdvisor, RuleBasedAdvisor, ScriptedAdvisor
from .copilot import ExperimentCopilot
from .cli import main, run_cli

__all__ = [
    "ColumnProfile",
    "ConstraintTier",
    "DataProfile",
    "Diagnosis",
    "ExperimentSpec",
    "Issue",
    "Turn",
    "VariableDecision",
    "spec_from_dict",
    "load_data",
    "profile_dataframe",
    "Advisor",
    "OpenAICompatibleAdvisor",
    "RuleBasedAdvisor",
    "ScriptedAdvisor",
    "ExperimentCopilot",
    "main",
    "run_cli",
]
