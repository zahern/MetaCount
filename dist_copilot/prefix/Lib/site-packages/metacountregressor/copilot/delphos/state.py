"""Explicit state representation for count-model specification.

A state is a set of modelling terms plus global model-family decisions.
Each term records *how* a variable enters the model:

    (variable, transformation, heterogeneity role, distribution, block)

This mirrors the DeepSet state of Delphos (a set of modelling terms) while
adding the extra decisions that matter for heterogeneous count models and
hierarchical Crash Modification Factor (CMF) models.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

# Role codes used by metacountregressor (main_hpc / experiment_package).
ROLE_CODES: Dict[str, int] = {
    "excluded": 0,
    "fixed": 1,
    "random_ind": 2,
    "random_cor": 3,
    "grouped": 4,
    "hetero": 5,
    "zi": 6,
    "membership": 7,
    "membership_fixed": 8,
}
ROLE_NAMES: Dict[int, str] = {v: k for k, v in ROLE_CODES.items()}
RANDOM_ROLES = ("random_ind", "random_cor", "grouped")

# Default evaluator order (populate_allowed_distributions in main_hpc.py).
DISTRIBUTIONS: Tuple[str, ...] = (
    "normal",
    "uniform",
    "lognormal",
    "triangular",
    "neglognormal",
)


@dataclass(frozen=True)
class ModelTerm:
    """One modelling term in the specification state."""

    variable: str
    transform: str = "identity"
    role: str = "excluded"
    distribution: str = "normal"
    block: str = "outcome"
    active: bool = False

    def __post_init__(self) -> None:
        if self.role not in ROLE_CODES:
            raise ValueError(f"unknown role {self.role!r}")
        if self.role == "excluded" and self.active:
            object.__setattr__(self, "active", False)
        if self.role != "excluded" and not self.active:
            object.__setattr__(self, "active", True)


@dataclass(frozen=True)
class VariableSpec:
    """A searchable variable and the columns that implement its transforms.

    ``transform_columns`` maps a human-readable transform name to the actual
    dataframe column used by the evaluator.  A variable with one transform
    (e.g. a binary dummy) has a single entry.
    """

    name: str
    transform_columns: Tuple[Tuple[str, str], ...]
    allowed_roles: Tuple[str, ...]
    allowed_distributions: Tuple[str, ...] = DISTRIBUTIONS
    block: str = "outcome"
    forced_role: Optional[str] = None
    kind: str = "continuous"

    def __post_init__(self) -> None:
        if not self.transform_columns:
            raise ValueError(f"{self.name}: at least one transform column required")
        if self.forced_role is not None:
            if self.forced_role not in ROLE_CODES:
                raise ValueError(f"{self.name}: bad forced_role {self.forced_role!r}")
            if self.forced_role not in self.allowed_roles:
                object.__setattr__(
                    self,
                    "allowed_roles",
                    tuple(dict.fromkeys((self.forced_role, *self.allowed_roles))),
                )
        if not self.allowed_roles:
            object.__setattr__(self, "allowed_roles", ("excluded",))

    @property
    def default_transform(self) -> str:
        return self.transform_columns[0][0]

    @property
    def default_distribution(self) -> str:
        return self.allowed_distributions[0]

    def column_for_transform(self, transform: str) -> str:
        for name, column in self.transform_columns:
            if name == transform:
                return column
        return self.transform_columns[0][1]

    def transform_for_column(self, column: str) -> Optional[str]:
        for name, col in self.transform_columns:
            if col == column:
                return name
        return None

    def coerce_role(self, role: str) -> str:
        if self.forced_role is not None:
            return self.forced_role
        return role if role in self.allowed_roles else "excluded"

    def coerce_distribution(self, distribution: str) -> str:
        return (
            distribution
            if distribution in self.allowed_distributions
            else self.default_distribution
        )


@dataclass(frozen=True)
class CountModelState:
    """Full specification state for one count-model candidate."""

    terms: Tuple[ModelTerm, ...]
    dispersion: int = 0
    latent_classes: int = 1
    family: str = "count"
    task: str = ""

    def term(self, index: int) -> ModelTerm:
        return self.terms[index]

    def with_term(self, index: int, **changes) -> "CountModelState":
        return replace(
            self,
            terms=tuple(
                replace(t, **changes) if i == index else t
                for i, t in enumerate(self.terms)
            ),
        )

    def with_terms(self, terms: Sequence[ModelTerm]) -> "CountModelState":
        return replace(self, terms=tuple(terms))

    @property
    def active_terms(self) -> Tuple[ModelTerm, ...]:
        return tuple(t for t in self.terms if t.active)

    @property
    def n_active(self) -> int:
        return sum(1 for t in self.terms if t.active)

    @property
    def has_random(self) -> bool:
        return any(t.active and t.role in RANDOM_ROLES for t in self.terms)

    @property
    def has_zi(self) -> bool:
        return any(t.active and t.role == "zi" for t in self.terms)

    @property
    def has_membership(self) -> bool:
        return any(
            t.active and t.role in ("membership", "membership_fixed")
            for t in self.terms
        )

    def derived_family(self) -> str:
        """Human-readable model family implied by the state."""
        base = "nb" if self.dispersion else "poisson"
        if self.latent_classes > 1:
            base = f"lc{self.latent_classes}_{base}"
        if self.has_random:
            base = f"rp_{base}"
        if self.has_zi:
            base = f"zi_{base}"
        if self.family == "cmf":
            base = f"cmf_{base}"
        return base

    def signature(self) -> Tuple:
        return (
            tuple(
                (
                    t.variable,
                    t.transform,
                    t.role,
                    t.distribution if t.role in RANDOM_ROLES else "",
                    t.block,
                    bool(t.active),
                )
                for t in self.terms
            ),
            int(self.dispersion),
            int(self.latent_classes),
            self.family,
        )

    def pretty(self, max_terms: int = 30) -> str:
        lines = [
            f"{self.derived_family()} | dispersion={self.dispersion} | "
            f"latent_classes={self.latent_classes}"
        ]
        active = self.active_terms
        if not active:
            lines.append("  (no active terms)")
        for t in active[:max_terms]:
            dist = f" ~ {t.distribution}" if t.role in RANDOM_ROLES else ""
            lines.append(
                f"  {t.variable:<24} {t.transform:<10} {t.role:<16}{dist}"
                + (f"  [{t.block}]" if t.block != "outcome" else "")
            )
        if len(active) > max_terms:
            lines.append(f"  ... {len(active) - max_terms} more")
        return "\n".join(lines)


def make_variable_specs_from_evaluator(
    evaluator,
    *,
    base_variables: Optional[Mapping[str, Sequence[Tuple[str, str]]]] = None,
    forced_roles: Optional[Mapping[str, str]] = None,
    blocks: Optional[Mapping[str, str]] = None,
    role_overrides: Optional[Mapping[str, Sequence[str]]] = None,
    kind: Optional[Mapping[str, str]] = None,
    name_map: Optional[Mapping[str, str]] = None,
) -> Tuple[VariableSpec, ...]:
    """Build a VariableSpec catalogue from an existing evaluator.

    Parameters
    ----------
    evaluator
        A ``StructureEvaluatorLC`` / CMF evaluator with ``vars``,
        ``allowed_roles`` and ``allowed_distributions``.
    base_variables
        Optional mapping ``display_name -> ((transform, column), ...)``.
        When omitted every evaluator column becomes its own variable with an
        identity transform.
    forced_roles, blocks, role_overrides, kind, name_map
        Optional per-display-name overrides.
    """
    forced_roles = dict(forced_roles or {})
    blocks = dict(blocks or {})
    role_overrides = dict(role_overrides or {})
    kind = dict(kind or {})
    name_map = dict(name_map or {})

    specs = []
    if base_variables is None:
        base_variables = {
            name_map.get(col, col): (("identity", col),) for col in evaluator.vars
        }

    for display_name, transform_columns in base_variables.items():
        columns = tuple(transform_columns)
        primary_col = columns[0][1]
        allowed = list(
            evaluator.allowed_roles.get(primary_col, list(ROLE_CODES))
        )
        allowed_names = [ROLE_NAMES.get(int(r), "excluded") for r in allowed]
        if display_name in role_overrides:
            allowed_names = [
                r for r in role_overrides[display_name] if r in allowed_names
            ] or allowed_names
        dists = tuple(
            evaluator.allowed_distributions.get(primary_col, DISTRIBUTIONS)
        )
        specs.append(
            VariableSpec(
                name=display_name,
                transform_columns=columns,
                allowed_roles=tuple(dict.fromkeys(allowed_names)),
                allowed_distributions=dists,
                block=blocks.get(display_name, "outcome"),
                forced_role=forced_roles.get(display_name),
                kind=kind.get(display_name, "continuous"),
            )
        )
    return tuple(specs)


@dataclass(frozen=True)
class SearchSpace:
    """The catalogue of possible modelling terms for one search task."""

    variables: Tuple[VariableSpec, ...]
    max_latent_classes: int = 1
    allow_dispersion: bool = True
    family: str = "count"
    task: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        columns: list[str] = []
        column_index: dict[str, int] = {}
        column_to_var: dict[str, int] = {}
        for i, spec in enumerate(self.variables):
            for _, column in spec.transform_columns:
                if column in column_index:
                    raise ValueError(f"duplicate evaluator column {column!r}")
                column_index[column] = len(columns)
                column_to_var[column] = i
                columns.append(column)
        object.__setattr__(self, "columns", tuple(columns))
        object.__setattr__(self, "column_index", column_index)
        object.__setattr__(self, "column_to_var", column_to_var)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    def null_state(self) -> CountModelState:
        """Intercept-and-offset null model used as the reward baseline."""
        terms = []
        for spec in self.variables:
            role = spec.forced_role or "excluded"
            terms.append(
                ModelTerm(
                    variable=spec.name,
                    transform=spec.default_transform,
                    role=role,
                    distribution=spec.default_distribution,
                    block=spec.block,
                    active=role != "excluded",
                )
            )
        return CountModelState(
            terms=tuple(terms),
            dispersion=0,
            latent_classes=1,
            family=self.family,
            task=self.task,
        )

    def default_state(self) -> CountModelState:
        """Linear additive starting specification, as in Delphos.

        Every variable enters linearly and homogeneously, the count family is
        NB when allowed, and there is one latent class.  The agent then
        removes / transforms / re-heterogenises terms.
        """
        terms = []
        for spec in self.variables:
            if spec.forced_role is not None:
                role = spec.forced_role
            elif "fixed" in spec.allowed_roles:
                role = "fixed"
            else:
                role = next(
                    (r for r in spec.allowed_roles if r != "excluded"),
                    "excluded",
                )
            terms.append(
                ModelTerm(
                    variable=spec.name,
                    transform=spec.default_transform,
                    role=role,
                    distribution=spec.default_distribution,
                    block=spec.block,
                    active=role != "excluded",
                )
            )
        return CountModelState(
            terms=tuple(terms),
            dispersion=1 if self.allow_dispersion else 0,
            latent_classes=1,
            family=self.family,
            task=self.task,
        )

    def from_decision(self, decision: Sequence[float]) -> CountModelState:
        """Decode the integer decision vector used by metacountregressor."""
        d = self.n_columns
        dec = np.asarray(decision, dtype=float).reshape(-1)
        roles = dec[:d].astype(int)
        dists = dec[d : 2 * d].astype(int) if len(dec) >= 2 * d else np.zeros(d, int)
        dispersion = (
            int(dec[2 * d]) % 2
            if self.allow_dispersion and len(dec) > 2 * d
            else 0
        )
        latent = 1
        if self.max_latent_classes > 1 and len(dec) > 2 * d + 1:
            latent = int(dec[2 * d + 1]) % self.max_latent_classes + 1

        terms = []
        for spec in self.variables:
            chosen = None
            for transform, column in spec.transform_columns:
                idx = self.column_index[column]
                if int(roles[idx]) != 0:
                    chosen = (transform, column, idx)
                    break
            if spec.forced_role is not None:
                if chosen is None:
                    transform = spec.default_transform
                    column = spec.column_for_transform(transform)
                    idx = self.column_index[column]
                else:
                    transform, column, idx = chosen
                role = spec.forced_role
                dist = spec.default_distribution
            elif chosen is None:
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
            else:
                transform, column, idx = chosen
                role = ROLE_NAMES.get(int(roles[idx]) % 9, "excluded")
                role = spec.coerce_role(role)
                dist = spec.coerce_distribution(
                    spec.allowed_distributions[int(dists[idx]) % len(spec.allowed_distributions)]
                )

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

        return CountModelState(
            terms=tuple(terms),
            dispersion=int(dispersion),
            latent_classes=int(latent),
            family=self.family,
            task=self.task,
        )

    def to_decision(self, state: CountModelState) -> np.ndarray:
        """Encode a state as the integer decision vector the evaluator wants."""
        d = self.n_columns
        roles = np.zeros(d, dtype=float)
        dists = np.zeros(d, dtype=float)
        for i, spec in enumerate(self.variables):
            term = state.terms[i]
            if spec.forced_role is not None:
                role = spec.forced_role
                transform = term.transform
                dist = term.distribution
            else:
                role = term.role if term.active else "excluded"
                transform = term.transform
                dist = term.distribution
            if role == "excluded":
                continue
            column = spec.column_for_transform(transform)
            idx = self.column_index[column]
            roles[idx] = ROLE_CODES[role]
            allowed_dists = spec.allowed_distributions
            if dist in allowed_dists:
                dists[idx] = allowed_dists.index(dist)

        parts = [roles, dists, np.array([float(state.dispersion if self.allow_dispersion else 0)])]
        if self.max_latent_classes > 1:
            lc = min(max(int(state.latent_classes), 1), self.max_latent_classes)
            parts.append(np.array([float(lc - 1)]))
        return np.concatenate(parts)

    def normalize(self, state: CountModelState) -> CountModelState:
        """Clamp a state to the catalogue (roles, transforms, distributions)."""
        terms = []
        for spec, term in zip(self.variables, state.terms):
            transform = (
                term.transform
                if term.transform in dict(spec.transform_columns)
                else spec.default_transform
            )
            role = spec.coerce_role(term.role if term.active else "excluded")
            dist = spec.coerce_distribution(term.distribution)
            terms.append(
                ModelTerm(
                    variable=spec.name,
                    transform=transform,
                    role=role,
                    distribution=dist,
                    block=spec.block,
                    active=role != "excluded" or spec.forced_role is not None,
                )
            )
        return replace(
            state,
            terms=tuple(terms),
            dispersion=int(bool(state.dispersion)) if self.allow_dispersion else 0,
            latent_classes=min(
                max(int(state.latent_classes), 1), max(1, self.max_latent_classes)
            ),
            family=self.family,
            task=self.task,
        )
