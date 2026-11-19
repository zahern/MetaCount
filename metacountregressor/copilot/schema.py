"""Validated data structures for the experiment copilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_missing: int
    n_unique: int
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    is_integer: bool = False
    is_nonnegative: bool = False
    is_binary: bool = False
    samples: List[Any] = field(default_factory=list)

    @property
    def missing_frac(self) -> float:
        return 0.0  # filled by DataProfile.to_dict if needed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataProfile:
    path: str
    n_rows: int
    n_cols: int
    columns: List[ColumnProfile]
    response_candidates: List[str] = field(default_factory=list)
    id_candidates: List[str] = field(default_factory=list)
    offset_candidates: List[str] = field(default_factory=list)
    group_candidates: List[str] = field(default_factory=list)

    def column(self, name: str) -> Optional[ColumnProfile]:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.columns]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
            "response_candidates": list(self.response_candidates),
            "id_candidates": list(self.id_candidates),
            "offset_candidates": list(self.offset_candidates),
            "group_candidates": list(self.group_candidates),
        }


@dataclass
class VariableDecision:
    """One variable's proposed treatment."""

    name: str
    include: bool = True
    role_hint: str = "auto"  # fixed | random | exclude | zi | membership | offset
    reason: str = ""
    allowed_roles: Tuple[str, ...] = ()
    allowed_distributions: Tuple[str, ...] = ()
    transform: str = "identity"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstraintTier:
    """A named layer of the constraint hierarchy.

    ``rules`` are method calls on ``ModelConstraints``.  For example::

        {"method": "no_random", "variables": ["URB", "GRADEBR"]}
        {"method": "force_fixed", "variables": ["OFFSET"]}
        {"method": "allow_random", "variables": ["CURVES"],
         "distributions": ["lognormal"]}
        {"method": "mutual_exclusion", "groups": [["AADT", "ADTLANE"]]}
    """

    tier: str  # hard | structural | plausibility | search
    rules: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentSpec:
    data_path: str
    response: str = ""
    id_col: Optional[str] = None
    offset_col: Optional[str] = None
    group_id_col: Optional[str] = None
    family: str = "count"  # count | cmf | duration | linear | multivariate | survival
    variables: List[VariableDecision] = field(default_factory=list)
    constraints: List[ConstraintTier] = field(default_factory=list)
    max_latent_classes: int = 1
    R: int = 50
    objective: str = "bic"
    algo: str = "sa"
    max_iter: int = 500
    seed: int = 0
    output_dir: str = "metacount_results"
    cmf: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def included_variables(self) -> List[str]:
        return [v.name for v in self.variables if v.include and v.role_hint != "offset"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Turn:
    question: str
    field: str
    options: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    done: bool = False
    hint: str = ""


@dataclass
class Issue:
    severity: str  # info | warning | error
    code: str
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    proposed_patch: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Diagnosis:
    summary: str
    issues: List[Issue] = field(default_factory=list)
    next_spec: Optional[ExperimentSpec] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "issues": [i.to_dict() for i in self.issues],
            "next_spec": self.next_spec.to_dict() if self.next_spec else None,
        }


def spec_from_dict(data: Dict[str, Any]) -> ExperimentSpec:
    """Rebuild a validated spec from JSON (used by the CLI's /load)."""
    variables = [
        VariableDecision(
            name=str(v["name"]),
            include=bool(v.get("include", True)),
            role_hint=str(v.get("role_hint", "auto")),
            reason=str(v.get("reason", "")),
            allowed_roles=tuple(v.get("allowed_roles", ())),
            allowed_distributions=tuple(v.get("allowed_distributions", ())),
            transform=str(v.get("transform", "identity")),
        )
        for v in data.get("variables", [])
    ]
    constraints = [
        ConstraintTier(
            tier=str(c.get("tier", "search")),
            rules=list(c.get("rules", [])),
            rationale=str(c.get("rationale", "")),
        )
        for c in data.get("constraints", [])
    ]
    return ExperimentSpec(
        data_path=str(data["data_path"]),
        response=str(data.get("response", "")),
        id_col=data.get("id_col"),
        offset_col=data.get("offset_col"),
        group_id_col=data.get("group_id_col"),
        family=str(data.get("family", "count")),
        variables=variables,
        constraints=constraints,
        max_latent_classes=int(data.get("max_latent_classes", 1)),
        R=int(data.get("R", 50)),
        objective=str(data.get("objective", "bic")),
        algo=str(data.get("algo", "sa")),
        max_iter=int(data.get("max_iter", 500)),
        seed=int(data.get("seed", 0)),
        output_dir=str(data.get("output_dir", "metacount_results")),
        cmf=dict(data.get("cmf", {})),
        notes=list(data.get("notes", [])),
    )


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "include", "keep"}:
        return True
    if text in {"no", "n", "false", "0", "exclude", "drop"}:
        return False
    raise ValueError(f"cannot interpret {value!r} as yes/no")


def parse_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    for sep in (",", ";", "\n", "|"):
        if sep in text:
            return [p.strip() for p in text.split(sep) if p.strip()]
    return [text.strip()] if text.strip() else []
