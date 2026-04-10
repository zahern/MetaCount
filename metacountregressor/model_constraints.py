"""
model_constraints.py
--------------------
Builder-style API for expressing structural constraints on variables
before passing them to ExperimentBuilder.build_evaluator().

Example
-------
>>> from metacountregressor import ModelConstraints
>>> c = ModelConstraints()
>>> (c
...     .force_fixed('AADT')            # must be fixed (role 1) or excluded
...     .no_zi('SPEED', 'WIDTH')        # cannot be zero-inflation terms
...     .membership_only('FC_ENCODED')  # can only drive class membership
...     .allow_random('LENGTH', distributions=['normal', 'lognormal'])
...     .no_random('URB')
...     .force_include('OFFSET')        # cannot be excluded
...     .exclude('YEAR', 'ID')
... )
>>> evaluator = builder.build_evaluator(constraints=c, ...)
"""

from __future__ import annotations

from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Role constants (mirrors main_hpc.py)
# ---------------------------------------------------------------------------
_ROLE_EXCLUDED        = 0
_ROLE_FIXED           = 1
_ROLE_RANDOM_IND      = 2
_ROLE_RANDOM_COR      = 3
_ROLE_GROUPED         = 4
_ROLE_HETRO           = 5
_ROLE_ZI              = 6
_ROLE_MEMBERSHIP      = 7
_ROLE_MEMBERSHIP_FIX  = 8

_ALL_ROLES = list(range(9))
_OUTCOME_ROLES   = [0, 1, 2, 3, 4, 5, 6]   # roles that affect outcome eq
_RANDOM_ROLES    = [2, 3, 4]
_MEMBERSHIP_ROLES = [7, 8]

_DIST_OPTIONS = ["normal", "lognormal", "triangular", "uniform"]

_ROLE_LABELS = {
    0: "Excluded",
    1: "Fixed",
    2: "Random (ind.)",
    3: "Random (corr.)",
    4: "Grouped",
    5: "Heterogeneity",
    6: "Zero Inflation",
    7: "Membership only",
    8: "Membership + Fixed",
}


class ModelConstraints:
    """
    Fluent builder for per-variable structural constraints.

    All methods return ``self`` so calls can be chained.  Call
    ``to_evaluator_kwargs()`` to convert to the dict expected by
    ``ExperimentBuilder.build_evaluator()``.
    """

    def __init__(self) -> None:
        # var -> set of allowed role codes
        self._roles: dict[str, set[int]] = {}
        # var -> list of allowed distributions (for random roles)
        self._dists: dict[str, list[str]] = {}
        # variables to exclude entirely
        self._excluded: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure(self, var: str) -> set[int]:
        """Return (creating if needed) the allowed-role set for *var*."""
        if var not in self._roles:
            self._roles[var] = set(_ALL_ROLES)
        return self._roles[var]

    def _apply(self, vars: Sequence[str], fn) -> "ModelConstraints":
        for v in vars:
            fn(self._ensure(v))
        return self

    # ------------------------------------------------------------------
    # Public constraint methods
    # ------------------------------------------------------------------

    def force_fixed(self, *variables: str) -> "ModelConstraints":
        """
        Variable may only be **excluded** (0) or **fixed** (1).

        Use this for mandatory exposure offsets, forced AADT terms, etc.
        """
        return self._apply(variables, lambda s: s.intersection_update({0, 1}))

    def force_include(self, *variables: str) -> "ModelConstraints":
        """
        Variable **must** appear in the model (role 0 / excluded is banned).

        Useful for exposure/offset terms that should never drop out.
        """
        return self._apply(variables, lambda s: s.discard(_ROLE_EXCLUDED) or s)

    def no_zi(self, *variables: str) -> "ModelConstraints":
        """
        Variable **cannot** enter the zero-inflation equation (role 6).

        Apply broadly when zero-inflation is theoretically implausible for
        a covariate (e.g. geometric road features in a crash-frequency model).
        """
        return self._apply(variables, lambda s: s.discard(_ROLE_ZI))

    def no_random(self, *variables: str) -> "ModelConstraints":
        """
        Variable **cannot** have a random parameter (roles 2, 3, 4 banned).

        Use for categorical indicators or binary dummies where individual
        taste variation is not meaningful.
        """
        return self._apply(
            variables,
            lambda s: [s.discard(r) for r in _RANDOM_ROLES],
        )

    def allow_random(
        self,
        *variables: str,
        distributions: Optional[list[str]] = None,
    ) -> "ModelConstraints":
        """
        Variable **may** be a random parameter.

        Optionally restrict the allowed *distributions* to a subset of
        ``["normal", "lognormal", "triangular", "uniform"]``.

        Does *not* force the variable to be random — the search can still
        choose fixed or excluded if the data do not support a random term.
        """
        dists = distributions if distributions is not None else list(_DIST_OPTIONS)
        invalid = [d for d in dists if d not in _DIST_OPTIONS]
        if invalid:
            raise ValueError(
                f"Unknown distributions {invalid}. "
                f"Choose from {_DIST_OPTIONS}."
            )
        # Ensure random roles are in the allowed set
        for v in variables:
            s = self._ensure(v)
            s.update(_RANDOM_ROLES)
            self._dists[v] = dists
        return self

    def membership_only(self, *variables: str) -> "ModelConstraints":
        """
        Variable may only drive **class membership** (role 7) or be excluded.

        The variable will have no direct effect on the outcome equation.
        Has no effect when the model has a single latent class.
        """
        return self._apply(
            variables,
            lambda s: s.intersection_update({_ROLE_EXCLUDED, _ROLE_MEMBERSHIP}),
        )

    def allow_membership(self, *variables: str) -> "ModelConstraints":
        """
        Variable **may** enter the class-membership equation (roles 7 or 8),
        in addition to whatever outcome roles it already allows.
        """
        return self._apply(
            variables,
            lambda s: s.update(_MEMBERSHIP_ROLES),
        )

    def outcome_only(self, *variables: str) -> "ModelConstraints":
        """
        Variable **cannot** enter the class-membership equation (roles 7, 8 banned).

        Inverse of :meth:`membership_only`.
        """
        return self._apply(
            variables,
            lambda s: [s.discard(r) for r in _MEMBERSHIP_ROLES],
        )

    def exclude(self, *variables: str) -> "ModelConstraints":
        """
        Completely exclude *variables* from the search space.

        Equivalent to always assigning role 0, but also removes them from
        the variable list passed to ``build_evaluator``.
        """
        for v in variables:
            if v not in self._excluded:
                self._excluded.append(v)
        return self

    def set_roles(self, variable: str, roles: list[int]) -> "ModelConstraints":
        """
        Directly set the allowed role codes for *variable*.

        Lower-level escape hatch when the named methods are not expressive
        enough.  ``roles`` must be a non-empty subset of 0–8.
        """
        invalid = [r for r in roles if r not in range(9)]
        if invalid:
            raise ValueError(f"Invalid role codes {invalid}. Valid: 0–8.")
        self._roles[variable] = set(roles)
        return self

    def set_distributions(
        self,
        variable: str,
        distributions: list[str],
    ) -> "ModelConstraints":
        """
        Directly set allowed distributions for *variable* (for random roles).
        """
        invalid = [d for d in distributions if d not in _DIST_OPTIONS]
        if invalid:
            raise ValueError(
                f"Unknown distributions {invalid}. Choose from {_DIST_OPTIONS}."
            )
        self._dists[variable] = list(distributions)
        return self

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_evaluator_kwargs(self) -> dict:
        """
        Convert to the kwargs dict expected by
        ``ExperimentBuilder.build_evaluator()``.

        Returns a dict with keys ``fixed_override``, ``membership_override``,
        ``exclude``, and optionally ``dist_override`` — only non-empty entries
        are included.

        Routing rules
        -------------
        - Variable is *pure membership* (only roles in {0, 7, 8}):
          → ``membership_override``
        - Variable has any outcome role (0–6), possibly also membership roles:
          → ``fixed_override`` (full set including membership roles)
        Each variable appears in at most one override dict because
        ``build_evaluator`` merges them with membership winning.
        """
        fixed_override: dict[str, list[int]] = {}
        membership_override: dict[str, list[int]] = {}

        for var, role_set in self._roles.items():
            if var in self._excluded:
                continue  # handled via 'exclude' key

            if set(role_set) == set(_ALL_ROLES):
                continue  # no restriction — omit entirely

            sorted_roles = sorted(role_set)
            # Roles 1–6 are actual outcome-equation roles (role 0 = excluded,
            # roles 7–8 are membership-only; neither counts as "outcome" here)
            _ACTUAL_OUTCOME = {1, 2, 3, 4, 5, 6}
            has_outcome_role = bool(role_set & _ACTUAL_OUTCOME)

            if has_outcome_role:
                # Emit into fixed_override (full set — outcome + any membership)
                fixed_override[var] = sorted_roles
            else:
                # Pure membership/excluded variable (only subsets of {0, 7, 8})
                membership_override[var] = sorted_roles

        result: dict = {}
        if fixed_override:
            result["fixed_override"] = fixed_override
        if membership_override:
            result["membership_override"] = membership_override
        if self._excluded:
            result["exclude"] = list(self._excluded)
        if self._dists:
            result["dist_override"] = dict(self._dists)
        return result

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = ["ModelConstraints("]
        all_vars = sorted(
            set(list(self._roles.keys()) + self._excluded),
            key=lambda v: (v in self._excluded, v),
        )
        for var in all_vars:
            if var in self._excluded:
                lines.append(f"  {var!r}: EXCLUDED")
                continue
            roles = sorted(self._roles.get(var, set(_ALL_ROLES)))
            role_str = ", ".join(
                f"{r}={_ROLE_LABELS[r]}" for r in roles
            )
            dist_str = ""
            if var in self._dists:
                dist_str = f"  dists={self._dists[var]}"
            lines.append(f"  {var!r}: [{role_str}]{dist_str}")
        lines.append(")")
        return "\n".join(lines)

    def summary(self) -> None:
        """Print a human-readable summary of all constraints."""
        print(self)
