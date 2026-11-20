"""Conversational experiment setup with a validated, deterministic backend."""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, replace
from typing import Any, Dict, List, Optional

import numpy as np

from .advisor import Advisor, RuleBasedAdvisor
from .derived import evaluate_expression, is_log_offset
from .profile import load_and_profile, load_data, profile_dataframe
from .schema import (
    ConstraintTier,
    DataProfile,
    ExperimentSpec,
    Turn,
    VariableDecision,
    parse_bool,
    parse_list,
)


class ExperimentCopilot:
    """Walk a user from a data path to a runnable ``ExperimentSpec``.

    The copilot is a small state machine.  Each call to :meth:`answer`
    advances one field, validates the answer against the profiled data, and
    returns the next question.  The LLM (when configured) only interprets the
    free-text answer; the state transition and validation are deterministic.
    """

    def __init__(
        self,
        advisor: Optional[Advisor] = None,
        *,
        max_latent_classes: int = 1,
        R: int = 50,
        max_iter: int = 300,
        seed: int = 0,
        output_dir: str = "metacount_results",
    ):
        self.advisor = advisor or RuleBasedAdvisor()
        self.profile: Optional[DataProfile] = None
        self.spec: Optional[ExperimentSpec] = None
        self.stage = "not_started"
        self.turns: List[Turn] = []
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_diagnosis: Optional[Dict[str, Any]] = None
        self._pending_variables: Optional[List[VariableDecision]] = None
        self._pending_constraints: Optional[List[ConstraintTier]] = None
        self._defaults = dict(
            max_latent_classes=max_latent_classes,
            R=R,
            max_iter=max_iter,
            seed=seed,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------
    def start(self, data_path: str) -> Turn:
        self.profile = load_and_profile(data_path)
        self.spec = ExperimentSpec(
            data_path=str(data_path),
            max_latent_classes=self._defaults["max_latent_classes"],
            R=self._defaults["R"],
            max_iter=self._defaults["max_iter"],
            seed=self._defaults["seed"],
            output_dir=self._defaults["output_dir"],
        )
        # Seed sensible defaults from the profile; the user can override.
        if self.profile.response_candidates:
            self.spec.response = self.profile.response_candidates[0]
        if self.profile.id_candidates:
            self.spec.id_col = self.profile.id_candidates[0]
        if self.profile.offset_candidates:
            self.spec.offset_col = self.profile.offset_candidates[0]
        self.stage = "response"
        return self._turn(
            "Which column is the count response?",
            "response",
            options=self.profile.response_candidates,
            examples=[self.spec.response] if self.spec.response else [],
            hint="I inferred this from integer, non-negative columns.",
        )

    # ------------------------------------------------------------------
    # Answer / advance
    # ------------------------------------------------------------------
    def answer(self, text: str) -> Turn:
        if self.spec is None or self.profile is None:
            raise RuntimeError("call start(data_path) first")
        if self.stage == "done":
            return self._turn("The experiment is ready. Call run().", "done", done=True)

        parsed = self.advisor.interpret_answer(
            self.stage, self._current_question(), text, self._context()
        )
        value = parsed.get("value", text) if isinstance(parsed, dict) else text

        handler = getattr(self, f"_handle_{self.stage}")
        handler(value, text)
        return self._advance()

    def _advance(self) -> Turn:
        order = ["response", "family", "id_col", "offset_col", "group_id_col"]
        if self.spec.family == "cmf":
            order += ["aadt_col"]
        order += ["variables"]
        if self.spec.family == "cmf":
            order += ["baseline_vars", "local_vars"]
        order += [
            "random_params",
            "constraints",
            "objective",
            "algo",
            "max_latent_classes",
            "max_iter",
            "R",
            "done",
        ]
        idx = order.index(self.stage)
        self.stage = order[idx + 1]
        return self._question_for_stage()

    # ------------------------------------------------------------------
    # Field handlers
    # ------------------------------------------------------------------
    def _handle_response(self, value, raw: str) -> None:
        name = str(value).strip()
        if name not in self.profile.names:
            # Allow a natural-language answer that contains the column name.
            matches = [n for n in self.profile.names if n.lower() in str(raw).lower()]
            if len(matches) == 1:
                name = matches[0]
            else:
                raise ValueError(
                    f"response {name!r} is not a column; candidates: "
                    f"{self.profile.response_candidates[:10]}"
                )
        self.spec.response = name
        col = self.profile.column(name)
        if self.spec.family == "count" and col is not None:
            if not (col.is_integer and col.is_nonnegative):
                raise ValueError(
                    f"{name!r} is not an integer, non-negative count column"
                )

    def _handle_family(self, value, raw: str) -> None:
        family = str(value).strip().lower()
        if family not in {"count", "cmf", "duration", "linear", "multivariate", "survival"}:
            raise ValueError(f"unsupported family {family!r}")
        self.spec.family = family
        if family == "cmf":
            # CMF needs an AADT column and baseline/local blocks; ask for the
            # AADT column after the optional ID/offset questions.
            self._pending_cmf = True

    def _handle_id_col(self, value, raw: str) -> None:
        self.spec.id_col = self._optional_name(value, "id_col")

    def _handle_offset_col(self, value, raw: str) -> None:
        text = str(value).strip()
        if text.lower() in {"none", "no", "n/a", "na", "skip", ""}:
            self.spec.offset_col = None
            self.spec.offset_expr = None
            return
        if text in self.profile.names:
            # An existing column is taken as the already-transformed offset.
            self.spec.offset_col = text
            self.spec.offset_expr = None
            return

        # Otherwise treat it as an exposure/offset expression.
        df = load_data(self.profile.path)
        values = evaluate_expression(text, df)
        expr = text
        if not is_log_offset(expr):
            if np.all(values > 0):
                expr = f"log({text})"
                print(
                    f"  treating {text!r} as exposure; using offset "
                    f"log({text})",
                    flush=True,
                )
            else:
                raise ValueError(
                    "expression is not logged and has non-positive values; "
                    "write an explicit log expression, e.g. "
                    "log(LENGTH*AADT*365/1e8)"
                )
        # Re-evaluate the final expression to validate it.
        evaluate_expression(expr, df)
        self.spec.offset_expr = expr
        self.spec.offset_col = "__mcr_offset__"

    def _handle_group_id_col(self, value, raw: str) -> None:
        self.spec.group_id_col = self._optional_name(value, "group_id_col")

    def _handle_aadt_col(self, value, raw: str) -> None:
        name = self._optional_name(value, "aadt_col")
        if not name:
            raise ValueError("CMF family requires an AADT column")
        col = self.profile.column(name)
        if col is not None and (col.min is None or col.min <= 0):
            raise ValueError(f"{name!r} must be strictly positive for log(AADT)")
        self.spec.cmf["aadt_col"] = name

    def _handle_baseline_vars(self, value, raw: str) -> None:
        names = [n for n in parse_list(value) if n in self.profile.names]
        if not names:
            raise ValueError("baseline_vars cannot be empty")
        self.spec.cmf["baseline_vars"] = names

    def _handle_local_vars(self, value, raw: str) -> None:
        names = [n for n in parse_list(value) if n in self.profile.names]
        if not names:
            raise ValueError("local_vars cannot be empty")
        self.spec.cmf["local_vars"] = names

    def _handle_random_params(self, value, raw: str) -> None:
        included = [v.name for v in self.spec.variables if v.include]
        local_labels = [
            f"{v}@logAADT"
            for v in (self.spec.cmf.get("local_vars", []) or [])
        ]
        allowed = list(dict.fromkeys(included + local_labels))
        text = str(raw).strip().lower()
        if text in {"", "accept", "auto", "default", "advisor"}:
            self.spec.random_params = {}
            return
        if text in {"none", "no", "fixed", "fixed only"}:
            self.spec.random_params = {}
            self.spec.constraints.append(
                ConstraintTier(
                    tier="plausibility",
                    rationale="User selected a fixed-effects-only model.",
                    rules=[{"method": "no_random", "variables": allowed}],
                )
            )
            return

        mapping: Dict[str, List[str]] = {}
        for token in parse_list(raw):
            if "=" in token:
                name, dist_text = token.split("=", 1)
                dists = [
                    d.strip()
                    for d in re.split(r"[|/]", dist_text)
                    if d.strip()
                ]
            else:
                name, dists = token, []
            name = name.strip()
            if name not in allowed:
                continue
            mapping[name] = dists or ["normal", "lognormal", "triangular"]
        if not mapping:
            raise ValueError(
                "no known included variable names; expected e.g. "
                "'CURVES=lognormal, WIDTH=normal' or 'none'"
            )
        self.spec.random_params = mapping
        rules = []
        for name, dists in mapping.items():
            rules.append(
                {
                    "method": "allow_random",
                    "variables": [name],
                    "distributions": dists,
                }
            )
        baseline = set(self.spec.cmf.get("baseline_vars", []) or [])
        local = list(self.spec.cmf.get("local_vars", []) or [])
        # Bare names belong to the upper (baseline) block; lower-level terms
        # use the explicit NAME@logAADT label.  Keep the two scopes separate.
        others_baseline = [
            n for n in baseline if n in included and n not in mapping
        ]
        others_local = [
            f"{v}@logAADT"
            for v in local
            if f"{v}@logAADT" not in mapping
        ]
        others_plain = [
            n
            for n in included
            if n not in mapping
            and n not in baseline
            and n not in local
        ]
        for group in (others_baseline, others_local, others_plain):
            if group:
                rules.append({"method": "no_random", "variables": group})
        self.spec.constraints.append(
            ConstraintTier(
                tier="plausibility",
                rationale="User-specified random-parameter scope.",
                rules=rules,
            )
        )

    def _handle_variables(self, value, raw: str) -> None:
        proposals = self._pending_variables
        if proposals is None:
            proposals = self.advisor.suggest_variables(
                self.profile, self.spec.response, self.spec.family
            )
        proposals = self._validate_variable_proposals(proposals)
        text = str(raw).strip().lower()
        if text not in {"accept", "ok", "yes", "default", "looks good"}:
            if text.startswith(("exclude", "drop", "remove")):
                drop = set(_mentioned_names(text, self.profile.names))
                if not drop:
                    raise ValueError(
                        "no known variable names found; use "
                        "'exclude NAME1, NAME2' or 'include NAME3'"
                    )
                for v in proposals:
                    if v.name in drop:
                        v.include = False
                        v.role_hint = "exclude"
                        v.reason = "excluded by user"
            elif "include" in text or "add " in text:
                add = _mentioned_names(text, self.profile.names)
                for v in proposals:
                    if v.name in add:
                        col = self.profile.column(v.name)
                        v.include = True
                        v.role_hint = "fixed"
                        v.reason = "added back by user"
                        v.allowed_roles = (
                            ("excluded", "fixed", "hetero")
                            if col is not None and col.is_binary
                            else (
                                "excluded",
                                "fixed",
                                "random_ind",
                                "random_cor",
                                "hetero",
                                "zi",
                            )
                        )
                        v.allowed_distributions = (
                            ("normal", "lognormal", "triangular")
                            if col is None or not col.is_binary
                            else ()
                        )
            else:
                names = parse_list(value if not isinstance(value, str) else raw)
                drop = {n for n in names if n in self.profile.names}
                for v in proposals:
                    if v.name in drop:
                        v.include = False
                        v.role_hint = "exclude"
                        v.reason = "excluded by user"
        # If the offset is already an exposure/log-VMT column, AADT is
        # already inside it; keeping raw AADT as a covariate double-counts
        # exposure.  CMF keeps AADT for the local block instead.
        if self.spec.family == "count" and self.spec.offset_col:
            for v in proposals:
                if "aadt" in v.name.lower():
                    v.include = False
                    v.role_hint = "exclude"
                    v.reason = "AADT is already contained in the exposure offset"
        self.spec.variables = proposals
        if self.spec.family == "cmf":
            included = [v.name for v in proposals if v.include]
            aadt = self.spec.cmf.get("aadt_col")
            binary = [
                v.name
                for v in proposals
                if v.include and (self.profile.column(v.name) or _dummy()).is_binary
            ]
            default_baseline = binary or included[:2]
            default_local = [
                n for n in included if n not in default_baseline and n != aadt
            ][:2]
            self.spec.cmf.setdefault("baseline_vars", default_baseline)
            self.spec.cmf.setdefault("local_vars", default_local or included[:2])

    def _handle_constraints(self, value, raw: str) -> None:
        if not self.spec.variables:
            raise RuntimeError("variables must be set before constraints")
        tiers = self._pending_constraints
        if tiers is None:
            tiers = self.advisor.suggest_constraints(self.profile, self.spec)
        text = str(raw).strip().lower()
        if text not in {"accept", "ok", "yes", "default", "looks good"}:
            # Apply simple natural-language edits on top of the proposal.
            tiers = self._apply_constraint_feedback(tiers, raw)
        # Keep the random-parameter scope the user chose earlier; this turn
        # is allowed to add/remove plausibility rules, not to erase it.
        user_tiers = [
            t
            for t in self.spec.constraints
            if t.rationale.startswith("User selected a fixed-effects-only")
            or t.rationale.startswith("User-specified random-parameter")
        ]
        self.spec.constraints = list(tiers) + user_tiers

    def _handle_objective(self, value, raw: str) -> None:
        objective = str(value).strip().lower()
        if objective not in {"bic", "aic", "loglik"}:
            raise ValueError("objective must be BIC, AIC or loglik")
        # The package's ExperimentBuilder.run() optimises BIC internally; keep
        # the choice for reporting / RL reward and warn if it differs.
        self.spec.objective = objective

    def _handle_algo(self, value, raw: str) -> None:
        algo = str(value).strip().lower()
        if algo not in {"sa", "ga", "hs", "de", "delphos_rl"}:
            raise ValueError("algo must be sa, ga, hs, de or delphos_rl")
        self.spec.algo = algo
        if algo == "delphos_rl" and self.spec.max_iter > 50:
            # Each RL episode is one full model fit.  A cold-start run of
            # several hundred episodes is a research experiment, not a
            # scoping run.
            self.spec.max_iter = 10

    def _handle_max_latent_classes(self, value, raw: str) -> None:
        k = int(value)
        if not 1 <= k <= 5:
            raise ValueError("max_latent_classes must be between 1 and 5")
        self.spec.max_latent_classes = k

    def _handle_max_iter(self, value, raw: str) -> None:
        n = int(value)
        if n < 1:
            raise ValueError("max_iter must be positive")
        self.spec.max_iter = n

    def _handle_R(self, value, raw: str) -> None:
        n = int(value)
        if n < 1:
            raise ValueError("R must be positive")
        self.spec.R = n

    # ------------------------------------------------------------------
    # Questions
    # ------------------------------------------------------------------
    def _offset_details(self) -> List[str]:
        lines = []
        for name in self.profile.offset_candidates:
            col = self.profile.column(name)
            if col is not None and col.min is not None:
                lines.append(
                    f"{name}: min={col.min:.3g}, max={col.max:.3g}, "
                    f"mean={col.mean:.3g}"
                )
        names = set(self.profile.names)
        if "EXPOSE" in names and "OFFSET" in names:
            lines.append(
                "EXPOSE is raw exposure; OFFSET may not be its log. "
                "Use log(EXPOSE) unless you know OFFSET is already logged."
            )
        return lines

    def _variable_table(self, proposals: List[VariableDecision]) -> List[str]:
        lines = [
            "#   variable         kind      miss  uniq  proposal   reason"
        ]
        for i, v in enumerate(proposals, start=1):
            col = self.profile.column(v.name)
            if col is None:
                kind = "?"
                miss = 0
                uniq = 0
            else:
                kind = (
                    "binary"
                    if col.is_binary
                    else ("count" if col.is_integer else "numeric")
                )
                miss = col.n_missing
                uniq = col.n_unique
            status = "INCLUDE" if v.include else "exclude"
            reason = (v.reason or "").replace("\n", " ")[:48]
            lines.append(
                f"{i:2d}  {v.name:<16} {kind:<9} {miss:>4} {uniq:>5}  "
                f"{status:<8} {v.role_hint:<10} {reason}"
            )
        return lines

    def _question_for_stage(self) -> Turn:
        if self.stage == "family":
            return self._turn(
                "What model family should the search use?",
                "family",
                options=["count", "cmf"],
                examples=["count"],
                hint=(
                    "count = heterogeneous Poisson/NB with fixed, random, "
                    "grouped or latent-class terms; cmf = hierarchical "
                    "Crash Modification Factor with baseline and log-AADT blocks."
                ),
            )
        if self.stage == "id_col":
            return self._turn(
                "Which column identifies the site / panel unit?",
                "id_col",
                options=self.profile.id_candidates,
                examples=["ID", "none"],
                hint="Type 'none' for cross-sectional data.",
            )
        if self.stage == "offset_col":
            return self._turn(
                "How should the exposure offset be defined?",
                "offset_col",
                options=self.profile.offset_candidates,
                examples=[
                    "OFFSET",
                    *(["log(EXPOSE)"] if "EXPOSE" in self.profile.names else []),
                    "log(LENGTH*AADT*365/1e8)",
                    "LENGTH*AADT*365/1e8",
                    "none",
                ],
                details=[
                    "A column name is used as-is (it must already be log scale).",
                    "An expression is evaluated row-wise.",
                    "A positive, non-logged expression (exposure) is automatically wrapped in log(...) .",
                    "Allowed: + - * / ** and log, log1p, log10, sqrt, exp, abs, minimum, maximum, clip.",
                    *self._offset_details(),
                ],
                hint=(
                    "Example: LENGTH*AADT*365/1e8 becomes "
                    "log(LENGTH*AADT*365/1e8). Use 'none' for no offset."
                ),
            )
        if self.stage == "group_id_col":
            return self._turn(
                "Is there a grouping column for grouped random effects?",
                "group_id_col",
                options=self.profile.group_candidates,
                examples=["none"],
                hint="Usually 'none' unless you want role 4 (grouped) effects.",
            )
        if self.stage == "aadt_col":
            cand = [
                n for n in self.profile.names if "aadt" in n.lower()
            ]
            return self._turn(
                "Which column is AADT (traffic volume) for the CMF local block?",
                "aadt_col",
                options=cand,
                examples=cand[:1] or ["AADT"],
                hint="The model uses log(AADT); the column must be strictly positive.",
            )
        if self.stage == "baseline_vars":
            included = [v.name for v in self.spec.variables if v.include]
            default = self.spec.cmf.get("baseline_vars", included[:3])
            return self._turn(
                "Which variables enter the upper (baseline) CMF block?",
                "baseline_vars",
                options=included,
                examples=[", ".join(default)],
                hint=(
                    "Upper-level effects are independent of AADT. A variable "
                    "may appear in both blocks; the local block uses a "
                    "separate log(AADT) interaction."
                ),
            )
        if self.stage == "local_vars":
            included = [v.name for v in self.spec.variables if v.include]
            aadt = self.spec.cmf.get("aadt_col", "AADT")
            candidates = [n for n in included if n != aadt]
            default = self.spec.cmf.get("local_vars", candidates[:2])
            return self._turn(
                "Which variables enter the lower (local) CMF block?",
                "local_vars",
                options=candidates,
                examples=[", ".join(default)],
                hint=(
                    "Lower-level effects are multiplied by log(AADT). A "
                    "variable can be in both blocks; use NAME@logAADT when "
                    "scoping random parameters for this block."
                ),
            )
        if self.stage == "variables":
            proposals = self.advisor.suggest_variables(
                self.profile, self.spec.response, self.spec.family
            )
            self._pending_variables = proposals
            included = [v.name for v in proposals if v.include]
            excluded = [v.name for v in proposals if not v.include]
            examples = ["accept"]
            if included:
                examples.append(f"exclude {included[-1]}")
            if excluded:
                examples.append(f"include {excluded[0]}")
            return self._turn(
                "Review the variable catalogue.",
                "variables",
                options=included,
                examples=examples,
                details=self._variable_table(proposals),
                hint=(
                    f"{len(included)} included / {len(excluded)} excluded. "
                    "Type 'accept', 'exclude NAME1, NAME2', or "
                    "'include NAME3'. Roles and distributions come next."
                ),
            )
        if self.stage == "random_params":
            included = [v.name for v in self.spec.variables if v.include]
            local_labels = [
                f"{v}@logAADT"
                for v in (self.spec.cmf.get("local_vars", []) or [])
            ]
            choices = list(dict.fromkeys(included + local_labels))
            examples = ["none", "accept"]
            if choices:
                examples.insert(
                    0,
                    f"{choices[0]}=lognormal, "
                    f"{choices[-1]}=normal",
                )
            hint = (
                "Use NAME=dist or NAME (defaults to normal/lognormal/"
                "triangular). 'none' forces a fixed-effects-only model; "
                "'accept' keeps the advisor's role proposals."
            )
            if self.spec.family == "cmf":
                hint += (
                    " For the lower (log-AADT) block, use the label shown "
                    "in the variable list, e.g. CURVES@logAADT."
                )
            return self._turn(
                "Which variables may be random parameters?",
                "random_params",
                options=choices,
                examples=examples,
                hint=hint,
            )
        if self.stage == "constraints":
            self._pending_constraints = self.advisor.suggest_constraints(
                self.profile, self.spec
            )
            return self._turn(
                "Here is the proposed constraint hierarchy. Accept it, or edit it.",
                "constraints",
                options=["accept"],
                examples=[
                    "accept",
                    "no random on URB, GRADEBR; allow random CURVES lognormal",
                ],
                hint=(
                    "Tiers: hard (IDs/response/offset) -> structural "
                    "(binary no-random/no-ZI) -> plausibility "
                    "(distributions) -> search (mutual exclusion / sparsity)."
                ),
            )
        if self.stage == "objective":
            return self._turn(
                "Which objective should the search and diagnosis use?",
                "objective",
                options=["bic", "aic", "loglik"],
                examples=["bic"],
                hint="ExperimentBuilder.run() ranks candidates by BIC internally.",
            )
        if self.stage == "algo":
            return self._turn(
                "Which search driver should run?",
                "algo",
                options=["sa", "ga", "hs", "de", "delphos_rl"],
                examples=["sa"],
                hint=(
                    "SA is the recommended single-objective driver; "
                    "delphos_rl runs the DeepSet-Q specification agent from "
                    "metacountregressor.copilot.delphos (one model fit per episode)."
                ),
            )
        if self.stage == "max_latent_classes":
            return self._turn(
                "Maximum latent classes?",
                "max_latent_classes",
                examples=[str(self.spec.max_latent_classes)],
                hint="1 disables latent classes; 2-3 are typical.",
            )
        if self.stage == "max_iter":
            if self.spec.algo == "delphos_rl":
                return self._turn(
                    "How many Delphos RL episodes? (one full model fit each)",
                    "max_iter",
                    examples=[str(self.spec.max_iter), "5", "10"],
                    hint=(
                        "Use 5-10 for scoping; 100+ is a long research run. "
                        "Each episode estimates one complete specification."
                    ),
                )
            return self._turn(
                "Search budget (iterations / generations)?",
                "max_iter",
                examples=[str(self.spec.max_iter)],
                hint="For SA this is the number of proposals, not unique fits.",
            )
        if self.stage == "R":
            return self._turn(
                "How many simulation draws for random parameters?",
                "R",
                examples=[str(self.spec.R)],
                hint="Use 50-200 for exploration, 500-1000 for final inference.",
            )
        return self._turn(
            "Setup complete. The validated ExperimentSpec is ready; call run().",
            "done",
            done=True,
        )

    # ------------------------------------------------------------------
    # Build / run / diagnose
    # ------------------------------------------------------------------
    def build_spec(self) -> ExperimentSpec:
        if self.spec is None:
            raise RuntimeError("call start(data_path) first")
        self._validate_spec()
        return self.spec

    def _validate_spec(self) -> None:
        if not self.spec.response:
            raise ValueError("response is required")
        if not self.spec.variables:
            raise ValueError("at least one variable must be included")
        if self.spec.max_latent_classes <= 1:
            # Membership-only terms are inert without a class-membership
            # equation; drop them rather than carrying dead columns.
            for v in self.spec.variables:
                if v.role_hint == "membership":
                    v.include = False
                    v.role_hint = "exclude"
        if self.spec.family == "cmf":
            if not self.spec.cmf.get("aadt_col"):
                # Infer a plausible AADT column.
                cand = [
                    n for n in self.profile.names if n.lower() in {"aadt", "aadt_total"}
                ]
                if not cand:
                    cand = [n for n in self.profile.names if "aadt" in n.lower()]
                if not cand:
                    raise ValueError("CMF family needs an AADT column; set spec.cmf['aadt_col']")
                self.spec.cmf["aadt_col"] = cand[0]
            aadt = self.spec.cmf["aadt_col"]
            # AADT is forced into the local block as log(AADT); keeping the
            # raw column in the ordinary variable list would duplicate it.
            for v in self.spec.variables:
                if v.name == aadt:
                    v.include = False
                    v.role_hint = "exclude"
            included = [v.name for v in self.spec.variables if v.include]
            if not self.spec.cmf.get("baseline_vars"):
                self.spec.cmf["baseline_vars"] = [
                    v.name
                    for v in self.spec.variables
                    if v.include and (self.profile.column(v.name) or _dummy()).is_binary
                ] or included[:2]
            if not self.spec.cmf.get("local_vars"):
                self.spec.cmf["local_vars"] = [
                    n for n in included if n not in self.spec.cmf["baseline_vars"]
                ][:2]
            self.spec.cmf["baseline_vars"] = [
                n for n in self.spec.cmf["baseline_vars"] if n in included
            ]
            self.spec.cmf["local_vars"] = [
                n for n in self.spec.cmf["local_vars"] if n in included
            ]
            if not self.spec.cmf["baseline_vars"] or not self.spec.cmf["local_vars"]:
                raise ValueError("CMF baseline_vars and local_vars must be non-empty")

    def run(self, *, verbose: bool = False, refit: bool = False) -> Dict[str, Any]:
        from .runner import run_search

        spec = self.build_spec()
        self.last_result = run_search(spec, verbose=verbose, refit=refit)
        return self.last_result

    def diagnose(self) -> Dict[str, Any]:
        from .runner import diagnose_result

        if self.last_result is None:
            raise RuntimeError("call run() first")
        self.last_diagnosis = diagnose_result(
            self.build_spec(), self.last_result, advisor=self.advisor
        )
        return self.last_diagnosis

    def apply_diagnosis(self) -> ExperimentSpec:
        if not self.last_diagnosis or not self.last_diagnosis.get("next_spec"):
            raise RuntimeError("call diagnose() first")
        self.spec = self.last_diagnosis["next_spec"]
        return self.spec

    def revise_constraints(self, feedback: str) -> List[ConstraintTier]:
        """Ask the advisor for a revised hierarchy given user feedback."""
        tiers = self.advisor.suggest_constraints(self.profile, self.spec)
        self.spec.constraints = self._apply_constraint_feedback(tiers, feedback)
        return self.spec.constraints

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _optional_name(self, value, field: str) -> Optional[str]:
        text = str(value).strip()
        if text.lower() in {"none", "no", "n/a", "na", "skip", ""}:
            return None
        if text not in self.profile.names:
            matches = [n for n in self.profile.names if n.lower() == text.lower()]
            if not matches:
                raise ValueError(f"{field} {text!r} is not a column")
            text = matches[0]
        return text

    def _validate_variable_proposals(
        self, proposals: List[VariableDecision]
    ) -> List[VariableDecision]:
        known = set(self.profile.names)
        out = []
        for v in proposals:
            if v.name not in known:
                continue
            if v.name == self.spec.response:
                v.include = False
                v.role_hint = "exclude"
            if v.role_hint == "offset":
                v.include = False
            out.append(v)
        if not any(v.include for v in out):
            raise ValueError("the advisor excluded every variable")
        return out

    def _apply_constraint_feedback(
        self, tiers: List[ConstraintTier], feedback: str
    ) -> List[ConstraintTier]:
        """Parse a few common natural-language constraint edits."""
        text = str(feedback)
        extra_rules: List[Dict[str, Any]] = []
        names = [v.name for v in self.spec.variables if v.include]
        # Parse clause by clause so "no random on URB; allow random CURVES"
        # does not attach CURVES to the no-random rule.
        for clause in re.split(r"[;\n.]+", text):
            low = clause.lower().strip()
            if not low:
                continue
            if "no random" in low:
                vars_ = _names_after(low, "no random", names)
                if vars_:
                    extra_rules.append({"method": "no_random", "variables": vars_})
            if "no zi" in low or "no zero" in low:
                vars_ = _names_after(low, "no zi", names)
                if vars_:
                    extra_rules.append({"method": "no_zi", "variables": vars_})
            if "force fixed" in low:
                vars_ = _names_after(low, "force fixed", names)
                if vars_:
                    extra_rules.append({"method": "force_fixed", "variables": vars_})
            if "allow random" in low:
                vars_ = _names_after(low, "allow random", names)
                dists = [
                    d for d in ("normal", "lognormal", "triangular", "uniform")
                    if d in low
                ] or None
                if vars_:
                    extra_rules.append(
                        {
                            "method": "allow_random",
                            "variables": vars_,
                            "distributions": dists,
                        }
                    )
            if "exclude" in low:
                vars_ = _names_after(low, "exclude", names)
                if vars_:
                    extra_rules.append({"method": "exclude", "variables": vars_})
        if extra_rules:
            tiers = list(tiers) + [
                ConstraintTier(
                    tier="search",
                    rationale=f"User feedback: {feedback}",
                    rules=extra_rules,
                )
            ]
        return tiers

    def _current_question(self) -> str:
        return self.turns[-1].question if self.turns else ""

    def _context(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "spec": self.spec.to_dict() if self.spec else None,
            "profile": self.profile.to_dict() if self.profile else None,
        }

    def _turn(self, question: str, field: str, **kwargs) -> Turn:
        turn = Turn(question=question, field=field, **kwargs)
        self.turns.append(turn)
        return turn


class _Dummy:
    is_binary = False


def _dummy() -> _Dummy:
    return _Dummy()


def _names_after(text: str, phrase: str, known: List[str]) -> List[str]:
    """Extract known variable names from the text following a phrase."""
    idx = text.find(phrase)
    if idx < 0:
        return []
    return _mentioned_names(text[idx + len(phrase) :], known)


def _mentioned_names(text: str, known: List[str]) -> List[str]:
    found = []
    for name in known:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(name)
    return found

