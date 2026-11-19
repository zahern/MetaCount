"""Advisors: deterministic rules or an LLM, behind one small interface.

The advisor is allowed to *propose*.  The copilot validates every proposal
against the data profile and the package's configuration objects before it is
used.  This is what keeps a hallucinated variable name or an impossible role
from ever reaching the estimator.
"""

from __future__ import annotations

import copy
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from .profile import load_data
from .schema import (
    ConstraintTier,
    DataProfile,
    ExperimentSpec,
    VariableDecision,
    parse_bool,
    parse_list,
)


class Advisor(Protocol):
    def interpret_answer(
        self, field: str, question: str, answer: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        ...

    def suggest_variables(
        self, profile: DataProfile, response: str, family: str
    ) -> List[VariableDecision]:
        ...

    def suggest_constraints(
        self, profile: DataProfile, spec: ExperimentSpec
    ) -> List[ConstraintTier]:
        ...

    def diagnose(
        self, profile: DataProfile, spec: ExperimentSpec, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------
class RuleBasedAdvisor:
    """No-API-key fallback with conservative, explainable heuristics."""

    def __init__(self, max_variables: int = 15):
        self.max_variables = int(max_variables)
        self._variable_cache: Dict[tuple, List[VariableDecision]] = {}

    def interpret_answer(
        self, field: str, question: str, answer: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        text = str(answer).strip()
        if field in {"confirm", "include"}:
            return {"value": parse_bool(text)}
        if field in {"max_latent_classes", "R", "max_iter", "seed"}:
            m = re.search(r"-?\d+", text)
            return {"value": int(m.group()) if m else None}
        if field in {"variables_exclude", "variables_include", "local_vars"}:
            return {"value": parse_list(text)}
        if field == "family":
            for choice in ("count", "cmf", "duration", "linear", "multivariate", "survival"):
                if choice in text.lower():
                    return {"value": choice}
            return {"value": "count"}
        if field == "algo":
            # Match the longest names first: "delphos_rl" contains "de".
            for choice in ("delphos_rl", "sa", "ga", "hs", "de"):
                if choice in text.lower():
                    return {"value": choice}
            return {"value": "sa"}
        if field == "objective":
            for choice in ("bic", "aic", "loglik"):
                if choice in text.lower():
                    return {"value": choice}
            return {"value": "bic"}
        if field in {"response", "id_col", "offset_col", "group_id_col", "aadt_col"}:
            return {"value": text}
        return {"value": text}

    def suggest_variables(
        self, profile: DataProfile, response: str, family: str
    ) -> List[VariableDecision]:
        key = (profile.path, response, family, self.max_variables)
        if key in self._variable_cache:
            return copy.deepcopy(self._variable_cache[key])

        try:
            df = load_data(profile.path)
        except Exception:
            df = None

        excluded = {response}
        for name in (
            profile.id_candidates
            + profile.offset_candidates
            + profile.group_candidates
        ):
            excluded.add(name)

        ranked: List[tuple] = []
        decisions: List[VariableDecision] = []
        for col in profile.columns:
            name = col.name
            low = name.lower()
            if name in excluded:
                decisions.append(_drop(name, "identifier / response / offset / grouping key"))
                continue
            if col.n_unique <= 1:
                decisions.append(_drop(name, "constant column"))
                continue
            if not _looks_numeric(col):
                decisions.append(_drop(name, "non-numeric or free-text column"))
                continue
            if any(tok in low for tok in ("_id", "id_", "label", "name", "date", "year")):
                decisions.append(_drop(name, "identifier / label / date column"))
                continue
            if any(tok in low for tok in ("fc", "functional", "class", "type")):
                decisions.append(
                    VariableDecision(
                        name=name,
                        include=True,
                        role_hint="membership",
                        reason="categorical code: class-membership equation, not a linear slope",
                        allowed_roles=("excluded", "membership"),
                    )
                )
                continue
            score = _univariate_score(df, name, response)
            score += _domain_bonus(low)
            ranked.append((score, col))

        ranked.sort(key=lambda item: item[0], reverse=True)
        keep = {col.name for _, col in ranked[: self.max_variables]}
        for score, col in ranked:
            name = col.name
            low = name.lower()
            if name not in keep:
                decisions.append(
                    _drop(
                        name,
                        f"pruned by univariate screen (score={score:.3f}); ask to add it back",
                    )
                )
                continue
            if col.is_binary:
                decisions.append(
                    VariableDecision(
                        name=name,
                        include=True,
                        role_hint="fixed",
                        reason=f"binary indicator (screen score={score:.3f}); fixed effect only",
                        allowed_roles=("excluded", "fixed", "hetero"),
                    )
                )
                continue
            geometry = any(
                tok in low
                for tok in ("curve", "width", "grade", "speed", "lane", "shoulder", "slope")
            )
            decisions.append(
                VariableDecision(
                    name=name,
                    include=True,
                    role_hint="fixed",
                    reason=(
                        f"geometry/design covariate (screen score={score:.3f})"
                        if geometry
                        else f"numeric covariate (screen score={score:.3f})"
                    ),
                    allowed_roles=(
                        ("excluded", "fixed", "random_ind", "random_cor", "hetero")
                        if geometry
                        else (
                            "excluded",
                            "fixed",
                            "random_ind",
                            "random_cor",
                            "hetero",
                            "zi",
                        )
                    ),
                    allowed_distributions=("normal", "lognormal", "triangular"),
                )
            )

        # Preserve original column order for readable turns.
        order = {c.name: i for i, c in enumerate(profile.columns)}
        decisions.sort(key=lambda v: order.get(v.name, 10**9))
        self._variable_cache[key] = copy.deepcopy(decisions)
        return decisions

    def suggest_constraints(
        self, profile: DataProfile, spec: ExperimentSpec
    ) -> List[ConstraintTier]:
        binary = [
            v.name
            for v in spec.variables
            if v.include and (profile.column(v.name) or _dummy()).is_binary
        ]
        geometry = [
            v.name
            for v in spec.variables
            if v.include
            and any(
                tok in v.name.lower()
                for tok in ("curve", "width", "grade", "speed", "lane", "shoulder")
            )
        ]
        continuous = [
            v.name
            for v in spec.variables
            if v.include and v.name not in binary and v.name not in geometry
        ]
        tiers = [
            ConstraintTier(
                tier="hard",
                rationale="Structural identifiers and the response never enter the search.",
                rules=[
                    {
                        "method": "exclude",
                        "variables": list(
                            dict.fromkeys(
                                [spec.response]
                                + ([spec.id_col] if spec.id_col else [])
                                + ([spec.offset_col] if spec.offset_col else [])
                                + ([spec.group_id_col] if spec.group_id_col else [])
                            )
                        ),
                    }
                ],
            ),
            ConstraintTier(
                tier="structural",
                rationale="Binary indicators and geometry are not plausible random/ZI terms.",
                rules=[
                    {"method": "no_random", "variables": binary},
                    {"method": "no_zi", "variables": list(dict.fromkeys(binary + geometry))},
                ],
            ),
            ConstraintTier(
                tier="plausibility",
                rationale="Continuous covariates may carry random taste variation; one-signed effects get lognormal.",
                rules=[
                    {
                        "method": "allow_random",
                        "variables": continuous,
                        "distributions": ["normal", "lognormal", "triangular"],
                    }
                ],
            ),
        ]
        # Mutual-exclusion groups for near-collinear numeric pairs.
        try:
            df = load_data(spec.data_path)
            numeric = [
                c.name
                for c in profile.columns
                if c.name in spec.included_variables
                and pd.api.types.is_numeric_dtype(df[c.name])
            ]
            if len(numeric) >= 2:
                corr = df[numeric].corr().abs()
                groups = []
                seen = set()
                for i, a in enumerate(numeric):
                    for b in numeric[i + 1 :]:
                        if corr.loc[a, b] > 0.95 and (a, b) not in seen:
                            groups.append([a, b])
                            seen.add((a, b))
                if groups:
                    tiers.append(
                        ConstraintTier(
                            tier="search",
                            rationale="Near-collinear pairs are mutually exclusive to avoid unstable coefficients.",
                            rules=[{"method": "mutual_exclusion", "groups": groups}],
                        )
                    )
        except Exception:
            pass
        return tiers

    def diagnose(
        self, profile: DataProfile, spec: ExperimentSpec, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        issues: List[Dict[str, Any]] = []
        best = result.get("best_score")
        if best is None or not np.isfinite(float(best)):
            issues.append(
                {
                    "severity": "error",
                    "code": "no_convergence",
                    "message": "No finite BIC was found. Relax the constraint hierarchy or increase the budget.",
                    "proposed_patch": {"method": "loosen_roles"},
                }
            )
        n_active = result.get("n_active")
        n_obs = result.get("n_obs") or profile.n_rows
        if n_active is not None and n_obs and n_active > max(5, n_obs // 10):
            issues.append(
                {
                    "severity": "warning",
                    "code": "overparameterised",
                    "message": (
                        f"{n_active} active terms for {n_obs} observations; "
                        "consider pruning or stronger sparsity."
                    ),
                    "proposed_patch": {"method": "prune_weak_terms"},
                }
            )
        return {"summary": "Deterministic diagnostic pass.", "issues": issues}


class _Dummy:
    is_binary = False


def _dummy() -> _Dummy:
    return _Dummy()


def _drop(name: str, reason: str) -> VariableDecision:
    return VariableDecision(
        name=name, include=False, role_hint="exclude", reason=reason
    )


def _looks_numeric(col) -> bool:
    dtype = str(col.dtype).lower()
    return any(tok in dtype for tok in ("int", "float", "uint", "bool"))


def _univariate_score(df: Optional[pd.DataFrame], name: str, response: str) -> float:
    """Absolute Spearman correlation with the response, in [0, 1]."""
    if df is None or name not in df.columns or response not in df.columns:
        return 0.0
    try:
        pair = df[[name, response]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 5 or pair[name].nunique() <= 1 or pair[response].nunique() <= 1:
            return 0.0
        corr = pair.corr(method="spearman").iloc[0, 1]
        return float(abs(corr)) if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0


def _domain_bonus(name: str) -> float:
    # Small, transparent prior for road-safety/CMF covariates.  It only
    # breaks ties in the univariate screen; the user can override every
    # inclusion decision in the next turn.
    if name in {"aadt", "length", "offset"}:
        return 0.25
    if any(tok in name for tok in ("speed", "urb", "curve", "access", "grade", "width", "avepre", "slope")):
        return 0.12
    if any(tok in name for tok in ("lane", "friction", "shoulder", "median")):
        return 0.06
    return 0.0


# ---------------------------------------------------------------------------
# LLM advisor (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
@dataclass
class OpenAICompatibleAdvisor:
    """Minimal OpenAI-compatible chat client using only the standard library.

    Works with OpenAI, Azure OpenAI-compatible gateways, vLLM, LM Studio,
    Ollama's OpenAI endpoint, etc.  Configure with::

        METACOUNT_LLM_BASE_URL=http://localhost:11434/v1
        METACOUNT_LLM_API_KEY=not-needed
        METACOUNT_LLM_MODEL=qwen2.5:14b
    """

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: int = 60
    temperature: float = 0.1

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or os.environ.get("METACOUNT_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = self.api_key or os.environ.get("METACOUNT_LLM_API_KEY", "")
        self.model = self.model or os.environ.get("METACOUNT_LLM_MODEL", "")
        if not self.base_url or not self.model:
            raise ValueError(
                "set METACOUNT_LLM_BASE_URL and METACOUNT_LLM_MODEL "
                "(or pass base_url/model)"
            )

    def _chat(self, messages: List[Dict[str, str]], json_mode: bool = True) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or 'not-needed'}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Some local servers reject response_format; retry plain.
            if json_mode and exc.code in (400, 422):
                return self._chat(messages, json_mode=False)
            raise
        return body["choices"][0]["message"]["content"]

    def _json(self, messages: List[Dict[str, str]]) -> Any:
        text = self._chat(messages)
        return _parse_json(text)

    def interpret_answer(
        self, field: str, question: str, answer: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt = (
            "You are configuring a statistical count-model experiment. "
            "Return only JSON with keys value, confidence, message. "
            f"Field: {field}. Question: {question}. User answer: {answer!r}. "
            f"Context: {json.dumps(context, default=str)[:4000]}"
        )
        try:
            out = self._json(
                [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ]
            )
            return out if isinstance(out, dict) else {"value": answer}
        except Exception:
            return RuleBasedAdvisor().interpret_answer(field, question, answer, context)

    def suggest_variables(
        self, profile: DataProfile, response: str, family: str
    ) -> List[VariableDecision]:
        prompt = (
            "Propose which columns should enter a count-model search. "
            "Return JSON {\"variables\":[{\"name\",\"include\",\"role_hint\","
            "\"reason\",\"allowed_roles\",\"allowed_distributions\"}]}. "
            "role_hint is one of fixed/random/exclude/zi/membership/offset. "
            "allowed_roles use excluded/fixed/random_ind/random_cor/grouped/"
            "hetero/zi/membership/membership_fixed. "
            f"Response column: {response}. Family: {family}. "
            f"Data profile: {json.dumps(profile.to_dict(), default=str)[:12000]}"
        )
        try:
            out = self._json(
                [
                    {"role": "system", "content": "You are a conservative road-safety statistician. Return strict JSON."},
                    {"role": "user", "content": prompt},
                ]
            )
            raw = out.get("variables", []) if isinstance(out, dict) else []
            decisions = []
            for item in raw:
                name = str(item.get("name", ""))
                if name not in profile.names:
                    continue
                decisions.append(
                    VariableDecision(
                        name=name,
                        include=bool(item.get("include", True)),
                        role_hint=str(item.get("role_hint", "fixed")),
                        reason=str(item.get("reason", "LLM proposal")),
                        allowed_roles=tuple(item.get("allowed_roles", ())),
                        allowed_distributions=tuple(item.get("allowed_distributions", ())),
                    )
                )
            return decisions or RuleBasedAdvisor().suggest_variables(profile, response, family)
        except Exception:
            return RuleBasedAdvisor().suggest_variables(profile, response, family)

    def suggest_constraints(
        self, profile: DataProfile, spec: ExperimentSpec
    ) -> List[ConstraintTier]:
        prompt = (
            "Build a tiered constraint hierarchy for a count-model search. "
            "Return JSON {\"tiers\":[{\"tier\",\"rationale\",\"rules\":[...]}]} "
            "where rules use methods force_fixed/force_include/no_zi/no_random/"
            "allow_random/membership_only/outcome_only/exclude/mutual_exclusion. "
            f"Spec: {json.dumps(spec.to_dict(), default=str)[:12000]}"
        )
        try:
            out = self._json(
                [
                    {"role": "system", "content": "Return strict JSON. Never invent variable names."},
                    {"role": "user", "content": prompt},
                ]
            )
            tiers = []
            for item in (out.get("tiers", []) if isinstance(out, dict) else []):
                tiers.append(
                    ConstraintTier(
                        tier=str(item.get("tier", "search")),
                        rationale=str(item.get("rationale", "")),
                        rules=list(item.get("rules", [])),
                    )
                )
            return tiers or RuleBasedAdvisor().suggest_constraints(profile, spec)
        except Exception:
            return RuleBasedAdvisor().suggest_constraints(profile, spec)

    def diagnose(
        self, profile: DataProfile, spec: ExperimentSpec, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt = (
            "Diagnose this fitted count-model search. Identify convergence, "
            "sign, identifiability, over-parameterisation and constraint issues. "
            "Return JSON {\"summary\",\"issues\":[{\"severity\",\"code\","
            "\"message\",\"evidence\",\"proposed_patch\"}]}. "
            f"Spec: {json.dumps(spec.to_dict(), default=str)[:8000]}. "
            f"Result: {json.dumps(result, default=str)[:12000]}"
        )
        try:
            out = self._json(
                [
                    {"role": "system", "content": "Return strict JSON. Propose only valid ModelConstraints methods."},
                    {"role": "user", "content": prompt},
                ]
            )
            return out if isinstance(out, dict) else {"summary": "", "issues": []}
        except Exception:
            return RuleBasedAdvisor().diagnose(profile, spec, result)


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------
class ScriptedAdvisor:
    """Deterministic advisor used by tests and offline demos."""

    def __init__(
        self,
        variables: Optional[List[VariableDecision]] = None,
        constraints: Optional[List[ConstraintTier]] = None,
        answers: Optional[Dict[str, Any]] = None,
        diagnosis: Optional[Dict[str, Any]] = None,
    ):
        self.variables = variables
        self.constraints = constraints
        self.answers = answers or {}
        self.diagnosis = diagnosis

    def interpret_answer(
        self, field: str, question: str, answer: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if field in self.answers:
            return {"value": self.answers[field]}
        return RuleBasedAdvisor().interpret_answer(field, question, answer, context)

    def suggest_variables(
        self, profile: DataProfile, response: str, family: str
    ) -> List[VariableDecision]:
        return self.variables or RuleBasedAdvisor().suggest_variables(
            profile, response, family
        )

    def suggest_constraints(
        self, profile: DataProfile, spec: ExperimentSpec
    ) -> List[ConstraintTier]:
        return self.constraints or RuleBasedAdvisor().suggest_constraints(profile, spec)

    def diagnose(
        self, profile: DataProfile, spec: ExperimentSpec, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.diagnosis or RuleBasedAdvisor().diagnose(profile, spec, result)


def _parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min(
            (i for i in (text.find("{"), text.find("[")) if i >= 0),
            default=-1,
        )
        if start >= 0:
            end = max(text.rfind("}"), text.rfind("]"))
            if end > start:
                return json.loads(text[start : end + 1])
        raise
