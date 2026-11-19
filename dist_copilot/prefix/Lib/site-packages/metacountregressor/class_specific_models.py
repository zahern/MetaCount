"""
Class-Specific Models and Variables for Latent-Class Analysis

Supports:
  1. Different models per class (e.g., Poisson vs NB2)
  2. Class-specific variable sets (e.g., SPEED only in class 1)
  3. Perturbations that swap variables in/out per class

HOW TO USE
──────────
    # In your spec dict or when building manually:
    spec_dict = {
        ...
        "fixed_terms": ["URB", "SPEED"],
        "class_models": ("poisson", "nb"),  # class 0 Poisson, class 1 NB
        # Optional: specify which variables appear in which classes
        "class_variable_masks": (
            frozenset(["URB", "SPEED"]),      # Class 0 variables
            frozenset(["URB"]),               # Class 1 variables (no SPEED)
        ),
    }

    # Then call:
    data, spec = build_model_from_manual_spec(df, spec_dict, ...)

The parameter vector will now have:
    - theta_0: parameters for class 0's outcome model (includes SPEED)
    - theta_1: parameters for class 1's outcome model (excludes SPEED)
    - gamma: class membership parameters
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, replace
from typing import Optional, Tuple, FrozenSet, Dict, List, Any
from functools import reduce
import operator


# ═══════════════════════════════════════════════════════════════════════
# Helper: Class-specific variable masks
# ═══════════════════════════════════════════════════════════════════════

def compute_class_variable_masks(
    fixed_terms: List[str],
    rdm_ind_terms: List[str],
    rdm_cor_terms: List[str],
    grouped_terms: List[str],
    hetro_terms: List[str],
    zi_terms: List[str],
    latent_classes: int,
    class_specific_rules: Optional[Dict[str, List[int]]] = None,
) -> Tuple[Tuple[FrozenSet[str], ...], Dict[str, List[int]]]:
    """
    Compute which variables appear in which class.

    Returns:
      (class_variable_masks, effective_rules)

    where:
      - class_variable_masks: tuple of frozensets, one per class
      - effective_rules: {var_name: [class_indices_where_it_appears], ...}

    By default, all variables appear in all classes.
    Override with class_specific_rules = {"SPEED": [0], "URB": [0, 1]}
    """
    if latent_classes == 1:
        # Single class: no class-specific masking needed
        all_vars = set(fixed_terms + rdm_ind_terms + rdm_cor_terms + grouped_terms + hetro_terms + zi_terms)
        return (frozenset(all_vars),), {v: [0] for v in all_vars}

    all_vars = set(fixed_terms + rdm_ind_terms + rdm_cor_terms + grouped_terms + hetro_terms + zi_terms)

    # Default: all variables in all classes
    if class_specific_rules is None:
        class_specific_rules = {v: list(range(latent_classes)) for v in all_vars}

    # Ensure all variables are in the rules dict
    for v in all_vars:
        if v not in class_specific_rules:
            class_specific_rules[v] = list(range(latent_classes))

    # Build masks (one frozenset per class)
    class_masks = []
    for c in range(latent_classes):
        vars_in_c = frozenset(v for v, classes in class_specific_rules.items() if c in classes)
        class_masks.append(vars_in_c)

    return tuple(class_masks), class_specific_rules


def apply_class_variable_masks(
    spec_dict: Dict[str, Any],
    class_variable_masks: Tuple[FrozenSet[str], ...],
) -> List[Dict[str, Any]]:
    """
    Split a global spec_dict into per-class specs, respecting class_variable_masks.

    Returns a list of spec dicts, one per class, with variables filtered accordingly.

    Example:
      Input spec_dict has fixed_terms=["URB", "SPEED"]
      masks = ({"URB", "SPEED"}, {"URB"})
      Output[0] = {fixed_terms: ["URB", "SPEED"], ...}
      Output[1] = {fixed_terms: ["URB"], ...}
    """
    if not class_variable_masks:
        return [spec_dict]

    C = len(class_variable_masks)
    per_class_specs = []

    for c in range(C):
        mask_c = class_variable_masks[c]
        spec_c = {}

        # Copy scalar keys
        for k in ["dispersion", "latent_classes", "min_class_proportion"]:
            if k in spec_dict:
                spec_c[k] = spec_dict[k]

        # Filter list keys by mask
        for key in ["fixed_terms", "rdm_terms", "rdm_cor_terms", "grouped_terms",
                    "hetro_in_means", "zi_terms", "membership_terms"]:
            terms = spec_dict.get(key, [])

            # For distributed terms (rdm_terms, etc.), extract variable name before ":"
            filtered = []
            for term in terms:
                var_name = term.split(":")[0] if ":" in term else term
                if var_name in mask_c:
                    filtered.append(term)

            spec_c[key] = filtered

        per_class_specs.append(spec_c)

    return per_class_specs


# ═══════════════════════════════════════════════════════════════════════
# Perturbation strategies for class-specific variables
# ═══════════════════════════════════════════════════════════════════════

class ClassSpecificPerturbation:
    """
    Proposes per-class variable inclusion/exclusion swaps during search.

    Strategies:
      1. "swap_one_var": remove one variable from one class, add to another
      2. "toggle_class": toggle a variable on/off in a specific class
      3. "conservative": only allow additions/removals that minimize param changes
    """

    @staticmethod
    def swap_one_variable(
        class_masks: Tuple[FrozenSet[str], ...],
        all_variables: List[str],
        rng: np.random.Generator,
    ) -> Tuple[FrozenSet[str], ...]:
        """
        Remove a variable from one class and add it to another (or exclude it).
        """
        C = len(class_masks)
        new_masks = list(class_masks)

        # Pick a class with at least 2 variables
        classes_with_vars = [c for c in range(C) if len(new_masks[c]) > 1]
        if not classes_with_vars:
            return class_masks  # Can't remove from any class

        # Remove from one class
        remove_from = rng.choice(classes_with_vars)
        var_to_remove = rng.choice(list(new_masks[remove_from]))
        new_masks[remove_from] = new_masks[remove_from] - {var_to_remove}

        # Try to add to another class (or leave out entirely)
        if rng.random() < 0.6 and C > 1:  # 60% chance to add to another class
            add_to = rng.choice([c for c in range(C) if c != remove_from])
            new_masks[add_to] = new_masks[add_to] | {var_to_remove}

        return tuple(new_masks)

    @staticmethod
    def toggle_variable_in_class(
        class_masks: Tuple[FrozenSet[str], ...],
        all_variables: List[str],
        rng: np.random.Generator,
    ) -> Tuple[FrozenSet[str], ...]:
        """
        Toggle one variable on/off in a random class.
        """
        C = len(class_masks)
        class_c = rng.integers(0, C)
        new_masks = list(class_masks)

        # Pick a variable to toggle
        var_to_toggle = rng.choice(all_variables)

        if var_to_toggle in new_masks[class_c]:
            # Remove it (if class still has variables)
            if len(new_masks[class_c]) > 1:
                new_masks[class_c] = new_masks[class_c] - {var_to_toggle}
        else:
            # Add it
            new_masks[class_c] = new_masks[class_c] | {var_to_toggle}

        return tuple(new_masks)

    @staticmethod
    def perturb_masks(
        class_masks: Tuple[FrozenSet[str], ...],
        all_variables: List[str],
        strategy: str = "swap_one_var",
        seed: int = 0,
    ) -> Tuple[FrozenSet[str], ...]:
        """
        Apply a perturbation strategy to class variable masks.

        Args:
          class_masks: current tuple of frozensets (one per class)
          all_variables: list of available variables
          strategy: "swap_one_var", "toggle", or "balanced"
          seed: random seed

        Returns:
          Perturbed class_masks (tuple of frozensets)
        """
        rng = np.random.default_rng(seed)

        if strategy == "swap_one_var":
            return ClassSpecificPerturbation.swap_one_variable(class_masks, all_variables, rng)
        elif strategy == "toggle":
            return ClassSpecificPerturbation.toggle_variable_in_class(class_masks, all_variables, rng)
        elif strategy == "balanced":
            # 50-50 between swap and toggle
            if rng.random() < 0.5:
                return ClassSpecificPerturbation.swap_one_variable(class_masks, all_variables, rng)
            else:
                return ClassSpecificPerturbation.toggle_variable_in_class(class_masks, all_variables, rng)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


# ═══════════════════════════════════════════════════════════════════════
# Integration with StructureEvaluatorLC
# ═══════════════════════════════════════════════════════════════════════

def extend_decision_vector_for_class_masks(
    decision: np.ndarray,
    class_variable_masks: Tuple[FrozenSet[str], ...],
    all_variables: List[str],
    encoding: str = "binary",
) -> np.ndarray:
    """
    Encode class variable masks into the decision vector.

    Args:
      decision: existing decision vector (roles, dists, disp, lc_code)
      class_variable_masks: per-class variable membership
      all_variables: all available variables
      encoding: "binary" = one bit per (class, variable) pair

    Returns:
      Extended decision vector
    """
    if encoding == "binary":
        # For each (class, variable) pair, 1 = include, 0 = exclude
        C = len(class_variable_masks)
        D = len(all_variables)

        mask_bits = []
        for c in range(C):
            for v in all_variables:
                bit = 1 if v in class_variable_masks[c] else 0
                mask_bits.append(bit)

        return np.concatenate([decision, np.array(mask_bits, dtype=float)])
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def extract_class_masks_from_decision(
    decision: np.ndarray,
    all_variables: List[str],
    base_dim: int,
    latent_classes: int,
) -> Tuple[FrozenSet[str], ...]:
    """
    Extract class variable masks from the extended decision vector.

    Args:
      decision: full decision vector (including mask bits)
      all_variables: all available variables
      base_dim: dimension of base decision vector (before masks)
      latent_classes: number of latent classes

    Returns:
      class_variable_masks: tuple of frozensets
    """
    if len(decision) <= base_dim:
        # No mask encoding; use all variables for all classes
        return tuple(frozenset(all_variables) for _ in range(latent_classes))

    mask_bits = decision[base_dim:]
    D = len(all_variables)
    C = latent_classes

    class_masks = []
    idx = 0
    for c in range(C):
        vars_in_c = frozenset(
            all_variables[v]
            for v in range(D)
            if int(mask_bits[idx + c * D + v]) == 1
        )
        class_masks.append(vars_in_c)

    return tuple(class_masks)
