"""Genetic algorithm over count-model structures.

The search space is the same integer decision vector the other
metaheuristics in this package optimise::

    [ roles(D) | dists(D) | dispersion_bit | lc_code? | class_masks? ]

where

``roles(D)``      one structural role code per candidate variable
                  (0 excluded, 1 fixed, 2 random independent, 3 random
                  correlated, 4 grouped, 5 heterogeneity in means, 6 zero
                  inflation, 7 membership only, 8 membership + fixed,
                  9 heterogeneity in variances)
``dists(D)``      random-parameter distribution per variable
                  (0 normal, 1 lognormal, 2 triangular, 3 uniform)
``dispersion_bit``  0 = Poisson, 1 = negative binomial
``lc_code``       latent-class count (only when ``max_latent_classes > 1``)
``class_masks``   per-variable latent-class membership masks (only with LC)

Every individual is scored by ``evaluator.fitness(decision)``, which fits
the actual count model and returns its BIC, so the search is entirely
statistical: the winner is a concrete combination of variables, each of
which carries its own estimated coefficient once refitted.  No permutation
importance, no impurity ranking -- structure is chosen by fit criterion.

Fitness is minimised (BIC).  Selection, crossover and mutation all operate
on integer genes, and every operator repairs offspring against the same
feasibility rules the simulated-annealing solver uses (role 5 requires a
role 2/3 partner, role 9 requires a role 2/3/4 partner, and mutual-exclusion
groups keep at most one active variable), so offspring handed to
``evaluator.fitness`` are always buildable.

Entry point: :class:`MultiStartGA`, which mirrors
:class:`~Solvers_METAJAX.MultiStartSA` so
``ExperimentBuilder.run(algo="ga", ...)`` can drop it in unchanged.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Prior mass used when seeding a role gene.  Mirrors Solvers_METAJAX's
# ROLE_PROBS so a GA started from scratch explores a comparable region of
# the space to the SA it sits beside (exclusion is the most likely single
# role, active structural roles are the minority).
ROLE_PROBS = np.array(
    [
        0.35,  # 0 excluded
        0.12,  # 1 fixed
        0.15,  # 2 random independent
        0.14,  # 3 random correlated
        0.00,  # 4 grouped
        0.05,  # 5 heterogeneity in means
        0.05,  # 6 zero inflation
        0.05,  # 7 membership only
        0.05,  # 8 membership + fixed
        0.04,  # 9 heterogeneity in variances
    ]
)
ROLE_PROBS = ROLE_PROBS / ROLE_PROBS.sum()

#: Fitness assigned to an individual the evaluator cannot score.  Large
#: enough to lose every tournament, small enough to survive arithmetic.
_BAD_FITNESS = 1e13

#: Rejection threshold shared with the SA solver: a score at or beyond this
#: means the structure failed to fit, not that it is genuinely terrible.
_MAX_ALLOWED_FITNESS = 1e12


def _scalar_score(score: Any) -> float:
    """Reduce an evaluator fitness to one comparable float.

    The evaluators return a scalar for single-objective families and a
    vector for the multi-objective (BIC, complexity) fronts.  This GA is
    single-objective, so a vector is collapsed to its first column, which
    is BIC in every front the package builds.
    """
    arr = np.asarray(score, dtype=float).ravel()
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def is_valid_fitness(score: Any, max_allowed: float = _MAX_ALLOWED_FITNESS) -> bool:
    """Mirror of the SA feasibility test, applied to a raw evaluator score."""
    arr = np.asarray(score, dtype=float).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return False
    if np.any(arr < 0):
        return False
    if np.any(arr >= max_allowed):
        return False
    return True


class GeneticAlgorithmSearch:
    """Single-population GA over model-structure decision vectors.

    Parameters
    ----------
    evaluator :
        Object exposing ``vars``, ``allowed_roles``, ``build_spec`` and
        ``fitness(decision)`` (lower is better).  The
        :class:`~metacountregressor.StructureEvaluatorLC` built by
        ``ExperimentBuilder.build_count_evaluator`` satisfies this.
    dimension :
        Length of the decision vector, i.e. ``2*D + 1``, or ``3*D + 2``
        when latent classes are enabled.
    population_size :
        Number of individuals kept per generation.
    max_iter :
        Generations to run.
    crossover_rate :
        Probability a pair of parents produces offspring at all.
    mutation_rate :
        Per-gene probability of mutation.
    elite_size :
        Best individuals copied unchanged into the next generation.  Kept
        >= 1 so the incumbent fitness is monotone across generations.
    tournament_size :
        Individuals sampled per tournament-selection draw.
    seed :
        Base RNG seed for this run.
    min_active :
        Fewest non-zero role genes an initial individual may carry.
    max_time :
        Optional wall-clock budget in seconds.
    on_iteration :
        Optional ``f(gen, best_decision, best_score, elapsed)`` hook fired
        whenever the incumbent improves, for checkpoint writers.
    cache_clear_every :
        Call ``jax.clear_caches()`` every N generations.  Each new
        structure compiles fresh XLA executables, so long searches grow
        RSS without this.
    init_solutions :
        Optional list of known-good decision vectors (e.g. an incumbent
        restored from a checkpoint) to seed the initial population with.
    """

    def __init__(
        self,
        evaluator,
        dimension: int,
        population_size: int = 60,
        max_iter: int = 200,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.15,
        elite_size: int = 4,
        tournament_size: int = 3,
        seed: int = 0,
        min_active: int = 2,
        max_time: Optional[float] = None,
        on_iteration: Optional[Callable[..., None]] = None,
        cache_clear_every: int = 0,
        init_solutions: Optional[List[np.ndarray]] = None,
    ) -> None:
        self.evaluator = evaluator
        self.dim = int(dimension)
        self.dim_core = len(evaluator.vars)
        if self.dim <= self.dim_core:
            raise ValueError(
                f"dimension ({self.dim}) must exceed the variable count "
                f"({self.dim_core}); the decision vector carries role, "
                f"distribution and dispersion genes after the roles."
            )

        self.population_size = max(int(population_size), 4)
        self.max_iter = int(max_iter)
        self.crossover_rate = float(np.clip(crossover_rate, 0.0, 1.0))
        self.mutation_rate = float(np.clip(mutation_rate, 0.0, 1.0))
        # Never let elitism plus mutation-rate bookkeeping produce an empty
        # or oversized gene pool.
        self.elite_size = int(np.clip(elite_size, 1, self.population_size - 1))
        self.tournament_size = max(int(tournament_size), 2)
        self.min_active = max(int(min_active), 1)
        self.max_time = None if max_time is None else float(max_time)
        self.on_iteration = on_iteration if callable(on_iteration) else None
        self.cache_clear_every = int(cache_clear_every or 0)

        # Warm-start incumbents, repaired to the feasible region up front so
        # seeding cannot inject an unbuildable structure.
        self.init_solutions: List[np.ndarray] = []
        for cand in init_solutions or []:
            arr = np.rint(np.asarray(cand, dtype=float)).reshape(-1).astype(int)
            if arr.size == self.dim:
                self.init_solutions.append(self.repair(arr))

        self.max_latent_classes = int(getattr(evaluator, "max_latent_classes", 1) or 1)
        self.mutual_exclusion = getattr(evaluator, "mutual_exclusion", None)
        self._rng = np.random.default_rng(int(seed))

        self.runtime: Optional[float] = None
        self.convergence: List[float] = []
        self.search_stats: List[Dict[str, Any]] = []
        self.archive: List[np.ndarray] = []
        self.archive_scores: List[float] = []
        self.best_solution: Optional[np.ndarray] = None
        self.best_score: float = float("inf")
        self._cache: Dict[bytes, float] = {}

    # ------------------------------------------------------------------
    # gene helpers
    # ------------------------------------------------------------------
    def _role_slice(self) -> slice:
        return slice(0, self.dim_core)

    def _dist_slice(self) -> slice:
        return slice(self.dim_core, 2 * self.dim_core)

    def _dispersion_index(self) -> Optional[int]:
        idx = 2 * self.dim_core
        return idx if self.dim > idx else None

    def _lc_index(self) -> Optional[int]:
        idx = 2 * self.dim_core + 1
        return idx if self.max_latent_classes > 1 and self.dim > idx else None

    def _mask_indices(self) -> range:
        start = 2 * self.dim_core + 2
        return range(start, min(3 * self.dim_core + 2, self.dim))

    def _sample_role(self, var_index: int, force_active: bool = False) -> int:
        """Draw one role code from the variable's allowed set."""
        var_name = self.evaluator.vars[var_index]
        allowed = list(self.evaluator.allowed_roles.get(var_name, [0, 1, 2, 3, 5, 6]))

        banned = getattr(self.evaluator, "_banned_variables", set())
        forced = getattr(self.evaluator, "_force_included_vars", set())
        if var_name in banned and var_name not in forced:
            return 0

        if force_active:
            allowed = [r for r in allowed if r != 0]
        if not allowed:
            return 0
        if len(allowed) == 1:
            return int(allowed[0])

        probs = ROLE_PROBS[np.asarray(allowed, dtype=int)]
        total = probs.sum()
        if total <= 0:
            # Every allowed role carries zero prior mass (e.g. role 4):
            # fall back to a uniform draw rather than a biased one.
            return int(self._rng.choice(allowed))
        return int(self._rng.choice(allowed, p=probs / total))

    def _random_individual(self, min_active: Optional[int] = None) -> np.ndarray:
        """Sample one feasible individual with at least ``min_active`` roles."""
        target = int(min_active if min_active is not None else self.min_active)
        target = int(np.clip(target, 1, self.dim_core))
        individual = np.zeros(self.dim, dtype=int)

        # Activate a random number of role genes, then draw each from the
        # variable's allowed set.  A few spare draws cover the case where
        # a variable's only allowed active role is absent from ROLE_PROBS.
        n_active = int(self._rng.integers(target, self.dim_core + 1))
        chosen = self._rng.choice(self.dim_core, size=n_active, replace=False)
        for j in chosen:
            individual[int(j)] = self._sample_role(int(j), force_active=True)

        for j in range(self.dim_core, 2 * self.dim_core):
            individual[j] = int(self._rng.integers(0, 4))

        disp = self._dispersion_index()
        if disp is not None:
            individual[disp] = int(self._rng.integers(0, 2))

        lc = self._lc_index()
        if lc is not None:
            individual[lc] = int(self._rng.integers(0, self.max_latent_classes))

        for j in self._mask_indices():
            individual[j] = int(self._rng.integers(0, self.max_latent_classes + 1))

        return self.repair(individual)

    # ------------------------------------------------------------------
    # feasibility
    # ------------------------------------------------------------------
    def _mutex_violations(self, roles: np.ndarray) -> int:
        if not self.mutual_exclusion:
            return 0
        violations = 0
        for group in self.mutual_exclusion:
            active = [i for i in group if int(roles[i]) != 0]
            if len(active) > 1:
                violations += len(active) - 1
        return violations

    def _repair_mutex(self, roles: np.ndarray) -> np.ndarray:
        """Keep only the first active variable in each exclusion group."""
        if not self.mutual_exclusion:
            return roles
        for group in self.mutual_exclusion:
            active = [i for i in group if int(roles[i]) != 0]
            for i in active[1:]:
                roles[i] = 0
        return roles

    def repair(self, individual: np.ndarray) -> np.ndarray:
        """Force an individual into the feasible region.

        Applies the same rules as ``AdvancedSimulatedAnnealing.repair``:
        heterogeneity-in-means (5) needs a random-coefficient partner
        (2 or 3), heterogeneity-in-variances (9) needs a random or grouped
        partner (2, 3 or 4), and each mutual-exclusion group keeps at most
        one active variable.  Structural repairs borrow a currently
        excluded variable; with none available they convert an existing
        one, so the active count is preserved either way.
        """
        roles = individual[self._role_slice()].astype(int).copy()

        if 5 in roles and not (2 in roles or 3 in roles):
            roles = self._promote(roles, (2, 3))

        if 9 in roles and not (2 in roles or 3 in roles or 4 in roles):
            roles = self._promote(roles, (2, 3, 4))

        roles = self._repair_mutex(roles)
        individual[self._role_slice()] = roles

        # Distribution codes only matter for variables that actually carry
        # a random term; snap the rest to 0 so two structures that differ
        # only in an irrelevant dist gene hash to the same cache entry.
        active_random = {2, 3, 4}
        for i, role in enumerate(roles):
            dist_idx = self.dim_core + i
            if dist_idx < self.dim and role not in active_random:
                individual[dist_idx] = 0

        return individual

    def _promote(self, roles: np.ndarray, candidates: tuple) -> np.ndarray:
        """Give a structural dependency a partner, preferring an excluded var."""
        zeros = np.where(roles == 0)[0]
        if zeros.size > 0:
            roles[int(self._rng.choice(zeros))] = int(self._rng.choice(candidates))
        else:
            roles[int(self._rng.integers(len(roles)))] = int(self._rng.choice(candidates))
        return roles

    # ------------------------------------------------------------------
    # operators
    # ------------------------------------------------------------------
    def crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Recombine two parents.

        Role genes use two-point crossover so a child tends to inherit a
        contiguous block of each parent's chosen variable set, which is how
        good structures cluster (a few strong terms plus context).  The
        trailing distribution / dispersion / latent-class genes ride along
        per-variable from whichever parent supplied the role.
        """
        child = parent_a.copy()

        D = self.dim_core
        if D >= 2:
            cuts = sorted(self._rng.choice(np.arange(1, D), size=2, replace=False))
            lo, hi = int(cuts[0]), int(cuts[1])
            child[lo:hi] = parent_b[lo:hi]

        # Per-variable auxiliary genes follow the role gene they belong to.
        aux_lo = D
        aux_hi = min(2 * D, self.dim)
        child[aux_lo:aux_hi] = np.where(
            child[:aux_lo] == parent_a[:aux_lo],
            parent_a[aux_lo:aux_hi],
            parent_b[aux_lo:aux_hi],
        )

        # Global genes: keep whichever parent's value this child inherits
        # more of, then let mutation explore the rest.
        for idx in range(2 * D, self.dim):
            child[idx] = (
                parent_a[idx] if self._rng.random() < 0.5 else parent_b[idx]
            )

        return self.repair(child)

    def mutate(self, individual: np.ndarray, rate: Optional[float] = None) -> np.ndarray:
        """Per-gene mutation, redrawing from each gene's legal range."""
        rate = self.mutation_rate if rate is None else float(rate)
        child = individual.copy()

        for i in range(self.dim_core):
            if self._rng.random() < rate:
                child[i] = self._sample_role(i)
            else:
                # Occasional legal-role-only nudge: re-draw from this
                # variable's allowed set without forcing activation, so a
                # fixed term can quietly become excluded and vice versa.
                if self._rng.random() < rate * 0.25:
                    child[i] = self._sample_role(i, force_active=False)

        for j in range(self.dim_core, min(2 * self.dim_core, self.dim)):
            if self._rng.random() < rate:
                child[j] = int(self._rng.integers(0, 4))

        disp = self._dispersion_index()
        if disp is not None and self._rng.random() < rate:
            child[disp] = int(self._rng.integers(0, 2))

        lc = self._lc_index()
        if lc is not None and self._rng.random() < rate:
            child[lc] = int(self._rng.integers(0, self.max_latent_classes))

        for j in self._mask_indices():
            if self._rng.random() < rate:
                child[j] = int(self._rng.integers(0, self.max_latent_classes + 1))

        return self.repair(child)

    def _select(self, population: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Tournament selection: sample k, return the fittest."""
        idx = self._rng.integers(0, len(population), size=self.tournament_size)
        best = int(idx[np.argmin(scores[idx])])
        return population[best]

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def evaluate(self, individual: np.ndarray) -> float:
        """Score one individual, memoised on the gene vector."""
        key = np.asarray(individual, dtype=np.int64).tobytes()
        if key in self._cache:
            return self._cache[key]

        try:
            raw = self.evaluator.fitness(np.asarray(individual, dtype=int))
        except Exception:
            # A structure the evaluator cannot build is simply a bad
            # structure; the search repairs and moves on.
            self._cache[key] = _BAD_FITNESS
            return _BAD_FITNESS

        score = _scalar_score(raw)
        if not is_valid_fitness(raw):
            score = _BAD_FITNESS
        self._cache[key] = score
        return score

    def _init_population(self) -> tuple[np.ndarray, np.ndarray]:
        population: List[np.ndarray] = []
        seen = set()
        attempts = 0
        max_attempts = 200 * self.population_size

        # Warm-start structures go in first so a resumed search cannot lose
        # its incumbent to random draws.
        for warm in self.init_solutions:
            if len(population) >= self.population_size:
                break
            population.append(warm)
            seen.add(warm.tobytes())

        # Seed with distinct structures; fall back to permitting duplicates
        # rather than spinning if the space is small.
        while len(population) < self.population_size and attempts < max_attempts:
            attempts += 1
            candidate = self._random_individual()
            key = candidate.tobytes()
            if key in seen:
                continue
            seen.add(key)
            population.append(candidate)

        while len(population) < self.population_size:
            population.append(self._random_individual())

        pop = np.asarray(population, dtype=int)
        scores = np.asarray([self.evaluate(ind) for ind in pop], dtype=float)
        return pop, scores

    # ------------------------------------------------------------------
    # driver
    # ------------------------------------------------------------------
    def optimize(self) -> tuple[np.ndarray, np.ndarray]:
        """Run the GA and return the full ``(solutions, scores)`` archive."""
        started = time.time()
        population, scores = self._init_population()

        self.archive = [row.copy() for row in population]
        self.archive_scores = [float(s) for s in scores]

        gen_best_idx = int(np.argmin(scores))
        self.best_score = float(scores[gen_best_idx])
        self.best_solution = population[gen_best_idx].copy()
        self.convergence = [self.best_score]
        self._notify(0)

        for gen in range(1, self.max_iter + 1):
            if self.max_time is not None and (time.time() - started) > self.max_time:
                break

            order = np.argsort(scores)
            elites = [population[i].copy() for i in order[: self.elite_size]]

            children: List[np.ndarray] = list(elites)
            while len(children) < self.population_size:
                parent_a = self._select(population, scores)
                if self._rng.random() < self.crossover_rate:
                    parent_b = self._select(population, scores)
                    child = self.crossover(parent_a, parent_b)
                else:
                    child = parent_a.copy()
                children.append(self.mutate(child))

            population = np.asarray(children, dtype=int)
            scores = np.asarray([self.evaluate(ind) for ind in population], dtype=float)

            self.archive.extend(row.copy() for row in population)
            self.archive_scores.extend(float(s) for s in scores)

            gen_best_idx = int(np.argmin(scores))
            gen_best = float(scores[gen_best_idx])
            self.convergence.append(min(self.convergence[-1], gen_best))

            if gen_best < self.best_score:
                self.best_score = gen_best
                self.best_solution = population[gen_best_idx].copy()
                self._notify(gen)

            self.search_stats.append(
                {
                    "iter": gen,
                    "best": self.best_score,
                    "gen_best": gen_best,
                    "mean": float(np.mean(scores)),
                    "population_size": len(population),
                    "elapsed": time.time() - started,
                }
            )

            if self.cache_clear_every and gen % self.cache_clear_every == 0:
                self._clear_device_cache()

        self.runtime = time.time() - started
        return np.asarray(self.archive, dtype=float), np.asarray(
            self.archive_scores, dtype=float
        )

    def _notify(self, gen: int) -> None:
        if self.on_iteration is not None and self.best_solution is not None:
            try:
                self.on_iteration(
                    gen, self.best_solution, self.best_score, self.runtime or 0.0
                )
            except Exception:
                pass

    @staticmethod
    def _clear_device_cache() -> None:
        try:
            import jax

            jax.clear_caches()
        except Exception:
            pass


class MultiStartGA:
    """Run :class:`GeneticAlgorithmSearch` once per seed, merge the archives.

    Mirrors :class:`~Solvers_METAJAX.MultiStartSA` so
    ``ExperimentBuilder.run`` can treat a GA exactly like an SA run.
    """

    def __init__(
        self,
        evaluator,
        dimension: int,
        n_starts: int = 1,
        n_jobs: int = 1,
        init_solutions: Optional[List[np.ndarray]] = None,
        on_iteration: Optional[Callable[..., None]] = None,
        **ga_kwargs,
    ) -> None:
        self.evaluator = evaluator
        self.dimension = dimension
        self.n_starts = max(int(n_starts), 1)
        self.n_jobs = int(n_jobs)
        self.init_solutions = (
            [np.rint(np.asarray(s, dtype=float)).astype(int) for s in init_solutions]
            if init_solutions is not None
            else None
        )
        self.on_iteration = on_iteration if callable(on_iteration) else None
        self.ga_kwargs = ga_kwargs
        self.results: List[Dict[str, Any]] = []
        self._best_score = float("inf")
        self._best_decision: Optional[np.ndarray] = None

    def run_single(self, seed: int, start_index: int = 0) -> Dict[str, Any]:
        search = GeneticAlgorithmSearch(
            evaluator=self.evaluator,
            dimension=self.dimension,
            seed=int(seed),
            on_iteration=self._forward,
            init_solutions=(
                [self.init_solutions[start_index % len(self.init_solutions)]]
                if self.init_solutions
                else None
            ),
            **self.ga_kwargs,
        )

        archive, scores = search.optimize()
        return {
            "archive": archive,
            "scores": scores,
            "convergence": list(search.convergence),
            "search_stats": list(search.search_stats),
            "runtime": search.runtime,
            "best_score": search.best_score,
            "best_solution": search.best_solution,
        }

    def _forward(self, gen: int, decision, score: float, elapsed: float) -> None:
        if float(score) < self._best_score:
            self._best_score = float(score)
            self._best_decision = np.asarray(decision, dtype=float).copy()
            if self.on_iteration is not None:
                try:
                    self.on_iteration(gen, self._best_decision, self._best_score, elapsed)
                except Exception:
                    pass

    def optimize(self) -> tuple[np.ndarray, np.ndarray]:
        results = [self.run_single(i, start_index=i) for i in range(self.n_starts)]
        self.results = results

        solutions: List[np.ndarray] = []
        scores: List[float] = []
        for r in results:
            solutions.extend(np.asarray(r["archive"]).tolist())
            scores.extend(np.asarray(r["scores"], dtype=float).ravel().tolist())

        return np.asarray(solutions, dtype=float), np.asarray(scores, dtype=float)


__all__ = ["GeneticAlgorithmSearch", "MultiStartGA", "is_valid_fitness"]
