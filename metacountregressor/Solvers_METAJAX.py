import numpy as np
import matplotlib.pyplot as plt
import jax
import gc

import os
import psutil
import time




def save_plot(fig, filename, folder="results"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved plot: {path}")


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def print_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"[MEMORY] {tag} RSS: {mem_mb:.2f} MB")

def force_gc():
    gc.collect()


import numpy as np
from joblib import Parallel, delayed


import pickle


import numpy as np
ROLE_PROBS = np.array([
    0.40,  # Excluded
    0.15,  # Fixed
    0.20,  # Random Independent
    0.2,  # Random Correlated
    0.0,  # Grouped
    0.0,  # Heterogeneity
    0.05   # Zero Inflation
])

ROLE_PROBS = ROLE_PROBS / ROLE_PROBS.sum()



class AdvancedSimulatedAnnealing:

    def __init__(self,
                 evaluator,
                 dimension,
                 max_iter=3000,
                 T0=None,
                 alpha=0.995,
                 mutation_rate=0.3,
                 min_changes=1,
                 max_changes=3,
                 adaptive=True,
                 step_size=1,
                 archive_limit=100,
                 restart_threshold=500, patience =400, tol = 1e-6):

        
        self.tol = tol
        self.patience = patience
        self.mutation_rate = mutation_rate
        self.min_changes = min_changes
        self.max_changes = max_changes
        self.step_size = step_size
        self.evaluator = evaluator
        self.dim = dimension
        self.dim_core = (dimension-1)//2
        self.max_iter = max_iter

        self.T0 = T0
        self.alpha = alpha
        self.adaptive = adaptive
        self.step_size = step_size

        self.archive_limit = archive_limit
        self.restart_threshold = restart_threshold
        self.runtime = time.time()
        self.archive = []
        self.archive_scores = []
        self.hypervolume_history = []
        self.convergence = []
        self.search_stats = []
        self.runtime = None
        self.total_iterations = 0

    
    def finalize_plots(self, algo="sa", seed=0, folder="results"):

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # ==========================
        # SINGLE OBJECTIVE
        # ==========================
        if len(self.convergence) > 0 and not self.is_multiobjective(self.archive_scores[0]):

            fig = plt.figure()
            plt.plot(self.convergence)
            plt.xlabel("Iteration")
            plt.ylabel("Best Fitness")
            plt.title("SA Convergence (Single Objective)")
            plt.grid(True)

            save_plot(fig, f"{algo}_convergence_seed{seed}_{timestamp}.png", folder)

        # ==========================
        # MULTI OBJECTIVE
        # ==========================
        if len(self.hypervolume_history) > 0:

            # Hypervolume trace
            fig = plt.figure()
            plt.plot(self.hypervolume_history)
            plt.xlabel("Iteration")
            plt.ylabel("Hypervolume")
            plt.title("SA Hypervolume Convergence")
            plt.grid(True)

            save_plot(fig, f"{algo}_hypervolume_seed{seed}_{timestamp}.png", folder)

            # Pareto front
            if len(self.archive_scores) > 0:
                scores = np.array(self.archive_scores)

                if scores.ndim > 1 and scores.shape[1] >= 2:
                    fig = plt.figure()
                    plt.scatter(scores[:, 0], scores[:, 1])
                    plt.xlabel("Objective 1")
                    plt.ylabel("Objective 2")
                    plt.title("Final Pareto Front (SA)")
                    plt.grid(True)

                    save_plot(fig, f"{algo}_pareto_seed{seed}_{timestamp}.png", folder)
        
    # =========================================================
    # Objective handling
    # =========================================================
    def is_multiobjective(self, score):
        return isinstance(score, np.ndarray) and score.ndim > 0

    def dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    # =========================================================
    # Temperature
    # =========================================================
    def temperature(self, gen):
        return self.T0 * (self.alpha ** gen)

    def is_feasible(self, solution):

        D = self.dim_core
        roles = solution[:D]

        # Rule 1:
        # If role 5 exists → must have at least one 2 or 3
        if 5 in roles:
            if not (2 in roles or 3 in roles):
                return False

        return True
    
    
    def repair(self, solution):
                                              
        D = self.dim_core
        roles = solution[:D]

        # Rule:
        # If role 5 exists → must also have at least one 2 or 3
        if 5 in roles and not (2 in roles or 3 in roles):

            zero_idx = np.where(roles == 0)[0]

            if len(zero_idx) > 0:
                # Activate one zero as 2 or 3
                roles[np.random.choice(zero_idx)] = np.random.choice([2, 3])
            else:
                # Force-convert one variable
                idx = np.random.randint(D)
                roles[idx] = np.random.choice([2, 3])

        solution[:D] = roles
           
        return solution
    
    
    
    def sample_allowed_role(self, var_index, force_active=False):
        var_name = self.evaluator.vars[var_index]
        allowed = self.evaluator.allowed_roles[var_name]

        # If we require non-zero role
        if force_active:
            allowed = [r for r in allowed if r != 0]

        if len(allowed) == 0:
            return 0  # fallback safety

        if len(allowed) == 1:
            return allowed[0]

        allowed = np.array(allowed)
        probs = ROLE_PROBS[allowed]
        probs = probs / probs.sum()

        return np.random.choice(allowed, p=probs)
    
    
    # =========================================================
    # Neighbor
    # =========================================================
    def generate_neighbor(self, solution, T=None, max_attempts=20, min_active=2):

        for _ in range(max_attempts):

            neighbor = solution.copy()

            # ---------------------------------
            # Determine mutation rate
            # ---------------------------------
            if T is not None and self.T0 is not None:
                temp_scale = T / self.T0
                mutation_rate = self.mutation_rate * temp_scale
            else:
                mutation_rate = self.mutation_rate

            mutation_rate = min(1.0, max(0.0, mutation_rate))

            # ---------------------------------
            # Decide how many variables to change
            # ---------------------------------
            n_changes = np.random.randint(
                self.min_changes,
                self.max_changes + 1
            )

            indices = np.random.choice(
                self.dim,
                size=n_changes,
                replace=False
            )

            changed = False

            # ---------------------------------
            # Apply mutations
            # ---------------------------------
            for idx in indices:

                if np.random.rand() < mutation_rate:

                    old_value = neighbor[idx]

                    # Force real step
                    step = 0
                    while step == 0:
                        step = np.random.randint(
                            -self.step_size,
                            self.step_size + 1
                        )

                    D = self.dim_core

                    if idx < D:
                        # role mutation → weighted
                        #neighbor[idx] = np.random.choice(np.arange(7), p=ROLE_PROBS)
                        neighbor[idx] = self.sample_allowed_role(idx)
                    elif idx < 2*self.dim_core:
                        # distribution or dispersion
                        neighbor[idx] = np.random.randint(0, 6)

                    changed = True

            # ---------------------------------
            # ✅ Enforce minimum active constraint
            # ---------------------------------
            D = self.dim_core
            active_count = np.sum(neighbor[:D] != 0)

            if active_count < min_active:

                zero_indices = np.where(neighbor[:D] == 0)[0]

                if len(zero_indices) > 0:
                    activate = np.random.choice(
                        zero_indices,
                        size=min_active - active_count,
                        replace=False
                    )
                    for j in activate:
                        neighbor[j] = self.sample_allowed_role(j, force_active=True)
                    

            # ---------------------------------
            # Return if valid and changed
            # ---------------------------------
            
            neighbor = self.repair(neighbor)
            if (
                    changed
                    and not self.is_same(neighbor, solution)
                ):
                return neighbor

        # ---------------------------------
        # 🔴 Fallback: force valid change
        # ---------------------------------
        neighbor = solution.copy()

        # ensure at least min_active nonzeros
        #active_count = np.sum(neighbor != 0)
        active_count = np.sum(neighbor[:self.dim_core] != 0)
        if active_count < min_active:
            zero_indices = np.where(neighbor == 0)[0]
            activate = np.random.choice(
                zero_indices,
                size=min_active - active_count,
                replace=False
            )
            neighbor[activate] = np.random.randint(1, 6, size=len(activate))

        else:
            # force one mutation
            idx = np.random.randint(0, self.dim_core)

            var_name = self.evaluator.vars[idx]
            allowed = self.evaluator.allowed_roles[var_name]

            old = neighbor[idx]

            possible = [v for v in allowed if v != old]

            if len(possible) == 0:
                return neighbor  # nothing else allowed

            neighbor[idx] = np.random.choice(possible)
        
        if self.is_same(solution, neighbor):
            return self.generate_neighbor(solution, T, min_active=min_active+1)
        return neighbor
    
    def is_same(self, nbr, sol):
        nbr_spec = self.evaluator.build_spec(nbr)
        sol_spec = self.evaluator.build_spec(sol)
        if nbr_spec is None or sol_spec is None:
            return False

        a = self.evaluator.structural_signature(nbr_spec)
        b = self.evaluator.structural_signature(sol_spec)
        if a is None or b is None:
            return False

        if a == b:
            return True
        else:
            return False
        
    #========================================
    # Crowding pruning
    # =========================================================
    def prune_archive(self):

        if len(self.archive_scores) <= self.archive_limit:
            return

        scores = np.array(self.archive_scores)

        distances = np.zeros(len(scores))

        for m in range(scores.shape[1]):
            order = np.argsort(scores[:, m])
            distances[order[0]] = distances[order[-1]] = np.inf

            min_val = scores[order[0], m]
            max_val = scores[order[-1], m]

            if max_val - min_val == 0:
                continue

            for i in range(1, len(scores)-1):
                distances[order[i]] += (
                    scores[order[i+1], m] -
                    scores[order[i-1], m]
                ) / (max_val - min_val)

        keep = np.argsort(-distances)[:self.archive_limit]

        self.archive = [self.archive[i] for i in keep]
        self.archive_scores = [self.archive_scores[i] for i in keep]

    # =========================================================
    # Hypervolume (2D minimization)
    # =========================================================
    def compute_hypervolume(self):

        if len(self.archive_scores) < 2:
            return 0.0

        scores = np.array(self.archive_scores)

        if scores.ndim == 1:
            return 0.0  # single objective

        ref = np.max(scores, axis=0) * 1.1

        sorted_pts = scores[np.argsort(scores[:, 0])]

        hv = 0.0
        prev_f1 = ref[0]

        for f1, f2 in reversed(sorted_pts):
            width = prev_f1 - f1
            height = ref[1] - f2
            if width > 0 and height > 0:
                hv += width * height
            prev_f1 = f1

        return hv

    # =========================================================
    # Archive update
    # =========================================================
    def update_archive(self, solution, score):

        if not self.is_multiobjective(score):
            # Single objective
            if len(self.archive) == 0 or score < self.archive_scores[0]:
                self.archive = [solution.copy()]
                self.archive_scores = [score]
            return

        keep = []

        for s, sc in zip(self.archive, self.archive_scores):
            if self.dominates(score, sc):
                continue
            keep.append((s, sc))

        self.archive = [k[0] for k in keep]
        self.archive_scores = [k[1] for k in keep]

        dominated = False
        for sc in self.archive_scores:
            if self.dominates(sc, score):
                dominated = True
                break

        if not dominated:
            self.archive.append(solution.copy())
            self.archive_scores.append(score.copy())

        self.prune_archive()

    
    def estimate_initial_temperature_single(self,
                                        solution,
                                        samples=100,
                                        target_accept=0.8):

        deltas = []

        f_current = self.evaluator.fitness(solution)
        if not np.isfinite(f_current):
            return 1.0

        for _ in range(samples):

            neighbor = self.generate_neighbor(solution)
            f_neighbor = self.evaluator.fitness(neighbor)
            if not np.isfinite(f_neighbor):
                continue

            delta = f_neighbor - f_current

            if delta > 0:
                deltas.append(delta)

        if len(deltas) == 0:
            return 1.0

        avg_delta = np.mean(deltas)

        return -avg_delta / np.log(target_accept)
    
    def estimate_initial_temperature_multi(self,
                                       solution,
                                       samples=100,
                                       target_accept=0.8):

        deltas = []

        f_current = self.evaluator.fitness(solution)

        for _ in range(samples):

            neighbor = self.generate_neighbor(solution)
            f_neighbor = self.evaluator.fitness(neighbor)

            if self.dominates(f_current, f_neighbor):
                # neighbor is worse
                delta = np.linalg.norm(f_neighbor - f_current)
                deltas.append(delta)

        if len(deltas) == 0:
            return 1.0
        
        avg_delta = np.mean(deltas)
        avg_delta = np.mean(deltas[np.isfinite(deltas)])

        return -avg_delta / np.log(target_accept)
    
    def auto_temperatureold(self,
                     solution,
                     samples=100,
                     target_accept=0.8):

        score = self.evaluator.fitness(solution)

        if isinstance(score, np.ndarray):
            return self.estimate_initial_temperature_multi(
                solution,
                samples,
                target_accept
            )
        else:
            return self.estimate_initial_temperature_single(
                solution,
                samples,
                target_accept
            )
    
    
    def is_valid_fitness(self, score, max_allowed=1e12):

        if not np.all(np.isfinite(score)):
            return False
        
        if np.any(score < 0):
            return False   # <--- add this

        if np.any(score >= max_allowed):
            return False

        return True
    
    def auto_temperature(self, solution, samples=25, target_accept=0.8):
        
        f_current = self.evaluator.fitness(solution)
        deltas = []

        for _ in range(samples):

            neighbor = self.generate_neighbor(solution)
            neighbor = self.repair(neighbor)
            f_neighbor = self.evaluator.fitness(neighbor)
            #neighbor = self.repair(neighbor)
            if not self.is_valid_fitness(f_neighbor):
                continue

            delta = self.energy_difference(f_current, f_neighbor)

            if delta > 0:
                deltas.append(delta)

        if len(deltas) == 0:
            return 1.0

        avg_delta = np.mean(deltas)

        return -avg_delta / np.log(target_accept)
        
    def energy_difference(self, current_score, neighbor_score):

        # SINGLE OBJECTIVE (minimization)
        if not self.is_multiobjective(current_score):

            delta = neighbor_score - current_score
            return max(0.0, delta)

        # MULTI OBJECTIVE (minimization)
        else:

            if self.dominates(current_score, neighbor_score):
                # neighbor worse
                return np.linalg.norm(neighbor_score - current_score)

            elif self.dominates(neighbor_score, current_score):
                # neighbor better
                return 0.0

            else:
                # non-dominated → treat as small penalty
                return np.linalg.norm(neighbor_score - current_score)
    
    
    
    # =========================================================
    # Safe Initialization
    # =========================================================
    def initialize_valid_solution(self, max_attempts=200, min_active=5):

        print(f"🔎 Searching for valid solution with ≥ {min_active} active variables")

        # Start from minimum allowed complexity (2 instead of 1)
        for complexity in range(min_active, self.dim_core):

            for _ in range(max_attempts):

                solution = np.zeros(self.dim, dtype=int)

                # Activate exactly `complexity` positions
                idx = np.random.choice(
                    self.dim_core,
                    size=complexity,
                    replace=False
                )

                D = self.dim_core

                for j in idx:
                    if j < D:
                        # weighted role sampling (exclude 0 since we force active)
                   
                        solution[j] = self.sample_allowed_role(j, force_active=True)
                    else:
                        solution[j] = np.random.randint(0, 7)
                solution = self.repair(solution)
                score = self.evaluator.fitness(solution)

                if self.is_valid_fitness(score):
                    print(f"✅ Found valid model with {complexity} active variables")
                    return solution, score

        raise ValueError("❌ Could not find valid initial solution.")
    
    
    
    def is_valid_fitness(self, score, max_allowed=1e12):

        if not np.all(np.isfinite(score)):
            return False

        if np.any(score == -1000000000000.0):
            return False
        
        if np.any(score >= max_allowed):
            return False

        return True
    
    
  

    def sa_accept(self, delta, T):
        """
        Stable simulated annealing acceptance rule.
        Prevents overflow in exp(-delta/T).
        """

        # Always accept downhill moves
        if delta <= 0:
            return True

        # If temperature is zero or extremely small → reject uphill moves
        if T <= 1e-12:
            return False

        # Compute exponent safely
        exponent = -delta / T

        # Clip to avoid overflow (float64 safe range)
        exponent = np.clip(exponent, -700, 700)

        # Acceptance probability
        p = np.exp(exponent)

        # Draw random uniform
        return np.random.rand() < p
    
    
    def save_search_stats_txt(
        self,
        algo="sa",
        seed=0,
        config_id=0,
        folder="results"
    ):

        os.makedirs(folder, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        filename = f"{algo}_search_stats_seed{seed}_config{config_id}_{timestamp}.txt"
        filepath = os.path.join(folder, filename)

        with open(filepath, "w") as f:

            f.write("=====================================================\n")
            f.write("SIMULATED ANNEALING SEARCH STATISTICS\n")
            f.write("=====================================================\n")
            f.write(f"Algorithm : {algo}\n")
            f.write(f"Seed      : {seed}\n")
            f.write(f"Config ID : {config_id}\n")
            f.write(f"Runtime   : {self.runtime:.2f} seconds\n")
            f.write(f"Iterations: {self.total_iterations}\n")
            f.write("=====================================================\n\n")

            if len(self.search_stats) == 0:
                f.write("No search statistics recorded.\n")
                return

            first = self.search_stats[0]

            # MULTI OBJECTIVE
            if "hypervolume" in first:

                f.write("Iter | Temp | Hypervolume | ParetoSize | BestObj1 | BestObj2\n")
                f.write("-------------------------------------------------------------\n")

                for row in self.search_stats:
                    f.write(
                        f"{row['iter']:4d} | "
                        f"{row['temperature']:8.4f} | "
                        f"{row['hypervolume']:12.6f} | "
                        f"{row['pareto_size']:10d} | "
                        f"{row['best_obj1']:10.4f} | "
                        f"{row['best_obj2']:10.4f}\n"
                    )

            # SINGLE OBJECTIVE
            else:

                f.write("Iter | Temp | Best | ArchiveSize\n")
                f.write("------------------------------------------\n")

                for row in self.search_stats:
                    f.write(
                        f"{row['iter']:4d} | "
                        f"{row['temperature']:8.4f} | "
                        f"{row['best']:12.6f} | "
                        f"{row['archive_size']:10d}\n"
                    )

        print(f"✅ SA search stats saved to {filepath}")
    
    
    # =========================================================
    # Optimize
    # =========================================================
    def optimize(self):
        start_time = time.time()   # ✅
        last_best = None
        current, current_score = self.initialize_valid_solution()

        if self.T0 is None:
            self.T0 = self.auto_temperature(current)
            print(f"Auto T0: {self.T0:.4f}")
        
        self.archive = []
        self.archive_scores = []
        archive_sizes = []
        self.update_archive(current, current_score)

        no_improve = 0

        for gen in range(self.max_iter):
            
            elapsed = time.time() - start_time
            avg_time_per_iter = elapsed / (gen + 1)
            remaining = avg_time_per_iter * (self.max_iter - gen - 1)

            if gen % 10 == 0:
                print(
                    f"[Gen {gen}/{self.max_iter}] "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"ETA: {remaining:.1f}s"
                )
            if gen % 50 == 0:
                force_gc()
                jax.clear_caches()
            T = self.temperature(gen)

            neighbor = self.generate_neighbor(current, T)
            neighbor = self.repair(neighbor)
            neighbor_score = self.evaluator.fitness(neighbor)
            print(f'gen{gen}, nbr fitness:{neighbor_score}')

            accept = False

            if not self.is_multiobjective(current_score):
                ## TERMINATON STAGNATON #################
                best = self.archive_scores[0]
        
                if last_best is None:
                    last_best = best

                if abs(last_best - best) < self.tol:
                    no_improve += 1
                else:
                    no_improve = 0

                last_best = best

                if no_improve > self.patience:
                    print("⏹ Early stop: no improvement")
                    break
                #####################################
                delta = self.energy_difference(current_score, neighbor_score)

                if delta == 0:
                    accept = True
                else:
                    if T < 1e-12:
                        accept = False
                    else:
                        accept = np.random.rand() < np.exp(-delta / T)

            else:

                if self.dominates(neighbor_score, current_score):
                    accept = True
                elif self.dominates(current_score, neighbor_score):
                    delta = np.sum(np.maximum(
                        0, neighbor_score - current_score))
                    accept = self.sa_accept(delta, T)
                else:
                    dist = np.linalg.norm(
                        neighbor_score - current_score)
                    accept = self.sa_accept(dist, T)

            if accept:
                current = neighbor
                current_score = neighbor_score
                self.update_archive(current, current_score)
                archive_sizes.append(len(self.archive))
                no_improve = 0
            else:
                no_improve += 1

            # ✅ Adaptive cooling
            if self.adaptive and no_improve > 50:
                self.alpha *= 0.99

            # ✅ Restart
            if no_improve > self.restart_threshold:
                D = self.dim_core

                roles = np.zeros(D, dtype=int)
                for j in range(D):
                    roles[j] = self.sample_allowed_role(j)
                dists = np.random.randint(0, 6, size=D)
                disp = np.random.randint(0, 2, size=1)

                current = np.concatenate([roles, dists, disp])
                current_score = self.evaluator.fitness(current)
                no_improve = 0

            # ✅ Hypervolume tracking
            if self.is_multiobjective(current_score):
                hv = self.compute_hypervolume()
                self.hypervolume_history.append(hv)
                self.convergence.append(hv)
                front_scores = np.array(self.archive_scores)

                best_obj1 = float(np.min(front_scores[:, 0]))
                best_obj2 = float(np.min(front_scores[:, 1]))

                self.search_stats.append({
                    "iter": gen,
                    "temperature": float(T),
                    "hypervolume": float(hv),
                    "pareto_size": len(self.archive),
                    "best_obj1": best_obj1,
                    "best_obj2": best_obj2
                })
                # Multiobjective stagnation check
                if (
                    len(self.archive) >= 2 and
                    len(self.hypervolume_history) > 40
                ):

                    recent_hv = self.hypervolume_history[-20:]
                    recent_size = archive_sizes[-20:]

                    if (
                        max(recent_hv) - min(recent_hv) < self.tol and
                        max(recent_size) == min(recent_size)
                    ):
                        print("⏹ Early stop: multiobjective stagnation")
                        break
                '''
                if max(recent) - min(recent) < self.tol:
                    print("⏹ Early stop: hypervolume stagnated")
                    break
                if len(archive_sizes) > 50:

                    recent_sizes = archive_sizes[-30:]

                    if max(recent_sizes) == min(recent_sizes):
                        print("⏹ Early stop: archive not growing")
                        break
                '''
            else:
                # single objective
                best = self.archive_scores[0]
                self.convergence.append(best)
                #self.convergence.append(best)
                self.search_stats.append({
                    "iter": gen,
                    "temperature": float(T),
                    "best": float(best),
                    "archive_size": len(self.archive)
                })
            
        self.runtime = time.time() - start_time
        self.total_iterations = gen + 1
        return np.array(self.archive), np.array(self.archive_scores)
    
    



class MultiStartSA:

    def __init__(self,
                 evaluator,
                 dimension,
                 n_starts=10,
                 n_jobs=1,
                 **sa_kwargs):

        self.evaluator = evaluator
        self.dimension = dimension
        self.n_starts = n_starts
        self.n_jobs = n_jobs
        self.sa_kwargs = sa_kwargs

    def run_single(self, seed):

        np.random.seed(seed)

        sa = AdvancedSimulatedAnnealing(
            evaluator=self.evaluator,
            dimension=self.dimension,
            **self.sa_kwargs
        )

        archive, scores = sa.optimize()
        sa.save_search_stats_txt(
                algo='sa',
                seed=seed,
                config_id=0
        )

        sa.finalize_plots(
            algo="sa",
            seed=seed
        )
        
        return {
        "archive": archive,
        "scores": scores,
        "convergence": sa.convergence
        }

    def optimize(self):

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.run_single)(i)
            for i in range(self.n_starts)
        )

        self.results = results

        # Merge archives
        all_solutions = []
        all_scores = []

        for r in results:
            for s, sc in zip(r["archive"], r["scores"]):
                all_solutions.append(s)
                all_scores.append(sc)

        return np.array(all_solutions), np.array(all_scores)
    
    def plot_convergence(self):

        all_curves = [r["convergence"] for r in self.results]

        min_len = min(len(c) for c in all_curves)
        trimmed = np.array([c[:min_len] for c in all_curves])

        mean_curve = np.mean(trimmed, axis=0)
        std_curve = np.std(trimmed, axis=0)

        plt.plot(mean_curve, label="Mean")
        plt.fill_between(range(min_len),
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.3)

        plt.xlabel("Iteration")
        plt.ylabel("Hypervolume / Best Fitness")
        plt.title("Convergence Across Starts")
        plt.legend()
        plt.show()
        
    def final_pareto(self, all_scores):

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        scores = np.array(all_scores)

        if scores.ndim == 1:
            # single objective
            idx = np.argmin(scores)
            return idx, scores[idx]

        keep = []

        for i in range(len(scores)):
            dominated = False
            for j in range(len(scores)):
                if j != i and dominates(scores[j], scores[i]):
                    dominated = True
                    break
            if not dominated:
                keep.append(i)

        return keep, scores[keep]


class NSGA2Engine:

    def __init__(self,
                 evaluator,
                 operator,
                 dimension,
                 pop_size=30,
                 max_iter=50,
                 n_jobs=1,
                 save_history=False):

        self.evaluator = evaluator
        self.operator = operator
        self.dim = dimension
        self.dim_core = (dimension-1)//2
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.generations = int(max_iter/pop_size)
        self.n_jobs = n_jobs
        self.mutation_prob = 0.2
        self.save_history = save_history

        # ---- tracking ----
        self.fitness_history = []
        self.hypervolume_history = []
        self.population_history = []
        self.pareto_history = []
        # ---- generation statistics ----
        self.search_stats = []

    
    def finalize_plots(self, algo="nsga2", seed=0, folder="results"):

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # ==========================
        # SINGLE OBJECTIVE
        # ==========================
        if len(self.fitness_history) > 0:

            fig = plt.figure()
            plt.plot(self.fitness_history)
            plt.xlabel("Generation")
            plt.ylabel("Best Fitness")
            plt.title("NSGA-II Convergence (Single Objective)")
            plt.grid(True)

            save_plot(fig, f"{algo}_convergence_seed{seed}_{timestamp}.png", folder)

        # ==========================
        # MULTI OBJECTIVE
        # ==========================
        if len(self.hypervolume_history) > 0:

            # Hypervolume trace
            fig = plt.figure()
            plt.plot(self.hypervolume_history)
            plt.xlabel("Generation")
            plt.ylabel("Hypervolume")
            plt.title("NSGA-II Hypervolume Convergence")
            plt.grid(True)

            save_plot(fig, f"{algo}_hypervolume_seed{seed}_{timestamp}.png", folder)

            # Pareto front
            if len(self.pareto_history) > 0:

                front = self.pareto_history[-1]

                fig = plt.figure()
                plt.scatter(front[:, 0], front[:, 1])
                plt.xlabel("Objective 1")
                plt.ylabel("Objective 2")
                plt.title("Final Pareto Front (NSGA-II)")
                plt.grid(True)

                save_plot(fig, f"{algo}_pareto_seed{seed}_{timestamp}.png", folder)
    
    # =========================================================
    # Evaluation
    # =========================================================
    def evaluate_population(self, pop):

        if self.n_jobs == 1:
            return np.array([
                self.evaluator.fitness(ind)
                for ind in pop
            ])

        return np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(self.evaluator.fitness)(ind)
                for ind in pop
            )
        )

    # =========================================================
    def is_multiobjective(self, scores):
        return scores.ndim > 1 and scores.shape[1] > 1

    def dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    # =========================================================
    def fast_non_dominated_sort(self, scores):

        fronts = [[]]
        domination_count = np.zeros(len(scores))
        dominated_sets = [[] for _ in range(len(scores))]

        for i in range(len(scores)):
            for j in range(len(scores)):
                if i == j:
                    continue
                if self.dominates(scores[i], scores[j]):
                    dominated_sets[i].append(j)
                elif self.dominates(scores[j], scores[i]):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        current = 0
        while fronts[current]:
            next_front = []
            for i in fronts[current]:
                for j in dominated_sets[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current += 1
            fronts.append(next_front)

        return fronts[:-1]

    # =========================================================
    def crowding_distance(self, scores, front):

        distance = np.zeros(len(front))
        front_scores = scores[front]
        num_obj = front_scores.shape[1]

        for m in range(num_obj):

            sorted_idx = np.argsort(front_scores[:, m])
            distance[sorted_idx[0]] = distance[sorted_idx[-1]] = np.inf

            min_val = front_scores[sorted_idx[0], m]
            max_val = front_scores[sorted_idx[-1], m]

            if max_val - min_val == 0:
                continue

            for i in range(1, len(front)-1):
                distance[sorted_idx[i]] += (
                    front_scores[sorted_idx[i+1], m] -
                    front_scores[sorted_idx[i-1], m]
                ) / (max_val - min_val)

        return distance

    # =========================================================
    def compute_hypervolume(self, front_scores, reference):

        # Sort by first objective ascending (since minimizing)
        sorted_pts = front_scores[np.argsort(front_scores[:, 0])]

        hv = 0.0
        prev_f1 = reference[0]

        for f1, f2 in reversed(sorted_pts):
            width = prev_f1 - f1
            height = reference[1] - f2

            if width > 0 and height > 0:
                hv += width * height

            prev_f1 = f1

        return hv
    
    def sample_allowed_role(self, var_index, force_active=False):
        var_name = self.evaluator.vars[var_index]
        allowed = self.evaluator.allowed_roles[var_name]

        # If we require non-zero role
        if force_active:
            allowed = [r for r in allowed if r != 0]

        if len(allowed) == 0:
            return 0  # fallback safety

        if len(allowed) == 1:
            return allowed[0]

        allowed = np.array(allowed)
        probs = ROLE_PROBS[allowed]
        probs = probs / probs.sum()

        return np.random.choice(allowed, p=probs)
    def _initialise_start_pop(self):
        D = self.dim_core   # ✅ define it here
        pop = []
        while len(pop) < self.pop_size:

            roles = np.zeros(D, dtype=int)

            for j in range(D):
                roles[j] = self.sample_allowed_role(j)

            dists = np.random.randint(0, 6, size=D)
            disp = np.random.randint(0, 2, size=1)

            candidate = np.hstack((roles, dists, disp))
            candidate = self.repair(candidate)

            pop.append(candidate)
        pop = np.array(pop)
        return pop

    # =========================================================
    def optimize(self):
        start_time = time.time()  # ✅ start timer

        D = self.dim_core
        pop  = self._initialise_start_pop()

      
        scores = self.evaluate_population(pop)
        scores = self.sanitize_scores(scores)

        single = not self.is_multiobjective(scores)
        
        # -------------------------------------------------
        # ✅ Auto-estimate SA initial temperature
        # -------------------------------------------------
        

        for gen in range(self.generations):
            elapsed = time.time() - start_time
            avg_time_per_gen = elapsed / (gen + 1)
            remaining = avg_time_per_gen * (self.generations - gen - 1)

            print(
                f"[Gen {gen}/{self.max_iter}] "
                f"Elapsed: {format_time(elapsed)} | "
                f"ETA: {format_time(remaining)}"
            )
            #print_jax_compilation_cache()
            offspring = []

            for i in range(self.pop_size):
                child = self.operator.generate(pop, i, gen, self.max_iter)
                child = self.repair(child)   # 
                offspring.append(child)

            offspring = np.array(offspring)

            offspring_scores = self.evaluate_population(offspring)
            offspring_scores = self.sanitize_scores(offspring_scores)

            if single:

                combined = np.vstack((pop, offspring))
                combined_scores = np.hstack((scores, offspring_scores))

                idx = np.argsort(combined_scores)[:self.pop_size]
                pop = combined[idx]
                scores = combined_scores[idx]

                best = np.min(scores)
                self.fitness_history.append(best)

            else:

                pop, scores = self.environmental_selection(
                    pop, scores, offspring, offspring_scores
                )

                front = self.fast_non_dominated_sort(scores)[0]
                front_scores = scores[front]
                # ✅ Remove NaNs
                front_scores = front_scores[~np.isnan(front_scores).any(axis=1)]

                if len(front_scores) == 0:
                    hv = 0.0
                else:
                    ref = np.nanmax(scores, axis=0) + 1
                    hv = self.compute_hypervolume(front_scores, ref)
                
                
               # ref = np.max(scores, axis=0) + 1
                #hv = self.compute_hypervolume(front_scores, ref)

                self.hypervolume_history.append(hv)
                self.pareto_history.append(front_scores.copy())

            if self.save_history:
                self.population_history.append(pop.copy())

            
            
            if single:

                best = np.min(scores)
                mean = np.mean(scores)
                std = np.std(scores)
                self.search_stats.append({
                        "gen": gen,
                        "best": float(best),
                        "mean": float(mean),
                        "std": float(std)
                })

                print(
                    f"Gen {gen:03d} | "
                    f"Best: {best:.4f} | "
                    f"Mean: {mean:.4f} | "
                    f"Std: {std:.4f}"
                )
                print_memory_usage(f"end gen before gc {gen}")
                force_gc()
                jax.clear_caches()
                print_memory_usage(f"end gen {gen}")
                print('cool')
            else:

                pareto_size = len(front_scores)

                best_bic = np.min(front_scores[:, 0])
                best_rmse = np.min(front_scores[:, 1])
                self.search_stats.append({
                    "gen": gen,
                    "hypervolume": float(hv),
                    "pareto_size": int(pareto_size),
                    "best_obj1": float(best_bic),
                    "best_obj2": float(best_rmse)
                })
                print(
                    f"Gen {gen:03d} | "
                    f"HV: {hv:.4f} | "
                    f"Pareto: {pareto_size} | "
                    f"Best BIC: {best_bic:.2f} | "
                    f"Best RMSE: {best_rmse:.4f}"
                )
                print_memory_usage(f"end gen before gc {gen}")
                force_gc()
                print_memory_usage(f"end gen {gen}")
                print('cool')

        if single:
            best = np.argmin(scores)
            return pop[best], scores[best]
        else:
            front = self.fast_non_dominated_sort(scores)[0]
            return pop[front], scores[front]
        
    def save_search_stats_txt(
        self,
        algo="nsga2",
        seed=0,
        config_id=0,
        folder="results"
    ):

        os.makedirs(folder, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        filename = f"{algo}_search_stats_seed{seed}_config{config_id}_{timestamp}.txt"
        filepath = os.path.join(folder, filename)

        with open(filepath, "w") as f:

            f.write("=====================================================\n")
            f.write("NSGA-II SEARCH STATISTICS\n")
            f.write("=====================================================\n")
            f.write(f"Algorithm : {algo}\n")
            f.write(f"Seed      : {seed}\n")
            f.write(f"Config ID : {config_id}\n")
            f.write(f"Timestamp : {timestamp}\n")
            f.write("=====================================================\n\n")

            if len(self.search_stats) == 0:
                f.write("No search statistics recorded.\n")
                return

            # Detect mode
            first = self.search_stats[0]

            if "hypervolume" in first:
                f.write("Gen | Hypervolume | ParetoSize | BestObj1 | BestObj2\n")
                f.write("-----------------------------------------------------\n")

                for row in self.search_stats:
                    f.write(
                        f"{row['gen']:3d} | "
                        f"{row['hypervolume']:12.6f} | "
                        f"{row['pareto_size']:10d} | "
                        f"{row['best_obj1']:10.4f} | "
                        f"{row['best_obj2']:10.4f}\n"
                    )
            else:
                f.write("Gen | Best | Mean | Std\n")
                f.write("---------------------------------\n")

                for row in self.search_stats:
                    f.write(
                        f"{row['gen']:3d} | "
                        f"{row['best']:10.6f} | "
                        f"{row['mean']:10.6f} | "
                        f"{row['std']:10.6f}\n"
                    )

        print(f"✅ Search stats saved to {filepath}")
    
    def constraint_violation(self, solution):

        D = self.dim_core
        roles = solution[:D]

        violation = 0.0

        # Rule:
        # If role 5 exists but no 2 or 3
        if 5 in roles and not (2 in roles or 3 in roles):
            violation += 1.0   # magnitude of violation

        return violation
    
    
    def repair(self, solution):

        D = self.dim_core
        roles = solution[:D]

        if 5 in roles and not (2 in roles or 3 in roles):

            zero_idx = np.where(roles == 0)[0]

            if len(zero_idx) > 0:
                roles[np.random.choice(zero_idx)] = np.random.choice([2,3])
            else:
                # force convert one variable
                idx = np.random.randint(D)
                roles[idx] = np.random.choice([2,3])

        solution[:D] = roles
        return solution

    def environmental_selection(self,
                            pop, scores,
                            offspring, offspring_scores):

        # Combine parent + offspring
        combined_pop = np.vstack((pop, offspring))
        combined_scores = np.vstack((scores, offspring_scores))

        # Fast non-dominated sort
        fronts = self.fast_non_dominated_sort(combined_scores)

        new_pop = []
        new_scores = []

        for front in fronts:

            if len(new_pop) + len(front) <= self.pop_size:
                # Take full front
                for idx in front:
                    new_pop.append(combined_pop[idx])
                    new_scores.append(combined_scores[idx])
            else:
                # Need partial front → crowding distance
                front_scores = combined_scores[front]
                distances = self.crowding_distance(combined_scores, front)

                sorted_idx = np.argsort(-distances)

                remaining = self.pop_size - len(new_pop)

                for i in sorted_idx[:remaining]:
                    idx = front[i]
                    new_pop.append(combined_pop[idx])
                    new_scores.append(combined_scores[idx])

                break

        return np.array(new_pop), np.array(new_scores)
    
    
    # =========================================================
    # ---------------- PLOTTING ----------------
    # =========================================================

    def plot_convergence(self):

        if len(self.fitness_history) > 0:
            plt.plot(self.fitness_history)
            plt.xlabel("Generation")
            plt.ylabel("Best Fitness")
            plt.title("Convergence (Single Objective)")
            plt.show()

        elif len(self.hypervolume_history) > 0:
            plt.plot(self.hypervolume_history)
            plt.xlabel("Generation")
            plt.ylabel("Hypervolume")
            plt.title("Hypervolume Convergence")
            plt.show()

    def plot_pareto_front(self):

        if len(self.pareto_history) == 0:
            print("Not multi-objective.")
            return

        last_front = self.pareto_history[-1]

        plt.scatter(last_front[:, 0], last_front[:, 1])
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.title("Final Pareto Front")
        plt.show()

    def plot_pareto_evolution(self):

        if len(self.pareto_history) == 0:
            print("Not multi-objective.")
            return

        for front in self.pareto_history:
            plt.scatter(front[:, 0], front[:, 1], alpha=0.2)

        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.title("Pareto Front Evolution")
        plt.show()

    # =========================================================
    # ---------------- SAVE / LOAD ----------------
    # =========================================================
    
    def sanitize_scores(self, scores, max_allowed=1e12):

        scores = np.array(scores)

        invalid = (
            ~np.isfinite(scores) |
            (scores >= max_allowed) |
            (scores < 0)   # <-- reject negatives
        )

        if scores.ndim == 1:
            scores[invalid] = max_allowed
        else:
            scores[invalid.any(axis=1)] = max_allowed

        return scores
    
    
    

    def save_history(self, filename="nsga_history.pkl"):

        data = {
            "fitness_history": self.fitness_history,
            "hypervolume_history": self.hypervolume_history,
            "population_history": self.population_history,
            "pareto_history": self.pareto_history
        }

        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load_history(self, filename="nsga_history.pkl"):

        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.fitness_history = data["fitness_history"]
        self.hypervolume_history = data["hypervolume_history"]
        self.population_history = data["population_history"]
        self.pareto_history = data["pareto_history"]



class DynamicHarmony:

    def __init__(self,
                 hmcr=0.9,
                 par_min=0.1,
                 par_max=0.9,
                 bw_min=1,
                 bw_max=3):

        self.hmcr = hmcr
        self.par_min = par_min
        self.par_max = par_max
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.F = par_max-par_min

    def generate(self, pop, i, gen, max_iter):

        PAR = self.par_min + (
            (self.par_max - self.par_min)
            * gen / max_iter
        )

        BW = int(
            self.bw_max -
            (self.bw_max - self.bw_min)
            * gen / max_iter
        )

        child = pop[i].copy()

        for j in range(len(child)):

            if np.random.rand() < self.hmcr:
                idx = np.random.randint(len(pop))
                child[j] = pop[idx][j]

            if np.random.rand() < PAR:
                step = np.random.randint(-BW, BW+1)
                child[j] =  np.random.randint(0,6)

        return child


class AdaptiveDE:

    def __init__(self, F=0.5, CR=0.7,
                 tau1=0.1, tau2=0.1):

        self.F = F
        self.CR = CR
        self.tau1 = tau1
        self.tau2 = tau2

    def generate(self, pop, i, gen, max_iter):

        if np.random.rand() < self.tau1:
            self.F = 0.1 + 0.9*np.random.rand()

        if np.random.rand() < self.tau2:
            self.CR = np.random.rand()

        idxs = list(range(len(pop)))
        idxs.remove(i)
        a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

        mutant = a.copy()
        #mutant = np.random.randint(0,6,size=len(pop[i]))
        for j in range(len(mutant)):
            if b[j] != c[j]:
                if np.random.rand() < self.F:
                    mutant[j] = b[j]
                else:
                    mutant[j] = c[j]
            else:
                if np.random.rand() < self.F:
                    mutant[j] = a[j]
                else:
                    mutant[j] = np.random.randint(0,6)

        trial = pop[i].copy()
        trial = pop[i].copy()

        dim = len(trial)

        # force at least one mutation
        j_rand = np.random.randint(dim)
        
        for j in range(len(trial)):
            if np.random.rand() < self.CR or j == j_rand:
                trial[j] = mutant[j]

        
        if np.array_equal(trial, pop[i]):
            idx = np.random.randint(dim)
            possible = [v for v in range(6) if v != trial[idx]]
            trial[idx] = np.random.choice(possible)
        
        return trial




    

class StepwiseStructureSolver:

    def __init__(
        self,
        fitness_function,
        all_variables,
        allowed_roles,
        allowed_distributions,
        max_iter=20,
        mode="aic"
    ):
        self.fitness = fitness_function
        self.vars = all_variables
        self.allowed_roles = allowed_roles
        self.allowed_distributions = allowed_distributions
        self.max_iter = max_iter
        self.mode = mode

        self.D = len(all_variables)

    # ---------------------------------------
    # Initial solution (all OFF)
    # ---------------------------------------
    def initialize(self):
        roles = np.zeros(self.D, dtype=int)
        dists = np.zeros(self.D, dtype=int)
        return np.concatenate([roles, dists])

    # ---------------------------------------
    # Generate neighbors (add/remove/modify)
    # ---------------------------------------
    def generate_neighbors(self, solution):

        neighbors = []
        roles = solution[:self.D]
        dists = solution[self.D:]

        for i, var in enumerate(self.vars):

            # Try changing role
            for new_role in self.allowed_roles.get(var, [0]):

                if new_role != roles[i]:
                    new_roles = roles.copy()
                    new_roles[i] = new_role

                    neighbors.append(
                        np.concatenate([new_roles, dists])
                    )

            # If random role, try changing distribution
            if roles[i] in [2,3,4]:
                allowed = self.allowed_distributions.get(var, ["normal"])
                for dist_idx in range(len(allowed)):

                    if dist_idx != dists[i]:
                        new_dists = dists.copy()
                        new_dists[i] = dist_idx

                        neighbors.append(
                            np.concatenate([roles, new_dists])
                        )

        return neighbors

    # ---------------------------------------
    # Optimize
    # ---------------------------------------
    def optimize(self):

        current = self.initialize()
        current_score = self.fitness(current)

        for _ in range(self.max_iter):
            print(current_score)

            neighbors = self.generate_neighbors(current)

            improved = False
            best_neighbor = current
            best_score = current_score

            for candidate in neighbors:

                score = self.fitness(candidate)

                if score > best_score:
                    best_score = score
                    best_neighbor = candidate
                    improved = True

            if improved:
                current = best_neighbor
                current_score = best_score
            else:
                break  # no improvement

        return current, current_score