"""
    ALGORITHMS, HS, DE and SA
"""
#

import copy
import multiprocessing as mp
import os
import random
import statistics as st
import time
import warnings
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd

warnings.warn(
    "metaheuristics.py is a legacy interface and will be removed in a future release. "
    "Use ExperimentBuilder.run(...) on the JAX-first path instead.",
    FutureWarning,
    stacklevel=2,
)

try:
    from .pareto_file import Pareto, Solution
except ImportError:
    from pareto_file import Pareto, Solution

try:
    from .solution import ObjectiveFunction
except ImportError:
    print
    from solution import ObjectiveFunction


HarmonySearchResults = namedtuple('HarmonySearchResults',
                                  ['elapsed_time', 'best_harmony', 'best_fitness', 'harmony_memories',
                                   'harmony_histories'])
DifferentialEvolutionResults = namedtuple('DifferentialEvolutionResults',
                                          ['elapsed_time', 'iteration', 'iter_solution', 'best_solutions',
                                           'best_fitness', 'best_struct', 'average_best'])
SimulatedAnnealingResults = namedtuple('SimulatedAnnealingResults',
                                       ['elapsed_time', 'iteration', 'iter_solution', 'best_solutions', 'best_fitness',
                                        'best_struct', 'average_best'])

DifferentialEvolutionMulti = namedtuple('DifferentialEvolutionMulti',
                                        ['elapsed_time', 'best_solutions', 'population_solutions'])


# helper function to plot the bic
def _plot(x, y, z, xlabel=None, ylabel=None, zlabel=None, filename=None):
    from matplotlib import pyplot as plt

    # data_tuples = list(zip(x, y, z))
    fig, ax = plt.subplots()
    ax.plot(x, y, color='red', label=ylabel)
    ax.plot(x, z, color='blue', label=zlabel)
    ax.set(xlabel=xlabel)
    ax.grid()
    # plt.ylim([0, 2000])
    if filename is not None:
        plt.savefig(filename + 'bic.svg', format='svg', dpi=1200)
        plt.savefig(filename + 'bic.eps', format='eps', dpi=1200)
        plt.savefig(filename + 'bic.png')
        plt.show()
    else:
        plt.savefig('bic.svg', format='svg', dpi=1200)
        plt.savefig('bic.eps', format='eps', dpi=1200)
        plt.savefig('bic.png')
        plt.show()


# helper function to grab dictionary means
def dict_mean(dict_list,
              ignore=None):
    if ignore is None:
        ignore = ['AIC', 'layout', 'fixed_fit', 'rdm_fit', 'rdm_cor_fit', 'zi_fit', 'simple', 'pvalues', "CAIC"]
    mean_dict = {}
    if ignore is None:
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
        return mean_dict
    else:
        mean_dict = {}
        for key in dict_list[0].keys():
            if key in ignore:
                continue
            # Filter out dictionaries that don't have the key
            filtered_values = [d[key] for d in dict_list if key in d]
            
            if filtered_values:  # Ensure there are values to compute the mean
                mean_dict[key] = sum(filtered_values) / len(filtered_values)
            else:
                mean_dict[key] = None  # Or handle missing data differently if needed
        return mean_dict


def pareto_logger(pareto_set, iteration, complexity, folder=None):
    if folder is not None:
        name = folder + '/pareto_logger_complex' + str(complexity) + ".csv"

    else:
        name = 'pareto_logger_complex' + str(complexity) + ".csv"

    st22 = pd.DataFrame(pareto_set)
    st22.to_csv(name, mode='a', index=False, header=True)


def logger(iteration, incumbent1, best1=None, alt_method=True, name=None, multi=0, local_best1=None):
    if name is None:
        name = 'log.csv'
    if alt_method:
        if multi:
            if best1 is not None:
                best = dict_mean(best1)
            else:
                best = None

        else:
            if best1 is not None:
                best = best1.copy()
            else:
                best = None

        if isinstance(incumbent1, dict):
            incumbent = incumbent1.copy()
            incumbent.pop('layout')
            incumbent.pop('fixed_fit')
            incumbent.pop('rdm_fit')
            incumbent.pop('rdm_cor_fit')
            incumbent.pop('zi_fit')
            try:
                incumbent.pop('pvalues')
            except Exception as e:
                print(e, 'cant pop')

            incumbent = pd.DataFrame(incumbent, index=[0])
            incumbent = incumbent.add_prefix('incumbent_')
        else:
            raise Exception

        if local_best1 is not None:
            local_best = local_best1.copy()
            local_best.pop('layout')
            local_best.pop('fixed_fit')
            local_best.pop('rdm_fit')
            local_best.pop('rdm_cor_fit')
            local_best.pop('zi_fit')
            local_best.pop('pvalues')
            local_best = pd.DataFrame(local_best, index=[0])
            local_best = local_best.add_prefix('localb_')
        else:
            local_best = None
        if not multi:
            if isinstance(best1, dict):
                best = best1.copy()
                best.pop('layout')
                best.pop('fixed_fit')
                best.pop('rdm_fit')
                best.pop('rdm_cor_fit')
                best.pop('zi_fit')
                best.pop('pvalues')
                best = pd.DataFrame(best, index=[0])
                best = best.add_prefix('best_')
            else:
                raise Exception
        else:
            if isinstance(best, dict):

                best = pd.DataFrame(best, index=[0])
                best = best.add_prefix('best_')
            elif best is not None:
                raise Exception

        data_tuples = list([iteration])
        iter_df = pd.DataFrame(data_tuples, columns=['iteration'])
        if best is not None:
            if local_best is None:
                df = pd.concat([iter_df, incumbent, best], axis=1)
            else:
                df = pd.concat([iter_df, incumbent, best, local_best], axis=1)
        else:
            if local_best is None:
                df = pd.concat([iter_df, incumbent], axis=1)
            else:
                df = pd.concat([iter_df, incumbent, local_best], axis=1)

        if os.path.isfile(name):
            df.to_csv(name, mode='a', index=False, header=False)
        else:
            df.to_csv(name, mode='a', index=False, header=True)

    else:
        if best1 is not None:
            data_tuples = np.vstack(list([iteration, incumbent1, best1])).transpose()
            df = pd.DataFrame(data_tuples, columns=['iteration', 'incumbent', 'best'])

            if os.path.isfile(name):
                df.to_csv(name, mode='a', index=False, header=False)
            else:
                df.to_csv(name, mode='a', index=False, header=True)
        else:
            data_tuples = np.vstack(list([iteration, incumbent1])).transpose()
            df = pd.DataFrame(data_tuples, columns=['iteration', 'incumbent'])

            if os.path.isfile(name):
                df.to_csv(name, mode='a', index=False, header=False)
            else:
                df.to_csv(name, mode='a', index=False, header=True)


def different_evolution(objective_function, initial_slns=None, **kwargs):
    if not isinstance(objective_function, ObjectiveFunction):
        raise TypeError
    de = DifferentialEvolution(objective_function, **kwargs)
    results = de.differential_evolution_run(initial_slns=initial_slns)
    x = list()
    y = list()
    z = list()
    for it, sln, best, struct in results:
        x.append(it)
        y.append(sln)
        z.append(best)
    _plot(x, y, z, 'Iterations', 'Incumbent', 'Best')
    return results


def differential_evolution(objective_function, initial_slns=None, **kwargs):

    start = datetime.now()

    man = None
    if 'Manual_Fit' in kwargs:
        if kwargs['Manual_Fit'] is not None:
            man = kwargs['Manual_Fit']

    if objective_function.is_multi:

        if 'MP' in kwargs and kwargs['MP'] == True:

            de = Mutlithreaded_Meta(objective_function, **kwargs)
            best, pare = de.run_mp(initial_slns=initial_slns, mod_init=man)
        else:
            print('Not Multi Threaded')
            de = DifferentialEvolution(objective_function, **kwargs)
            best, pare = de.differential_evolution_run(initial_slns=initial_slns, mod_init=man)

        end = datetime.now()
        elapsed_time = end - start
        return DifferentialEvolutionMulti(elapsed_time=elapsed_time, best_solutions=best, population_solutions=pare)
    else:
        de = DifferentialEvolution(objective_function, **kwargs)

        iterations, solutions, best_solutions, best_fitness, best_struct = de.differential_evolution_run(
            initial_slns=initial_slns, mod_init=man)
        AVERAGE_BEST = st.mean(best_solutions)
        end = datetime.now()
    
        elapsed_time = end - start
        print(f'elapsed time{elapsed_time}')
        return DifferentialEvolutionResults(elapsed_time=elapsed_time, iteration=iterations,
                                            iter_solution=solutions, best_solutions=best_solutions,
                                            best_fitness=best_fitness,
                                            best_struct=best_struct, average_best=AVERAGE_BEST)


def simulated_annealing(objective_function, initial_slns=None, **kwargs):
    # if hyperparameters is None:
    #    TEMP_ALPHA = 0.99; MAX_STEPS = 10; INTL_ACCEPT = 0.5; STEPS = 20; SWAP_PERC = 0.2; NUM_INTL_SLNS = 25; IS_MULTI = 0
    # else:
    #   TEMP_ALPHA, MAX_STEPS, INTL_ACCEPT, STEPS, SWAP_PERC, NUM_INTL_SLNS, IS_MULTI= hyperparameters
    man = None
    try:
        objective_function.instance_name = str(0)
    except:
        pass
    if 'Manual_Fit' in kwargs:
        if kwargs['Manual_Fit'] is not None:
            man = kwargs['Manual_Fit']
    if 'MP' in kwargs:
        if kwargs['MP']:
            sa = Mutlithreaded_Meta(objective_function, **kwargs)
            return sa.run_sa(initial_slns=initial_slns, mod_init=man)
    sa = SimulatedAnnealing(objective_function, **kwargs)
    return sa.run(initial_slns=initial_slns, mod_init=man)


def harmony_search(objective_function, initial_harmonies=None, hyperparameters=None, **kwargs):
    """ Worker for hamony search, 1 cpu process

    Args:
        objective_function (_type_): _description_
        initial_harmonies (_type_, optional): _description_. Defaults to None.
        hyperparameters (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: results for the harmony search algorithm in single or multi objective
    """
    #set up objective function hyper
    if kwargs.get('_hms') is not None:
        objective_function._hms = kwargs.get('_hms')
    if kwargs.get('_hmcr') is not None:
        objective_function._hmcr = kwargs.get('_hmcr')
    try:
        objective_function.instance_name = f"run_hs_{str(0)}"
    except:
        pass

    man = None
    if 'Manual_Fit' in kwargs:
        if kwargs['Manual_Fit'] is not None:
            man = kwargs['Manual_Fit']
    start = datetime.now()
    if 'MP' in kwargs and kwargs['MP']:
        hs = Mutlithreaded_Meta(objective_function)
        results = hs.run_hs_mp(initial_harmonies, mod_init=man)
    else:

        hs = HarmonySearch(objective_function)
        results = hs.run(initial_harmonies, mod_init=man)
    end = datetime.now()
    elapsed_time = end - start
    if objective_function.is_multi:
        harmony, fitness, harmony_memory, harmony_history = results
        return HarmonySearchResults(elapsed_time=elapsed_time, best_harmony=harmony, best_fitness=fitness,
                                    harmony_memories=harmony_memory, harmony_histories=harmony_history)
    else:
        harmony, fitness, harmony_memory, harmony_history, iter, iter_fitness, fitness_collection = results
        return HarmonySearchResults(elapsed_time, harmony, fitness, harmony_memory, harmony_history)


# ---------------------------------------------------------------------- #
#  SparseEA-AGDS: NSGA-III reference-point helpers (numpy, self-contained)#
# ---------------------------------------------------------------------- #
def _agds_reference_points(n_obj, divisions):
    """Das-Dennis structured reference points on the (n_obj-1)-simplex."""
    def _recurse(left, depth):
        if depth == n_obj - 1:
            return [[left]]
        pts = []
        for i in range(left + 1):
            for tail in _recurse(left - i, depth + 1):
                pts.append([i] + tail)
        return pts
    if n_obj == 1:
        return np.array([[1.0]])
    raw = _recurse(divisions, 0)
    return np.array(raw, dtype=float) / float(divisions)


def _agds_perp_distance(points, ref, assoc):
    """Perpendicular distance of each point to its associated reference line."""
    d = np.empty(len(points))
    for i, p in enumerate(points):
        w = ref[assoc[i]]
        wn = w / (np.linalg.norm(w) + 1e-12)
        proj = np.dot(p, wn) * wn
        d[i] = np.linalg.norm(p - proj)
    return d


def _agds_associate(points, ref):
    """Associate each point with the nearest reference line (index into ref)."""
    if len(points) == 0:
        return np.array([], dtype=int)
    ref_norm = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
    assoc = np.empty(len(points), dtype=int)
    for i, p in enumerate(points):
        proj = points[i] @ ref_norm.T                     # scalar projections
        perp = np.linalg.norm(
            p[None, :] - proj[:, None] * ref_norm, axis=1)
        assoc[i] = int(np.argmin(perp))
    return assoc


def sparse_ea_agds(objective_function, initial_slns=None, **kwargs):
    """ Worker for the SparseEA-AGDS solution algorithm (single process).

    Mirrors :func:`harmony_search`: builds a :class:`SparseEA_AGDS`, runs it,
    and returns a :class:`HarmonySearchResults` so ``helperprocess.results_printer``
    works unchanged for both single- and multi-objective searches.
    """
    if kwargs.get('_hms') is not None:
        objective_function._hms = kwargs.get('_hms')
    try:
        objective_function.instance_name = f"run_agds_{str(0)}"
    except Exception:
        pass

    man = None
    if kwargs.get('Manual_Fit') is not None:
        man = kwargs['Manual_Fit']

    start = datetime.now()
    agds = SparseEA_AGDS(objective_function, **kwargs)
    results = agds.run(initial_slns, mod_init=man)
    elapsed_time = datetime.now() - start

    if objective_function.is_multi:
        best_layout, best_struct, pareto_memory, history = results
        return HarmonySearchResults(elapsed_time=elapsed_time, best_harmony=best_layout,
                                    best_fitness=best_struct, harmony_memories=pareto_memory,
                                    harmony_histories=history)
    else:
        best_layout, best_fitness, memory, history, _it, _inc, _bst = results
        return HarmonySearchResults(elapsed_time, best_layout, best_fitness, memory, history)


class Metaheuristic(object):
    def __init__(self, objective_function, **kwargs):
        self._obj_fun = objective_function
        self._num_intl_slns = 10
        self._sa_memory = list()

        self._pop_size = kwargs.get('_pop_size', 20)

        self.F = kwargs['_AI']  # mustation scale
        self.iter = kwargs.get('_max_iter', 10000)
        self.cr = kwargs.get('_crossover_perc') or kwargs.get('_cr', 0.2)
        self.instance_name = str(kwargs.get('instance_name', 1))
        if objective_function.is_multi:

            self.obj_1 = objective_function._obj_1
            self.obj_2 = objective_function._obj_2

            self.pf = Pareto(self.obj_1, self.obj_2, True)

            self._pareto_population = list()
        else:
            self.obj_1 = objective_function._obj_1
            self.obj_2 = objective_function._obj_2
            self.pf = Pareto(self.obj_1, self.obj_2, False)

    def _random_selection(self, sln, i):
        """
            Choose a note according to get_value(). Remember that even if a note is not variable, get_value() must still
            return a valid value.
        """
        sln.append(self._obj_fun.get_value(i))

    def _initialize(self, initial_slns=None, model_nature=None):
        if model_nature is not None:
            a = self._obj_fun.modify_initial_fit(model_nature)
            vector = self._obj_fun.reconstruct_vector(a)

        if initial_slns is not None:
            # verify that the initial harmonies are provided correctly
            if len(initial_slns) != self._num_intl_slns:
                raise ValueError('Number of initial solutions does not equal initial solution size.')
            num_parameters = self._obj_fun.get_num_parameters()
            for i in range(len(initial_slns)):
                num_parameters_initial_harmonies = len(initial_slns[i])
                if num_parameters_initial_harmonies != num_parameters:
                    raise ValueError('Number of parameters in initial solutions does not match that defined.')
        else:
            initial_slns = list()
            for i in range(0, self._num_intl_slns):
                sln = list()
                for j in range(0, self._obj_fun.get_num_parameters()):
                    self._random_selection(sln, j)
                initial_slns.append(sln)

        initial_list = list()
        if model_nature is not None:
            initial_slns[0] = vector
            # embedding Krishnas model
        print('number of initial solns', self._num_intl_slns)
        for i in range(0, self._num_intl_slns):
            print('evaluating initial sln', i)
            try:
                fitness = self._obj_fun.get_fitness(initial_slns[i], multi=self.pf.get_objective_is_multi(),
                                                    max_routine=2)
                if self.pf.get_objective_is_multi():
                    self.pf.add_Structs(fitness)
                else:
                    if np.isnan(fitness[self._obj_fun._obj_1]):
                        fitness = 10 ** 9
            except Exception as e:
                fitness = 10 ** 9
                print(e, 'fitness eror meta')
                # print('solution struct', initial_slns[i])
            print('the final fitness is', fitness)
            if self.pf.get_objective_is_multi():
                initial_list.append((initial_slns[i], fitness))
            else:
                if isinstance(fitness, dict):
                    initial_list.append((initial_slns[i], fitness))
                else:
                    initial_list.append((initial_slns[i], fitness))
                self._sa_memory.append(self._obj_fun.Last_Sol)

        # print(initial_list)
        if self.pf.get_objective_is_multi():
            self.pf.run()
            print('running initial pareto sort')
        else:
            self.pf.pareto_run(self._sa_memory, True)
        return initial_list


class DifferentialEvolution(object):
    """
        This class implements the Differential Evolution Algorithm
    """

    def __init__(self, objective_function, **kwargs):
        objective_function.algorithm = 'de'
        try:
            objective_function.instance_name = str(0)
        except:
            pass
        self._obj_fun = objective_function
        if self._obj_fun._obj_1 is None:
            print('no objective found, automatically selecting BIC')
            self._obj_fun._obj_1 = 'bic'

        self._pop_size = kwargs.get('_pop_size', 20)
        print('Population size is', self._pop_size)
        if not isinstance(self._pop_size, int):
            raise ValueError("_pop_size must be an integer")
        elif self._pop_size <= 3:
            raise ValueError("_pop_size must be greater than 4")

        self.F = kwargs.get('_AI', 2)  # mutation scale
        self.iter = kwargs.get('_max_iter', 10000)
        self.cr = kwargs.get('_crossover_perc') or kwargs.get('_cr', 0.2)
        self.instance_name = str(kwargs.get('instance_name', 1))
        self.instance_name = objective_function.instance_name
        self.get_directory()

        self._population = list()
        self.it_process = 1
        if objective_function.is_multi:
            self.obj_1 = objective_function._obj_1
            self.obj_2 = objective_function._obj_2
            self.pf = Pareto(self.obj_1, self.obj_2, True)
            self._pareto_population = list()
        else:
            self.obj_1 = objective_function._obj_1
            self.obj_2 = objective_function._obj_2
            self.pf = Pareto(self.obj_1, self.obj_2, False)

    def get_directory(self):
        # checking if the directory demo_folder2 
        # exist or not.
        if not os.path.isdir(self.instance_name):
            # if the demo_folder2 directory is
            # not present then create it.
            os.makedirs(self.instance_name)

    def get_instance_name(self):
        name = str(self.instance_name) + '/log.csv'
        return name

    def _random_selection(self, sln, i):
        """
            Choose a note according to get_value(). Remember that even if a note is not variable, get_value() must still
            return a valid value.
        """
        sln.append(self._obj_fun.get_value(i))

    def choose(self, j):
        candidates = [candidate for candidate in range(self._pop_size) if candidate != j]
        # print(candidates)
        aa, bb, cc = np.random.choice(candidates, 3, replace=False)
        # print(aa, ' and ', bb, ' and ', cc)
        # print(self._population[aa][0])
        if self._obj_fun.is_multi:
            a = self._population[j]['layout']
            b = self._population[bb]['layout']
            c = self._population[cc]['layout']
            return a, b, c
        else:

            a = self._population[aa]['layout']
            b = self._population[bb]['layout']
            c = self._population[cc]['layout']
            return a, b, c

    def mixed_list_checker(self, a, b):
        mixed_list_a = [x for x in a if not isinstance(x, str)]
        mixed_list_b = [x for x in b if not isinstance(x, str)]
        if mixed_list_a == mixed_list_b:
            mixed_str_a = [x for x in a if isinstance(x, str)]
            mixed_str_b = [x for x in b if isinstance(x, str)]
            if mixed_str_a == mixed_str_b:
                return True
        else:
            return False

    def hard_mutate_index_and_value(self):
        mutate_possible = self._obj_fun.get_indexes_of_ints()
        random_index = random.choice(mutate_possible)

        inject_this = self._obj_fun._discrete_values[random_index][
            random.randint(0, len(self._obj_fun._discrete_values[random_index]) - 1)]
        return random_index, inject_this

    def mutation(self, struct_a, struct_b, struct_c) -> list:

        new_a = struct_a.copy()
        endless = 0
        same = True

        while same == True:
            for i in range(0, len(struct_a)):

                try:
                    current_index = self._obj_fun.get_index(i, struct_a[i])
                    new_index = current_index + self.F * (
                            self._obj_fun.get_index(i, struct_b[i]) - self._obj_fun.get_index(i, struct_c[i]))
                except Exception as e:
                    print(e, 'index errpr [rpb;es jere]')
                    print(struct_b)
                    print(struct_c)
                    new_index = 0
                if new_index not in range(0, len(self._obj_fun._discrete_values[i])):
                    new_index = self._obj_fun.modulo_or_divisor(new_index, len(self._obj_fun._discrete_values[i])) - 1

                new_a[i] = self._obj_fun.get_value(i, new_index)

            endless += 1
            same = self.mixed_list_checker(new_a, struct_a)
            if same:
                random_index, inject_this = self.hard_mutate_index_and_value()
                print(new_a, 'before')
                new_a[random_index] = inject_this
                print(new_a, 'after')
                same = self.mixed_list_checker(new_a, struct_a)
            if endless > 10:
                print('endless loop', endless, 'lets break')
                break
        return new_a

    # defintion of the crossover operator
    def does_it_appear(self, new):
        for d in self._population:
            if self.mixed_list_checker(d['layout'], new):
                return True
        return False

    def crossover(self, mutated, target, dims, cr_r):
        # generate a uniform random value for every dimension
        repeats = 0
        # generate trial vector by binomial crossover
        unique = True
        trial = target
        cr = cr_r
        while unique:
            p = np.random.rand(dims)
            trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
            unique = self.does_it_appear(trial)
            if unique:
                rdm_i, inject_this = self.hard_mutate_index_and_value()
                trial[rdm_i] = inject_this
                unique = self.does_it_appear(trial)

            cr += .1
            repeats += 1
            if repeats > 10:
                print('breaking.. Too Many repeats')
                break

        return trial

    def differential_evolution_run(self, initial_slns=None, mod_init=None):
        # set optional random seed
        average_iteration = 0
        iterations_without_improvement = 0

        start_time = datetime.now()
        if self._obj_fun.use_random_seed():
            self._obj_fun.set_random_seed()

        # fill randomly, but with initial_slns if provided
        self._population = self._initialize(initial_slns, mod_init)
        if self._obj_fun.is_multi:
            self._pareto_population = self._population.copy()
            self._pareto_population = self.pf.pareto_run(self._pareto_population)  # FIXME shrinking length
            self.pf.update_means()
        # print(self._population)
        self._sort_memory()
        if self._obj_fun.is_multi:
            best_solution = self._population[0]
            best_struct = self._population[0]['layout']
            prev_solution = best_solution
        else:
            best_solution = self._population[0]
            best_struct = self._population[0]['layout']
            prev_solution = best_solution
        it_best = Solution()
        # initialise list to store the objective function value at each iteration
        obj_iter = list()
        # run iterations of the algorithm
        i = 0
        time_elapsed = 0

        print('The maximum run time of this algorithm is ', self._obj_fun.get_max_time())

        while (i <= self.iter + 1) and (time_elapsed <= self._obj_fun.get_max_time()) and (
                iterations_without_improvement < self._obj_fun.get_termination_iter()):
            time_elapsed = (datetime.now() - start_time).total_seconds()
            print('current time elapsed is', time_elapsed)

            # iterate over population

            for j in range(self._pop_size):
                pop_a, pop_b, pop_c = self.choose(j)
                # perform mutation
                mutated = self.mutation(pop_a, pop_b, pop_c)
                if self._obj_fun.is_multi:
                    trial = self.crossover(mutated, self._population[j]['layout'], len(self._population[j]['layout']),
                                           self.cr)
                else:
                    trial = self.crossover(mutated, self._population[j]['layout'], len(self._population[j]['layout']),
                                           self.cr)
                try:
                    obj_trial = self._obj_fun.get_fitness(trial, self.pf.get_objective_is_multi())
                    if not self.pf.get_objective_is_multi():
                        average_iteration += obj_trial[self._obj_fun._obj_1]
                        try:

                            logger(i, obj_trial, None, True, self.get_instance_name(), 1)
                        except:
                            pass
                    else:
                        try:
                            logger(i, obj_trial, None, True, self.get_instance_name(), self.pf.get_objective_is_multi())
                        except:    
                            pass
                except Exception as e:
                    print('why is there an exception')
                    print(e)
                    continue

                i += 1
                if self.pf.get_objective_is_multi():
                    is_updated, self._pareto_population = self.pf.check_if_dominance(self._pareto_population, obj_trial,
                                                                                     1)

                    if len(self._pareto_population) == 1:
                        print('Pareto Population Size is only 1')
                    if self.pf.check_dominance([obj_trial[self.pf.obj_key_1], obj_trial[self.pf.obj_key_2]],
                                               [self._population[j][self.pf.obj_key_1], self._population[j][
                                                   self.pf.obj_key_2]]):  # if solution dominates existing #FIXME some error here true but not entering

                        iterations_without_improvement = 0
                        self._population[j] = obj_trial
                        
                        logger(self.it_process, obj_trial, self._population, True,
                               self.instance_name + '/population_logger_strict_non_pareto.csv', 1)
                        logger(self.it_process, obj_trial, self._pareto_population, True,
                               self.instance_name + '/population_logger_pareto.csv', 1)
                    else:
                        if self.pf.calculate_difference(obj_trial, self._population[j]):
                            iterations_without_improvement = 0
                            self._population[j] = obj_trial
                            self._pareto_population = self.pf.Pareto_F
                            logger(self.it_process, obj_trial, self._population, True,
                                   self.instance_name + '/population_logger_strict_non_pareto.csv', 1)
                            logger(self.it_process, obj_trial, self._pareto_population, True,
                                   self.instance_name + '/population_logger_pareto.csv', 1)

                    if it_best is None:
                        it_best = obj_trial
                    else:
                        if self.pf.weighted_obj(obj_trial) < self.pf.weighted_obj(it_best):
                            it_best = obj_trial




                else:
                    if obj_trial[self._obj_fun._obj_1] < self._population[j][self._obj_fun._obj_1]:
                        iterations_without_improvement = 0
                        self._population[j] = obj_trial

                    if it_best is None:
                        it_best = obj_trial
                    else:
                        if obj_trial[self._obj_fun._obj_1] < it_best[self.obj_1]:
                            iterations_without_improvement = 0
                            it_best = obj_trial
                    if obj_trial[self._obj_fun._obj_1] < best_solution[self.obj_1]:
                        best_solution = obj_trial

                    logger(self.it_process, obj_trial, best_solution, name=self.get_instance_name(),
                           local_best1=it_best)

                self.it_process += 1

                # self._sort_memory()  # should I sort
            self._sort_memory()
            best_solution = self._population[0]
            best_struct = self._population[0]['layout']

            iterations_without_improvement += 1

            obj_iter.append((i, it_best[self.obj_1], best_solution[self.obj_1], average_iteration / self._pop_size))
            average_iteration = 0
            # print('Iteration: ', i, ' with best objective: ', best_solution)
            # print("Iterations without improvement currently", iterations_without_improvement)
            it_best = Solution()
        if self._obj_fun.is_multi:
            self._pareto_population = self.pf.pareto_run(self._pareto_population)
            return self._pareto_population, self._population

        else:
            output_a = list()
            output_b = list()
            output_c = list()
            output_d = list()
            for a, b, c, d in obj_iter:
                output_a.append(a)
                output_b.append(b)
                output_c.append(c)
                output_d.append(d)
            return output_a, output_b, output_c, best_solution, best_struct

    def _sort_memory(self):
        '''sorts the population into a list of good solutions'''
        if self._obj_fun.is_multi:
            b_size = len(self._population)
            self._population = self.pf.non_dominant_sorting(self._population)
            if b_size != len(self._population):
                raise Exception
        else:
            if self._obj_fun.maximize():
                self._population.sort(key=lambda x: x[self._obj_fun._obj_1], reverse=True)
            else:
                self._population.sort(key=lambda x: x[self._obj_fun._obj_1])

    def _initialize(self, initial_slns=None, model_nature=None):
        """
            Initialize slns, the matrix (list of lists) containing the various harmonies (solution vectors). Note
            that we aren't actually doing any matrix operations, so a library like NumPy isn't necessary here. The matrix
            merely stores previous harmonies.
            If harmonies are provided, then use them instead of randomly initializing them.
            Populate harmony_history with initial harmony memory.
        """
        if model_nature is not None:
            a = self._obj_fun.modify_initial_fit(model_nature)
            # self._obj_fun.makeRegression(a)
            vector = self._obj_fun.reconstruct_vector(a)

        if initial_slns is not None:
            # verify that the initial harmonies are provided correctly
            if len(initial_slns) != self._pop_size:
                raise ValueError('Number of initial solutions does not equal initial solution size.')
            num_parameters = self._obj_fun.get_num_parameters()
            for i in range(len(initial_slns)):
                num_parameters_initial_harmonies = len(initial_slns[i])
                if num_parameters_initial_harmonies != num_parameters:
                    raise ValueError('Number of parameters in initial solutions does not match that defined.')
        else:
            initial_slns = list()
            for i in range(0, self._pop_size):
                sln = list()
                for j in range(0, self._obj_fun.get_num_parameters()):
                    self._random_selection(sln, j)

                initial_slns.append(sln)

        initial_list = list()
        if model_nature is not None:
            initial_slns[0] = vector

        for i in range(0, self._pop_size):

            fitness = self._obj_fun.get_fitness(initial_slns[i], self.pf.get_objective_is_multi(), max_routine=2)

            if i % 10 == 10:
                print('evaluating initial sln', i)
                print('the final fitness is', fitness)
            if self.pf.get_objective_is_multi():
                initial_list.append(fitness)
                logger(i, fitness, None, True, self.get_instance_name(), self.pf.get_objective_is_multi())
            else:
                if isinstance(fitness, dict):
                    initial_list.append(fitness)
                else:
                    raise Exception('should not be possible')

        return initial_list


class SimulatedAnnealing(object):
    """
            This class implements the simulated annealing algorithm. In general, what you'll do is this:
            1. Implement an objective function that inherits from ObjectiveFunctionInterface.
        """

    def __init__(self, objective_function, **kwargs):
        objective_function.algorithm = 'sa'
        self._STEPS_PER_TEMP = int(kwargs.get('STEPS_PER_TEMP', 2)) or int(kwargs.get('_ts', 2))
        self._INITAL_ACCEPT_RATE = float(kwargs.get('INTL_ACPT', 0.5))
        self._NUM_INITIAL_SLNS = int(kwargs.get('_num_intl_slns', 20))
        self._alpha = float(kwargs.get('alpha', 0.95)) or float(kwargs.get('a', 0.95))
        self.best_energy = None
        self.best_struct = None
        self._current_energy = 10000
        self.current_struct = None
        self._temp_max = 1000000
        self.temp_min = 0.05
        self._MAX_ITERATIONS = int(kwargs.get('MAX_ITERATIONS', 10000)) or int(kwargs.get('_max_iter', 10000))

        self.instance_name = str(objective_function.instance_name)
        self.accept = 0
        self.profiler = []
        self.update_t = self.cooling_linear_m
        self.get_directory()
        self._crossover_perc = float(kwargs.get('_crossover_perc', 0.2)) or float(kwargs.get('_cr', 0.2))
        self._obj_fun = objective_function
        if objective_function.is_multi:  # TODO Define more specific objectives in the intialiser
            self.obj_1 = objective_function._obj_1
            self.obj_2 = objective_function._obj_2
            self.pf = Pareto(self.obj_1, self.obj_2)
        else:
            self.obj_1 = objective_function._obj_1
            self.obj_2 = objective_function._obj_2

            self.pf = Pareto(self.obj_1, self.obj_2, False)
            self._sa_memory = list()

    def get_directory(self):
        # checking if the directory demo_folder2 
        # exist or not.
        if not os.path.isdir(self.instance_name):
            # not present then create it.
            os.makedirs(self.instance_name)

    def get_instance_name(self):
        name = str(self.instance_name) + '/log.csv'
        return name

    def run(self, initial_slns=None, mod_init=None):
        """
            This is the main SA loop. It initializes the random solutions and then continually generates new solutions until the stopping
            criteria is reached.
        """
        # generational
        generation_best = 10 ** 10
        iterations_without_improvement = 0
        start_time = datetime.now()
        # fill randomly, but with initial_slns if provided
        initial_slns_struct = self._initialize(initial_slns, mod_init)
        # Calculate the init   ial temperature, and seed this instance so it is always the same
        if self._obj_fun.use_random_seed():
            self._obj_fun.set_random_seed()
        self._temp_max = self.Calculate_Temp(initial_slns_struct, acceptance_prob=self._INITAL_ACCEPT_RATE,
                                             multi=self.pf.get_objective_is_multi())
        if self.best_energy is None:
            raise Exception
        else:
            self.current_struct, self._current_energy = initial_slns_struct[0]

        current_temperature = self._temp_max
        print('The Temperature is', current_temperature)
        # set optional random seed

        iteration = 0
        temperature_iteration = 0  # counting mechanism to update the temperature
        # Get Current Solution and Fitness
        for struct, fitness in initial_slns_struct:
            if self.pf.get_objective_is_multi():
                try:
                    fit_ob_1 = fitness.get(self.obj_1)  # TODO better way to get the objectives
                    fit_ob_2 = fitness.get(self.obj_2)
                    self.pf.evaluate_frontier_against_new_sln(fitness)
                except Exception as e:
                    print(e, 'sa error part')



            else:
                if fitness[self._obj_fun._obj_1] <= self.best_energy[self._obj_fun._obj_1]:
                    self.best_energy = fitness
                    self.best_struct = struct
                    self._current_energy = fitness
                    self.current_struct = struct
        # start from the best solution if analyst did not specify
        if self._obj_fun.solution_analyst is not None:
            self._current_energy = self._obj_fun.solution_analyst[self.obj_1]  # TODO handle both objectives
            self.current_struct = self._obj_fun.solution_analyst['layout']
        else:
            if self.pf.get_objective_is_multi():  # TODO deterimine the best solion in the pareto
                best_sln = self.pf.Pareto_F[0]
                # best_sln = self.pf.find_bestsol(self.pf.Pareto_F)
                self.current_struct = best_sln['layout']
                self._current_energy = best_sln
                self.best_struct = best_sln['layout']
                self.best_energy = best_sln
                # print('does this work')

            else:

                self.pf.run(self._obj_fun.Last_Sol)
                self._current_energy = self.best_energy  # TODO refactor to solution
                self.current_struct = self.best_struct
        # Main Algorithm
        elapsed_time = 0

        if not isinstance(self.current_struct, list):
            raise Exception
        while (iteration <= self._MAX_ITERATIONS) and (
                elapsed_time <= self._obj_fun.get_max_time()) and (
                iterations_without_improvement < self._obj_fun.get_termination_iter()):
            elapsed_time = (datetime.now() - start_time).total_seconds()
            for i in range(self._STEPS_PER_TEMP):
                # Get Neighbouring solution and Energy level
                # print('the current struct is ' ,self.current_struct)
                nbr_struct = self._get_neighbour(self.current_struct,
                                                 round(self._crossover_perc * len(self.current_struct)))
                nbr_energy = self._obj_fun.get_fitness(nbr_struct, multi=self.pf.get_objective_is_multi())

                # Acceptance Criteria
                if random.uniform(0, 1) <= self.Energy_Acceptance_Criteria(nbr_energy, self._current_energy,
                                                                           current_temperature,
                                                                           self.pf.get_objective_is_multi()):
                    # print('Accepted New with fitness of', nbr_energy)
                    didchange = self.pf.did_it_change()
                    if didchange:
                        pareto_logger(self.pf.Pareto_F, iteration, self._obj_fun.complexity_level,
                                      self._obj_fun.instance_name)
                    self._current_energy = nbr_energy
                    self.current_struct = nbr_struct
                    self.accept += 1
                    iterations_without_improvement = 0
                    # Check if neigbouring solution is the best      
                    if not self.pf.get_objective_is_multi():  # if this is the global best (show) #TODO determine global best for multiobjective
                        if nbr_energy[self._obj_fun._obj_1] < self.best_energy[self._obj_fun._obj_1]:
                            print('new best at iteration ', iteration, 'with fitness of', nbr_energy)
                            self.best_energy = nbr_energy
                            self.best_struct = nbr_struct
                            iterations_without_improvement = 0
                    else:
                        if nbr_energy['layout'] == self.pf.Pareto_F[0]['layout']:
                            print('new best at iteration ', iteration, 'with fitness of', nbr_energy)
                            self.best_energy = nbr_energy
                            self.best_struct = nbr_struct
                            iterations_without_improvement = 0
                    # log data
                    iteration += 1
                    self.profiler.append([iteration, self._current_energy, self.best_energy])
                else:

                    iteration += 1

                try:
                    logger(iteration, nbr_energy, self.best_energy, name=self.get_instance_name(),
                           local_best1=self._current_energy)
                except Exception as e:
                    print('THEREE IS AN ERRPR OM TJE ;PGGER', e)
                    # update temperature
            temperature_iteration += 1
            current_temperature = self.update_t(temperature_iteration)
            iterations_without_improvement += 1
        acceptance_rate = self.accept / iteration
        print('the acceptance rate is', acceptance_rate * 100, '%')
        self._obj_fun.get_fitness(self.best_struct, multi=self.pf.get_objective_is_multi(), verbose=True)
        print('best', self.best_struct)
        output_step = list()
        output_energy = list()
        output_best_energy = list()

        for a, b, c in self.profiler:
            output_step.append(a)
            output_energy.append(b)
            output_best_energy.append(c)

        return {'elapsed_time': elapsed_time, 'Iteration': iteration}  # TODO make this reachavble
        # return output_step, output_energy, output_best_energy, self.best_energy, self.best_struct

    def _get_neighbour(self, current, mutations=None):
        neighbour = copy.deepcopy(current)
        # get the values form the number of paramaters
        prmVect = neighbour[:self._obj_fun._characteristics]
        if not isinstance(prmVect, list):
            raise Exception

        # number of paramaters in the model #TODO get the last value if 2

        num_of_changeablePARMs = 0

        self._obj_fun.nbr_routine(current)
        nParm = self._obj_fun.get_param_num()

        if mutations is None:
            rdm_i = random.randint(0, self._obj_fun.get_num_parameters() - 1)
            if nParm >= self._obj_fun._max_characteristics:
                # weights = nParm.resize(neighbour.shape, refcheck = False)
                weights = prmVect
                if np.all(weights) == 0:
                    weights = [1 for _ in weights]

                if not isinstance(weights, list):
                    raise Exception
                rdm_i = random.choices(range(len(prmVect)), weights=weights)[0]

            while self._obj_fun.get_num_discrete_values(rdm_i) <= 1:
                rdm_i = random.randint(0, self._obj_fun.get_num_parameters() - 1)
            # print('index is', rdm_i)
            nbr_i = neighbour[rdm_i]
            new_nbr_i = neighbour[rdm_i]
            while nbr_i == new_nbr_i:
                # new_nbr_i = self._obj_fun.get_value(rdm_i)
                get_rdm_j = random.randint(0, self._obj_fun.get_num_discrete_values(rdm_i) - 1)
                # print(get_rdm_j)
                new_nbr_i = self._obj_fun.get_value(rdm_i, get_rdm_j)
            neighbour[rdm_i] = new_nbr_i
        else:
            for i in range(mutations):
                allGood = 0
                while allGood == 0:
                    rdm_i = random.randint(0, self._obj_fun.get_num_parameters() - 1)
                    if nParm >= self._obj_fun._max_characteristics:
                        weights = prmVect
                        rdm_i = random.choices(range(len(prmVect)), weights=weights, k=1)[0]
                    elif nParm <= self._obj_fun._min_characteristics:
                        weights = [1 if x == 0 else 0 for x in prmVect]
                        if np.all(weights) == 0:
                            weights = [1 for _ in weights]

                        rdm_i = random.choices(range(len(prmVect)), weights=weights, k=1)[0]
                    elif num_of_changeablePARMs == 0:
                        rdm_i = random.choice(range(len(prmVect)))
                        if self._obj_fun.get_num_discrete_values(rdm_i) <= 1:
                            print('retry, not a enough choices')

                    while self._obj_fun.get_num_discrete_values(rdm_i) <= 1:
                        rdm_i = random.randint(0, self._obj_fun.get_num_parameters() - 1)

                    nbr_i = neighbour[rdm_i]
                    new_nbr_i = neighbour[rdm_i]

                    if rdm_i < (len(prmVect)):

                        if nParm + 1 > self._obj_fun._max_characteristics:

                            if nbr_i != 0:
                                if rdm_i == len(prmVect) - 1:  # if it is the dispersion paramater force Poisson
                                    neighbour[rdm_i] = 0
                                else:
                                    neighbour[rdm_i] -= 1  # else redued 1
                                allGood = 1
                                num_of_changeablePARMs += 1
                                nParm -= 1
                                prmVect = neighbour[:self._obj_fun._characteristics]
                            else:
                                allGood = 0

                        else:
                            while nbr_i == new_nbr_i:
                                # new_nbr_i = self._obj_fun.get_value(rdm_i)
                                get_rdm_j = random.randint(0, self._obj_fun.get_num_discrete_values(rdm_i) - 1)

                                new_nbr_i = self._obj_fun.get_value(rdm_i, get_rdm_j)
                                neighbour[rdm_i] = new_nbr_i
                                allGood = 1
                            num_of_changeablePARMs += 1
                            nParm += new_nbr_i - nbr_i


                    else:
                        while nbr_i == new_nbr_i:
                            # new_nbr_i = self._obj_fun.get_value(rdm_i)
                            get_rdm_j = random.randint(0, self._obj_fun.get_num_discrete_values(rdm_i) - 1)
                            if (self._obj_fun.get_num_discrete_values(
                                    rdm_i) - 1) < 1:  # TODO: remove this is just a test
                               
                                break
                            new_nbr_i = self._obj_fun.get_value(rdm_i, get_rdm_j)
                            neighbour[rdm_i] = new_nbr_i
                            allGood = 1
                        if rdm_i == len(prmVect) - 1:
                            if new_nbr_i == 2 or new_nbr_i == 1:  # check for number of dispersion paramaters
                                bi = 1
                            else:
                                bi = 0
                            if nbr_i == 2 or nbr_i == 1:
                                ai = 1
                            else:
                                ai = 0
                            nParm += bi - ai

            if num_of_changeablePARMs == 0:
                print('check this line in particular 631')
        return neighbour

    def _initialize(self, initial_slns=None, model_nature=None):
        """
            Initialize slns, the matrix (list of lists) containing the various harmonies (solution vectors). Note
            that we aren't actually doing any matrix operations, so a library like NumPy isn't necessary here. The matrix
            merely stores previous harmonies.
            If harmonies are provided, then use them instead of randomly initializing them.
            Populate harmony_history with initial harmony memory.
        """
        if model_nature is not None:
            a = self._obj_fun.modify_initial_fit(model_nature)
            # self._obj_fun.makeRegression(a)
            vector = self._obj_fun.reconstruct_vector(a)

            # vector = self._obj_fun.get_solution_vector(a['fixed_fit'], a['rdm_fit'], [], a['dispersion'])
            # print(1)

        if initial_slns is not None:
            # verify that the initial harmonies are provided correctly
            if len(initial_slns) != self._NUM_INITIAL_SLNS:
                raise ValueError('Number of initial solutions does not equal initial solution size.')
            num_parameters = self._obj_fun.get_num_parameters()
            for i in range(len(initial_slns)):
                num_parameters_initial_harmonies = len(initial_slns[i])
                if num_parameters_initial_harmonies != num_parameters:
                    raise ValueError('Number of parameters in initial solutions does not match that defined.')
        else:
            initial_slns = list()
            for i in range(0, self._NUM_INITIAL_SLNS):
                sln = list()
                for j in range(0, self._obj_fun.get_num_parameters()):
                    self._random_selection(sln, j)
                initial_slns.append(sln)
        initial_list = list()
        if model_nature is not None:
            initial_slns[0] = vector
            # embedding Krishnas model
        print('number of initial solns', self._NUM_INITIAL_SLNS)
        for i in range(0, self._NUM_INITIAL_SLNS):
            print('evaluating initial sln', i)
            try:
                fitness = self._obj_fun.get_fitness(initial_slns[i], multi=self.pf.get_objective_is_multi(),
                                                    max_routine=2)
                if self.pf.get_objective_is_multi():
                    self.pf.add_Structs(fitness)
                else:
                    if np.isnan(fitness[self._obj_fun._obj_1]):
                        fitness = 10 ** 9
            except Exception as e:
                fitness = 10 ** 9
                print(e, 'sa here co')
                # print('solution struct', initial_slns[i])
            print('the final fitness is', fitness)
            if self.pf.get_objective_is_multi():
                initial_list.append((initial_slns[i], fitness))
            else:
                if isinstance(fitness, dict):
                    initial_list.append((initial_slns[i], fitness))
                else:
                    initial_list.append((initial_slns[i], fitness))
                self._sa_memory.append(self._obj_fun.Last_Sol)

        # print(initial_list)
        if self.pf.get_objective_is_multi():
            self.pf.run()
            print('running initial pareto sort')
        else:
            self.pf.pareto_run(self._sa_memory, True)
        return initial_list

    def _random_selection(self, sln, i):
        """
            Choose a note according to get_value(). Remember that even if a note is not variable, get_value() must still
            return a valid value.
        """
        sln.append(self._obj_fun.get_value(i))

    def Calculate_Temp(self, slns, acceptance_prob, multi=False):
        fitness_list = list()

        feasibility = 20000
        if multi is False:
            for gene, fitness in slns:

                if fitness.get(self._obj_fun._obj_1) < feasibility:
                    fitness_list.append(fitness.get(self._obj_fun._obj_1))
                    if self.best_energy and fitness.get(self._obj_fun._obj_1) <= self.best_energy.get(
                            self._obj_fun._obj_1):
                        self.best_energy = fitness
                    elif self.best_energy is None:
                        self.best_energy = fitness
            "Initial Temp is"

            return (-st.stdev(fitness_list)) / np.log(1 - acceptance_prob)
        else:
            fitness_list_2 = list()
            for gene, fitness in slns:
                try:
                    fitness_list.append(
                        fitness.get(self._obj_fun._obj_1))  # TODO handle specific swithhing of objectives.
                    fitness_list_2.append(fitness.get(self._obj_fun._obj_2))
                except Exception as e:
                    print(e, 
                          'fitness list eror')

                    # Temp1
            Temp1 = (-st.stdev(fitness_list)) / np.log(1 - acceptance_prob)
            Temp2 = (-st.stdev(fitness_list_2)) / np.log(1 - acceptance_prob)

            # Weight of the temperatures
            w_1 = sum(fitness_list) / (sum(fitness_list) + sum(fitness_list_2))
            w_2 = sum(fitness_list_2) / (sum(fitness_list) + sum(fitness_list_2))
            Temp3 = (Temp1 + Temp2) / (w_1 * Temp2 + w_2 * Temp1)

            self.best_energy = slns[0]
            low_best = 1e5
            for i, val in enumerate(fitness_list):
                low = w_1 * fitness_list[i] + w_2 * fitness_list_2[i]
                if low < low_best:
                    low_best = low
                    self.best_energy = slns[i]

            return Temp3  # return the first temperature for now #TODO 3 phase temperature schema

            # we have too objectives

    def Energy_Acceptance_Criteria(self, proposed, current, temperature, multi_objective=False):
        if not multi_objective:

            if proposed[self._obj_fun._obj_1] < current[self._obj_fun._obj_1]:
                return 1

            try:
                return np.exp(-(proposed[self._obj_fun._obj_1] - current[self._obj_fun._obj_1]) / temperature)
            except:
                return 0
        else:
            dominance = self.pf.check_dominance([proposed[self.pf.obj_key_1], proposed[self.pf.obj_key_2]],
                                                [current[self.pf.obj_key_1], current[
                                                    self.pf.obj_key_2]])  # where proposed and current are list objectives

            if dominance:
                self.pf.evaluate_frontier_against_new_sln(proposed)  # add to prontier
                return 1
            elif self.pf.is_pareto_efficient(proposed, self.pf.Pareto_F):
                self.pf.add_Structs(proposed)
                self.pf.run()
                return 1



            else:

                delta = self.pf.delta_calc(proposed, current)
                return np.exp(-(delta) / temperature)

    def cooling_linear_m(self, step):
        return self._temp_max / (1 + step * self._alpha)


class HarmonySearch(object):
    """
        This class implements the harmony search (HS) global optimization algorithm. In general, what you'll do is this:
        1. Implement an objective function that inherits from ObjectiveFunctionInterface.
        2. Initialize HarmonySearch with this objective function (e.g., hs = HarmonySearch(objective_function)).
        3. Run HarmonySearch (e.g., results = hs.run()).
    """

    def __init__(self, objective_function, **kwargs):
        """
            Initialize HS with the specified objective function. Note that this objective function must implement ObjectiveFunctionInterface.
        """
        objective_function.algorithm = 'hs'
        self._obj_fun = objective_function
        ## NEW CODE, TRYING TO EXCTACT OUT THE PARAMATERS
        self._hms = kwargs.get('_hms', 20)
        self._par = kwargs.get('_par', .30)
        self.F = kwargs.get('_AI', 2)  # mutation scale
        self.iter = kwargs.get('_max_iter', 10000)
        self.cr = kwargs.get('_crossover_perc') or kwargs.get('_cr', 0.2)
        self.instance_name = str(kwargs.get('instance_name', 1))



        # for printing basics metrics
        self.print_verbose = kwargs.get('verbose', False)
        # harmony_memory stores the best hms harmonies
        self._harmony_memory = list()
        # harmony_history stores all hms harmonies every nth improvisations (i.e., one 'generation')
        self._harmony_history = list()
        # saves the best fitness
        self.instance_name = str(objective_function.instance_name)
        self.get_directory()
        self._harmony_trace_best = list()
        self._harmony_trace_incumbent = list()
        if self._obj_fun.is_multi:  # TODO Define more specific objectives in the intialiser
            self.obj_1 = objective_function._get_obj1()
            self.obj_2 = objective_function._get_obj2()

            self.pf = Pareto(self.obj_1, self.obj_2, True)
            self._pareto_harmony_memory = self._harmony_memory.copy()

        else:
            self.obj_1 = objective_function._get_obj1()
            self.obj_2 = objective_function._get_obj2()

            self.pf = Pareto(self.obj_1, self.obj_2, False)

    def get_directory(self):
        # checking if the directory demo_folder2 
        # exist or not.
        if not os.path.isdir(self.instance_name):
            # if the demo_folder2 directory is
            # not present then create it.
            os.makedirs(self.instance_name)

    def get_instance_name(self):
        name = str(self.instance_name) + '/log.csv'
        return name

    def hard_mutate_index_and_value(self):
        mutate_possible = self._obj_fun.get_indexes_of_ints()
        random_index = random.choice(mutate_possible)
        inject_this = random.randint(0, self._obj_fun.get_num_discrete_values(random_index) - 1)
        return random_index, inject_this

    def mixed_list_chescker(self, a, b):
        mixed_list_a = [x for x in a if not isinstance(x, str)]
        mixed_list_b = [x for x in b if not isinstance(x, str)]
        if mixed_list_a == mixed_list_b:
            return True
        else:
            return False

    def does_it_appear(self, new):
        for d in self._harmony_memory:
            if self.mixed_list_chescker(d['layout'], new):
                # print('same sln appears in population')
                return True

        return False

    def run(self, initial_harmonies=None, mod_init=None):
        """
            This is the main HS loop. It initializes the harmony memory and then continually generates new harmonies
            until the stopping criterion (max_imp iterations) is reached.
        """
        # generational
        # generation_best = 10 ** 10
        # improvisation_best = 10 ** 10
        start_time = datetime.now()
        iterations_without_improvement = 0
        # set optional random seed
        if self._obj_fun.use_random_seed():
            self._obj_fun.set_random_seed()
        # fill harmony_memory using random parameter values by default, but with initial_harmonies if provided
        self._initialize(initial_harmonies, mod_init)
        if self.print_verbose: print('Initialization complete')
        if self.pf.get_objective_is_multi():
            self._pareto_harmony_memory = self.pf.non_dominant_sorting(self._harmony_memory)
            generation_best = self._pareto_harmony_memory[0]
            improvisation_best = self._pareto_harmony_memory[0]

        else:
            self._sort_memory()
            generation_best = self._harmony_memory[0]
            improvisation_best = self._harmony_memory[0]
            # create max_imp improvisations
        generation = 0
        num_imp = 0
        elapsed_time = 0

        while (num_imp < self._obj_fun.get_max_imp()) and (
                elapsed_time <= self._obj_fun.get_max_time()) and (
                iterations_without_improvement < self._obj_fun.get_termination_iter()):
            # generate new harmony
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if self.print_verbose:
                print('Time: ', elapsed_time)
                print('Improvisation: ', num_imp)
            harmony = list()

            for i in range(0, self._obj_fun.get_num_parameters()):
                if random.random() < self._obj_fun.get_hmcr():
                    self._memory_consideration(harmony, i)
                    if random.random() < self._obj_fun.get_par():
                        # print('this is', i)
                        # print('i am just testing this, change back to tru')
                        self._pitch_adjustment(harmony, i, True)
                else:
                    self._random_selection(harmony, i)
                    if random.random() < self._obj_fun.get_par():
                        # print('this is', i)
                        # print('i am just testing this, change back to tru')
                        self._pitch_adjustment(harmony, i, False)

            # check if harmony is in memory

            if self.does_it_appear(harmony):

                rand_idx, rand_inj = self.hard_mutate_index_and_value()
                harmony[rand_idx] = rand_inj
                appear = self.does_it_appear(harmony)
                while appear == 1:
                    rand_idx, rand_inj = self.hard_mutate_index_and_value()
                    harmony[rand_idx] = rand_inj
                    appear = self.does_it_appear(harmony)

            if self.pf.get_objective_is_multi():
                fitness = self._obj_fun.get_fitness(harmony, multi=self.pf.get_objective_is_multi())


            else:

                fitness = self._obj_fun.get_fitness(harmony)

            iterations_without_improvement = self._update_harmony_memory(harmony, fitness,
                                                                         iterations_without_improvement,
                                                                         self.pf.get_objective_is_multi())
            num_imp += 1
            if iterations_without_improvement == 0:  # if there is any kind of improvement updae the logs
                if self.print_verbose: print('improvement found at improvisation', num_imp)
                if self.pf.get_objective_is_multi():
                    try:
                        logger(num_imp, fitness, self._harmony_memory, True, self.get_instance_name(),
                               1)  # for consistency
                    except Exception as e:
                        print(e, 'logger run hs')
                    # logger(num_imp, fitness, self._pareto_harmony_memory, True, self.instance_name +'/log_for_pareto_harmony_memory.csv', 1)


                else:
                    # update generation best
                    if generation_best is None:
                        generation_best = fitness
                    if fitness[self.obj_1] < generation_best[self.obj_1]:
                        generation_best = fitness
                        # iterations_without_improvement =0
                    if fitness[self.obj_1] < improvisation_best[self.obj_1]:
                        improvisation_best = fitness
                    # update logs
                    try:
                        logger(num_imp, fitness, improvisation_best, name=self.get_instance_name(),
                               local_best1=generation_best)
                    except Exception as e:
                        print('log broken', e)
            elif not self.pf.get_objective_is_multi():
                # update generation best
                if generation_best is None:
                    generation_best = fitness
                if fitness[self.obj_1] < generation_best[self.obj_1]:
                    generation_best = fitness
                # update logs
                try:
                    logger(num_imp, fitness, improvisation_best, name=self.get_instance_name())
                except Exception as e:
                    print('log broken', e)
                    # save harmonies every nth improvisations (i.e., one 'generation')
            if num_imp % self._obj_fun.get_hms() == 0:

                # self._sort_memory()
                harmony_list = {'gen': generation, 'harmonies': copy.deepcopy(self._harmony_memory)}
                self._harmony_history.append(harmony_list)
                if self.pf.get_objective_is_multi() is False:
                    generation_list = {'gen': generation, 'fitness': generation_best}
                    generation_incumbent_list = {'gen': generation, 'fitness': improvisation_best}

                    self._harmony_trace_best.append(generation_list)
                    self._harmony_trace_incumbent.append(generation_incumbent_list)
                    generation_best = None

                else:
                    pareto_logger(self._pareto_harmony_memory, num_imp / self._obj_fun.get_hms(),
                                  self._obj_fun.complexity_level, self._obj_fun.instance_name)
                generation += 1
                iterations_without_improvement += 1

            # print('the iterations_without improvement', iterations_without_improvement)
        # find out why it terminated
        print('The number of imps is', num_imp, 'the max imps is', self._obj_fun.get_max_imp(),
              ' elapsed time is', elapsed_time, 'max time', self._obj_fun.get_max_time(), 'no improvement',
              iterations_without_improvement, 'max_improvement', self._obj_fun.get_termination_iter())

        # return best harmony
        best_harmony = None
        best_fitness = float('-inf') if self._obj_fun.maximize() else float('+inf')
        if not self.pf.get_objective_is_multi():

            for fitness in self._harmony_memory:
                if (self._obj_fun.maximize() and fitness[self._obj_fun] > best_fitness) or (
                        not self._obj_fun.maximize() and fitness[self.obj_1] < best_fitness):
                    best_harmony = fitness['layout']
                    best_fitness = fitness[self.obj_1]
            # self._plot_harmony_history()
            iterations_for_plotting, incumbent_for_plotting, best_for_plotting = self._retrieve_results()
            print('best harmony', best_harmony)
            self._obj_fun.get_fitness(best_harmony, verbose=True)
        if self.pf.get_objective_is_multi():

            self._pareto_harmony_memory = self.pf.pareto_run(self._pareto_harmony_memory)

            return self._harmony_memory[0]['layout'], self._harmony_memory[
                0], self._pareto_harmony_memory, self._harmony_history
        else:
            return best_harmony, best_fitness, self._harmony_memory, self._harmony_history, iterations_for_plotting, incumbent_for_plotting, best_for_plotting

    def _retrieve_results(self):
        x = list();
        x2 = list();
        y = list();
        z = list()
        print('best_results in harmony trace', self._harmony_trace_best)
        for dic_item in self._harmony_trace_best:
            for key in dic_item:
                if key == 'gen':
                    x.append(dic_item[key])
                else:
                    z.append(dic_item[key])
        print(x)
        print(z)
        for dic_item2 in self._harmony_trace_incumbent:
            for key2 in dic_item2:
                if key2 == 'gen':
                    x2.append(dic_item2[key2])
                else:
                    y.append(dic_item2[key2])
        print(x2)
        print(y)
        if len(x2) == len(y):
            print('yes')
        else:
            print('no')
        return x, y, z

    def _plot_harmony_history(self):
        from matplotlib import pyplot as plt
        x = list();
        x2 = list();
        y = list();
        z = list()
        print(self._harmony_trace_best)
        for dic_item in self._harmony_trace_best:
            for key in dic_item:
                if key == 'gen':
                    x.append(dic_item[key])
                else:
                    z.append(dic_item[key])
        print(x)
        print(z)
        for dic_item2 in self._harmony_trace_incumbent:
            for key2 in dic_item2:
                if key2 == 'gen':
                    x2.append(dic_item2[key2])
                else:
                    y.append(dic_item2[key2])
        print(x2)
        print(y)
        if len(x2) == len(y):
            print('yes')
        else:
            print('no')
        _plot(x, y, z, 'Iterations', 'Incumbent', 'Best')
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='Generation', ylabel='BIC')
        ax.grid()
        plt.show()
        return x, y, z

    def _initialize(self, initial_harmonies=None, model_nature=None):

        """
            Initialize harmony_memory, the matrix (list of lists) containing the various harmonies (solution vectors). Note
            that we aren't actually doing any matrix operations, so a library like NumPy isn't necessary here. The matrix
            merely stores previous harmonies.
            If harmonies are provided, then use them instead of randomly initializing them.
            Populate harmony_history with initial harmony memory.
        """
        if model_nature is not None:
            a = self._obj_fun.modify_initial_fit(model_nature)
            # self._obj_fun.makeRegression(a)
            vector = self._obj_fun.reconstruct_vector(a)

        if initial_harmonies is not None:

            # verify that the initial harmonies are provided correctly
            if len(initial_harmonies) != self._obj_fun.get_hms():

                if len(initial_harmonies) > self._obj_fun.get_hms():
                    if isinstance(initial_harmonies, dict):
                        raise Exception
                    while (initial_harmonies) > self._obj_fun.get_hms():
                        initial_harmonies.pop()
                stuff = initial_harmonies.copy()
                initial_harmonies = list()

                for i in range(0, np.min((self._obj_fun.get_hms(), len(stuff)))):
                    if len(stuff) < self._obj_fun.get_hms():
                        if not isinstance(stuff, dict):
                            for j, b in enumerate(stuff[i]['layout'][self._obj_fun._characteristics]):
                                if b:
                                    stuff[i]['layout'][j] = random.randint(1,
                                                                           self._obj_fun.get_num_discrete_values(j) - 1)
                    initial_harmonies.append(stuff[i]['layout'])
                while len(initial_harmonies) < self._obj_fun.get_hms():
                    pop_i = random.choice(initial_harmonies)
                    harmony = list()
                    for j, b in enumerate(pop_i):
                        if isinstance(b, int) and b:
                            harmony.append(random.randint(1, self._obj_fun.get_num_discrete_values(j) - 1))
                        else:
                            harmony.append(b)
                            # self._obj_fun.hasher_check(harmony)
                    initial_harmonies.append(harmony)


            else:
                initial_harmonies = [a['layout'] for a in initial_harmonies]
                for i, j in enumerate(initial_harmonies):
                    for a, b in enumerate(j):
                        if isinstance(b, int) and b:
                            initial_harmonies[i][a] = random.randint(1, self._obj_fun.get_num_discrete_values(a) - 1)



        else:
            initial_harmonies = list()

            # embedding Krishnas model
            while len(initial_harmonies) < self._obj_fun.get_hms():
                harmony = list()
                for j in range(0, self._obj_fun.get_num_parameters()):
                    self._random_selection(harmony, j)

                initial_harmonies.append(harmony)
        if model_nature is not None:
            initial_harmonies[0] = vector

        for i in range(0, self._obj_fun.get_hms()):
            # print(initial_harmonies[i])
            fitness = self._obj_fun.get_fitness(initial_harmonies[i], self.pf.get_objective_is_multi(), max_routine=2)
            # print(initial_harmonies[i])
            if self.pf.get_objective_is_multi():

                self._harmony_memory.append(fitness)

            else:

                self._harmony_memory.append(fitness)
        harmony_list = {'gen': 0, 'harmonies': self._harmony_memory}

        self._harmony_history.append(harmony_list)

    def _random_selection(self, harmony, i):
        """
            Choose a note according to get_value(). Remember that even if a note is not variable, get_value() must still
            return a valid value.
        """
        harmony.append(self._obj_fun.get_value(i))

    def _memory_consideration(self, harmony, i):
        """
            Randomly choose a note previously played.
        """
        if self.pf.get_objective_is_multi():
            memory_index = random.randint(0, self._obj_fun.get_hms() - 1)
            harmony.append(self._harmony_memory[memory_index]['layout'][i])


        else:
            memory_index = random.randint(0, self._obj_fun.get_hms() - 1)
            harmony.append(self._harmony_memory[memory_index]['layout'][i])

    def _pitch_adjustment(self, harmony, i, global_best_harmony=True):
        """
            If variable, randomly adjust the pitch up or down by some amount. This is the only place in the algorithm where there
            is an explicit difference between continuous and discrete variables.
            The probability of adjusting the pitch either up or down is fixed at 0.5. The maximum pitch adjustment proportion (mpap)
            and maximum pitch adjustment index (mpai) determine the maximum amount the pitch may change for continuous and discrete
            variables, respectively.
            For example, suppose that it is decided via coin flip that the pitch will be adjusted down. Also suppose that mpap is set to 0.25.
            This means that the maximum value the pitch can be dropped will be 25% of the difference between the lower bound and the current
            pitch. mpai functions similarly, only it relies on indices of the possible values instead.
        """
        if global_best_harmony:
            if self._obj_fun.is_multi:

                harmony[i] = self._pareto_harmony_memory[0]['layout'][i]
            else:
                harmony[i] = self._harmony_memory[0]['layout'][i]  # this is the best


        else:
            if self._obj_fun.is_variable(i):
                if self._obj_fun.is_discrete(i):
                    current_index = self._obj_fun.get_index(i, harmony[i])
                    try:
                        new_index = current_index + random.randint(1, self._obj_fun.get_mpai()) * random.choice([-1, 1])
                    except Exception as e:
                        print(e, 'index error, this is probs it')
                        new_index = current_index
                    if new_index not in range(0, len(self._obj_fun._discrete_values[i])):
                        new_index = self._obj_fun.modulo_or_divisor(new_index,
                                                                    len(self._obj_fun._discrete_values[i])) - 1

                    harmony[i] = self._obj_fun.get_value(i, new_index)
                    # harmony[i] = self._rdm_best_feature(i)
                    # old turn off for now
                    # current_index + min(self.F * difference, self._obj_fun.get_num_discrete_values(i) - current_index)
                    # discrete variable

                else:
                    # continuous variable
                    if random.random() < 0.5:
                        # adjust pitch down
                        harmony[i] -= (harmony[i] - self._obj_fun.get_lower_bound(
                            i)) * random.random() * self._obj_fun.get_mpap()
                    else:
                        # adjust pitch up
                        harmony[i] += (self._obj_fun.get_upper_bound(i) - harmony[
                            i]) * random.random() * self._obj_fun.get_mpap()(self._obj_fun.get_upper_bound(i) - harmony[
                            i]) * random.random() * self._obj_fun.get_mpap()

    def _update_harmony_memory(self, considered_harmony, considered_fitness, iterations_improv=0, IS_MULTI=True):
        """
            Update the harmony memory if necessary with the given harmony. If the given harmony is better than the worst
            harmony in memory, replace it. This function doesn't allow duplicate harmonies in memory.
            Multi-objective sorting available if declared 'IS_MULTI = True'
        """

        if IS_MULTI:

            if considered_fitness['layout'] not in [a['layout'] for a in
                                                    self._harmony_memory]:  # TODO check just for integer values

                # update the pareto memory

                # self._pareto_harmony_memory = self.pf.check_if_dominance(self._pareto_harmony_memory, considered_fitness)
                self._pareto_harmony_memory = self.pf.pareto_run(self._pareto_harmony_memory, True, considered_fitness)
                copy_harmony = self._harmony_memory.copy()
                # insert and then sort
                copy_harmony.insert(0, considered_fitness)
                copy_harmony = self.pf.non_dominant_sorting(copy_harmony)
                if len(copy_harmony) != len(self._harmony_memory) + 1:
                    raise Exception

                if considered_fitness == copy_harmony[-1]:

                    if self.pf.is_pareto_efficient(considered_fitness, self.pf.Pareto_F):
                        return 0
                    return iterations_improv
                else:
                    copy_harmony.pop()
                    self._harmony_memory = copy_harmony
                    return 0







            else:
                print('same solution')
                return iterations_improv


        else:
            if considered_fitness['layout'] not in [a['layout'] for a in self._harmony_memory]:

                if (self._obj_fun.maximize() and considered_fitness[self.obj_1] < self._harmony_memory[-1][
                    self.obj_1]) or (
                        not self._obj_fun.maximize() and considered_fitness[self.obj_1] > self._harmony_memory[-1][
                    self.obj_1]):
                    self._harmony_memory[-1] = considered_fitness
                    self._sort_memory()
                    return 0
                else:
                    return iterations_improv
            else:
                return iterations_improv

    def _sort_memory(self):
        '''sorts the harmony memories into order from best solutions'''
        if self.pf.get_objective_is_multi():
            self._harmony_memory = self.pf.non_dominant_sorting(self._harmony_memory)
        else:
            if self._obj_fun.maximize():
                self._harmony_memory.sort(key=lambda x: x[self._obj_fun._get_obj1()], reverse=True)
            else:
                self._harmony_memory.sort(key=lambda x: x[self._obj_fun._get_obj1()])



    def _get_best_feature(self, j):
        self._sort_memory()
        if self.pf.get_objective_is_multi():
            for harmony in self._harmony_memory[0]['layout']:
                print('best feauture', harmony[j])
            return harmony[j]
        else:
            for harmony in self._harmony_memory[0]['layout']:
                print('harm', harmony[j])
                return harmony[j]

    def _rdm_best_feature(self, j):
        self._sort_memory()
        # print(self._harmony_memory)

        choice = list()  # make an empty list

        if self.pf.get_objective_is_multi():
            for harmony in self._harmony_memory[0]['layout']:
                choice.append(harmony[j])
            return random.choice(choice)
        else:
            for harmony, fitness in self._harmony_memory:
                choice.append(harmony[j])
            return random.choice(choice)


class SparseEA_AGDS(object):
    """
        SparseEA-AGDS: Evolution algorithm with Adaptive Genetic operator and
        Dynamic Scoring mechanism, after Wang et al. (Scientific Reports, 2025,
        "Evolution algorithm with adaptive genetic operator and dynamic scoring
        mechanism for large-scale sparse many-objective optimization").

        The SparseEA representation X = dec x mask is mapped onto the model
        specification search: ``mask_d`` is whether term ``d`` is included
        (solution value != 0) and ``dec_d`` is which discrete choice
        (distribution / transform / model-type) that term takes.  Three
        stackable strategies are implemented:

          (A) Adaptive genetic operator - per-individual crossover / mutation
              probabilities scale with the individual's non-dominated rank:
              ``P_s,i = (maxr - r_i + 1)/maxr`` (Eq. 5),
              ``P_c,i = pc0 * P_s,i`` (Eq. 6), ``P_m,i = pm0 * P_s,i`` (Eq. 7).
          (B) Dynamic scoring mechanism - each generation the decision-variable
              score is recomputed from the current layers:
              ``S_i_r = maxr - r_i + 1`` (Eq. 8), ``sumS = S_r^T x mask``
              (Eq. 9), ``S_d = maxS - sumS_d + 1`` (Eq. 10); a binary
              tournament on ``S_d`` picks which mask bits to flip.
          (C) Reference-point-based environmental selection (NSGA-III niching)
              maintains selection pressure under many objectives; for the
              single-objective case it degenerates to best-N-by-objective.

        The class mirrors the ``HarmonySearch`` interface: it takes an
        ``ObjectiveFunction`` and exposes ``run()`` returning the same shape the
        module-level wrapper expects.
    """

    def __init__(self, objective_function, **kwargs):
        objective_function.algorithm = 'agds'
        self._obj_fun = objective_function
        if self._obj_fun._obj_1 is None:
            print('no objective found, automatically selecting BIC')
            self._obj_fun._obj_1 = 'bic'

        self._pop_size = int(kwargs.get('_pop_size', kwargs.get('_hms', 20)))
        if self._pop_size < 4:
            self._pop_size = 4
        self._num_intl_slns = self._pop_size
        self.iter = int(kwargs.get('_max_iter', kwargs.get('MAX_ITERATIONS', 100)))
        self._D = self._obj_fun.get_num_parameters()
        # pc0 / pm0 are the paper's fixed base probabilities (Figs. 1-2 use
        # 0.9 and 0.03); pm0 defaults to 1/D following Table 2's SparseEA row.
        self.pc0 = float(kwargs.get('_pc0', 0.9))
        self.pm0 = float(kwargs.get('_pm0', 1.0 / max(self._D, 1)))
        self.ref_divisions = int(kwargs.get('_ref_divisions', 12))
        self.print_verbose = kwargs.get('verbose', False)

        try:
            self.instance_name = str(objective_function.instance_name)
        except Exception:
            self.instance_name = str(kwargs.get('instance_name', 1))
            objective_function.instance_name = self.instance_name
        self.get_directory()

        self._sa_memory = list()
        self._history = list()
        if objective_function.is_multi:
            self.obj_1 = objective_function._get_obj1()
            self.obj_2 = objective_function._get_obj2()
            self.pf = Pareto(self.obj_1, self.obj_2, True)
        else:
            self.obj_1 = objective_function._get_obj1()
            self.obj_2 = objective_function._get_obj2()
            self.pf = Pareto(self.obj_1, self.obj_2, False)

    # ------------------------------------------------------------------ #
    #  bookkeeping (mirrors HarmonySearch / DifferentialEvolution)       #
    # ------------------------------------------------------------------ #
    def get_directory(self):
        if not os.path.isdir(self.instance_name):
            os.makedirs(self.instance_name)

    def get_instance_name(self):
        return str(self.instance_name) + '/log.csv'

    def _random_selection(self, sln, i):
        sln.append(self._obj_fun.get_value(i))

    # ------------------------------------------------------------------ #
    #  dec x mask representation helpers                                 #
    # ------------------------------------------------------------------ #
    def _maskable(self, i):
        """A position is a sparsity gene (can be switched off) when 0 is one of
        its discrete choices - e.g. feature-selection positions.  Positions
        such as the constant or the model-type carry no 0 and stay active."""
        try:
            if not self._obj_fun.is_discrete(i):
                return False
            return 0 in self._obj_fun._discrete_values[i]
        except Exception:
            return False

    def _nonzero_choices(self, i):
        try:
            choices = [v for v in self._obj_fun._discrete_values[i] if v != 0]
        except Exception:
            choices = []
        return choices

    @staticmethod
    def _is_active(val):
        return val != 0

    def _maskable_indices(self):
        return [i for i in range(self._D) if self._maskable(i)]

    def _mask_vector(self, layout):
        return [1 if self._is_active(layout[i]) else 0 for i in range(self._D)]

    # ------------------------------------------------------------------ #
    #  fitness evaluation -> "Struct" dict (layout + objective keys)      #
    # ------------------------------------------------------------------ #
    def _penalty_struct(self, layout):
        pen = {'layout': list(layout), self.obj_1: 10 ** 9}
        if self._obj_fun.is_multi:
            pen[self.obj_2] = 10 ** 9
        return pen

    def _evaluate(self, layout):
        try:
            fitness = self._obj_fun.get_fitness(
                layout, multi=self._obj_fun.is_multi, max_routine=2)
            if not isinstance(fitness, dict):
                return self._penalty_struct(layout)
            if self.obj_1 not in fitness or fitness[self.obj_1] is None \
                    or (isinstance(fitness[self.obj_1], float) and np.isnan(fitness[self.obj_1])):
                return self._penalty_struct(layout)
            if 'layout' not in fitness:
                fitness['layout'] = list(layout)
            try:
                self._sa_memory.append(self._obj_fun.Last_Sol)
            except Exception:
                pass
            return fitness
        except Exception as e:
            if self.print_verbose:
                print('agds fitness error', e)
            return self._penalty_struct(layout)

    def _obj_vector(self, struct):
        if self._obj_fun.is_multi:
            return [float(struct[self.obj_1]), float(struct[self.obj_2])]
        return [float(struct[self.obj_1])]

    # ------------------------------------------------------------------ #
    #  non-dominated ranking (r_i, maxr)  -- Eq. 5 / Eq. 8               #
    # ------------------------------------------------------------------ #
    def _compute_ranks(self, structs):
        """Return (ranks, maxr) where ranks[i] is the 1-indexed non-dominated
        layer of ``structs[i]``.  Multi-objective uses the package's front
        builder; single-objective groups equal-objective ties into layers."""
        n = len(structs)
        if n == 0:
            return [], 1
        if self._obj_fun.is_multi:
            try:
                fronts = self.pf.get_fronts(structs)
                ranks = [1] * n
                for layer, (_, idxs) in enumerate(fronts.items(), start=1):
                    for idx in idxs:
                        ranks[idx] = layer
                maxr = max(ranks)
                return ranks, maxr
            except Exception:
                pass
        order = sorted(range(n), key=lambda i: self._obj_vector(structs[i])[0])
        ranks = [1] * n
        layer = 1
        prev = None
        for pos, i in enumerate(order):
            val = self._obj_vector(structs[i])[0]
            if prev is not None and val > prev:
                layer += 1
            ranks[i] = layer
            prev = val
        return ranks, max(ranks)

    # ------------------------------------------------------------------ #
    #  (B) dynamic decision-variable score  -- Eqs. 8-10                 #
    # ------------------------------------------------------------------ #
    def _dynamic_scores(self, structs, ranks, maxr):
        si_r = [maxr - r + 1 for r in ranks]                 # Eq. 8
        sumS = [0.0] * self._D
        for i, struct in enumerate(structs):
            mask = self._mask_vector(struct['layout'])
            for d in range(self._D):
                sumS[d] += si_r[i] * mask[d]                  # Eq. 9
        maxS = max(sumS) if sumS else 0.0
        S_d = [maxS - sumS[d] + 1 for d in range(self._D)]    # Eq. 10 (lower is better)
        return S_d

    def _tournament_dim(self, S_d, candidates):
        """Binary tournament on the (dynamic) score - the smaller S_d wins,
        i.e. dimensions frequently non-zero among superior individuals are
        favoured for crossover / mutation of the mask (paper, Figs. 3-4)."""
        if not candidates:
            return None
        a = random.choice(candidates)
        b = random.choice(candidates)
        return a if S_d[a] <= S_d[b] else b

    # ------------------------------------------------------------------ #
    #  (A) adaptive genetic operator + mask operator  -> one offspring   #
    # ------------------------------------------------------------------ #
    def _make_offspring(self, parent, partner, ps_i, S_d):
        pc_i = self.pc0 * ps_i                                # Eq. 6
        pm_i = self.pm0 * ps_i                                # Eq. 7
        child = list(parent['layout'])
        p_layout = partner['layout']

        # ---- dec: adaptive per-position crossover / mutation ----------
        # position index matrix K (Alg. 1): each dimension independently
        # selected; whether it actually changes is gated by pc_i / pm_i.
        for d in range(self._D):
            if random.random() < pc_i:                        # crossover of dec
                child[d] = p_layout[d]
            if random.random() < pm_i:                        # mutation of dec
                nz = self._nonzero_choices(d)
                if nz:
                    child[d] = random.choice(nz)
                else:
                    child[d] = self._obj_fun.get_value(d)

        # ---- mask: dynamic-scoring driven crossover + mutation --------
        maskable = self._maskable_indices()
        # mask crossover: at a tournament-chosen dimension where activity
        # differs from the partner, adopt the partner's value.
        differing = [d for d in maskable
                     if self._is_active(child[d]) != self._is_active(p_layout[d])]
        if differing and random.random() < self.pc0 * ps_i:
            d = self._tournament_dim(S_d, differing)
            if d is not None:
                child[d] = p_layout[d]
        # mask mutation: invert activity at a tournament-chosen dimension.
        if maskable and random.random() < max(pm_i, self.pm0):
            d = self._tournament_dim(S_d, maskable)
            if d is not None:
                if self._is_active(child[d]):
                    child[d] = 0
                else:
                    nz = self._nonzero_choices(d)
                    child[d] = random.choice(nz) if nz else child[d]
        return child

    def _reproduce(self, structs):
        ranks, maxr = self._compute_ranks(structs)
        S_d = self._dynamic_scores(structs, ranks, maxr)
        n = len(structs)
        children = []
        for i in range(n):
            ps_i = (maxr - ranks[i] + 1) / maxr               # Eq. 5
            j = random.randrange(n)
            while j == i and n > 1:
                j = random.randrange(n)
            child_layout = self._make_offspring(structs[i], structs[j], ps_i, S_d)
            children.append(self._evaluate(child_layout))
        return children

    # ------------------------------------------------------------------ #
    #  (C) reference-point-based environmental selection (NSGA-III)       #
    # ------------------------------------------------------------------ #
    def _environmental_selection(self, combined, n_select):
        if len(combined) <= n_select:
            return list(combined)
        if not self._obj_fun.is_multi:
            combined = sorted(combined, key=lambda s: self._obj_vector(s)[0])
            return combined[:n_select]

        # non-dominated fronts (reuse the package builder)
        fronts = self.pf.get_fronts(combined)
        selected, critical = [], None
        for _, idxs in fronts.items():
            if len(selected) + len(idxs) <= n_select:
                selected.extend(idxs)
            else:
                critical = idxs
                break
        if len(selected) == n_select or critical is None:
            return [combined[i] for i in selected[:n_select]]

        F = np.array([self._obj_vector(combined[i]) for i in range(len(combined))],
                     dtype=float)
        M = F.shape[1]
        ideal = F.min(axis=0)
        Fn = F - ideal
        nadir = Fn.max(axis=0)
        nadir[nadir == 0] = 1.0
        Fn = Fn / nadir

        ref = _agds_reference_points(M, self.ref_divisions)
        sel_assoc = _agds_associate(Fn[selected], ref) if selected else np.array([], dtype=int)
        niche = np.zeros(len(ref), dtype=int)
        for a in sel_assoc:
            niche[a] += 1
        crit_assoc = _agds_associate(Fn[critical], ref)
        crit_dist = _agds_perp_distance(Fn[critical], ref, crit_assoc)

        chosen = list(selected)
        available = list(range(len(critical)))
        while len(chosen) < n_select and available:
            rmin = None
            for k in set(crit_assoc[c] for c in available):
                if rmin is None or niche[k] < niche[rmin]:
                    rmin = k
            members = [c for c in available if crit_assoc[c] == rmin]
            if niche[rmin] == 0:
                pick = min(members, key=lambda c: crit_dist[c])
            else:
                pick = random.choice(members)
            chosen.append(critical[pick])
            niche[rmin] += 1
            available.remove(pick)
        return [combined[i] for i in chosen[:n_select]]

    # ------------------------------------------------------------------ #
    #  initial population (Alg. 3 lines 6-11)                            #
    # ------------------------------------------------------------------ #
    def _initialize(self, initial_slns=None, model_nature=None):
        vector = None
        if model_nature is not None:
            try:
                a = self._obj_fun.modify_initial_fit(model_nature)
                vector = self._obj_fun.reconstruct_vector(a)
            except Exception as e:
                if self.print_verbose:
                    print('agds model_nature seed failed', e)
        if initial_slns is None:
            initial_slns = []
            for _ in range(self._num_intl_slns):
                sln = []
                for j in range(self._D):
                    self._random_selection(sln, j)
                initial_slns.append(sln)
        if vector is not None and initial_slns:
            initial_slns[0] = vector

        population = []
        for k in range(len(initial_slns)):
            if self.print_verbose:
                print('evaluating initial sln', k)
            struct = self._evaluate(initial_slns[k])
            population.append(struct)
            if self._obj_fun.is_multi:
                self.pf.add_Structs(struct)
        if self._obj_fun.is_multi:
            self.pf.run()
        return population

    # ------------------------------------------------------------------ #
    #  main loop (Alg. 3 lines 12-17)                                    #
    # ------------------------------------------------------------------ #
    def run(self, initial_slns=None, mod_init=None):
        start_time = datetime.now()
        if self._obj_fun.use_random_seed():
            self._obj_fun.set_random_seed()

        population = self._initialize(initial_slns, mod_init)
        best_key = self.obj_1

        def _best_of(structs):
            return min(structs, key=lambda s: self._obj_vector(s)[0])

        incumbent = _best_of(population)
        best_collection, incumbent_collection, iter_axis = [], [], []

        for gen in range(self.iter):
            children = self._reproduce(population)
            if self._obj_fun.is_multi:
                for c in children:
                    self.pf.add_Structs(c)
            else:
                for c in children:
                    self._sa_memory.append(c)

            population = self._environmental_selection(
                population + children, self._pop_size)

            gen_best = _best_of(population)
            if self._obj_vector(gen_best)[0] < self._obj_vector(incumbent)[0]:
                incumbent = gen_best

            iter_axis.append(gen)
            best_collection.append(self._obj_vector(gen_best)[0])
            incumbent_collection.append(self._obj_vector(incumbent)[0])
            self._history.append({'gen': gen, 'pop': copy.deepcopy(population)})

            try:
                logger(gen, gen_best, incumbent, name=self.get_instance_name(),
                       local_best1=gen_best)
            except Exception:
                pass
            if self.print_verbose:
                print(f'gen {gen}: best {best_key}={self._obj_vector(gen_best)[0]:.4f}')

            elapsed = (datetime.now() - start_time).total_seconds()
            try:
                if elapsed > self._obj_fun.get_max_time():
                    break
            except Exception:
                pass

        if self._obj_fun.is_multi:
            pareto_memory = self.pf.pareto_run(population, True)
            best = _best_of(pareto_memory) if pareto_memory else incumbent
            return best['layout'], best, pareto_memory, self._history
        else:
            population = sorted(population, key=lambda s: self._obj_vector(s)[0])
            best = population[0]
            best_fitness = self._obj_vector(best)[0]
            return (best['layout'], best_fitness, population, self._history,
                    iter_axis, incumbent_collection, best_collection)


class Mutlithreaded_Meta(DifferentialEvolution, SimulatedAnnealing, HarmonySearch):

    def __init__(self, objective_function, **kwargs):
        DifferentialEvolution.__init__(self, objective_function, **kwargs)
        SimulatedAnnealing.__init__(self, objective_function, **kwargs)
        HarmonySearch.__init__(self, objective_function)

    def calculate_fitness(self, args):
        i, initial_slns = args
        return self._obj_fun.get_fitness(initial_slns[i], self.pf.get_objective_is_multi(), max_routine=2)

    def run_hs_mp(self, initial_harmonies=None, mod_init=None):
        """
            This is the main HS loop. It initializes the harmony memory and then continually generates new harmonies
            until the stopping criterion (max_imp iterations) is reached.
        """
        # generational
        # generation_best = 10 ** 10
        # improvisation_best = 10 ** 10
        start_time = datetime.now()
        iterations_without_improvement = 0
        # set optional random seed
        if self._obj_fun.use_random_seed():
            self._obj_fun.set_random_seed()
        # fill harmony_memory using random parameter values by default, but with initial_harmonies if provided
        initial_harmonies = self._initialize_mp(initial_harmonies, mod_init)
        if self.pf.get_objective_is_multi():
            self._pareto_harmony_memory = self.pf.non_dominant_sorting(initial_harmonies)
            generation_best = self._pareto_harmony_memory[0]
            improvisation_best = self._pareto_harmony_memory[0]
            self._harmony_memory = initial_harmonies

        else:
            self._sort_memory()
            generation_best = self._harmony_memory[0]
            improvisation_best = self._harmony_memory[0]
            # create max_imp improvisations
        generation = 0
        num_imp = 0
        elapsed_time = 0
        num_processors = np.min((mp.cpu_count(), self._obj_fun._hms))
        pool = mp.Pool(num_processors)

        while (num_imp < self._obj_fun.get_max_imp()) and (
                elapsed_time <= self._obj_fun.get_max_time()) and (
                iterations_without_improvement < self._obj_fun.get_termination_iter()):
            # generate new harmony
            elapsed_time = (datetime.now() - start_time).total_seconds()
            harmony_gen = list()
            for harmony_gen_i in range(self._obj_fun._hms):
                harmony = list()

                for i in range(0, self._obj_fun.get_num_parameters()):
                    if random.random() < self._obj_fun.get_hmcr():
                        self._memory_consideration(harmony, i)
                        if random.random() < self._obj_fun.get_par():
                            # print('this is', i)
                            # print('i am just testing this, change back to tru')
                            self._pitch_adjustment(harmony, i, True)
                    else:
                        self._random_selection(harmony, i)
                        if random.random() < self._obj_fun.get_par():
                            # print('this is', i)
                            # print('i am just testing this, change back to tru')
                            self._pitch_adjustment(harmony, i, False)

                # check if harmony is in memory

                if self.does_it_appear(harmony):

                    rand_idx, rand_inj = self.hard_mutate_index_and_value()
                    harmony[rand_idx] = rand_inj
                    appear = self.does_it_appear(harmony)
                    while appear == 1:
                        rand_idx, rand_inj = self.hard_mutate_index_and_value()
                        harmony[rand_idx] = rand_inj
                        appear = self.does_it_appear(harmony)
                harmony_gen.append(harmony)

            results_hs = pool.map(self.calculate_fitness, [(i, harmony_gen) for i in range(self._obj_fun._hms)])
            for hs_i, fitness in enumerate(results_hs):
                harmony = fitness['layout']

                iterations_without_improvement = self._update_harmony_memory(harmony, fitness,
                                                                             iterations_without_improvement,
                                                                             self.pf.get_objective_is_multi())
                num_imp += 1
                if iterations_without_improvement == 0:  # if there is any kind of improvement updae the logs
                    if self.pf.get_objective_is_multi():
                        logger(num_imp, fitness, self._harmony_memory, True, self.get_instance_name(),
                               1)  # for consistency
                        logger(num_imp, fitness, self._pareto_harmony_memory, True,
                               self.instance_name + '/log_for_pareto_harmony_memory.csv', 1)


                    else:
                        # update generation best
                        if generation_best is None:
                            generation_best = fitness
                        if fitness[self.obj_1] < generation_best[self.obj_1]:
                            generation_best = fitness
                            # iterations_without_improvement =0
                        if fitness[self.obj_1] < improvisation_best[self.obj_1]:
                            improvisation_best = fitness
                        # update logs
                        try:
                            logger(num_imp, fitness, improvisation_best, name=self.get_instance_name(),
                                   local_best1=generation_best)
                        except Exception as e:
                            print('log broken', e)
                elif not self.pf.get_objective_is_multi():
                    # update generation best
                    if generation_best is None:
                        generation_best = fitness
                    if fitness[self.obj_1] < generation_best[self.obj_1]:
                        generation_best = fitness
                    # update logs
                    try:
                        logger(num_imp, fitness, improvisation_best, name=self.get_instance_name())
                    except Exception as e:
                        print('log broken', e)
                        # save harmonies every nth improvisations (i.e., one 'generation')
            if num_imp % self._obj_fun.get_hms() == 0:

                # self._sort_memory()
                harmony_list = {'gen': generation, 'harmonies': copy.deepcopy(self._harmony_memory)}
                self._harmony_history.append(harmony_list)
                if self.pf.get_objective_is_multi() is False:
                    generation_list = {'gen': generation, 'fitness': generation_best}
                    generation_incumbent_list = {'gen': generation, 'fitness': improvisation_best}

                    self._harmony_trace_best.append(generation_list)
                    self._harmony_trace_incumbent.append(generation_incumbent_list)
                    generation_best = None

                else:
                    pareto_logger(self._pareto_harmony_memory, num_imp / self._obj_fun.get_hms(),
                                  self._obj_fun.complexity_level, self._obj_fun.instance_name)
                generation += 1
                iterations_without_improvement += 1

            # print('the iterations_without improvement', iterations_without_improvement)
        # find out why it terminated
        print('The number of imps is', num_imp, 'the max imps is', self._obj_fun.get_max_imp(),
              ' elapsed time is', elapsed_time, 'max time', self._obj_fun.get_max_time(), 'no improvement',
              iterations_without_improvement, 'max_improvement', self._obj_fun.get_termination_iter())

        # return best harmony
        best_harmony = None
        best_fitness = float('-inf') if self._obj_fun.maximize() else float('+inf')
        if not self.pf.get_objective_is_multi():

            for fitness in self._harmony_memory:
                if (self._obj_fun.maximize() and fitness[self._obj_fun] > best_fitness) or (
                        not self._obj_fun.maximize() and fitness[self.obj_1] < best_fitness):
                    best_harmony = fitness['layout']
                    best_fitness = fitness[self.obj_1]
            # self._plot_harmony_history()
            iterations_for_plotting, incumbent_for_plotting, best_for_plotting = self._retrieve_results()
            print('I Belive this falls over')
            print('best harmony', best_harmony)
            print('best_fitness,', best_fitness)
            self._obj_fun.get_fitness(best_harmony, verbose=True)
        if self.pf.get_objective_is_multi():

            self._pareto_harmony_memory = self.pf.pareto_run(self._pareto_harmony_memory)

            return self._harmony_memory[0]['layout'], self._harmony_memory[
                0], self._pareto_harmony_memory, self._harmony_history
        else:
            return best_harmony, best_fitness, self._harmony_memory, self._harmony_history, iterations_for_plotting, incumbent_for_plotting, best_for_plotting

    def run_sa(self, initial_slns=None, mod_init=None):
        """
            This is the main SA loop. It initializes the random solutions and then continually generates new solutions until the stopping
            criteria is reached.
        """
        # generational
        generation_best = 10 ** 10
        iterations_without_improvement = 0
        start_time = datetime.now()
        # fill randomly, but with initial_slns if provided
        initial_slns_struct = self._initialize_mp(initial_slns, mod_init)
        # Calculate the init   ial temperature, and seed this instance so it is always the same
        if self._obj_fun.use_random_seed():
            self._obj_fun.set_random_seed()
        self.temp_max = self.Calculate_Temp_mp(initial_slns_struct, acceptance_prob=self._INITAL_ACCEPT_RATE,
                                               multi=self.pf.get_objective_is_multi())

        current_temperature = self.temp_max
        print('The Temperature is', current_temperature)
        # set optional random seed

        iteration = 0
        temperature_iteration = 0  # counting mechanism to update the temperature
        # Get Current Solution and Fitness
        for fitness in initial_slns_struct:
            if self.pf.get_objective_is_multi():
                fit_ob_1 = fitness.get(self.obj_1)  # TODO better way to get the objectives
                fit_ob_2 = fitness.get(self.obj_2)
                self.pf.evaluate_frontier_against_new_sln(fitness)



            else:
                if fitness[self._obj_fun._obj_1] <= self.best_energy[self._obj_fun._obj_1]:
                    self.best_energy = fitness

                    self.current_energy = fitness

        # start from the best solution if analyst did not specify
        if self._obj_fun.solution_analyst is not None:
            self.current_energy = self._obj_fun.solution_analyst[self.obj_1]  # TODO handle both objectives
            self.current_struct = self._obj_fun.solution_analyst['layout']
        else:
            if self.pf.get_objective_is_multi():  # TODO deterimine the best solion in the pareto

                num_processors = np.min((mp.cpu_count(), self._STEPS_PER_TEMP))
                pool = mp.Pool(num_processors)

                current_energy = initial_slns_struct.copy()
                nbr_energy = initial_slns_struct.copy()  # todo somet
                # print('does this work')

            else:

                self.pf.run(self._obj_fun.Last_Sol)
                self.current_energy = self.best_energy  # TODO refactor to solution
                self.current_struct = self.best_struct
        # Main Algorithm
        elapsed_time = 0

        num_processors = np.min((mp.cpu_count(), self._STEPS_PER_TEMP))
        pool = mp.Pool(num_processors)

        while (iteration <= self._MAX_ITERATIONS) and (
                elapsed_time <= self._obj_fun.get_max_time()) and (
                iterations_without_improvement < self._obj_fun.get_termination_iter()):
            elapsed_time = (datetime.now() - start_time).total_seconds()
            for i in range(self._STEPS_PER_TEMP):
                # Get Neighbouring solution and Energy level
                # print('the current struct is ' ,self.current_struct)
                nbr_struct = list()
                for j in range(self._pop_size):
                    nbr_struct1 = self._get_neighbour(current_energy[j]['layout'],
                                                      round(self._crossover_perc * len(current_energy[j]['layout'])))
                    nbr_struct.append(nbr_struct1)

                results_sa = pool.map(self.calculate_fitness, [(i, nbr_struct) for i in range(self._pop_size)])

                for j, nbr_energy in enumerate(results_sa):

                    # Acceptance Criteria
                    if random.uniform(0, 1) <= self.Energy_Acceptance_Criteria(nbr_energy, current_energy[j],
                                                                               current_temperature,
                                                                               self.pf.get_objective_is_multi()):
                        # print('Accepted New with fitness of', nbr_energy)
                        didchange = self.pf.did_it_change()
                        if didchange:
                            pareto_logger(self.pf.Pareto_F, iteration, self._obj_fun.complexity_level,
                                          self._obj_fun.instance_name)
                        current_energy[j] = nbr_energy

                        self.accept += 1
                        iterations_without_improvement = 0
                        # Check if neigbouring solution is the best      
                        if not self.pf.get_objective_is_multi():  # if this is the global best (show) #TODO determine global best for multiobjective
                            if nbr_energy[j][self._obj_fun._obj_1] < self.best_energy[self._obj_fun._obj_1]:
                                print('new best at iteration ', iteration, 'with fitness of', nbr_energy)
                                self.best_energy = nbr_energy

                                iterations_without_improvement = 0
                        else:
                            if nbr_energy[j] in self.pf.Pareto_F:
                                print('new pareto at iteration ', iteration, 'with fitness of', nbr_energy[j])

                                iterations_without_improvement = 0
                        # log data
                        iteration += 1

                    else:

                        iteration += 1

                    try:
                        logger(iteration, nbr_energy, self.best_energy, name=self.get_instance_name(),
                               local_best1=self.current_energy)
                    except Exception as e:
                        print('THEREE IS AN ERRPR OM TJE ;PGGER', e)
                        # update temperature
            temperature_iteration += 1
            current_temperature = self.update_t(temperature_iteration)
            iterations_without_improvement += 1
        acceptance_rate = self.accept / iteration
        print('the acceptance rate is', acceptance_rate * 100, '%')
        self._obj_fun.get_fitness(self.best_struct, multi=self.pf.get_objective_is_multi(), verbose=True)
        print('best', self.best_struct)

        return {'elapsed_time': elapsed_time, }

    def _initialize_mp(self, initial_slns=None, model_nature=None):
        """
            Initialize slns, the matrix (list of lists) containing the various harmonies (solution vectors). Note
            that we aren't actually doing any matrix operations, so a library like NumPy isn't necessary here. The matrix
            merely stores previous harmonies.
            If harmonies are provided, then use them instead of randomly initializing them.
            Populate harmony_history with initial harmony memory.
        """
        if self._obj_fun.algorithm == 'hs':
            self._pop_size = self._obj_fun.get_hms()
        if model_nature is not None:
            a = self._obj_fun.modify_initial_fit(model_nature)
            # self._obj_fun.makeRegression(a)
            vector = self._obj_fun.reconstruct_vector(a)

        if initial_slns is not None:
            # verify that the initial harmonies are provided correctly
            if len(initial_slns) != self._pop_size:
                raise ValueError('Number of initial solutions does not equal initial solution size.')
            num_parameters = self._obj_fun.get_num_parameters()
            for i in range(len(initial_slns)):
                num_parameters_initial_harmonies = len(initial_slns[i])
                if num_parameters_initial_harmonies != num_parameters:
                    raise ValueError('Number of parameters in initial solutions does not match that defined.')
        else:
            initial_slns = list()
            for i in range(0, self._pop_size):
                sln = list()
                for j in range(0, self._obj_fun.get_num_parameters()):
                    self._random_selection(sln, j)

                initial_slns.append(sln)

        initial_list = list()
        if model_nature is not None:
            initial_slns[0] = vector
            # embedding Krishnas model

        time_before = time.time()
        # mp.set_start_method('fork')
        num_processors = np.min((mp.cpu_count(), self._pop_size))

        pool = mp.Pool(num_processors)
        # fitness = zip(*)
        results = pool.map(self.calculate_fitness, [(i, initial_slns) for i in range(self._pop_size)])

        for i, fitness in enumerate(results):

            if i % 10 == 10:
                print('evaluating initial sln', i)
                print('the final fitness is', fitness)
            if self.pf.get_objective_is_multi():
                initial_list.append(fitness)
                logger(i, fitness, None, True, self.get_instance_name(), self.pf.get_objective_is_multi())
            else:
                if isinstance(fitness, dict):
                    initial_list.append(fitness)


                else:
                    raise Exception('should not be possible')

        return initial_list

    def run_mp(self, initial_slns=None, mod_init=None):
        # set optional random seed
        average_iteration = 0
        iterations_without_improvement = 0
        # self.pop_size = pop_size
        # self.iter = iter
        # self.F = F
        # self.cr = cr

        start_time = datetime.now()
        if self._obj_fun.use_random_seed():
            self._obj_fun.set_random_seed()

        # fill randomly, but with initial_slns if provided
        self._population = self._initialize_mp(initial_slns, mod_init)
        if self._obj_fun.is_multi:
            self._pareto_population = self._population.copy()
            self._pareto_population = self.pf.pareto_run(self._pareto_population)  # FIXME shrinking length
            self.pf.update_means()
        # print(self._population)
        self._sort_memory()
        if self._obj_fun.is_multi:
            best_solution = self._population[0]
            best_struct = self._population[0]['layout']
            prev_solution = best_solution
        else:
            best_solution = self._population[0]
            best_struct = self._population[0]['layout']
            prev_solution = best_solution
        it_best = Solution()
        # initialise list to store the objective function value at each iteration
        obj_iter = list()
        # run iterations of the algorithm
        i = 0
        time_elapsed = 0

        print('The maximum run time of this algorithm is ', self._obj_fun.get_max_time())
        num_processors = np.min((mp.cpu_count(), self._pop_size))
        pool = mp.Pool(num_processors)
        # fitness = zip(*)
        # results = pool.map(self.calculate_fitness, [(i, initial_slns) for i in range(self._pop_size)])
        while (i <= self.iter + 1) and (time_elapsed <= self._obj_fun.get_max_time()) and (
                iterations_without_improvement < self._obj_fun.get_termination_iter()):
            time_elapsed = (datetime.now() - start_time).total_seconds()
            print('current time elapsed is', time_elapsed)

            # iterate over population
            storage_pop = list()
            for j in range(self._pop_size):
                pop_a, pop_b, pop_c = self.choose(j)
                # perform mutation
                mutated = self.mutation(pop_a, pop_b, pop_c)
                if self._obj_fun.is_multi:
                    trial = self.crossover(mutated, self._population[j]['layout'], len(self._population[j]['layout']),
                                           self.cr)
                else:
                    trial = self.crossover(mutated, self._population[j]['layout'], len(self._population[j]['layout']),
                                           self.cr)

                storage_pop.append(trial)
            results_population = pool.map(self.calculate_fitness, [(i, storage_pop) for i in range(self._pop_size)])

            for j, obj_trial in enumerate(results_population):

                try:

                    # obj_trial = self._obj_fun.get_fitness(trial, self.pf.get_objective_is_multi())
                    if not self.pf.get_objective_is_multi():
                        average_iteration += obj_trial[self._obj_fun._obj_1]
                        logger(i, obj_trial, None, True, self.get_instance_name(), 1)
                    else:
                        logger(i, obj_trial, None, True, self.get_instance_name(), self.pf.get_objective_is_multi())

                except Exception as e:
                    print('why is there an exception')
                    print(e)
                    continue

                i += 1
                if self.pf.get_objective_is_multi():
                    is_updated, self._pareto_population = self.pf.check_if_dominance(self._pareto_population, obj_trial,
                                                                                     1)

                    if len(self._pareto_population) == 1:
                        print('the size of the population is only 1')
                    if self.pf.check_dominance([obj_trial[self.pf.obj_key_1], obj_trial[self.pf.obj_key_2]],
                                               [self._population[j][self.pf.obj_key_1], self._population[j][
                                                   self.pf.obj_key_2]]):  # if solution dominates existing #FIXME some error here true but not entering

                        iterations_without_improvement = 0
                        self._population[j] = obj_trial

                        logger(self.it_process, obj_trial, self._population, True,
                               self.instance_name + '/population_logger_strict_non_pareto.csv', 1)
                        logger(self.it_process, obj_trial, self._pareto_population, True,
                               self.instance_name + '/population_logger_pareto.csv', 1)
                    else:
                        if self.pf.calculate_difference(obj_trial, self._population[j]):
                            iterations_without_improvement = 0
                            self._population[j] = obj_trial
                            self._pareto_population = self.pf.Pareto_F
                            logger(self.it_process, obj_trial, self._population, True,
                                   self.instance_name + '/population_logger_strict_non_pareto.csv', 1)
                            logger(self.it_process, obj_trial, self._pareto_population, True,
                                   self.instance_name + '/population_logger_pareto.csv', 1)

                    if it_best is None:
                        it_best = obj_trial
                    else:
                        if self.pf.weighted_obj(obj_trial) < self.pf.weighted_obj(it_best):
                            it_best = obj_trial




                else:
                    if obj_trial[self._obj_fun._obj_1] < self._population[j][self._obj_fun._obj_1]:
                        iterations_without_improvement = 0
                        self._population[j] = obj_trial

                    if it_best is None:
                        it_best = obj_trial
                    else:
                        if obj_trial[self._obj_fun._obj_1] < it_best[self.obj_1]:
                            iterations_without_improvement = 0
                            it_best = obj_trial
                    if obj_trial[self._obj_fun._obj_1] < best_solution[self.obj_1]:
                        best_solution = obj_trial

                    logger(self.it_process, obj_trial, best_solution, name=self.get_instance_name(),
                           local_best1=it_best)

                self.it_process += 1

                # self._sort_memory()  # should I sort
            self._sort_memory()
            best_solution = self._population[0]
            best_struct = self._population[0]['layout']

            iterations_without_improvement += 1

            obj_iter.append((i, it_best[self.obj_1], best_solution[self.obj_1], average_iteration / self._pop_size))
            average_iteration = 0
            # print('Iteration: ', i, ' with best objective: ', best_solution)
            # print("Iterations without improvement currently", iterations_without_improvement)
            it_best = Solution()
        if self._obj_fun.is_multi:
            self._pareto_population = self.pf.pareto_run(self._pareto_population)
            return self._pareto_population, self._population

        else:
            output_a = list()
            output_b = list()
            output_c = list()
            output_d = list()
            for a, b, c, d in obj_iter:
                output_a.append(a)
                output_b.append(b)
                output_c.append(c)
                output_d.append(d)
            return output_a, output_b, output_c, best_solution, best_struct

    def Calculate_Temp_mp(self, slns, acceptance_prob, multi=False):
        fitness_list = list()

        feasibility = 20000
        if multi is False:
            for fitness in slns:

                if fitness.get(self._obj_fun._obj_1) < feasibility:
                    fitness_list.append(fitness.get(self._obj_fun._obj_1))
                    if self.best_energy and fitness.get(self._obj_fun._obj_1) <= self.best_energy.get(
                            self._obj_fun._obj_1):
                        self.best_energy = fitness
                    elif self.best_energy is None:
                        self.best_energy = fitness
            "Initial Temp is"

            return (-st.stdev(fitness_list)) / np.log(1 - acceptance_prob)
        else:
            fitness_list_2 = list()
            for fitness in slns:
                fitness_list.append(fitness.get(self._obj_fun._obj_1))  # TODO handle specific swithhing of objectives.
                fitness_list_2.append(fitness.get(self._obj_fun._obj_2))

            # Temp1
            Temp1 = (-st.stdev(fitness_list)) / np.log(1 - acceptance_prob)
            Temp2 = (-st.stdev(fitness_list_2)) / np.log(1 - acceptance_prob)

            # Weight of the temperatures
            w_1 = sum(fitness_list) / (sum(fitness_list) + sum(fitness_list_2))
            w_2 = sum(fitness_list_2) / (sum(fitness_list) + sum(fitness_list_2))
            Temp3 = (Temp1 + Temp2) / (w_1 * Temp2 + w_2 * Temp1)

            self.best_energy = slns[0]
            low_best = 1e5
            for i, val in enumerate(fitness_list):
                low = w_1 * fitness_list[i] + w_2 * fitness_list_2[i]
                if low < low_best:
                    low_best = low
                    self.best_energy = slns[i]

            return Temp3  # return the first temperature for now #TODO 3 phase temperature schema
