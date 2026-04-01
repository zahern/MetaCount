import warnings
import argparse
import csv
import faulthandler
import sys
import timeit
from collections import namedtuple
import numpy as np
import pandas as pd
print('loaded standard packages')

from .helperprocess import*
from .data_split_helper import DataProcessor
print('loaded helper')
from .metaheuristics import (differential_evolution,
                            harmony_search, 
                            simulated_annealing)
print('loaded metaheuristics')
from .solution import ObjectiveFunction
from importlib import import_module

__all__ = [
    "DataProcessor",
    "ObjectiveFunction",
    "differential_evolution",
    "harmony_search",
    "simulated_annealing",
    "ExperimentBuilder",
    "StructureEvaluatorLC",
    "JAXMLE",
    "ModelSpec",
    "build_param_index",
    "build_jax_data",
    "mixed_model_loglik",
    "build_model_from_manual_spec",
    "print_summary",
]

_LAZY_EXPORTS = {
    "ExperimentBuilder": ("experiment_package", "ExperimentBuilder"),
    "StructureEvaluatorLC": ("experiment_package", "StructureEvaluatorLC"),
}

_JAX_EXPORTS = {
    "JAXMLE": ("solvers_meta", "JAXMLE"),
    "ModelSpec": ("main_hpc_lc_patch", "ModelSpec"),
    "build_param_index": ("main_hpc_lc_patch", "build_param_index"),
    "build_jax_data": ("main_hpc_lc_patch", "build_jax_data"),
    "mixed_model_loglik": ("main_hpc_lc_patch", "mixed_model_loglik"),
    "build_model_from_manual_spec": ("main_hpc_lc_patch", "build_model_from_manual_spec"),
    "print_summary": ("main_hpc_lc_patch", "print_summary"),
}

_LAZY_EXPORTS.update(_JAX_EXPORTS)


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    try:
        module = import_module(f".{module_name}", __name__)
    except ImportError as exc:
        raise ImportError(
            f"Unable to import {name!r}. Install the package with its JAX "
            f"dependencies available."
        ) from exc

    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


# import pandas as pd
# df = pd.read_csv("https://raw.githubusercontent.com/zahern/data/main/Ex-16-3.csv")


# y = df['FREQ'] #Frequency of crashes
# X =  df.drop(columns=['FREQ', 'ID']) #Drop Y, and ID as there are no panels
# X = pd.get_dummies(X, columns=['FC'], prefix=['FC'], prefix_sep='_').astype(int)
# X['Offset'] = np.log(1+X['AADT'] * X['LENGTH'] * 365 / 100000000)
# #X = interactions(X)    
# #X = pd.get_dummies(X, columns=['FC'], prefix=['FC'], prefix_sep='_')


# # Fit the model with metacountregressor

# other_data = 1
# if other_data:
#     df = pd.read_csv('panel_synth.csv')  # read in the data
#     y = df[['Y']].copy()  # only consider crashes
#     y.rename(columns={"crashes": "Y"}, inplace=True)
#     panels = df['ind_id']
            
#     X = df.drop(columns=['Y', 'alt'])
#     #Model Decisions, Specify for Intial Optimization
#     manual_fit_spec = {
#         'fixed_terms': ['added_fixed1', 'added_fixed2', 'added_fixed3', 'constant'],
#         'rdm_terms': [],
#         'rdm_cor_terms': ['added_random1:grpd| normal', 'added_random2:grpd| uniform', 'added_random3:grpd| triangular'],
#         'grouped_terms': [],
#         'hetro_in_means': [],
#         'transformations': ['no', 'no', 'no', 'no', 'no', 'no', 'no'],
#         'dispersion': 0
#     }
#     arguments = dict()
#     arguments['group'] = 'group'
#     arguments['panels'] =  'ind_id'
#     arguments['ID'] ='ind_id'
# else:
#     #Model Decisions, Specify for Intial Optimization
#     manual_fit_spec = {
#         'fixed_terms': ['const', 'FC_2'],
#         'rdm_terms': ['MXGRADE:triangular', 'AVEPRE:normal'],
#         'rdm_cor_terms': [],
#         'grouped_terms': [],
#         'hetro_in_means': ['URB:triangular', 'ACCESS:triangular', 'FC_1:triangular'],
#         'transformations': ['no', 'no', 'no', 'no', 'no', 'no', 'no'],
#         'dispersion': 0
#     }
        



# #select one of the algorithms
# alg = [harmony_search, differential_evolution, simulated_annealing]
# alg = alg[0] #harmony search





# #Search Arguments
# arguments = {
#     'algorithm': 'hs',
#     'test_percentage': 0.2,
#     'test_complexity': 6,
#     'instance_number': 'name',
#     'Manual_Fit': manual_fit_spec
# }

# arguments['group'] = 'group'
# arguments['panels'] =  'ind_id'
# arguments['ID'] ='ind_id'


# arguments_hyperparamaters = dict()

# # end default constructor
# obj_fun = ObjectiveFunction(X, y, **arguments)
# results = alg(obj_fun, None, **arguments_hyperparamaters)
