import warnings
import argparse
import csv
import faulthandler
import sys
import timeit
from collections import namedtuple
print('loaded standard packages')

import numpy as np

import pandas as pd
from .helperprocess import*
print('loaded helper')
from .metaheuristics import (differential_evolution,
                            harmony_search, 
                            simulated_annealing)
from .solution import ObjectiveFunction








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
