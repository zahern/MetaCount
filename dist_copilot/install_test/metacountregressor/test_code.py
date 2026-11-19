from solution import ObjectiveFunction
from metaheuristics import (harmony_search,differential_evolution,simulated_annealing)         
#from . import helperprocess
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/zahern/data/refs/heads/main/rural_int.csv")
y = df['crashes']  # Frequency of crashes

df.drop(columns=[ 'year', 'orig_ID',
                                    'jurisdiction', 'town', 'maint_region', 'weather_station', 'dummy_winter_2', 'month', 'inj.fat', 'PDO', 'zonal_ID', 'ln_AADT', 'ln_seg'], inplace=True)  # was dropped postcode

           

arguments_hs = {'_par': 0.3, '_hms': 20}
arguments = {'test_percentage': 0.2, 'complexity_level': 5, 'reg_penalty':0} #Objective args
# Step 2: Process Data
model_terms = {
    'Y': 'crashes',         # Dependent variable
    'group': 'county',       # Grouping column (if any)
    'panels': 'element_ID',      # Panel column (if any)
    'Offset': None       # Offset column (if any)
}


X = df.drop(columns=['crashes']) # setup X based on data
X.columns
print(X.columns)

manual_fit_spec = {
'fixed_terms': ['const', 'DP10'],
'rdm_terms':  [ 'DX32:normal'],
'rdm_cor_terms': [],
'group_rdm': ['DPO1:triangular'],
'hetro_in_means': [],
'transformations': ['no', 'no', 'no', 'no', 'no', 'no'],
'dispersion': 0
}
arguments = {'test_percentage': 0.2, 'complexity_level': 6, 'reg_penalty':0, 'group':'county', 'panels':'element_ID'} #Objective args
arguments['Manual_Fit'] = manual_fit_spec
#initial_solution = None
obj_fun = ObjectiveFunction(X, y, **arguments)
initial_solution = None
results_hs = harmony_search(obj_fun, initial_solution, **arguments_hs)