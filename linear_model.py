import numpy as np
import pandas as pd

import random

np.random.seed(42)  # Set the seed for NumPy
random.seed(42)     # Set the seed for the random module

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from metacountregressor.solution import ObjectiveFunction
from metacountregressor.metaheuristics import (harmony_search,
                                            differential_evolution,
                                            simulated_annealing)


from metacountregressor.helperprocess import delete_folder_and_contents

from metacountregressor import helperprocess

try:
    delete_folder_and_contents('1')
except Exception as e:
    print(e)
# Load the data
df = pd.read_csv('data/Real estate.csv')
y = df['Y house price of unit area']
X = df.drop(['No', 'Y house price of unit area'], axis =1)
#Florian Data
F_DATA = True
if F_DATA:
    scaler = StandardScaler()
    df = pd.read_csv('Parking analysis/mixed_data_for_nlogit_v2.csv')
    y = df['AmountDiff']
    X = df.drop(['AmountDiff'], axis =1)
  
    keep_these = ['StayDrtn', 'Weekend', 'StrtTmHr', 'AmountA', 'TrnvrAdj', 'LctnAlcS', 'LctnElzS', 'LctnTrbS','NtPd', 'TmTClrwy']
    X = X[keep_these]
    #X[keep_these] = scaler.fit_transform(X[keep_these])

print("X=", X.shape, "\ny=", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, random_state=101)


# Define example arguments (defaults for clarity); see documentation for details
manual_fit_spec = {
    'fixed_terms': ['const', 'Weekend', 'StrtTmHr', 'TmTClrwy', 
                    'TrnvrAdj', 'LctnElzS', 'NtPd'],
    'rdm_terms': ['StayDrtn:normal', 'LctnAlcS:normal'],
    'rdm_cor_terms': [],
    'grouped_terms': [],
    'hetro_in_means': [],
    'transformations': ['no', 'nil', 'nil', 'nil', 'nil', 'nil', 'nil', 'nil', 'nil'],
    'dispersion': 1
}
#### TEST 

DE_PARAM_GRID = {
            '_AI': [1],  # Adjustment Index
            '_crossover_perc': [5, 0.3, 0.4, 0.5, 0.6],  # Crossover Percentage
            '_max_iter': [10000000],  # Maximum Iterations
            '_pop_size': [10, 25, 50]  # Population Size
        }
de_combinations = helperprocess.generate_param_combinations(DE_PARAM_GRID)


args_de = de_combinations[0]







#####





# Define example arguments (defaults for clarity); see documentation for details
arguments = {
    'algorithm': 'de',           # Harmony Search algorithm
    'test_percentage': 0,     # 15% of data for testing
    'complexity_level': 3,        # Complexity of the test
    'instance_number': 1,        # Instance number
    'val_percentage': 0,      # 15% of data for validation
    '_obj_1': 'bic',              # First objective: Bayesian Information Criterion
    '_obj_2': 'MAE',       # Second objective: Root Mean Square Error on Test data
    '_max_time': 12,               # Maximum time for the process (seconds)
    'linear_model': True,        # Use linear model, False for Poisson regression
    '_transformations': ['nil'], # No transformations
    'is_multi': False,         # Single output
    'Manual_Fit': manual_fit_spec
}
# Fit the model using MetaCountRegressor
obj_fun = ObjectiveFunction(X, y, **arguments)

# Use Harmony Search metaheuristic to optimize the objective function
results = differential_evolution(obj_fun, None, **args_de)
helperprocess.results_printer(results, arguments['algorithm'], int(arguments['is_multi']))