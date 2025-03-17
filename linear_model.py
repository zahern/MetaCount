import numpy as np
import pandas as pd

import random

np.random.seed(42)  # Set the seed for NumPy
random.seed(42)     # Set the seed for the random module

from sklearn.model_selection import train_test_split
from metacountregressor.solution import ObjectiveFunction
from metacountregressor.metaheuristics import (harmony_search,
                                            differential_evolution,
                                            simulated_annealing)


from metacountregressor.helperprocess import delete_folder_and_contents



delete_folder_and_contents('1')
# Load the data
df = pd.read_csv('data/Real estate.csv')
y = df['Y house price of unit area']
X = df.drop(['No', 'Y house price of unit area'], axis =1)
#Florian Data
F_DATA = True
if F_DATA:
    df = pd.read_csv('Parking analysis/mixed_data_for_nlogit_v2.csv')
    y = df['AmountDiff']
    X = df.drop(['AmountDiff'], axis =1)
    keep_these = ['StayDrtn', 'Weekend', 'StrtTmHr', 'AmountA', 'TrnvrAdj', 'LctnAlcS']
    X = X[keep_these]

print("X=", X.shape, "\ny=", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



manual_fit_spec = {
    'fixed_terms': ['SINGLE', 'LENGTH'],
    'rdm_terms': ['AADT:normal'],
    'rdm_cor_terms': ['GRADEBR:normal', 'CURVES:normal'],
    'grouped_terms': [],
    'hetro_in_means': [],
    'transformations': ['no', 'no', 'log', 'no', 'no', 'no', 'no'],
    'dispersion': 0
}


# Define example arguments (defaults for clarity); see documentation for details
arguments = {
    'algorithm': 'hs',           # Harmony Search algorithm
    'test_percentage': 0,     # 15% of data for testing
    'complexity_level': 3,        # Complexity of the test
    'instance_number': 1,        # Instance number
    'val_percentage': 0,      # 15% of data for validation
    '_obj_1': 'bic',              # First objective: Bayesian Information Criterion
    '_obj_2': 'MAE',       # Second objective: Root Mean Square Error on Test data
    '_max_time': 600000,               # Maximum time for the process (seconds)
    'linear_model': True,        # Use linear model, False for Poisson regression
    '_transformations': ['nil'], # No transformations
    'is_multi': False            # Single output
}
# Fit the model using MetaCountRegressor
obj_fun = ObjectiveFunction(X, y, **arguments)

# Use Harmony Search metaheuristic to optimize the objective function
results = harmony_search(obj_fun)