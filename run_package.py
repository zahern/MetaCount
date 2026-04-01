import pandas as pd
import numpy as np
from metacountregressor.solution import ObjectiveFunction
from metacountregressor.metaheuristics import (harmony_search,
                                            differential_evolution,
                                            simulated_annealing)


## MCR ###

if __name__ == 'main':
    #This is a script to run the package version of MCR
    # Read data from CSV file
    df = pd.read_csv(
    "https://raw.githubusercontent.com/zahern/data/main/Ex-16-3.csv")
    X = df
    y = df['FREQ']  # Frequency of crashes
    X['Offset'] = np.log(df['AADT']) # Explicitley define how to offset the data, no offset otherwise
    # Drop Y, selected offset term and  ID as there are no panels
    X = df.drop(columns=['FREQ', 'ID'])  

    #some example argument, these are defualt so the following line is just for claritity. See the later agruments section for detials.
    arguments = {'algorithm': 'hs', 'test_percentage': 0.15, 'test_complexity': 6, 'instance_number':1,
                'val_percentage':0.15, 'obj_1': 'bic', '_obj_2': 'RMSE_TEST', "_max_time": 6}
    # Fit the model with metacountregressor
    obj_fun = ObjectiveFunction(X, y, **arguments)
    #replace with other metaheuristics if desired
    results = harmony_search(obj_fun)
