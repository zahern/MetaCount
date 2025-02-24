import argparse
import pandas as pd
import numpy as np
from metacountregressor.solution import ObjectiveFunction
from metacountregressor.metaheuristics import (harmony_search,
                                            differential_evolution,
                                            simulated_annealing)


from metacountregressor import helperprocess



''' PLEASE CHANGE THE MAIN CORE OF THE DATA HERE'''
def setup_main(**kwargs):
    # Step 1: Read data
    df = pd.read_csv("rural_int.csv")
    #drop irrelevant factors in our data we never want to consider
    df.drop(columns=['year', 'orig_ID',
                                    'jurisdiction', 'town', 'maint_region', 'weather_station', 'dummy_winter_2', 'month', 'inj.fat', 'PDO', 'zonal_ID', 'ln_AADT', 'ln_seg'], inplace=True)  # was dropped postcode

           
    df['rumble_install_year'] = df['rumble_install_year'].astype('category').cat.codes
    df.rename(columns={"rumble_install_year": "has_rumble"}, inplace=True)
    

    # Step 2: Process Data
    model_terms = {
        'Y': 'crashes',         # Dependent variable
        'group': 'county',       # Grouping column (if any)
        'panels': 'element_ID',      # Panel column (if any)
        'Offset': None       # Offset column (if any)
    }
    a_des, df = helperprocess.set_up_analyst_constraints(df, model_terms)
    print(df.head())
    # Step 3: Drop highly correlated features and keep required terms
    df = helperprocess.interactions(df, keep=['Y', 'group', 'panels', 'Offset', 'FAADT'])
    print(df.head())
    # Step 4: Define configuration dictionary
    config = {
        'FAADT': {
            'type': 'continuous',
            'bounds': [0.0, np.inf],
            'discrete': False,
            'apply_func': lambda x: np.log(x + 1)
        },
        'Y': {'type': 'none'},    # Predictor we don't want to change
        'group': {'type': 'none'},
        'Offset': {'type': 'none'},
        'panels': {'type': 'none'}

    }
    
    # Step 5: Infer types for remaining columns and update the configuration
    dataset = df
    for column in dataset.columns:
        if column not in config:
            config[column] = helperprocess.guess_low_medium_high(column, dataset[column])

    # Step 6: Transform the dataset based on the configuration
    data_new = helperprocess.transform_dataframe(dataset, config)
    print(data_new.head())

    instance = f"arg_combo{kwargs.get('alg')}, {kwargs.get('line')}"
    print(model_terms.get('group'))
    # Step 7: Define solution arguments
    model_terms = {
        'Y': 'Y',         # Dependent variable
        'group': 'group',       # Grouping column (if any)
        'panels': 'panels',      # Panel column (if any)
        'Offset': None       # Offset column (if any)
    }
    
    arguments = {
        'algorithm': 'hs',
        'is_multi': 1,
        'test_percentage': 0.2,
        'val_percenetage:': 0.2,
        'test_complexity': 6,
        'instance_number': instance, 
        'distribution': ['normal', 'ln_normal', 'triangular', 'uniform'],
        'Model': [[0], [1]],
        'transformations': ['no'],
        'method_ll': 'BFGS_2',
        '_max_time': 10,
        'decisions': None,
        'model_terms': model_terms
    }
    y = data_new[['Y']]
    X = data_new.drop(columns=['Y'])
    call_meta_args(kwargs.get('alg'), kwargs.get('line'), X, y, arguments)


def call_meta_args(alg,line, X, y, arguments):
    
    if alg == 'hs':
        # Harmony Search Hyperparameter Ranges
        HS_PARAM_GRID = {
            '_hms': [10, 20, 30],  # Harmony Memory Size
            '_max_imp': [1000],  # Maximum Improvisations
            '_hmcr': [0.3, 0.5, 0.7],  # Harmony Memory Consideration Rate
            '_mpai': [1, 2, 3],  # Minimum Pitch Adjustment Index
        }
        hs_combinations = helperprocess.generate_param_combinations(HS_PARAM_GRID)
        # Validate line number
        if line < 0 or line >= len(hs_combinations):
            raise ValueError(f"Invalid line number {line} for Harmony Search. "
                             f"Valid range: 0 to {len(hs_combinations) - 1}.")
        obj_fun = ObjectiveFunction(X, y, **arguments)
        args_hs = hs_combinations[line]
        results = harmony_search(obj_fun, None, args_hs)
        return results
    elif alg == 'de':
        # Differential Evolution Hyperparameter Ranges
        DE_PARAM_GRID = {
            '_AI': [1, 2, 3],  # Adjustment Index
            '_crossover_perc': [0.2, 0.3, 0.4, 0.5, 0.6],  # Crossover Percentage
            '_max_iter': [1000],  # Maximum Iterations
            '_pop_size': [10, 25, 50],  # Population Size
        }
        de_combinations = helperprocess.generate_param_combinations(DE_PARAM_GRID)
        # Validate line number
        if line < 0 or line >= len(de_combinations):
            raise ValueError(f"Invalid line number {line} for Differential Evolution. "
                             f"Valid range: 0 to {len(de_combinations) - 1}.")
        obj_fun = ObjectiveFunction(X, y, **arguments)
        args_de = de_combinations[line]
        results = differential_evolution(obj_fun, None, args_de)
        return results
    else:
        # Simulated Annealing Hyperparameter Ranges
        SA_PARAM_GRID = {
            'alpha': [0.9, 0.95, 0.99],  # Cooling Rate
            'STEPS_PER_TEMP': [5, 10, 20, 30],  # Steps Per Temperature
            'INTL_ACPT': [0.3, 0.5, 0.7],  # Initial Acceptance Rate
            '_crossover_perc': [0.2, 0.3, 0.4],  # Crossover Percentage
            'MAX_ITERATIONS': [1000],  # Fixed Value
            '_num_intl_slns': [10, 25, 50],  # Number of Initial Solutions
        }


    
        sa_combinations = helperprocess.generate_param_combinations(SA_PARAM_GRID)
        if line < 0 or line >= len(sa_combinations):
            raise ValueError(f"Invalid line number {line} for Differential Evolution. "
                             f"Valid range: 0 to {len(sa_combinations) - 1}.")
        args_sa = sa_combinations[line]
        #get string_id from combo
        obj_fun = ObjectiveFunction(X, y, **arguments)
        results = simulated_annealing(obj_fun, None, args_sa)
        return results
    



# Use argparse to handle command-line arguments
if __name__ == "__main__":
    #need to pass line number and args
    parser = argparse.ArgumentParser(description="Run specific model setup.")
    #need to pass some jobs to access the right infor
    #embed that into setup main
    parser.add_argument(
        '--alg', 
        type=str, 
        default='hs', 
        choices=['hs', 'de', 'sa'], 
        help="Algorithm to run (hs, de, sa)"
    )
    parser.add_argument(
        '--line', 
        type=int, 
        default=0, 
        help="Line number to specify parameter combination"
    )
    args = parser.parse_args()

    # Call setup_main with parsed arguments
    setup_main(alg=args.alg, line=args.line)