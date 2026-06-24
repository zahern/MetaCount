import warnings
import argparse
import csv
import faulthandler
import ast
from typing import Any
import cProfile
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
import helperprocess
from metaheuristics import (differential_evolution,
                            harmony_search,
                            simulated_annealing)
from solution import ObjectiveFunction


warnings.simplefilter("ignore")

faulthandler.enable()


def convert_df_columns_to_binary_and_wide(df):
    columns = list(df.columns)

    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df


def process_arguments():
    '''
    TRYING TO TURN THE CSV FILES INTO RELEVANT ARGS
    '''
    try:
        data_characteristic = pd.read_csv('problem_data.csv')
        analyst_d = pd.read_csv('decisions.csv')
        hyper = pd.read_csv('setup_hyper.csv')
    except Exception as e:
        print(e)
        print('Files Have Not Been Set Up Yet..')
        print('Run the App')
        exit()

    new_data = {'data': data_characteristic,
                'analyst':analyst_d,
                'hyper': hyper}
    return new_data

def main(args, **kwargs):
    '''METACOUNT REGRESSOR TESTING ENVIRONMENT'''




    print('the args is:', args)
    print('the kwargs is', kwargs)

    # removing junk files if specicified
    helperprocess.remove_files(args.get('removeFiles', True))

    # do we want to run a test


    data_info = process_arguments()
    data_info['hyper']
    data_info['analyst']
    data_info['data']['Y']
    #data_info['data']['Group'][0]
    #data_info['data']['Panel'][0]
    args['decisions'] = data_info['analyst']
    grouped_c = data_info['data']['Grouped'][0]
    if isinstance(data_info['data']['Grouped'][0],str):
        args['group'] = data_info['data']['Grouped'][0]
        args['ID'] = data_info['data']['Panel'][0]
    if isinstance(data_info['data']['Panel'][0],str):
        args['panels'] = data_info['data']['Panel'][0]

    df = pd.read_csv(str(data_info['data']['Problem'][0]))
    x_df = df.drop(columns=[data_info['data']['Y'][0]])
    # drop the columns of x_df where column is string exclude the column stype args['group']
    exclude_column = args['group']
    columns_to_keep = x_df.dtypes != 'object'
    columns_to_keep |= (x_df.columns == exclude_column)
    x_df = x_df.loc[:, columns_to_keep]
    y_df = df[[data_info['data']['Y'][0]]]
    y_df.rename(columns={data_info['data']['Y'][0]: "Y"}, inplace=True)

    manual_fit_spec = None #TODO add in manual fit
    if args['Keep_Fit'] == str(2) or args['Keep_Fit'] == 2:
        if manual_fit_spec is None:
            args['Manual_Fit'] = None
        else:
            print('fitting manually')
            args['Manual_Fit'] = manual_fit_spec
    if args['problem_number'] == str(8) or args['problem_number'] == 8:
        print('Maine County Dataset.')
        args['group'] = 'county'
        args['panels'] = 'element_ID'
        args['ID'] = 'element_ID'
        args['_max_characteristics'] = 55
    elif args['problem_number'] == str(9) or args['problem_number'] == 9:
        args['group'] = 'group'
        args['panels'] = 'ind_id'
        args['ID'] = 'ind_id'



    args['complexity_level'] = args.get('complexity_level', 6)


    # Initialize AnalystSpecs to None if not manually provided
    args['AnalystSpecs'] = args.get('AnalystSpecs', None)

    if args['algorithm'] == 'sa':
        args_hyperparameters = {'alpha': float(args['temp_scale']),
                                'STEPS_PER_TEMP': int(args['steps']),
                                'INTL_ACPT': 0.5,
                                '_crossover_perc': args['crossover'],
                                'MAX_ITERATIONS': int(args['_max_imp']),
                                '_num_intl_slns': 25,
                                'Manual_Fit': args['Manual_Fit'],
                                'MP': int(args['MP'])}
        helperprocess.entries_to_remove(('crossover', '_max_imp', '_hms', '_hmcr', '_par'), args)
        print(args)

        obj_fun = ObjectiveFunction(x_df, y_df, **args)

        results = simulated_annealing(obj_fun, None, **args_hyperparameters)

        helperprocess.results_printer(results, args['algorithm'], int(args['is_multi']))

        if args['dual_complexities']:
            args['complexity_level'] = args['secondary_complexity']
            obj_fun = ObjectiveFunction(x_df, y_df, **args)
            results = simulated_annealing(obj_fun, None, **args_hyperparameters)
            helperprocess.results_printer(results, args['algorithm'], int(args['is_multi']))

    elif args['algorithm'] == 'hs':
        args['_mpai'] = 1

        obj_fun = ObjectiveFunction(x_df, y_df, **args)
        args_hyperparameters = {
            'Manual_Fit': args['Manual_Fit'],
            'MP': int(args['MP'])
        }

        results = harmony_search(obj_fun, None, **args_hyperparameters)
        helperprocess.results_printer(results, args['algorithm'], int(args['is_multi']))

        if args.get('dual_complexities', 0):
            args['complexity_level'] = args['secondary_complexity']
            obj_fun = ObjectiveFunction(x_df, y_df, **args)
            results = harmony_search(obj_fun, None, **args_hyperparameters)
            helperprocess.results_printer(results, args['algorithm'], int(args['is_multi']))


    elif args['algorithm'] == 'de':
        # force variables
        args['must_include'] = args.get('force', [])

        args_hyperparameters = {'_AI': args.get('_AI', 2),
                                '_crossover_perc': float(args['crossover']),
                                '_max_iter': int(args['_max_imp'])
            , '_pop_size': int(args['_hms']), 'instance_number': int(args['line'])
            , 'Manual_Fit': args['Manual_Fit'],
                                'MP': int(args['MP'])
                                }

        args_hyperparameters = dict(args_hyperparameters)

        helperprocess.entries_to_remove(('crossover', '_max_imp', '_hms', '_hmcr', '_par'), args)
        obj_fun = ObjectiveFunction(x_df, y_df, **args)

        results = differential_evolution(obj_fun, None, **args_hyperparameters)

        helperprocess.results_printer(results, args['algorithm'], int(args['is_multi']))

        if args['dual_complexities']:
            args['complexity_level'] = args['secondary_complexity']
            obj_fun = ObjectiveFunction(x_df, y_df, **args)
            results = differential_evolution(obj_fun, None, **args_hyperparameters)
            helperprocess.results_printer(results, args['algorithm'], int(args['is_multi'])) #TODO FIX This


if __name__ == '__main__':
    """Loading in command line args.  """
    alg_parser = argparse.ArgumentParser(prog='algorithm', epilog='algorithm specific arguments')
    alg_parser.add_argument('-AI', default=2, help='adjustment index. For the allowable movement of the algorithm')
    alg_parser.print_help()
    parser = argparse.ArgumentParser(prog='main',
                                     epilog=main.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter, conflict_handler='resolve')

    parser.add_argument('-line', type=int, default=1,
                        help='line to read in csv to pass in argument')

    if vars(parser.parse_args())['line'] is not None:
        reader = csv.DictReader(open('set_data.csv', 'r'))
        args = list()
        line_number_obs = 0
        for dictionary in reader:  # TODO find a way to handle multiple args
            args = dictionary
            if line_number_obs == int(vars(parser.parse_args())['line']):
                break
            line_number_obs += 1
        args = dict(args)

        for key, value in args.items():
            try:
                # Attempt to parse the string value to a Python literal if value is a string.
                if isinstance(value, str):
                    value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # If there's a parsing error, value remains as the original string.
                pass

            # Add the argument to the parser with the potentially updated value.
            parser.add_argument(f'-{key}', default=value)

        for i, action in enumerate(parser._optionals._actions):
            if "-algorithm" in action.option_strings:
                parser._optionals._actions[i].help = "optimization algorithm"

        override = True
        if override:
            print('todo turn off, in testing phase')
            parser.add_argument('-problem_number', default='10')
            print('did it make it')
        if 'algorithm' not in args:
            parser.add_argument('-algorithm', type=str, default='hs',
                                help='optimization algorithm')
        elif 'Manual_Fit' not in args:
            parser.add_argument('-Manual_Fit', action='store_false', default=None,
                                help='To fit a model manually if desired.')

        parser.add_argument('-seperate_out_factors', action='store_false', default=False,
                            help='Trie of wanting to split data that is potentially categorical as binary'
                                 ' we want to split the data for processing')
        parser.add_argument('-supply_csv', type = str, help = 'enter the name of the csv, please include it as a full directorys')

    else:  # DIDN"T SPECIFY LINES TRY EACH ONE MANNUALY
        parser.add_argument('-com', type=str, default='MetaCode',
                            help='line to read csv')

    # Check the args
    parser.print_help()
    args = vars(parser.parse_args())
    print(type(args))
    # TODO add in chi 2 and df in estimation and compare degrees of freedom this needs to be done in solution

    # Print the args.
    profiler = cProfile.Profile()
    profiler.runcall(main,args)
    profiler.print_stats(sort='time')
    #TOO MAX_TIME


