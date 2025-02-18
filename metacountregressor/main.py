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






def process_arguments(**kwargs):
    '''
    TRYING TO TURN THE CSV FILES INTO RELEVANT ARGS
    '''
    #dataset
    '''
    if kwargs.get('dataset_file', False
    ):
        dataset = pd.read_csv(kwargs.get('dataset_file'))
        named_data_headers = dataset.columns.tolist()
        decision_constants = {name: list(range(7)) for name in named_data_headers}
        data_info = {


            'AADT': {
                'type': 'continuous',
                'bounds': [0.0, np.infty],
                'discrete': False,
                'apply_func': (lambda x: np.log(x + 1)),
            },
            'SPEED': {
                'type': 'continuous',
                'bounds': [0, 100],
                'enforce_bounds': True,
                'discrete': True
            },
            'TIME': {
                'type': 'continuous',
                'bounds': [0, 23.999],
                'discrete': False
            }
        }
        #remove ID CoLUMNS from dataset
        dataset = dataset.drop(columns = [
            'ID'
        ])
        for c in dataset.columns:
            if c not in data_info.keys():
                data_info[c] = {'type': 'categorical'}

        data_new  =helperprocess.transform_dataframe(dataset,data_info)

        update_constant = kwargs.get('analyst_constraints')
        #update the decision_constraints
    '''
    data_characteristic = pd.read_csv(kwargs.get('problem_data', 'problem_data.csv'))
    # Extract the column as a list of characteristic names
    #name_data_characteristics = data_characteristic.columns.tolist()

    # Create the dictionary
    #decision_constraints = {name: list(range(7)) for name in name_data_characteristics}

    #print('this gets all the features, I need to remove...')

    analyst_d = pd.read_csv(kwargs.get('decison_constraints', 'decisions.csv'))
    hyper = pd.read_csv('setup_hyper.csv')

    new_data = {'data': data_characteristic,
                'analyst':analyst_d,
                'hyper': hyper}
    return new_data

def process_package_arguments():

    new_data = {}
    pass


def main(args, **kwargs):

    '''METACOUNT REGRESSOR TESTING ENVIRONMENT'''

    '''
    TESTING_ENV = False
    if TESTING_ENV:

        import statsmodels.api as sm

        data = sm.datasets.sunspots.load_pandas().data
        # print(data.exog)
        data_exog = data['YEAR']
        data_exog = sm.add_constant(data_exog)
        data_endog = data['SUNACTIVITY']

        # Instantiate a gamma family model with the default link function.
        import numpy as np

        gamma_model = sm.NegativeBinomial(data_endog, data_exog)
        gamma_results = gamma_model.fit()

        print(gamma_results.summary())

        # NOW LET's COMPARE THIS TO METACOUNT REGRESSOR
        import metacountregressor
        from importlib.metadata import version
        print(version('metacountregressor'))
        import pandas as pd
        import numpy as np
        from metacountregressor.solution import ObjectiveFunction
        from metacountregressor.metaheuristics import (harmony_search,
                                                       differential_evolution,
                                                       simulated_annealing)

        # Model Decisions,
        manual_fit_spec = {

            'fixed_terms': ['const', 'YEAR'],
            'rdm_terms': [],
            'rdm_cor_terms': [],
            'grouped_terms': [],
            'hetro_in_means': [],
            'transformations': ['no', 'no'],
            'dispersion': 1  # Negative Binomial
        }

        # Arguments
        arguments = {
            'algorithm': 'hs',
            'test_percentage': 0,
            'test_complexity': 6,
            'instance_number': 'name',
            'Manual_Fit': manual_fit_spec
        }
        obj_fun = ObjectiveFunction(data_exog, data_endog, **arguments)
    '''


    print('the args is:', args)
    print('the kwargs is', kwargs)

    # removing junk files if specicified
    helperprocess.remove_files(args.get('removeFiles', True))

    # do we want to run a test
    if args.get('com', False) == 'MetaCode':
        print('Testing the Python Package')  # TODO add in python package import
        # Read data from CSV file
        df: TextFileReader | DataFrame | Any = pd.read_csv(
            "https://raw.githubusercontent.com/zahern/data/main/Ex-16-3.csv")
        X = df
        y = df['FREQ']  # Frequency of crashes
        X['Offset'] = np.log(df['AADT'])  # Explicitley define how to offset the data, no offset otherwise
        # Drop Y, selected offset term and  ID as there are no panels
        X = df.drop(columns=['FREQ', 'ID', 'AADT'])

        # some example argument, these are defualt so the following line is just for claritity
        args = {'algorithm': 'hs', 'test_percentage': 0.15, 'test_complexity': 6, 'instance_number': 1,
                'val_percentage': 0.15, 'obj_1': 'bic', '_obj_2': 'RMSE_TEST', "MAX_TIME": 6}
        # Fit the model with metacountregressor
        obj_fun = ObjectiveFunction(X, y, **args)
        # replace with other metaheuristics if desired
        results = harmony_search(obj_fun)
        print(results)
        print('exiting..')
        return 0

    dataset = int(args.get('problem_number', 3))
    print('the dataset is', dataset)
    manual_fit_spec = args.get('Manual_Fit', None)
    if dataset == 1:
        print('Stage 5 A Short.')
        df = pd.read_csv('./data/1848.csv')  # read in the data
        y_df = df[['FSI']]  # only consider crashes
        y_df.rename(columns={"FSI": "Y"}, inplace=True)
        x_df = df.drop(columns=['FSI'])
        x_df = helperprocess.as_wide_factor(x_df)

    elif dataset == 3:
        print('Stage 5 A Data Complete.')
        x_df = pd.read_csv('./data/Stage5A_1848_All_Initial_Columns.csv')  # drop the ID columns
        drop_these = ['Id', 'ID', 'old', 'G_N']
        for i in drop_these:
            x_df.drop(x_df.filter(regex=i).columns, axis=1, inplace=True)
        y_df = x_df[['Headon']].copy()  # only consider crashes
        y_df.rename(columns={"Headon": "Y"}, inplace=True)
        x_df['Offset'] = np.log(x_df['LEN_YR'] * 1000) / 10  # Offset

        x_df = x_df.drop(columns=['Headon', 'LEN_YR'])  # drop the main predictor
        drop_these_too = ['LEN', 'VS_Curve', 'FW_RS', 'RD', 'M', 'SP', 'FW']
        for i in drop_these_too:
            x_df.drop(x_df.filter(regex=i).columns, axis=1, inplace=True)

        helperprocess.as_wide_factor(x_df, args.get('separate_out_factors', False), keep_original=1)
        # x_df = helperprocess.interactions(x_df)
        manual_fit_spec = {
            'fixed_terms': ['Constant', 'US', 'RSMS', 'MCV'],
            'rdm_terms': ['RSHS:normal', 'AADT:normal', 'Curve50:normal'],
            'rdm_cor_terms': [],
            'grouped_terms': [],
            'hetro_in_means': [],
            'transformations': ['no', 'log', 'no', 'no', 'no', 'no', 'no'],
            'dispersion': 0
        }

        keep = ['Constant', 'US', 'RSMS', 'MCV', 'RSHS', 'AADT', 'Curve50', 'Offset']
        x_df = helperprocess.interactions(x_df, keep)

    elif dataset == 4:
        manual_fit_spec = {
            'fixed_terms': ['const', 'LOWPRE', 'GBRPM', 'FRICTION'],
            'rdm_terms': ['EXPOSE:normal', 'INTPM:normal', 'CPM:normal', 'HISNOW:normal'],
            'rdm_cor_terms': [],
            'grouped_terms': [],
            'hetro_in_means': [],
            'transformations': ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no'],
            'dispersion': 1
        }
        '''
        manual_fit_spec = {
            'fixed_terms': ['const', 'LOWPRE', 'GBRPM', 'FRICTION', 'EXPOSE', 'INTPM', 'CPM', 'HISNOW'],
            'rdm_terms': [],
            'rdm_cor_terms': [],
            'grouped_terms': [],
            'hetro_in_means': [],
            'transformations': ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no'],
            'dispersion': 1
        }
        '''


        '''
        print('overriding this delete, just want to test the NB')
        manual_fit_spec = {
            'fixed_terms': ['const'],
            'rdm_terms': [],
            'rdm_cor_terms': [],
            'grouped_terms': [],
            'hetro_in_means': [],
            'transformations': ['no'],
            'dispersion': 1
        }
        '''
        df = pd.read_csv('./data/Ex-16-3.csv')  # read in the data
        y_df = df[['FREQ']].copy()  # only consider crashes
        y_df.rename(columns={"FREQ": "Y"}, inplace=True)
        x_df = df.drop(columns=['FREQ', 'ID'])
        # grabbing the offset amount
        x_df['Offset'] = np.log(1 + x_df['AADT'] * x_df['LENGTH'] * 365 / 100000000)
        x_df = x_df.drop(columns=['AADT', 'LENGTH'])

        if args.get('seperate_out_factors', 0):

            x_df = helperprocess.as_wide_factor(x_df, keep_original=0,
                                                exclude=['INTECHAG', 'CURVES', 'MIMEDSH', 'MXMEDSH', 'SPEED'])
            x_df = pd.DataFrame(
                {col: x_df[col].astype(int) if x_df[col].dropna().isin([True, False]).all() else x_df[col] for col in
                 x_df})

        else:
            original_columns = x_df.columns
            x_df = x_df.astype({col: pd.Int64Dtype() if x_df[col].dtype == 'Int64' else float for col in x_df.columns})
            # Restore original columns and fill missing values with 0
            x_df = x_df.reindex(columns=original_columns, fill_value=0)
            x_df = pd.DataFrame(
                {col: x_df[col].astype(int) if x_df[col].dropna().isin([True, False]).all() else x_df[col] for col in
                 x_df})
            # x_df = pd.get_dummies(x_df, columns=['FC'], prefix=['FC'], prefix_sep='_')
        keep = ['Offset', 'LOWPRE', 'GBPRM', 'FRICTION', 'EXPOSE', 'INTPM', 'CPM', 'HISNOW']
        x_df = helperprocess.interactions(x_df, keep, drop_this_perc=0.8)



    elif dataset == 7:
        df = pd.read_csv('./data/artificial_mixed_corr_2023_MOOF.csv')  # read in the data
        y_df = df[['Y']].copy()  # only consider crashes

        x_df = df.drop(columns=['Y'])  # was dropped postcode

        # x_df1 = helperprocess.PCA_code(x_df, 10)
        x_df = helperprocess.as_wide_factor(x_df, keep_original=1)
        keep = ['X1', 'X2', 'X3', 'const']
        x_df = helperprocess.interactions(x_df, keep, drop_this_perc=0.8)
        manual_fit_spec = {
            'fixed_terms': ['const'],
            'rdm_terms': [],
            'rdm_cor_terms': ['X1:normal', 'X2:normal', 'X3:normal'],
            'grouped_terms': [],
            'hetro_in_means': [],
            'transformations': ['no', 'no', 'no', 'no'],
            'dispersion': 0
        }
    elif dataset == 8:
        print('Main County')
        df = pd.read_csv('./data/rural_int.csv')  # read in the data
        y_df = df[['crashes']].copy()  # only consider crashes
        y_df.rename(columns={"crashes": "Y"}, inplace=True)
        panels = df['orig_ID']
        try:
            x_df = df.drop(columns=['crashes', 'year', 'orig_ID',
                                    'jurisdiction', 'town', 'maint_region', 'weather_station', 'dummy_winter_2'])  # was dropped postcode
            print('dropping for test')
            x_df = x_df.drop(columns=['month', 'inj.fat', 'PDO'])
            x_df = x_df.drop(columns = [ 'zonal_ID', 'ln_AADT', 'ln_seg'])
            x_df['rumble_install_year'] = x_df['rumble_install_year'].astype('category').cat.codes
            x_df.rename(columns={"rumble_install_year": "has_rumble"}, inplace=True)
        except Exception as e:
            print(e)
            x_df = df.drop(columns=['Y'])  # was dropped postcode

        group_grab = x_df['county']
        x_df = x_df.drop(columns =['county'])
        x_df = helperprocess.interactions(x_df, drop_this_perc=0.8)
        x_df['county'] = group_grab

        print('benchmark specification')
        manual_fit_spec = {
            'fixed_terms': ['const', 'monthly_AADT', 'segment_length', 'speed', 'paved_shoulder', 'curve'],
            'rdm_terms': [],
            'rdm_cor_terms': [],
            'grouped_terms': ['DP01:normal', 'DX32:normal'],
            'hetro_in_means': [],
            'transformations': ['no', 'no', 'no', 'no', 'no', 'no'],
            'dispersion': 0
        }

    elif dataset == 9:
        df = pd.read_csv('panel_synth.csv')  # read in the data
        y_df = df[['Y']].copy()  # only consider crashes
        y_df.rename(columns={"crashes": "Y"}, inplace=True)
        panels = df['ind_id']

        x_df = df.drop(columns=['Y'])
        print(x_df)
        manual_fit_spec = {
            'fixed_terms': ['constant'],
            # 'rdm_terms': [],
            'rdm_terms': ['added_random1:grpd|normal', 'added_random2:grpd|normal', 'added_random3:grpd|normal'],
            'rdm_cor_terms': [],
            'grouped_terms': [],
            'hetro_in_means': [],
            # 'transformations': ['no'],
            'transformations': ['no', 'no', 'no', 'no'],
            'dispersion': 0
        }

        # x_df = helperprocess.as_wide_factor(x_df, keep_original=1)
        keep = ['group', 'constant', 'element_ID']

        x_df = helperprocess.interactions(x_df, keep)


    elif dataset ==10:  # the dataset has been selected in the program as something else
        data_info = process_arguments(**args)
        data_info['hyper']
        data_info['analyst']
        data_info['data']['Y']
        #data_info['data']['Group'][0]
        #data_info['data']['Panel'][0]
        args['decisions'] = data_info['analyst']

        if type(data_info['data']['Grouped'][0]) == str and len(data_info['data']['Grouped'][0]) >1:
            args['group'] = data_info['data']['Grouped'][0]
            args['ID'] = data_info['data']['Grouped'][0]
        if type(data_info['data']['Panel'][0]) == str and len(data_info['data']['Panel'][0])>1:
            args['panels'] = data_info['data']['Panel'][0]

        df = pd.read_csv(str(data_info['data']['Problem'][0]))
        x_df = df.drop(columns=[data_info['data']['Y'][0]])
        y_df = df[[data_info['data']['Y'][0]]]
        y_df.rename(columns={data_info['data']['Y'][0]: "Y"}, inplace=True)
        print('test') #FIXME
    else:
        print('PROCESS THE PACKAGE ARGUMENTS SIMULIAR TO HOW ONE WOULD DEFINE THE ENVIRONMENT')
        data_info =process_package_arguments()


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
    
    
    BATCH_JOB = True

    if BATCH_JOB:
        parser.add_argument('-dataset_file', default='data/Ex-16-3.csv', help='supply the path to the dataset')

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
                print('WARNING: TESTING ENVIRONMENT, TURN OFF FOR RELEASE')
                parser.add_argument('-problem_number', default='10')

            if 'algorithm' not in args:
                parser.add_argument('-algorithm', type=str, default='hs',
                                    help='optimization algorithm')
            elif 'Manual_Fit' not in args:
                parser.add_argument('-Manual_Fit', action='store_false', default=None,
                                    help='To fit a model manually if desired.')

            parser.add_argument('-seperate_out_factors', action='store_false', default=False,
                                help='Trie of wanting to split data that is potentially categorical as binary'
                                    ' we want to split the data for processing')
            parser.add_argument('-supply_csv', type = str, help = 'enter the name of the csv, please include it as a full directories')

    else:  # DIDN"T SPECIFY LINES TRY EACH ONE MANNUALY
        print("RUNNING WITH ARGS")
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


