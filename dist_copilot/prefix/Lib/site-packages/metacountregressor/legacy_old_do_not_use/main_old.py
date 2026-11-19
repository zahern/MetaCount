import warnings
import argparse
import csv
import faulthandler
import ast
import numpy as np
import pandas as pd
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


def main(args):
    reader = csv.DictReader(open('set_data.csv', 'r'))
    arguments = list()
    line_number_obs = 0
    for dictionary in reader:  # TODO find a way to handle multiple arguments
        arguments = dictionary
        if line_number_obs == int(args['-line']):
            break
        line_number_obs += 1
    arguments['instance_number'] = args['-line']
    arguments = dict(arguments)
    print('the arguments is:', arguments)

    initial_complexity = 6  # representes Hetrogeneity in the the means group random parameterrs
    dual_complexities = 0
    secondary_complexity = 6  # 5 Group Random Parameters
    forced_variables = None
    separate_out_factors = 0  # convert data into binary (long format)
    removeFiles = 1  # remove the tex files which store the saved models
    postprocess = 0  # postprocess the solutions..
    defineApp = 0
    helperprocess.remove_files(removeFiles)

    dataset = 7
    print('reading set data...')
    reader = csv.DictReader(open('set_data.csv', 'r'))
    arguments = list()
    loop_condition = 1
    line_number_obs = 0
    if '-com' in args and args['-com'] == 'MetaCode':
        print('do this')
        # Read data from CSV file
        df = pd.read_csv(
            "https://raw.githubusercontent.com/zahern/data/main/Ex-16-3.csv")
        X = df
        y = df['FREQ']  # Frequency of crashes
        X['Offset'] = np.log(df['AADT'])  # Explicitley define how to offset the data, no offset otherwise
        # Drop Y, selected offset term and  ID as there are no panels
        X = df.drop(columns=['FREQ', 'ID', 'AADT'])

        # some example argument, these are defualt so the following line is just for claritity
        arguments = {'algorithm': 'hs', 'test_percentage': 0.15, 'test_complexity': 6, 'instance_number': 1,
                     'val_percentage': 0.15, 'obj_1': 'bic', '_obj_2': 'RMSE_TEST', "MAX_TIME": 6}
        # Fit the model with metacountregressor
        obj_fun = ObjectiveFunction(X, y, **arguments)
        # replace with other metaheuristics if desired
        results = harmony_search(obj_fun)

    else:
        for dictionary in reader:  # TODO find a way to handle multiple arguments
            print('current line number', line_number_obs)
            arguments = dictionary
            if line_number_obs == int(args['line']):
                break
            line_number_obs += 1
        print('the arguments is:', arguments)
        dataset = int(arguments['problem_number'])

    print('the dataset is', dataset)
    manual_fit_spec = None
    if dataset == 1:
        df = pd.read_csv('1848.csv')  # read in the data
        y_df = df[['FSI']]  # only consider crashes
        y_df.rename(columns={"FSI": "Y"}, inplace=True)
        x_df = df.drop(columns=['FSI'])
        x_df = helperprocess.as_wide_factor(x_df)
    elif dataset == 2:
        df = pd.read_csv('4000.csv')  # read in the data
        y_df = df[['Y']].copy()  # only consider crashes
        x_df = df.drop(columns=['Y', 'CT'])
        x_df.rename(columns={"O": "Offset"}, inplace=True)
        x_df = helperprocess.as_wide_factor(x_df)
    elif dataset == 3:
        x_df = pd.read_csv('Stage5A_1848_All_Initial_Columns.csv')  # drop the ID columns
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

        helperprocess.as_wide_factor(x_df, separate_out_factors, keep_original=1)
        # x_df = helperprocess.interactions(x_df)
        manual_fit_spec = {
            'fixed_terms': ['Constant', 'US', 'RSMS', 'MCV'],
            'rdm_terms': ['RSHS:normal', 'AADT:normal', 'Curve50:normal'],
            'rdm_cor_terms': [],
            'grouped_terms': [],
            'hetro_in_means': [],
            'transformations': ['no', 'log', 'log', 'no', 'no', 'no', 'no'],
            'dispersion': 1
        }

        '''
        manual_fit_spec = {
        'fixed_terms': ['Constant', 'M_Curve', 'C50RT', 'Length','Mt_Ter', 'LW_RS', 'LW_US', 'SW', 'FW_US', 'ATLM', 'US', 'RSMS', 'RD', 'RSHS', 'RSLS'],
        'rdm_terms': ['AADT:normal', 'HSP:normal'],
        'rdm_cor_terms': [],
        'grouped_terms': [],
        'hetro_in_means': [],
        'transformations': ['no', 'arcsinh', 'no', 'no','no', 'no', 'arcsinh', 'no', 'no', 'no', 'no', 'no', 'no' , 'arcsinh', 'arcsinh', 'no', 'arcsinh'],
        'dispersion': 1
        }
        '''
        keep = ['Constant', 'US', 'RSMS', 'MCV', 'RSHS', 'AADT', 'Curve50', 'Offset']

        x_df = helperprocess.interactions(x_df, keep)
        separate_out_factors = 0
        if separate_out_factors:
            # x_df1 = helperprocess.PCA_code(x_df, 10)
            x_df = helperprocess.as_wide_factor(x_df, keep_original=1)
        else:
            original_columns = x_df.columns
        # manual_fit_spec = None

    elif dataset == 4:
        manual_fit_spec = {
            'fixed_terms': ['const', 'LOWPRE', 'GBRPM', 'FRICTION'],
            'rdm_terms': ['Expose:normal', 'INTPM:normal', 'CPM:normal', 'HISNOW:normal'],
            'rdm_cor_terms': [],
            'grouped_terms': [],
            'hetro_in_means': [],
            'transformations': ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no'],
            'dispersion': 1
        }

        df = pd.read_csv('Ex-16-3.csv')  # read in the data
        y_df = df[['FREQ']].copy()  # only consider crashes
        y_df.rename(columns={"FREQ": "Y"}, inplace=True)
        x_df = df.drop(columns=['FREQ', 'ID'])
        try:  # grabbing the offset amount
            x_df['Offset'] = np.log(1 + x_df['AADT'] * x_df['LENGTH'] * 365 / 100000000)
            x_df = x_df.drop(columns=['AADT', 'LENGTH'])
        except:
            raise Exception

        separate_out_factors = 1
        if separate_out_factors:

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
        # manual_fit_spec = None

        # manual_keep = ['const', 'LOWPRE', 'GBRPM', 'FRICTION', 'EXPOSE', 'INTPM', 'CPM', 'HISNOW']
    elif dataset == 5:
        df = pd.read_csv('artificial_ZA.csv')  # read in the data
        y_df = df[['Y']].copy()  # only consider crashes
        x_df = df.drop(columns=['Y'])  # was dropped postcode
        x_df = helperprocess.as_wide_factor(x_df, keep_original=1)
        x_df = helperprocess.interactions(x_df)
        print(x_df)

    elif dataset == 6:
        print('check here')
        df = pd.read_csv('ThaiAccident.csv')  # read in the data
        print('the lenght of the dataset is', len(df))
        print(df.head())
        print('true mean', np.mean(df['Death']))
        print(df.head())
        print('Mean after sampling:', np.mean(df['Death']))
        y_df = df[['Death']].copy()  # only consider crashes
        y_df.rename(columns={"Death": "Y"}, inplace=True)
        x_df = df.drop(columns=['Death', 'ID'])  # was dropped postcode
        x_df = convert_df_columns_to_binary_and_wide(x_df)

    elif dataset == 7:
        df = pd.read_csv('artificial_mixed_corr_2023_MOOF.csv')  # read in the data
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
        # manual_fit_spec = None


    elif dataset == 8:
        df = pd.read_csv('rural_int.csv')  # read in the data
        y_df = df[['crashes']].copy()  # only consider crashes
        y_df.rename(columns={"crashes": "Y"}, inplace=True)
        panels = df['orig_ID']

        x_df = df.drop(columns=['crashes', 'year', 'orig_ID',
                                'jurisdiction', 'town', 'maint_region', 'weather_station',
                                'dummy_winter_2'])  # was dropped postcode
        print('dropping for test')
        x_df = x_df.drop(columns=['month', 'inj.fat', 'PDO'])
        x_df = x_df.drop(columns=['zonal_ID', 'ln_AADT', 'ln_seg'])
        x_df['rumble_install_year'] = x_df['rumble_install_year'].astype('category').cat.codes
        x_df.rename(columns={"rumble_install_year": "has_rumble"}, inplace=True)

        # x_df = helperprocess.as_wide_factor(x_df, keep_original=1)
        group_grab = x_df['county']
        x_df = x_df.drop(columns=['county'])
        x_df = helperprocess.interactions(x_df, drop_this_perc=0.8)
        x_df['county'] = group_grab
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
    elif dataset == 10:
        x_df = pd.read_csv('tmrdata.csv')
        # Convert all values to their real parts
        df_real = x_df.select_dtypes(include=[np.number]).apply(np.real)

        # Replace the original DataFrame's numerical columns with real-valued ones
        x_df[df_real.columns] = df_real

        test_measure = arguments['main_analysis']
        arguments['Y_Predict'] = ast.literal_eval(arguments['Y_Predict'])
        # List of substrings to check for
        substrings = ['NA', '.', 'ID', 'killed', 'Tdist', 'Road Name', 'Type 2 Road Trains', 'Check 100%',
                      'formation width', 'length of road with Wide Centre Line', '_', 'AADT']
        more_substrings = ['Crash', 'kill', 'year', 'Length on USCR', 'crashes', 'Head-on', 'Pedestrian',
                           'Number of Serious', 'Injur', 'Rear', 'VKT', 'Total', 'Length', 'Single', 'Angle',
                           'Shoulder', 'Object', 'Side', 'Merge', 'Over', '1,000']
        more_substrings = more_substrings + ['Rural Dual', 'Urban']
        # Create a mask for columns that don't contain any of the substrings in their names
        mask = ~x_df.columns.to_series().apply(lambda x: any(sub in x for sub in substrings))
        force = ast.literal_eval(arguments['force_relationship'])

        # Open the file in write mode
        file_path = "./" + arguments['ModelN'] + "file.txt"  # Replace with your desired file path
        file = open(file_path, "w")

        # Write content to the file
        file.write("This is the content that will be written to the file.")

        # Close the file
        file.close()

        mask[arguments['Y_Predict']] = True

        # if arguments['main_analysis'] not in mask: #make sure its not masked out
        mask[arguments['main_analysis']] = True
        for a in force:
            mask[a] = True
            # mask = mask + arguments['main_analysis'] if arguments['main_analysis'] not in mask
        # Use the mask to select the columns
        x_df = x_df[x_df.columns[mask]]
        print(x_df)
        test_measure = 'Urban Single Carriageway Road and Gazetted for B-doubles'
        test_measure = 'Rural Single Carriageway Road and Gazetted for B-doubles'
        test_measure = arguments['main_analysis']
        # x_df = x_df.drop(columns=['NA'])
        # Drop rows where column 'A' has NaN
        # Assume df is your DataFrame
        # x_df = x_df.replace('nan', 0)
        x_df = x_df.fillna(0)
        x_df = x_df.replace('Y', 1)
        x_df = x_df[x_df[test_measure] != 0]
        # x_df.dropna(subset=test_measure,inplace=True)
        # x_df = x_df.dropna(axis=1)
        # x_df = x_df[x_df['Gazetted for Type 1 Road Trains'] == 'G']
        arguments['Y_Predict'] = arguments['Y_Predict'][0]

        y_df = x_df[[arguments['Y_Predict']]]
        y_df.rename(columns={"All Crash Types Costs": "Y"}, inplace=True)
        # x_df['Offset'] = x_df[arguments['OffsetPeriod']]* x_df['VKT']
        x_df['Offset'] = x_df[arguments['OffsetPeriod']]
        mask = ~x_df.columns.to_series().apply(lambda x: any(sub in x for sub in more_substrings))
        x_df = x_df[x_df.columns[mask]]

        # Find columns where all values are the same
        cols_to_drop = x_df.columns[x_df.nunique() <= 1]

        # Drop these columns
        x_df = x_df.drop(cols_to_drop, axis=1)
        # Convert all possible columns to numeric
        for col in x_df.columns:
            temp_col = pd.to_numeric(x_df[col], errors='coerce')

            # If there are no NaNs in the temporary Series, apply the conversion
            if not temp_col.isna().any():
                x_df[col] = temp_col

        # Get only columns of type object (i.e., strings)
        string_columns = x_df.select_dtypes(include='object')
        print(x_df)
        # Perform one-hot encoding
        x_df = pd.get_dummies(x_df, columns=string_columns.columns)

        # print(df_encoded)
        force = ast.literal_eval(arguments['force_relationship'])


        print('checkign if i can force this in')
        force = ['Roadway AADT', 'Class L AADT', 'Class J AADT', 'Motorways', 'Class K AADT', 'Articulated Truck AADT',
                 'Rigid Truck and Bus AADT', 'Offset', test_measure]

        x_df = helperprocess.interactions(x_df, force)
        # Convert all values to their real parts
        df_real = x_df.select_dtypes(include=[np.number]).apply(np.real)
        x_df = df_real
        arguments['must_include'] = force

    else:  # the dataset has been selected in the program as something else
        df = pd.read_csv('var_test.csv')
        y_df = df[['Y']].copy()  # only consider crashes
        y_df.rename(columns={"crashes": "Y"}, inplace=True)
        panels = df['ind_id']

        x_df = df.drop(columns=['Y', 'alt'])

        drop_this_many = list(range(0, int(arguments['drop'])))

        x_df = x_df.drop(x_df.columns[drop_this_many], axis=1)

        print(x_df)

    arguments['Manual_Fit'] = None

    if arguments['Keep_Fit'] == str(2):
        if manual_fit_spec is None:
            arguments['Manual_Fit'] = None
        else:

            arguments['Manual_Fit'] = manual_fit_spec

    if arguments['problem_number'] == str(8):
        arguments['group'] = 'county'
        arguments['panels'] = 'element_ID'
        arguments['ID'] = 'element_ID'
        arguments['_max_characteristics'] = 55
    elif arguments['problem_number'] == str(9):
        arguments['group'] = 'group'
        arguments['panels'] = 'ind_id'
        arguments['ID'] = 'ind_id'

    if not isinstance(arguments, dict):
        raise Exception
    else:

        if 'complexity_level' in arguments:
            arguments['complexity_level'] = int(arguments['complexity_level'])
        else:
            arguments['complexity_level'] = initial_complexity
    if not defineApp:  # if no manual input ALGORITHMS DEPEND ON The SET_DATA_CSV TO DEFINE HYPERPARAMATERS
        AnalystSpecs = None
        # if dataset != 9:
        # x_df = helperprocess.drop_correlations(x_df)
        arguments['AnalystSpecs'] = AnalystSpecs
        multi_threaded = 0
        if arguments['algorithm'] == 'sa':
            arguments_hyperparamaters = {'alpha': float(arguments['temp_scale']),
                                         'STEPS_PER_TEMP': int(arguments['steps']),
                                         'INTL_ACPT': 0.5,
                                         '_crossover_perc': arguments['crossover'],
                                         'MAX_ITERATIONS': int(arguments['_max_imp']),
                                         '_num_intl_slns': 25,
                                         'Manual_Fit': arguments['Manual_Fit'],
                                         'MP': int(arguments['MP'])}
            helperprocess.entries_to_remove(('crossover', '_max_imp', '_hms', '_hmcr', '_par'), arguments)
            print(arguments)
            # arguments['_distributions'] = ['normal', 'uniform', 'triangular']
            # arguments['model_types'] = [[1]]
            obj_fun = ObjectiveFunction(x_df, y_df, **arguments)

            # results = simulated_annealing(obj_fun, 1, 1, None, arguments_hyperparamaters)

            results = simulated_annealing(obj_fun, None, **arguments_hyperparamaters)
            try:
                helperprocess.results_printer(results, arguments['algorithm'], int(arguments['is_multi']))
            except Exception as e:
                print(e)
            if dual_complexities:
                arguments['complexity_level'] = secondary_complexity
                obj_fun = ObjectiveFunction(x_df, y_df, **arguments)

                # results = simulated_annealing(obj_fun, 1, 1, None, arguments_hyperparamaters)

                results = simulated_annealing(obj_fun, None, **arguments_hyperparamaters)
                helperprocess.results_printer(results, arguments['algorithm'], int(arguments['is_multi']))

        elif arguments['algorithm'] == 'hs':
            arguments['_mpai'] = 1
            # arguments['_distributions'] = ['normal', 'uniform', 'triangular']
            # arguments['model_types'] = [[1]]
            obj_fun = ObjectiveFunction(x_df, y_df, **arguments)
            arguments_hyperparamaters = {
                'Manual_Fit': arguments['Manual_Fit'],
                'MP': int(arguments['MP'])
            }

            results = harmony_search(obj_fun, None, **arguments_hyperparamaters)
            helperprocess.results_printer(results, arguments['algorithm'], int(arguments['is_multi']))

            if dual_complexities:
                arguments['complexity_level'] = secondary_complexity
                obj_fun = ObjectiveFunction(x_df, y_df, **arguments)
                # if multi_threaded:
                #    results = harmony_search(obj_fun, 1, 1, None)
                # else:
                results = harmony_search(obj_fun, None, **arguments_hyperparamaters)
                helperprocess.results_printer(results, arguments['algorithm'], int(arguments['is_multi']))


        elif arguments['algorithm'] == 'de':
            try:
                arguments['must_include'] = force

            except:
                print('no contributing factors are forced into the model')

            arguments_hyperparamaters = {'_AI': 2,
                                         '_crossover_perc': float(arguments['crossover']),
                                         '_max_iter': int(arguments['_max_imp'])
                , '_pop_size': int(arguments['_hms']), 'instance_number': int(args['line'])
                , 'Manual_Fit': arguments['Manual_Fit'],
                                         'MP': int(arguments['MP'])

                                         }
            # arguments['_distributions'] = ['normal', 'uniform', 'triangular']
            # arguments['model_types'] = [[1]]
            arguments_hyperparamaters = dict(arguments_hyperparamaters)

            helperprocess.entries_to_remove(('crossover', '_max_imp', '_hms', '_hmcr', '_par'), arguments)
            obj_fun = ObjectiveFunction(x_df, y_df, **arguments)
            # if multi_threaded:

            #  results = differential_evolution(obj_fun, 1, 1, None, **arguments_hyperparamaters)
            # else
            results = differential_evolution(obj_fun, None, **arguments_hyperparamaters)

            helperprocess.results_printer(results, arguments['algorithm'], int(arguments['is_multi']))

            if dual_complexities:
                arguments['complexity_level'] = secondary_complexity
                obj_fun = ObjectiveFunction(x_df, y_df, **arguments)
                # results = differential_evolution(obj_fun, 1, 1, None, **arguments_hyperparamaters)
                results = differential_evolution(obj_fun, None, **arguments_hyperparamaters)
                helperprocess.results_printer(results, arguments['algorithm'], int(arguments['is_multi']))


if __name__ == '__main__':
    """Loading in command line arguments.  """
    
    parser = argparse.ArgumentParser(prog='main',
                                     epilog=main.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-line', type=int, default=0,
                        help='line to read in csv to pass in argument')
    
    if vars(parser.parse_args())['line'] is not None:
        reader = csv.DictReader(open('set_data.csv', 'r'))
        arguments = list()
        line_number_obs = 0
        for dictionary in reader:  # TODO find a way to handle multiple arguments
            arguments = dictionary
            if line_number_obs == int(vars(parser.parse_args())['line']):
                break
            line_number_obs += 1
        arguments = dict(arguments)
        print('the arguments is:', arguments)
        ## Pass in Optimization Algorithm
        parser.add_argument('-algorithm', type=str, default=arguments['algorithm'],
                            help='optimization algorithm')
    else: #DIDN"T SPECIFY LINES TRY EACH ONE MANNUALLY
        parser.add_argument('-com', type=str, default='MetaCode',
                                help='line to read csv')

    
    # Check the arguments
    parser.print_help()
    args = vars(parser.parse_args())
    # Print the arguments.
    print(args)
    main(args)
