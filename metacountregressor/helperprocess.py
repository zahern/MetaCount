from os.path import exists
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy import stats as st
from sklearn.preprocessing import StandardScaler


plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')





from itertools import product

# Function to create a list of dictionaries from a parameter grid
def generate_param_combinations(param_grid):
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return combinations


##Select the best Features Based on RF
def select_features(X_train, y_train, n_f=16):
    try:
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression
        feature_names = X_train.columns
        # configure to select all features
        fs = SelectKBest(score_func=f_regression, k=16)

        # learn relationship from training data
        fs.fit(X_train, y_train)

        mask = fs.get_support()  # Boolean array of selected features
        selected_features = [feature for bool, feature in zip(mask, feature_names) if bool]
        X_train = X_train[selected_features]
    except:
        print('import error, not performing feature selection')
        fs = X_train.columns #TODO check if this is actually getting the names

    return X_train, fs


#Cutts off correlated data





def findCorrelation(corr, cutoff=0.9, exact=None):    """
    This function is the Python implementation of the R function
    `findCorrelation()`.

    Relies on numpy and pandas, so must have them pre-installed.

    It searches through a correlation matrix and returns a list of column names
    to remove to reduce pairwise correlations.

    For the documentation of the R function, see
    https://www.rdocumentation.org/packages/caret/topics/findCorrelation
    and for the source code of `findCorrelation()`, see
    https://github.com/topepo/caret/blob/master/pkg/caret/R/findCorrelation.R

    -----------------------------------------------------------------------------

    Parameters:
    -----------
    corr: pandas dataframe.
        A correlation matrix as a pandas dataframe.
    cutoff: float, default: 0.9.
        A numeric value for the pairwise absolute correlation cutoff
    exact: bool, default: None
        A boolean value that determines whether the average correlations be
        recomputed at each step
    -----------------------------------------------------------------------------
    Returns:
    --------
    list of column names
    -----------------------------------------------------------------------------
    Example:
    --------
    R1 = pd.DataFrame({
        'x1': [1.0, 0.86, 0.56, 0.32, 0.85],
        'x2': [0.86, 1.0, 0.01, 0.74, 0.32],
        'x3': [0.56, 0.01, 1.0, 0.65, 0.91],
        'x4': [0.32, 0.74, 0.65, 1.0, 0.36],
        'x5': [0.85, 0.32, 0.91, 0.36, 1.0]
    }, index=['x1', 'x2', 'x3', 'x4', 'x5'])

    findCorrelation(R1, cutoff=0.6, exact=False)  # ['x4', 'x5', 'x1', 'x3']
    findCorrelation(R1, cutoff=0.6, exact=True)   # ['x1', 'x5', 'x4']
    """

def _findCorrelation_fast(corr, avg, cutoff):

    combsAboveCutoff = corr.where(lambda x: (np.tril(x) == 0) & (x > cutoff)).stack().index

    rowsToCheck = combsAboveCutoff.get_level_values(0)
    colsToCheck = combsAboveCutoff.get_level_values(1)

    msk = avg[colsToCheck] > avg[rowsToCheck].values
    deletecol = pd.unique(np.r_[colsToCheck[msk], rowsToCheck[~msk]]).tolist()

    return deletecol

def _findCorrelation_exact(corr, avg, cutoff):

    x = corr.loc[(*[avg.sort_values(ascending=False).index] * 2,)]

    if (x.dtypes.values[:, None] == ['int64', 'int32', 'int16', 'int8']).any():
        x = x.astype(float)

    x.values[(*[np.arange(len(x))] * 2,)] = np.nan

    deletecol = []
    for ix, i in enumerate(x.columns[:-1]):
        for j in x.columns[ix + 1:]:
            if x.loc[i, j] > cutoff:
                if x[i].mean() > x[j].mean():
                    deletecol.append(i)
                    x.loc[i] = x[i] = np.nan
                else:
                    deletecol.append(j)
                    x.loc[j] = x[j] = np.nan




"""Funtion to Convert Data to Binaries """ 
def clean_data_types(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            # Attempt to convert the column to numeric type
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def drop_correlations(x_df, percentage=0.85):
    cor_matrix = x_df.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))  # type: ignore
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > percentage)]
    return x_df.drop(x_df[to_drop].columns, axis=1)


def remove_files(yes=1):
    if not yes:
        return
    else:
        import os

        directory = "./"
        files_in_directory = os.listdir(directory)
        filtered_files = [file for file in files_in_directory if file.endswith(".tex")]
        filtered_csv_files = [file for file in files_in_directory if 'log' in file and file.endswith(".csv")]

        for file in filtered_files:
            path_to_file = os.path.join(directory, file)
            os.remove(path_to_file)
        for file in filtered_csv_files:
            path_to_file = os.path.join(directory, file)
            os.remove(path_to_file)
        if os.path.exists('log.csv'):
            os.remove('log.csv')
        if os.path.exists('pop_log.csv'):
            os.remove('pop_log.csv')


# Function to process the DataFrame
'''
Example usuage
# Configuration dictionary
config = {
    'Age': {
        'type': 'bin',
        'bins': [0, 18, 35, 50, 100],
        'labels': ['Child', 'YoungAdult', 'MiddleAged', 'Senior'],
        'prefix': 'Age_Binned'
    },
    'Income': {
        'type': 'bin',
        'bins': [0, 2000, 5000, 10000],
        'labels': ['Low', 'Medium', 'High'],
        'prefix': 'Income_Binned'
    },
    'Gender': {
        'type': 'one-hot',
        'prefix': 'Gender'
    },
    'Score': {
        'type': 'none'
    }
}
'''
def null_handler(vari):
    if vari in locals():
        return vari
    else:
        print(f'{vari} does not exist, setting None..')
        return None


def set_up_analyst_constraints(data_characteristic, model_terms,  variable_decisions_alt = None):


    name_data_characteristics = data_characteristic.columns.tolist()
    # Get non-None values as a list
    non_none_terms = [value for value in model_terms.values() if value is not None]
    # how to make name_data_characteristics - non_none_terms

    result = [item for item in name_data_characteristics if item not in non_none_terms]
    distu = ['Normal', 'Uniform', 'Triangular']
    tra = ['no', 'sqrt', 'arcsinh']
    if model_terms.get('grouped') is None:
        print('cant have grouped rpm, removing level 4 from every item')
        MAKE_ALL_4_FALSE = True
    else:
        MAKE_ALL_4_FALSE = False

    variable_decisions = {
        name: {
            'levels': list(range(6)),
            'distributions': distu,
            'transformations': tra
        }
        for name in result
    }
    # Override elements in the original dictionary with the alt dictionary
    if variable_decisions_alt is not None:
        for key, alt_value in variable_decisions_alt.items():
            if key in variable_decisions:
                # Update the existing entry
                variable_decisions[key].update(alt_value)
            else:
                # Add new entry if it doesn't exist
                variable_decisions[key] = alt_value
    # Prepare the data for the DataFrame
    rows = []
    for column_name, details in variable_decisions.items():
        # Create a row dictionary
        row = {'Column': column_name}

        # Add levels as True/False for Level 0 through Level 5
        for level in range(6):  # Assuming Level 0 to Level 5

            if level == 4 and MAKE_ALL_4_FALSE:
                row[f'Level {level}'] = False
            else:
                row[f'Level {level}'] = level in details['levels']

        # Add distributions and transformations directly
        row['distributions'] = details['distributions']
        row['transformations'] = details['transformations']

        rows.append(row)

    # Create the DataFrame
    df = pd.DataFrame(rows)

    data_new = data_characteristic.rename(columns={v: k for k, v in model_terms.items() if v in data_characteristic.columns})
    return  df, data_new

# Function to guess Low, Medium, High ranges
def guess_low_medium_high(column_name, series):
    # Compute the tertiles (33rd and 66th percentiles)
    #print('did it make it...')
    #mode_value = st.mode(series)  # Get the most frequent value
    #print('good')
   # series = pd.to_numeric(series, errors='coerce').fillna(mode_value)
    low_threshold = np.quantile(series, 0.33)
    high_threshold = np.quantile(series,0.66)

    # Define the bins and labels
    bins = [np.min(series) - 1, low_threshold, high_threshold, np.max(series)]
    # Handle duplicate bins by adjusting labels
    if len(set(bins)) < len(bins):  # Check for duplicate bin edges
        if low_threshold == high_threshold:
            # Collapse to two bins (Low and High)
            bins = [np.min(series) - 1, low_threshold, np.max(series)]
            labels = ['Low', 'High']
        else:
            # Collapse to three unique bins
            bins = sorted(set(bins))  # Remove duplicate edges
            labels = [f'Bin {i + 1}' for i in range(len(bins) - 1)]
    else:
        # Standard case: Low, Medium, High
        labels = ['Low', 'Medium', 'High']

    return {
        'type': 'bin',
        'bins': bins,
        'labels': labels,
        'prefix': f'{column_name}'
    }

def transform_dataframe(df, config):
    output_df = pd.DataFrame()

    for column, settings in config.items():
        if settings['type'] == 'bin':
            # Apply binning
            binned = pd.cut(
                df[column],
                bins=settings['bins'],
                labels=settings['labels'],
                right=False,

            )
            # One-hot encode the binned column
            binned_dummies = pd.get_dummies(binned, prefix=settings['prefix'])
            output_df = pd.concat([output_df, binned_dummies], axis=1)

        elif settings['type'] == 'one-hot':
            # One-hot encode the column
            one_hot_dummies = pd.get_dummies(df[column], prefix=settings.get('prefix', column))
            output_df = pd.concat([output_df, one_hot_dummies], axis=1)

        elif settings['type'] == 'continuous':
            # Apply function to continuous data
            data = df[column]
            if 'bounds' in settings:
                # Apply bounds filtering
                lower, upper = settings['bounds']
                data = data[(data >= lower) & (data <= upper)]
            if 'apply_func' in settings:
                # Apply custom function
                data = data.apply(settings['apply_func'])
            output_df[column] = data

        elif settings['type'] == 'none':
            # Leave the column unchanged
            output_df = pd.concat([output_df, df[[column]]], axis=1)

    return output_df

# Helper function to guess column type and update `config`
def guess_column_type(column_name, series):

    if series.empty:
        raise ValueError(f"The column {column_name} contains no numeric data.")

    if series.dtype == 'object' or series.dtype.name == 'category':
        # If the column is categorical (e.g., strings), assume one-hot encoding
        return {'type': 'one-hot', 'prefix': column_name}
    elif pd.api.types.is_numeric_dtype(series):
        unique_values = series.nunique()

        if unique_values < 5:
            return {'type': 'one-hot', 'prefix': column_name}

        elif np.max(series) - np.min(series) > 20:
            print('made it through here')
            # If there are few unique values, assume binning with default bins
            return guess_low_medium_high(column_name,series)
        else:
           # # Otherwise, assume continuous data with normalization
            # Otherwise, fallback to continuous standardization
            return {
                'type': 'continuous',
                'apply_func': (lambda x: (x - series.mean()) / series.std())  # Z-Score Standardization
            }
    else:
        # Default fallback (leave the column unchanged)
        return {'type': 'none'}



def as_wide_factor(x_df, yes=1, min_factor=2, max_factor=8, keep_original=0, exclude=[]):
    if not yes:
        return x_df
    else:
        for col in x_df.columns:
            if col not in exclude:
                factor = len(set(x_df[col]))
                if factor > min_factor and factor < max_factor:
                    if keep_original:
                        x_df[col + str('orig')] = x_df[col]
                    x_df = pd.get_dummies(x_df, columns=[col], prefix=[col], prefix_sep='_')
        return x_df


def PCA_code(X, n_components=5):
    from sklearn.decomposition import PCA
    pca_code = PCA(n_components=n_components)
    principalComponents = pca_code.fit_transform(X)
    return principalComponents


def interactions(df, keep=None, drop_this_perc=0.6, interact = False):
    full_columns = df.columns
    if interact:
        interactions_list = []
        for i, var_i in enumerate(df.columns):
            for j, var_j in enumerate(df.columns):
                if i <= j:
                    continue
                interaction = df[var_i] * df[var_j]
                interactions_list.append(interaction)

        df_interactions = pd.concat(interactions_list, axis=1)
        df_interactions.columns = [f'{var_i}_{var_j}' for i, var_i in enumerate(df.columns) for j, var_j in
                                   enumerate(df.columns) if i < j]
        corr_matrix = df_interactions.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.3)]

        # Drop features
        df_interactions.drop(to_drop, axis=1, inplace=True)
        # to_drop = [column for column in correlation_matrix.columns if any(correlation_matrix[column] > 0.9)]

        df = pd.concat([df, df_interactions], axis=1, sort=False)

    # second
    # Remove `keep` columns from the correlation matrix
    if keep is not None:
        missing_columns = [col for col in keep if col not in df.columns]
     
        if missing_columns:
            print(f"The following columns are not in the DataFrame and will be ignored: {missing_columns}")
            keep = [col for col in keep if col not in df.columns]
        df_corr = df.drop(columns=keep, errors='ignore', inplace=False)  # Exclude `keep` columns
    else:
        df_corr = df

    # Compute the absolute correlation matrix
    corr_matrix = df_corr.corr().abs()

    # Keep only the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > drop_this_perc)]

    # Ensure `keep` columns are not dropped
    if keep is not None:
        to_drop = [column for column in to_drop if column not in full_columns]

    # Drop the identified features
    df.drop(to_drop, axis=1, inplace=True)

    return df


def check_list_type(lst, check_type):
    for element in lst:
        if not isinstance(element, check_type):
            raise TypeError(f"All elements in the list must be of type {check_type}")


def results_printer(results, algorithm='hs', is_multi=1, obj_1='bic', obj_2='MSE'):
    if algorithm == 'hs':
        plt.scatter([x['bic'] for x in results.harmony_memories], [x['MAE'] for x in results.harmony_memories])
        plt.savefig('bic.svg', format='svg', dpi=1200)
        print('Elapsed time: {}\nBest harmony: {}\nBest fitness: {}\nHarmony memories: {}'.format(results.elapsed_time,
                                                                                                  results.best_harmony,
                                                                                                  results.best_fitness,
                                                                                                  results.harmony_memories))
    elif algorithm == 'de':
        if is_multi:

            plt.plot([x[obj_1] for x in results.best_solutions], [x[obj_2] for x in results.best_solutions], 'o-r')
            plt.xlabel(f'{obj_1.upper()}')  # x label
            plt.ylabel(f'{obj_2.upper()}')  # y label
            plt.savefig(f'{obj_1}_vs_{obj_2}.svg', format='svg', dpi=1200)
            print('Elapsed time: {}\nPareto Solutions: {} \nPopulation Solutions: {}'.format(results.elapsed_time,
                                                                                             results.best_solutions,
                                                                                             results.population_solutions))
        else:

            print(
                'Elapsed time: {}\nIterations: {}\nIteration_Fitnesses: {}\nBest Fitnessses: {}\nBest Fitness: {}\nBest Struct: {}\nAverage Fitness: {}'.format(
                    results.elapsed_time,
                    results.iteration, results.iter_solution, results.best_solutions, results.best_fitness,
                    # type: ignore
                    results.best_struct, results.average_best))  # type: ignore
    elif algorithm == 'sa':
        print(
            'Elapsed time: {}\nIterations: {}'.format(
                results['elapsed_time'], results['Iteration']
            ))


def algorithm_set_data(algorithm='de'):
    POPULATION = 50
    MAX_ITER: int = 3600
    ADJ_INDX: int = 1
    CR_R: float = 0.2
    INTL_ACCEPT: float = 0.5
    STEPS: int = 20
    SWAP_PERC: float = 0.05
    ALPHA_TEMP:float = 0.99
    NUM_INTL_SLNS = 25
    IS_MULTI:bool = 1
    SHARED = list()
    if algorithm == 'de':
        with open('set_data.csv') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                algorithm = row['algorithm']
                POPULATION = int(row['population'])
                CR_R = float(row['crossover'])
                MAX_TIME = float(row['max_time'])
                SEED = int(row['seed'])
                MAX_ITER = int(row['max_iter'])
                IS_MULTI = int(row['is_multi'])
                TEST_SET_SIZE = float(row['test_size'])
                OBJ_1 = str(row['obj1'])
                OBJ_2 = str(row['obj2'])
                if TEST_SET_SIZE == 0:
                    TEST_SET_SIZE = 0
        csv_file.close()
        hyperparameters = [POPULATION, MAX_ITER, ADJ_INDX, CR_R]
        return hyperparameters
    elif algorithm == 'sa':
        with open('set_data.csv') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                algorithm = row['algorithm']
                ALPHA_TEMP = float(row['temp_scale'])
                STEPS = int(row['steps'])
                MAX_TIME = float(row['max_time'])
                SEED = int(row['seed'])
                MAX_ITER = int(row['max_iter'])
                SWAP_PERC = float(row['crossover'])
                IS_MULTI = int(row['is_multi'])
                TEST_SET_SIZE = float(row['test_size'])
                OBJ_1 = str(row['obj1'])
                OBJ_2 = str(row['obj2'])
                if TEST_SET_SIZE == 0:
                    TEST_SET_SIZE = 0
        csv_file.close()
        hyperparameters = [ALPHA_TEMP, MAX_ITER, INTL_ACCEPT, STEPS, SWAP_PERC, NUM_INTL_SLNS, IS_MULTI]
        return hyperparameters
    elif algorithm == 'hs':
        with open('set_data.csv') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                algorithm = row['algorithm']
                POPULATION = int(row['population'])
                CR_R = float(row['crossover'])
                MAX_TIME = float(row['max_time'])
                SEED = int(row['seed'])
                MAX_ITER = int(row['max_iter'])
                HMCR = float(row['hmcr'])
                PAR = float(row['par'])
                IS_MULTI = int(row['is_multi'])
                TEST_SET_SIZE = float(row['test_size'])
                OBJ_1 = str(row['obj1'])
                OBJ_2 = str(row['obj2'])
                if TEST_SET_SIZE == 0:
                    TEST_SET_SIZE = 0
        csv_file.close()


def entries_to_remove(entries, the_dict):
    for key in entries:
        if key in the_dict:
            del the_dict[key]


