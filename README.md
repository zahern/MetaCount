<div style="display: flex; align-items: center;">
    <img src="https://github.com/zahern/data/raw/main/m.png" alt="My Image" width="100">
    <h1 style="margin-left: 20px;">MetaCountRegressor</h1>
</div>

# Tutorial also available as a jupyter notebook
[Download Example Notebook](https://github.com/zahern/CountDataEstimation/blob/main/Tutorial.ipynb)

The tutorial provides more extensive examples on how to run the code and perform experiments. Further documentation is currently in development.

[Download Notebook Designed for Analysis Use Case](https://github.com/zahern/data/blob/main/Step%20By%20Step_Synthetic.ipynb)


# For an Application Setup Download the following GUI
[Download Application](https://github.com/zahern/MetaCount/tree/master/metacountregressor/application_gui/dist/meta_app)

The application involves setting up a problem instance to run the models.

### Entire [Git Repository](https://github.com/zahern/MetaCount.git) is available to clone.
#### Steps
1. Clone Project
2. Navigate to "metacountregressor/application_gui/dist/meta_app"
3. Run meta_app.exe
4. Navigate to metacountregressor/app_main.py
5. Run app_main.py


## Setup For Python Package Approach
The Below code demonstrates how to set up automatic optimization assisted by the harmony search algorithm. References to the Differential Evolution and Simulated Annealing has been mentioned (change accordingly)

## Install: Requires Python 3.10

Install `metacountregressor` using pip as follows:

```bash
pip install metacountregressor


```python
import pandas as pd
import numpy as np
from metacountregressor.solution import ObjectiveFunction
from metacountregressor.metaheuristics import (harmony_search,
                                            differential_evolution,
                                            simulated_annealing)


```

    loaded standard packages
    loaded helper
    testing
    

#### Basic setup. 
The initial setup involves reading in the data and selecting an optimization algorithm. As the runtime progresses, new solutions will be continually evaluated. Finally, at the end of the runtime, the best solution will be identified and printed out. In the case of multiple objectives all of the best solutions will be printed out that belong to the Pareto frontier.


```python
# Read data from CSV file
df = pd.read_csv(
"https://raw.githubusercontent.com/zahern/data/main/Ex-16-3.csv")
X = df
y = df['FREQ']  # Frequency of crashes
X['Offset'] = np.log(df['AADT']) # Explicitley define how to offset the data, no offset otherwise
# Drop Y, selected offset term and  ID as there are no panels
X = df.drop(columns=['FREQ', 'ID', 'AADT'])  

#some example argument, these are defualt so the following line is just for claritity. See the later agruments section for detials.
arguments = {'algorithm': 'hs', 'test_percentage': 0.15, 'test_complexity': 6, 'instance_name':1,
             'val_percentage':0.15, 'obj_1': 'bic', '_obj_2': 'RMSE_TEST', "_max_time": 6}
# Fit the model with metacountregressor
obj_fun = ObjectiveFunction(X, y, **arguments)
#replace with other metaheuristics if desired
results = harmony_search(obj_fun)


```

## Arguments to feed into the Objective Function:
### 
Note: Please Consider the main arguments to change.

- `algorithm`: This parameter has multiple choices for the algorithm, such as 'hs', 'sa', and 'de'. Only one choice should be defined as a string value.
- `test_percentage`: This parameter represents the percentage of data used for in-sample prediction of the model. The value 0.15 corresponds to 15% of the data.
- `val_percentage`: This parameter represents the percentage of data used to validate the model. The value 0.15 corresponds to 15% of the data.
- `test_complexity`: This parameter defines the complexity level for testing. The value 6 tests all complexities. Alternatively, you can provide a list of numbers to consider different complexities. The complexities are further explained later in this document.
- `instance_number`: This parameter is used to give a name to the outputs.
- `_obj_1`: This parameter has multiple choices for obj_1, such as 'bic', 'aic', and 'hqic'. Only one choice should be defined as a string value.
- `_obj_2`: This parameter has multiple choices for objective 2, such as 'RMSE_TEST', 'MSE_TEST', and 'MAE_TEST'.
- `_max_time`: This parameter specifies the maximum number of seconds for the total estimation before stopping.
- `distribution`: This parameter is a list of distributions to consider. Please select all of the available options and put them into a list of valid options if you want to to consider the distribution type for use when modellign with random parameters. The valid options include: 'Normal', 'LnNormal', 'Triangular', and 'Uniform'.
- `transformations`: This parameters is a list of transformations to consider. Plesee select all of the available options and put them into a list of valid options if you want to consider the transformation type. The valid options include 'Normal', 'LnNormal', 'Triangular', 'Uniform'.
- `method_ll`: This is a specificication on the type of solvers are avilable to solve the lower level maximum likilihood objective. The valid options include: 'Normal', 'LnNormal', 'Triangular', and 'Uniform'.



### Example of changing the arguments:
Modify the arguments according to your preferences using the commented code as a guide.


```python
#Solution Arguments
arguments = {
        'algorithm': 'hs', #alternatively input 'de', or 'sa'
        'is_multi': 1,
        'test_percentage': 0.2, # used in multi-objective optimisation only. Saves 20% of data for testing.
        'val_percenetage:': 0.2, # Saves 20% of data for testing.
        'test_complexity': 6, # Complexity level for testing (6 tests all) or a list to consider potential differences in complexity
        'instance_name': 'name', # used for creeting a named folder where your models are saved into from the directory
        'distribution': ['Normal', 'LnNormal', 'Triangular', 'Uniform'],
        'model_types': [[0,1]],  # or equivalently ['POS', 'NB']
        'transformations': ['no', 'sqrt', 'archsinh'],
        'method_ll': 'BFGS_2',
        '_max_time': 10
    }
obj_fun = ObjectiveFunction(X, y, **arguments)
results = harmony_search(obj_fun)
```

## Initial Solution Configurement
Listed below is an example of how to specify an initial solution within the framework. This initial solution will be used to calculate the fitness and considered in the objective-based search. However, as the search progresses, different hypotheses may be proposed, and alternative modeling components may completely replace the initial solution.


```python
 #Model Decisions, Specify for initial solution that will be optimised.
manual_fit_spec = {
    'fixed_terms': ['SINGLE', 'LENGTH'],
    'rdm_terms': ['AADT:normal'],
    'rdm_cor_terms': ['GRADEBR:normal', 'CURVES:normal'],
    'grouped_rdm': [],
    'hetro_in_means': ['ACCESS:normal', 'MINRAD:normal'],
    'transformations': ['no', 'no', 'log', 'no', 'no', 'no', 'no'],
    'dispersion': 0
}


#Search Arguments
arguments = {
    'algorithm': 'hs',
    'test_percentage': 0.2,
    'test_complexity': 6,
    'instance_name': 'name',
    'Manual_Fit': manual_fit_spec
}
obj_fun = ObjectiveFunction(X, y, **arguments)
```

    Setup Complete...
    Benchmaking test with Seed 42
    --------------------------------------------------------------------------------
    Log-Likelihood:  -1339.1862434675106
    --------------------------------------------------------------------------------
    bic: 2732.31
    --------------------------------------------------------------------------------
    MSE: 650856.32
    +--------------------------+--------+-------+----------+----------+------------+
    |          Effect          | $\tau$ | Coeff | Std. Err | z-values | Prob |z|>Z |
    +==========================+========+=======+==========+==========+============+
    | LENGTH                   | no     | -0.15 |   0.01   |  -12.98  | 0.00***    |
    +--------------------------+--------+-------+----------+----------+------------+
    | SINGLE                   | no     | -2.46 |   0.04   |  -50.00  | 0.00***    |
    +--------------------------+--------+-------+----------+----------+------------+
    | GRADEBR                  | log    | 4.23  |   0.10   |  42.17   | 0.00***    |
    +--------------------------+--------+-------+----------+----------+------------+
    | CURVES                   | no     | 0.51  |   0.01   |  34.78   | 0.00***    |
    +--------------------------+--------+-------+----------+----------+------------+
    |  Chol: GRADEBR (Std.     |        | 2.21  |   0.00   |  50.00   | 0.00***    |
    | Dev. normal) )           |        |       |          |          |            |
    +--------------------------+--------+-------+----------+----------+------------+
    |  Chol: CURVES (Std. Dev. |        | -0.51 |   0.00   |  -50.00  | 0.00***    |
    | normal) )                |        |       |          |          |            |
    +--------------------------+--------+-------+----------+----------+------------+
    |  Chol: CURVES (Std. Dev. | no     | 0.55  |   0.00   |  50.00   | 0.00***    |
    | normal) . GRADEBR (Std.  |        |       |          |          |            |
    | Dev.   normal )          |        |       |          |          |            |
    +--------------------------+--------+-------+----------+----------+------------+
    | main: MINRAD: hetro      | no     | -0.00 |   0.00   |  -44.36  | 0.00***    |
    | group 0                  |        |       |          |          |            |
    +--------------------------+--------+-------+----------+----------+------------+
    | ACCESS: hetro group 0    |        | 0.68  |   0.09   |   7.68   | 0.00***    |
    +--------------------------+--------+-------+----------+----------+------------+
    | main: MINRAD: hetro      |        | -0.00 |   0.00   |  -44.86  | 0.00***    |
    | group 0:normal:sd  hetro |        |       |          |          |            |
    | group 0                  |        |       |          |          |            |
    +--------------------------+--------+-------+----------+----------+------------+
    

 Simarly to return the results feed the objective function into a metaheuristic solution algorithm. An example of this is provided below:


```python
results = harmony_search(obj_fun)
print(results)
```

# Notes:
### Capabilities of the software include:
* Handling of Panel Data
* Support for Data Transformations
* Implementation of Models with Correlated and Non-Correlated Random Parameters
* A variety of mixing distributions for parameter estimations, including normal, lognormal, truncated normal, Lindley, Gamma, triangular, and uniform distributions
Capability to handle heterogeneity in the means of the random parameters
* Use of Halton draws for simulated maximum likelihood estimation
* Support for grouped random parameters with unbalanced groups
* Post-estimation tools for assessing goodness of fit, making predictions, and conducting out-of-sample validation
* Multiple parameter optimization routines, such as the BFGS method
* Comprehensive hypothesis testing using single objectives, such as in-sample BIC and log-likelihood
* Extensive hypothesis testing using multiple objectives, such as in-sample BIC and out-of-sample MAE (Mean Absolute Error), or in-sample AIC and out-of-sample MSPE (mean-square prediction errorr) 
* Features that allow analysts to pre-specify variables, interactions, and mixing distributions, among others
* Meta-heuristic Guided Optimization, including techniques like Simulated Annealing, Harmony Search, and Differential Evolution
* Customization of Hyper-parameters to solve problems tailored to your dataset
* Out-of-the-box optimization capability using default metaheuristics

### Intepreting the output of the model:
A regression table is produced. The following text elements are explained:
- Std. Dev.: This column appears for effects that are related to random paramters and displays the assument distributional assumption next to it
- Chol: This term refers to Cholesky decomposition element, to show the correlation between two random paramaters. The combination of the cholesky element on iyself is equivalent to a normal random parameter.
- hetro group: This term represents the heterogeneity group number, which refers all of the contributing factors that share hetrogentiy in the means to each other under the same numbered value.
- $\tau$: This column, displays the type of transformation that was applied to the specific contributing factor in the data.


## Arguments: 
#### In reference to the arguments that can be fed into the solution alrogithm, a dictionary system is utilised with relecant names these include


The following list describes the arguments available in this function. By default, all of the capabilities described are enabled unless specified otherwise as an argument. For list arguments, include all desired elements in the list to ensure the corresponding options are considered. Example code will be provided later in this guide.

1. **`complexity_level`**: This argument accepts an integer 1-6 or a list based of integegers between 0 to 5 eg might be a possible configuration [0, 2, 3]. Each integer represents a hierarchy level for estimable models associated with each explanatory variable. Here is a summary of the hierarchy:
    - 0: Null model
    - 1: Simple fixed effects model
    - 2: Random parameters model
    - 3: Random correlated parameters model
    - 4: Grouped random parameters model
    - 5: Heterogeneity in the means random parameter model

    **Note:** For the grouped random parameters model, groupings need to be defined prior to estimation. This can be achieved by including the following key-value pair in the arguments of the `ObjectiveFunction`: `'group': "Enter Column Grouping in data"`. Replace `"Enter Column Grouping in data"` with the actual column grouping in your dataset.

    Similarly, for panel data, the panel column needs to be defined using the key-value pair: `'panel': "enter column string covering panels"`. Replace `"enter column string covering panels"` with the appropriate column string that represents the panel information in your dataset.

2. **`distributions`**: This argument accepts a list of strings where each string corresponds to a distribution. Valid options include:
    - "Normal"
    - "Uniform"
    - "LogNormal"
    - "Triangular"
    - "TruncatedNormal"
    - Any of the above, concatenated with ":" (e.g., "Normal:grouped"; requires a grouping term defined in the model)

3. **`Model`**: This argument specifies the model form. It can be a list of integers representing different models to test:
    - 0: Poisson
    - 1: Negative-Binomial

4. **`transformations`**: This argument accepts a list of strings representing available transformations within the framework. Valid options include:
    - "no"
    - "square-root"
    - "logarithmic"
    - "archsinh"
    - "nil"

5. **`is_multi`**: This argument accepts an integer indicating whether single or multiple objectives are to be tested (0 for single, 1 for multiple).

6. **`test_percentage`**: This argument is used for multi-objective optimization. Define it as a decimal; for example, 0.2 represents 20% of the data for testing.

7.  **`val_percentage`**: This argument saves data for validation. Define it as a decimal; for example, 0.2 represents 20% of the data for validation.

8. **`_max_time`**: This argument is used to add a termination time in the algorithm. It takes values as seconds. Note the time is only dependenant on the time after intial population of solutions are generated.

## Example: Assistance by Harmony Search


Let's begin by fitting very simple models and use the structure of these models to define our objectives. Then, we can conduct a more extensive search on the variables that are more frequently identified. For instance, in the case below, the complexity is level 3, indicating that we will consider, at most randomly correlated parameters. This approach is useful for initially identifying a suitable set of contributing factors for our search.



```python

'''Setup Data'''
df = pd.read_csv(
"https://raw.githubusercontent.com/zahern/data/main/Ex-16-3.csv")
X = df
y = df['FREQ']  # Frequency of crashes
X['Offset'] = np.log(df['AADT']) # Explicitley define how to offset the data, no offset otherwise
# Drop Y, selected offset term and  ID as there are no panels
X = df.drop(columns=['FREQ', 'ID', 'AADT'])  
'''Aguments for Solution'''
arguments = {
        'is_multi': 1, #is two objectives considered
        'test_percentage': 0.2, # used in multi-objective optimisation only. Saves 20% of data for testing.
        'val_percentage:': 0.2, # Saves 20% of data for testing.
        'test_complexity': 3, # For Very simple Models
        'obj_1': 'BIC', '_obj_2': 'RMSE_TEST',
        'instance_name': 'hs_run', # used for creeating a named folder where your models are saved into from the directory
        'distribution': ['Normal'],
        'model_types': [0, 1],  # or equivalently ['POS', 'NB']
        'transformations': ['no', 'sqrt', 'archsinh'],
        '_max_time': 10000
} '''Arguments for the solution algorithm'''
argument_hs = {
    '_hms': 20, #harmony memory size,
    '_mpai': 1, #adjustement inded
    '_par': 0.3,
    '_hmcr': .5
}
obj_fun = ObjectiveFunction(X, y, **arguments)
results = harmony_search(obj_fun, None, argument_hs)
print(results)
```

## Example: Assistance by Differential Evololution and Simulated Annealing
Similiar to the above example we only need to change the hyperparamaters, the obj_fun can remane the same


```python
argument_de = {'_AI': 2,
            '_crossover_perc': .2,
            '_max_iter': 1000,
            '_pop_size': 25
}
de_results = differential_evolution(obj_fun, None, **argument_de)
print(de_results)


args_sa = {'alpha': .99,
        'STEPS_PER_TEMP': 10,
        'INTL_ACPT': 0.5,
        '_crossover_perc': .3,
        'MAX_ITERATIONS': 1000,
        '_num_intl_slns': 25,
}

sa_results = simulated_annealing(obj_fun, None, **args_sa)
print(sa_results)
```

## Comparing to statsmodels
The following example illustrates how the output compares to well-known packages, including Statsmodels."


```python
# Load modules and data
import statsmodels.api as sm

data = sm.datasets.sunspots.load_pandas().data
#print(data.exog)
data_exog = data['YEAR']
data_exog = sm.add_constant(data_exog)
data_endog = data['SUNACTIVITY']

# Instantiate a gamma family model with the default link function.
import numpy as np

gamma_model = sm.NegativeBinomial(data_endog, data_exog)
gamma_results = gamma_model.fit()

print(gamma_results.summary())




#NOW LET's COMPARE THIS TO METACOUNTREGRESSOR




 #Model Decisions, 
manual_fit_spec = {
    'fixed_terms': ['const','YEAR'],
    'rdm_terms': [],
    'rdm_cor_terms': [],
    'grouped_rdm': [],
    'hetro_in_means': [],
    'transformations': ['no', 'no'],
    'dispersion': 1 #Negative Binomial 
}


#Arguments
arguments = {
    'algorithm': 'hs',
    'test_percentage': 0,
    'test_complexity': 6,
    'instance': 'name',
    'Manual_Fit': manual_fit_spec
}
obj_fun = ObjectiveFunction(data_exog, data_endog, **arguments)







```

    Optimization terminated successfully.
             Current function value: 4.877748
             Iterations: 22
             Function evaluations: 71
             Gradient evaluations: 70
                         NegativeBinomial Regression Results                      
    ==============================================================================
    Dep. Variable:            SUNACTIVITY   No. Observations:                  309
    Model:               NegativeBinomial   Df Residuals:                      307
    Method:                           MLE   Df Model:                            1
    Date:                Tue, 13 Aug 2024   Pseudo R-squ.:                0.004087
    Time:                        14:13:22   Log-Likelihood:                -1507.2
    converged:                       True   LL-Null:                       -1513.4
    Covariance Type:            nonrobust   LLR p-value:                 0.0004363
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.2913      1.017      0.287      0.774      -1.701       2.284
    YEAR           0.0019      0.001      3.546      0.000       0.001       0.003
    alpha          0.7339      0.057     12.910      0.000       0.622       0.845
    ==============================================================================
    0.1.88
    Setup Complete...
    Benchmaking test with Seed 42
    1
    --------------------------------------------------------------------------------
    Log-Likelihood:  -1509.0683662284273
    --------------------------------------------------------------------------------
    bic: 3035.84
    --------------------------------------------------------------------------------
    MSE: 10000000.00
    +--------+--------+-------+----------+----------+------------+
    | Effect | $\tau$ | Coeff | Std. Err | z-values | Prob |z|>Z |
    +========+========+=======+==========+==========+============+
    | const  | no     | 0.10  |   0.25   |   0.39   | 0.70       |
    +--------+--------+-------+----------+----------+------------+
    | YEAR   | no     | 0.00  |   0.00   |  20.39   | 0.00***    |
    +--------+--------+-------+----------+----------+------------+
    | nb     |        | 1.33  |   0.00   |  50.00   | 0.00***    |
    +--------+--------+-------+----------+----------+------------+
    

---

# JAX-Accelerated Fitting (Advanced API)

The package includes a JAX-based back-end (`jax_engine`, `jax_engine_n`) that provides fast gradient computation via JIT compilation and automatic differentiation. The high-level `ExperimentBuilder` API wraps this back-end and drives a metaheuristic search over model structure — selecting which variables to include, what role they play, how many latent sub-populations exist, and what distributional assumptions to use.

---

## Quick Setup

```python
import pandas as pd
from metacountregressor.experiment_package import ExperimentBuilder

df = pd.read_csv("crash_data.csv")

builder = ExperimentBuilder(
    df           = df,
    id_col       = "SITE_ID",    # panel/site identifier column
    y_col        = "CRASHES",    # count outcome column
    offset_col   = "EXPOSURE",   # optional log-exposure offset
    group_id_col = "FC",         # optional grouping column (e.g. road class)
)

builder.describe()           # prints data summary, outcome stats, variable types
builder.suggest_config()     # auto-suggests roles and distributions per variable
```

If you do not have panel data, set `id_col` to any column of unique row identifiers.
`offset_col` and `group_id_col` are optional.

---

## Role Codes

Each variable in the search is assigned a **role** that determines how it enters the model. The search explores which role best fits each variable.

| Code | Name | Description |
|------|------|-------------|
| 0 | Excluded | Not in the model |
| 1 | Fixed | Single coefficient, constant across all observations |
| 2 | Random Independent | Individual-specific draws, independent across variables |
| 3 | Random Correlated | Individual-specific draws with joint covariance (Cholesky) |
| 4 | Grouped | Coefficient shared within a group (requires `group_id_col`) |
| 5 | Heterogeneity in means | Shifts the mean of a random-effect distribution |
| 6 | Zero Inflation | Enters the zero-inflation probability equation |
| 7 | Membership only | Enters class-membership equation only (latent class models) |
| 8 | Membership + Fixed | Enters both class-membership and outcome equations |

Roles 7 and 8 are only active when `max_latent_classes > 1`.

---

## Supported Distributions (Random Effects)

When a variable is assigned a random role (2, 3, or 4), the search also selects its mixing distribution:

| Name | Code | Use case |
|------|------|----------|
| `normal` | 0 | General-purpose, allows negative and positive values |
| `lognormal` | 1 | Strictly positive effects (e.g. exposure scaling) |
| `triangular` | 2 | Bounded symmetric effects |

---

## Complexity Levels

Complexity is controlled implicitly through which roles are allowed. You can restrict the search to simpler structures by limiting `allowed_roles`:

| Complexity | Roles allowed | Description |
|-----------|--------------|-------------|
| 1 — Fixed only | `[0, 1]` | Simple fixed-effects Poisson/NB |
| 2 — Random independent | `[0, 1, 2]` | Adds uncorrelated random parameters |
| 3 — Random correlated | `[0, 1, 2, 3]` | Adds Cholesky-parameterised covariance |
| 4 — Grouped | `[0, 1, 2, 3, 4]` | Adds group-level shared effects |
| 5 — Heterogeneity in means | `[0, 1, 2, 3, 4, 5]` | Adds variables that shift random-effect means |
| 6 — Latent class | `[0, 1, 2, 3, 5, 7, 8]` | Multiple latent sub-populations |

Pass `allowed_roles` to `build_evaluator()` to limit the search space and reduce runtime.

---

## Building the Evaluator

```python
evaluator = builder.build_evaluator(
    variables          = None,          # list of columns to search over; None = all
    fixed_override     = {"AADT": [1]}, # force specific roles for named variables
    membership_override= {"URB": [0, 7], "SPEED": [0, 7, 8]},  # latent class roles
    exclude            = ["ID"],        # always exclude these variables
    mode               = "single",      # "single" = minimise BIC; "multi" = BIC + test RMSE
    max_latent_classes = 1,             # set > 1 to enable latent class search
    R                  = 200,           # Halton simulation draws (higher = more stable)
    default_roles      = [0, 1, 2, 3],  # default allowed roles for unlisted variables
)
```

### `mode` options

- **`"single"`** — minimises BIC. Recommended for exploratory analyses or when a single best model is wanted.
- **`"multi"`** — minimises BIC and out-of-sample RMSE simultaneously using Pareto-front methods (NSGA-II). Returns multiple non-dominated solutions. Requires more data (20 % is held out for testing).

---

## Running the Search

```python
result = builder.run(evaluator, algo="sa", max_iter=3000, seed=42)
```

### Algorithm options

| Key | Algorithm | Best for |
|-----|-----------|----------|
| `"sa"` | Simulated Annealing | Single-objective; recommended starting point |
| `"hc"` | Hill Climbing | Fast, greedy; good for small problems |
| `"de"` | Adaptive Differential Evolution + NSGA-II | Multi-objective (`mode="multi"`) |
| `"hs"` | Dynamic Harmony Search + NSGA-II | Multi-objective (`mode="multi"`) |

### SA / HC hyperparameters

```python
result = builder.run(
    evaluator,
    algo          = "sa",
    max_iter      = 3000,
    seed          = 0,
    mutation_rate = 0.3,   # fraction of genes mutated per step
    alpha         = 0.995, # cooling schedule multiplier (closer to 1 = slower cooling)
    min_changes   = 1,     # minimum genes to change per neighbour
    max_changes   = 3,     # maximum genes to change per neighbour
    n_starts      = 1,     # number of independent restarts
)
```

### DE hyperparameters

```python
result = builder.run(
    evaluator,
    algo            = "de",
    max_iter        = 2000,
    population_size = 20,  # number of candidate solutions
    F               = 0.5, # scale factor for difference vectors
    CR              = 0.7, # crossover probability
)
```

### HS hyperparameters

```python
result = builder.run(
    evaluator,
    algo            = "hs",
    max_iter        = 2000,
    population_size = 20,
    hmcr            = 0.9,  # harmony memory consideration rate
    par_min         = 0.1,  # pitch adjustment rate (minimum)
    par_max         = 0.9,  # pitch adjustment rate (maximum)
)
```

---

## Latent Class Models

Latent class models allow the data to be explained by multiple unobserved sub-populations. The search automatically determines how many classes best fit the data.

```python
evaluator = builder.build_evaluator(
    mode               = "single",
    max_latent_classes = 3,              # search over 1, 2, or 3 classes
    membership_override= {
        "URB":   [0, 7],    # can influence class membership only
        "SPEED": [0, 7, 8], # can influence membership and the outcome equation
    },
    R = 150,
)

result = builder.run(evaluator, algo="sa", max_iter=3000, seed=42)
```

The EM warm-start is applied automatically when classes > 1: a single-class solution seeds the multi-class initialisation, improving convergence stability.

### What the search controls for latent classes

- **Number of classes** (1 up to `max_latent_classes`) — selected automatically
- **Class-membership variables** — which variables enter the softmax class-probability equation
- **Per-class fixed effects** — estimated separately for each latent class
- **Dispersion** — Poisson (0) or Negative Binomial (1) selected per run

---

## Manual Model Specification (no search)

If you want to fit a specific structure directly without running a metaheuristic:

```python
from metacountregressor.jax_engine_n import CountModel, build_model_from_manual_spec

manual_spec = {
    "fixed_terms"   : ["X1", "X2"],
    "rdm_terms"     : ["X3:normal", "X4:lognormal"],   # independent random
    "rdm_cor_terms" : ["X5:normal", "X6:triangular"],  # correlated random (Cholesky)
    "grouped_terms" : [],
    "hetro_in_means": ["X7"],                           # heterogeneity in means
    "dispersion"    : 1,                                # 0 = Poisson, 1 = Negative Binomial
}

data, spec = build_model_from_manual_spec(
    df         = df,
    manual_spec= manual_spec,
    id_col     = "SITE_ID",
    y_col      = "CRASHES",
    offset_col = "EXPOSURE",
    R          = 200,
)

model = CountModel(spec, data)
result = model.fit()

print(f"Log-likelihood: {model.loglik():.4f}")
print(f"BIC:            {model.bic():.4f}")
model.predict()   # returns mean predictions
```

---

## Full Example: Crash-Frequency Latent Class Search

```python
import pandas as pd
import numpy as np
from metacountregressor.experiment_package import ExperimentBuilder

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/zahern/data/main/Ex-16-3.csv")
df["OFFSET"] = np.log(df["AADT"] * df["LENGTH"] * 365 / 1e8)
df = df.drop(columns=["AADT"])

builder = ExperimentBuilder(
    df           = df,
    id_col       = "ID",
    y_col        = "FREQ",
    offset_col   = "OFFSET",
    group_id_col = "FC",
)

builder.describe()
builder.suggest_config(max_latent_classes=2)

evaluator = builder.build_evaluator(
    mode               = "single",
    max_latent_classes = 2,
    membership_override= {"URBAN": [0, 7], "SPEED": [0, 7, 8]},
    exclude            = ["FC"],   # already used as group_id_col
    R                  = 150,
)

result = builder.run(evaluator, algo="sa", max_iter=1000, seed=42)

print("Best BIC:", result["best_score"])
print("Best structure:", result["best_solution"])
```

---

## Simulation Draws (Halton)

The JAX engine uses **scrambled Halton sequences** for quasi-Monte Carlo integration over the random-effect distributions. The number of draws `R` controls the trade-off between speed and accuracy:

| R | Accuracy | Speed |
|---|----------|-------|
| 50 | Low — suitable for quick screening | Fast |
| 100 | Moderate — reasonable for search | Moderate |
| 200 | Good — recommended default | Moderate |
| 500+ | High — use for final model fitting | Slow |

Draws are pre-generated once per evaluator and reused across all model evaluations, ensuring consistent comparisons.

---

## Paper

The following tutorial is in conjunction with our latest paper. A link the current paper can be found here [MetaCountRegressor](https://www.overleaf.com/read/mszwpwzcxsng#c5eb0c)

## Contact
If you have any questions, ideas to improve MetaCountRegressor, or want to report a bug, just open a new issue in [GitHub repository](https://github.com/zahern/CountDataEstimation).

## Citing MetaCountRegressor
Please cite MetaCountRegressor as follows:

Ahern, Z., Corry P., Paz A. (2024). MetaCountRegressor [Computer software]. [https://pypi.org/project/metacounregressor/](https://pypi.org/project/metacounregressor/)

Or using BibTex as follows:

```bibtex
@misc{Ahern2024Meta,
   author = {Zeke Ahern, Paul Corry and Alexander Paz},
   journal = {PyPi},
   title = {metacountregressor · PyPI},
   url = {https://pypi.org/project/metacountregressor/0.1.80/},
   year = {2024},
}

