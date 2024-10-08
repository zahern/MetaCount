.. container::

   ::

      <img src="https://github.com/zahern/data/raw/main/m.png" alt="My Image" style="width: 100px; margin-right: 20px;">
      <p><span style="font-size: 60px;"><strong>MetaCountRegressor</strong></span></p>

Tutorial also available as a jupyter notebook
=============================================

`Download Example
Notebook <https://github.com/zahern/CountDataEstimation/blob/main/Tutorial.ipynb>`__

The tutorial provides more extensive examples on how to run the code and
perform experiments. Further documentation is currently in development.

For an Application Setup Download the following GUI
===================================================

`Download
Application <https://github.com/zahern/MetaCount/tree/master/metacountregressor/application_gui/dist/meta_app>`__

The application involves setting up a problem instance to run the
models.

Entire `Git Repository <https://github.com/zahern/MetaCount.git>`__ is available to clone.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Steps
^^^^^

1. Clone Project
2. Navigate to “metacountregressor/application_gui/dist/meta_app”
3. Run meta_app.exe
4. Navigate to metacountregressor/app_main.py
5. Run app_main.py

Setup For Python Package Approach
---------------------------------

The Below code demonstrates how to set up automatic optimization
assisted by the harmony search algorithm. References to the Differential
Evolution and Simulated Annealing has been mentioned (change
accordingly)

Install: Requires Python 3.10
-----------------------------

Install ``metacountregressor`` using pip as follows:

\```bash pip install metacountregressor

.. code:: ipython3

    import pandas as pd
    import numpy as np
    from metacountregressor.solution import ObjectiveFunction
    from metacountregressor.metaheuristics import (harmony_search,
                                                differential_evolution,
                                                simulated_annealing)
    
    


.. parsed-literal::

    loaded standard packages
    loaded helper
    testing
    

Basic setup.
^^^^^^^^^^^^

The initial setup involves reading in the data and selecting an
optimization algorithm. As the runtime progresses, new solutions will be
continually evaluated. Finally, at the end of the runtime, the best
solution will be identified and printed out. In the case of multiple
objectives all of the best solutions will be printed out that belong to
the Pareto frontier.

.. code:: ipython3

    # Read data from CSV file
    df = pd.read_csv(
    "https://raw.githubusercontent.com/zahern/data/main/Ex-16-3.csv")
    X = df
    y = df['FREQ']  # Frequency of crashes
    X['Offset'] = np.log(df['AADT']) # Explicitley define how to offset the data, no offset otherwise
    # Drop Y, selected offset term and  ID as there are no panels
    X = df.drop(columns=['FREQ', 'ID', 'AADT'])  
    
    #some example argument, these are defualt so the following line is just for claritity. See the later agruments section for detials.
    arguments = {'algorithm': 'hs', 'test_percentage': 0.15, 'test_complexity': 6, 'instance_number':1,
                 'val_percentage':0.15, 'obj_1': 'bic', '_obj_2': 'RMSE_TEST', "_max_time": 6}
    # Fit the model with metacountregressor
    obj_fun = ObjectiveFunction(X, y, **arguments)
    #replace with other metaheuristics if desired
    results = harmony_search(obj_fun)
    
    

Arguments to feed into the Objective Function:
----------------------------------------------

Note: Please Consider the main arguments to change.

-  ``algorithm``: This parameter has multiple choices for the algorithm,
   such as ‘hs’, ‘sa’, and ‘de’. Only one choice should be defined as a
   string value.
-  ``test_percentage``: This parameter represents the percentage of data
   used for in-sample prediction of the model. The value 0.15
   corresponds to 15% of the data.
-  ``val_percentage``: This parameter represents the percentage of data
   used to validate the model. The value 0.15 corresponds to 15% of the
   data.
-  ``test_complexity``: This parameter defines the complexity level for
   testing. The value 6 tests all complexities. Alternatively, you can
   provide a list of numbers to consider different complexities. The
   complexities are further explained later in this document.
-  ``instance_number``: This parameter is used to give a name to the
   outputs.
-  ``_obj_1``: This parameter has multiple choices for obj_1, such as
   ‘bic’, ‘aic’, and ‘hqic’. Only one choice should be defined as a
   string value.
-  ``_obj_2``: This parameter has multiple choices for objective 2, such
   as ‘RMSE_TEST’, ‘MSE_TEST’, and ‘MAE_TEST’.
-  ``_max_time``: This parameter specifies the maximum number of seconds
   for the total estimation before stopping.
-  ``distribution``: This parameter is a list of distributions to
   consider. Please select all of the available options and put them
   into a list of valid options if you want to to consider the
   distribution type for use when modellign with random parameters. The
   valid options include: ‘Normal’, ‘LnNormal’, ‘Triangular’, and
   ‘Uniform’.
-  ``transformations``: This parameters is a list of transformations to
   consider. Plesee select all of the available options and put them
   into a list of valid options if you want to consider the
   transformation type. The valid options include ‘Normal’, ‘LnNormal’,
   ‘Triangular’, ‘Uniform’.
-  ``method_ll``: This is a specificication on the type of solvers are
   avilable to solve the lower level maximum likilihood objective. The
   valid options include: ‘Normal’, ‘LnNormal’, ‘Triangular’, and
   ‘Uniform’.

Example of changing the arguments:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modify the arguments according to your preferences using the commented
code as a guide.

.. code:: ipython3

    #Solution Arguments
    arguments = {
            'algorithm': 'hs', #alternatively input 'de', or 'sa'
            'is_multi': 1,
            'test_percentage': 0.2, # used in multi-objective optimisation only. Saves 20% of data for testing.
            'val_percenetage:': 0.2, # Saves 20% of data for testing.
            'test_complexity': 6, # Complexity level for testing (6 tests all) or a list to consider potential differences in complexity
            'instance_number': 'name', # used for creeating a named folder where your models are saved into from the directory
            'distribution': ['Normal', 'LnNormal', 'Triangular', 'Uniform'],
            'Model': [0,1],  # or equivalently ['POS', 'NB']
            'transformations': ['no', 'sqrt', 'archsinh'],
            'method_ll': 'BFGS_2',
            '_max_time': 10
        }
    obj_fun = ObjectiveFunction(X, y, **arguments)
    results = harmony_search(obj_fun)

Initial Solution Configurement
------------------------------

Listed below is an example of how to specify an initial solution within
the framework. This initial solution will be used to calculate the
fitness and considered in the objective-based search. However, as the
search progresses, different hypotheses may be proposed, and alternative
modeling components may completely replace the initial solution.

.. code:: ipython3

     #Model Decisions, Specify for initial solution that will be optimised.
    manual_fit_spec = {
        'fixed_terms': ['SINGLE', 'LENGTH'],
        'rdm_terms': ['AADT:normal'],
        'rdm_cor_terms': ['GRADEBR:normal', 'CURVES:normal'],
        'grouped_terms': [],
        'hetro_in_means': ['ACCESS:normal', 'MINRAD:normal'],
        'transformations': ['no', 'no', 'log', 'no', 'no', 'no', 'no'],
        'dispersion': 0
    }
    
    
    #Search Arguments
    arguments = {
        'algorithm': 'hs',
        'test_percentage': 0.2,
        'test_complexity': 6,
        'instance_number': 'name',
        'Manual_Fit': manual_fit_spec
    }
    obj_fun = ObjectiveFunction(X, y, **arguments)


.. parsed-literal::

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
    

Simarly to return the results feed the objective function into a
metaheuristic solution algorithm. An example of this is provided below:

.. code:: ipython3

    results = harmony_search(obj_fun)
    print(results)

Notes:
======

Capabilities of the software include:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Handling of Panel Data
-  Support for Data Transformations
-  Implementation of Models with Correlated and Non-Correlated Random
   Parameters
-  A variety of mixing distributions for parameter estimations,
   including normal, lognormal, truncated normal, Lindley, Gamma,
   triangular, and uniform distributions Capability to handle
   heterogeneity in the means of the random parameters
-  Use of Halton draws for simulated maximum likelihood estimation
-  Support for grouped random parameters with unbalanced groups
-  Post-estimation tools for assessing goodness of fit, making
   predictions, and conducting out-of-sample validation
-  Multiple parameter optimization routines, such as the BFGS method
-  Comprehensive hypothesis testing using single objectives, such as
   in-sample BIC and log-likelihood
-  Extensive hypothesis testing using multiple objectives, such as
   in-sample BIC and out-of-sample MAE (Mean Absolute Error), or
   in-sample AIC and out-of-sample MSPE (mean-square prediction errorr)
-  Features that allow analysts to pre-specify variables, interactions,
   and mixing distributions, among others
-  Meta-heuristic Guided Optimization, including techniques like
   Simulated Annealing, Harmony Search, and Differential Evolution
-  Customization of Hyper-parameters to solve problems tailored to your
   dataset
-  Out-of-the-box optimization capability using default metaheuristics

Intepreting the output of the model:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A regression table is produced. The following text elements are
explained: - Std. Dev.: This column appears for effects that are related
to random paramters and displays the assument distributional assumption
next to it - Chol: This term refers to Cholesky decomposition element,
to show the correlation between two random paramaters. The combination
of the cholesky element on iyself is equivalent to a normal random
parameter. - hetro group: This term represents the heterogeneity group
number, which refers all of the contributing factors that share
hetrogentiy in the means to each other under the same numbered value. -
:math:`\tau`: This column, displays the type of transformation that was
applied to the specific contributing factor in the data.

Arguments:
----------

In reference to the arguments that can be fed into the solution alrogithm, a dictionary system is utilised with relecant names these include
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following list describes the arguments available in this function.
By default, all of the capabilities described are enabled unless
specified otherwise as an argument. For list arguments, include all
desired elements in the list to ensure the corresponding options are
considered. Example code will be provided later in this guide.

1. **``complexity_level``**: This argument accepts an integer 1-6 or a
   list based of integegers between 0 to 5 eg might be a possible
   configuration [0, 2, 3]. Each integer represents a hierarchy level
   for estimable models associated with each explanatory variable. Here
   is a summary of the hierarchy:

   -  0: Null model
   -  1: Simple fixed effects model
   -  2: Random parameters model
   -  3: Random correlated parameters model
   -  4: Grouped random parameters model
   -  5: Heterogeneity in the means random parameter model

   **Note:** For the grouped random parameters model, groupings need to
   be defined prior to estimation. This can be achieved by including the
   following key-value pair in the arguments of the
   ``ObjectiveFunction``: ``'group': "Enter Column Grouping in data"``.
   Replace ``"Enter Column Grouping in data"`` with the actual column
   grouping in your dataset.

   Similarly, for panel data, the panel column needs to be defined using
   the key-value pair:
   ``'panel': "enter column string covering panels"``. Replace
   ``"enter column string covering panels"`` with the appropriate column
   string that represents the panel information in your dataset.

2. **``distributions``**: This argument accepts a list of strings where
   each string corresponds to a distribution. Valid options include:

   -  “Normal”
   -  “Lindley”
   -  “Uniform”
   -  “LogNormal”
   -  “Triangular”
   -  “Gamma”
   -  “TruncatedNormal”
   -  Any of the above, concatenated with “:” (e.g., “Normal:grouped”;
      requires a grouping term defined in the model)

3. **``Model``**: This argument specifies the model form. It can be a
   list of integers representing different models to test:

   -  0: Poisson
   -  1: Negative-Binomial
   -  2: Generalized-Poisson

4. **``transformations``**: This argument accepts a list of strings
   representing available transformations within the framework. Valid
   options include:

   -  “no”
   -  “square-root”
   -  “logarithmic”
   -  “archsinh”
   -  “as_factor”

5. **``is_multi``**: This argument accepts an integer indicating whether
   single or multiple objectives are to be tested (0 for single, 1 for
   multiple).

6. **``test_percentage``**: This argument is used for multi-objective
   optimization. Define it as a decimal; for example, 0.2 represents 20%
   of the data for testing.

7. **``val_percentage``**: This argument saves data for validation.
   Define it as a decimal; for example, 0.2 represents 20% of the data
   for validation.

8. **``_max_time``**: This argument is used to add a termination time in
   the algorithm. It takes values as seconds. Note the time is only
   dependenant on the time after intial population of solutions are
   generated.

Example: Assistance by Harmony Search
-------------------------------------

Let’s begin by fitting very simple models and use the structure of these
models to define our objectives. Then, we can conduct a more extensive
search on the variables that are more frequently identified. For
instance, in the case below, the complexity is level 3, indicating that
we will consider, at most randomly correlated parameters. This approach
is useful for initially identifying a suitable set of contributing
factors for our search.

.. code:: ipython3

    
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
            'instance_number': 'hs_run', # used for creeating a named folder where your models are saved into from the directory
            'distribution': ['Normal'],
            'Model': [0, 1],  # or equivalently ['POS', 'NB']
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

Example: Assistance by Differential Evololution and Simulated Annealing
-----------------------------------------------------------------------

Similiar to the above example we only need to change the
hyperparamaters, the obj_fun can remane the same

.. code:: ipython3

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

Comparing to statsmodels
------------------------

The following example illustrates how the output compares to well-known
packages, including Statsmodels.”

.. code:: ipython3

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
        'grouped_terms': [],
        'hetro_in_means': [],
        'transformations': ['no', 'no'],
        'dispersion': 1 #Negative Binomial 
    }
    
    
    #Arguments
    arguments = {
        'algorithm': 'hs',
        'test_percentage': 0,
        'test_complexity': 6,
        'instance_number': 'name',
        'Manual_Fit': manual_fit_spec
    }
    obj_fun = ObjectiveFunction(data_exog, data_endog, **arguments)
    
    
    
    
    
    
    


.. parsed-literal::

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
    

Paper
-----

The following tutorial is in conjunction with our latest paper. A link
the current paper can be found here
`MetaCountRegressor <https://www.overleaf.com/read/mszwpwzcxsng#c5eb0c>`__

Contact
-------

If you have any questions, ideas to improve MetaCountRegressor, or want
to report a bug, just open a new issue in `GitHub
repository <https://github.com/zahern/CountDataEstimation>`__.

Citing MetaCountRegressor
-------------------------

Please cite MetaCountRegressor as follows:

Ahern, Z., Corry P., Paz A. (2024). MetaCountRegressor [Computer
software]. https://pypi.org/project/metacounregressor/

Or using BibTex as follows:

\```bibtex @misc{Ahern2024Meta, author = {Zeke Ahern, Paul Corry and
Alexander Paz}, journal = {PyPi}, title = {metacountregressor Â· PyPI},
url = {https://pypi.org/project/metacountregressor/0.1.80/}, year =
{2024}, }
