import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.stats import nbinom
# Define the size of your data
constant = -2

size = 2000

# Generate independent explanatory variables
X1 = np.random.normal(loc=0.0, scale=1.0, size=size)
X2 = np.random.normal(loc=0.0, scale=1.0, size=size)
X3 = np.random.normal(loc=0.0, scale=1.0, size=size)


# scale betwen 0 and 1
X1 = (X1 - np.min(X1)) / (np.max(X1) - np.min(X1))
X2 = (X2 - np.min(X2)) / (np.max(X2) - np.min(X2))
X3 = (X3 - np.min(X3)) / (np.max(X3) - np.min(X3))
X1_avg, X2_avg, X3_avg = np.zeros(size), np.zeros(size), np.zeros(size)
Y_avg = np.zeros(size)
noise = 0.1
X_non_relevant1 = X1+ np.random.normal(loc=0.0, scale=noise, size=size)
X_non_relevant2 = X2+2*X3++np.random.normal(loc=0.0, scale=noise, size=size)
X_non_relevant3 = X3+X1+np.random.normal(loc=0.0, scale=noise, size=size)
X_non_relevant4 = X1+X2+np.random.normal(loc=0.0, scale=noise, size=size)
X_non_relevant5 = 2*X1+X2+X3+np.random.normal(loc=0.0, scale=noise, size=size)

# Min-Max scaling 
X_non_relevant1 = (X_non_relevant1 - np.min(X_non_relevant1)) / (np.max(X_non_relevant1) - np.min(X_non_relevant1))
X_non_relevant2 = (X_non_relevant2 - np.min(X_non_relevant2)) / (np.max(X_non_relevant2) - np.min(X_non_relevant2))
X_non_relevant3 = (X_non_relevant3 - np.min(X_non_relevant3)) / (np.max(X_non_relevant3) - np.min(X_non_relevant3))
X_non_relevant4 = (X_non_relevant4 - np.min(X_non_relevant4)) / (np.max(X_non_relevant4) - np.min(X_non_relevant4))
X_non_relevant5 = (X_non_relevant5 - np.min(X_non_relevant5)) / (np.max(X_non_relevant5) - np.min(X_non_relevant5))


# Generate independent random variables
uniform_data = np.random.uniform(low=0.0, high=1.0, size=size)
normal_data = np.random.normal(loc=0.0, scale=1.0, size=size)
triangular_data = np.random.triangular(left=-1.0, mode=0.0, right=1.0, size=size)

# Define a correlation matrix
corr_matrix = np.array([
    [1.0, -0.6, -0.3],
    [-0.6, 1.0, -0.2],
    [-0.3, -0.2, 1.0]
])
# Perform Cholesky decomposition
cholesky_decomp = np.linalg.cholesky(corr_matrix)
num_trials = 10000
for trial in range(num_trials):

    # Generate independent explanatory variables
   # X1 = np.random.uniform(size=size)
   # X2 = np.random.uniform(size=size)
   # X3 = np.random.uniform(size=size)

    # Generate independent random variables
    uniform_data = 5*np.random.normal(size=size)
    normal_data = 2*np.random.normal(scale=3, size=size)
    triangular_data = 3*np.random.normal(size=size)

    # Create a matrix of independent variables
    independent_vars = np.array([uniform_data, normal_data, triangular_data])

    # Multiply by the Cholesky decomposition to get correlated variables
    correlated_vars = np.dot(cholesky_decomp, independent_vars)

    # Form linear combination (including constant)
    linear_combination = constant + correlated_vars[0, :] * X1 + correlated_vars[1, :] * X2 + correlated_vars[2, :] * X3

    # Apply link function
    lambda_ = np.exp(linear_combination)

    # Generate synthetic response variable
    
    # Assume a disper   sion parameter
    alpha = 0.5

# G# Convert parameters to the (n, p) parameterization used by scipy
    n = lambda_ / alpha
    p = alpha / (1 + alpha)

# Generate the negative binomial count data
    Y = stats.nbinom.rvs(n, p)
    
    
    Y = np.random.poisson(lam=lambda_, size=size)

    # Accumulate for averages
    X1_avg += X1
    X2_avg += X2
    X3_avg += X3
    Y_avg += Y


# Compute averages
X1_avg /= num_trials
X2_avg /= num_trials
X3_avg /= num_trials
Y_avg /= num_trials


# Convert to DataFrame
df = pd.DataFrame({
    'const': np.ones(size),
    'X1': X1_avg,
    'X2': X2_avg,
    'X3': X3_avg,
    'na1': X_non_relevant1,
    'na2':X_non_relevant2,
    'na3': X_non_relevant3,
    'na4': X_non_relevant4,
    'na5': X_non_relevant5,
    'Y': np.rint(Y_avg)
})

df.to_csv('artificial_mixed_corr_2023_MOOF.csv', index = False)