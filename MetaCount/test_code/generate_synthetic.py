import numpy as np
import pandas as pd
import random
import scipy.stats as ss
from scipy.stats import maxwell
#from scipy.stats import lindley
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import genpoisson_p
from scipy.stats import nbinom
import matplotlib.pyplot as plt

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
import rpy2.rinterface as rinterface
r = robjects.r
r['source']('./test_code/lindley.R')



N = int(10000/4)
np.random.seed(1213)


# Call the R function
rlindley_r = robjects.globalenv['R_LINDL']
samples = 0.5 + np.array(rlindley_r(N, .5))
coef_var2 = 1 + np.array(rlindley_r(N, .25))
coef_var2 = np.random.normal(loc = -1, scale = 1, size = N)

rgamma = robjects.globalenv['R_GAMMA']


from scipy.stats import rv_continuous


# generate random samples from the Lindley random variable
#samples = 2 + np.array(generate_lindley(3))

# print the mean and variance of the samples
print("Mean:", np.mean(samples))
print("Variance:", np.var(samples))



p = 9

# Generate the synthetic explanatory variables from a multivariate normal distribution
X = np.random.multivariate_normal(mean=[1, 1, 1, 1, 1, 5, 3, 2, 0], cov=np.eye(p), size=N)
print(X[0, :])
print(max(X[:, 3]))



def noise(n_obs, perc=.4):
    noise_vec = np.zeros(n_obs)
    rand_pos = np.random.randint(n_obs, size=(int(n_obs*perc)))
    noise_vec[rand_pos] = np.random.normal(size=(int(n_obs*perc)))
   
    return noise_vec


# Generate the first explanatory variable with mean 2 and random standard deviation of 1
coef_var1 = np.random.normal(loc=1, scale=.5, size=N)
#coef_var1 = samples
#coef_var2 = np.random.normal(loc = 3, scale = 2, size = N)
old_way = 0
if old_way:
    explanatory_var1 = np.random.normal(2, size=N)

    val = coef_var1*explanatory_var1
    # Generate three other explanatory variables
    explanatory_var2 = np.random.normal(size=N)
    explanatory_var3 =np.random.normal(size = N)
    explanatory_var4 = np.random.uniform(low = 0, high = 1, size=N)
    bad_1 =np.random.poisson(lam =3, size = N) + noise(N, 0.5)
    bad_2 = np.exp(1)*np.random.uniform(low =0, high = 10 , size = N) + noise(N)
    bad_3 =np.random.exponential(scale=5, size=N) + noise(N)
    bad_4 = np.random.gamma(shape =2 , scale=3, size = N) +  noise(N)
    bad_5 =-np.random.pareto(a = 2, size = N) +  noise(N)
else:
    explanatory_var1 = X[:, 0]

    val = coef_var1*explanatory_var1
    # Generate three other explanatory variables
    explanatory_var2 =X[:,1]
    explanatory_var3 =X[:, 2]
    explanatory_var4 = X[:,3]
    bad_1 =X[:, 4] + noise(N, 0.5)
    bad_2 =X[:, 5] + noise(N)
    bad_3 =X[:, 6] + noise(N)
    bad_4 = X[:, 7] +  noise(N)
    bad_5 =X[:, 8] +  noise(N)
        




const = np.ones(N)
y_param = -5*const+ val + coef_var2*explanatory_var2 + 2*explanatory_var3 + 1*explanatory_var4

# Convert betas to matrix for easy product
#response = np.random.poisson(lam=np.exp(y_param), size=N)
size = N
dispersion = 1

exb = np.exp(y_param)


xg = np.array(rgamma(N, dispersion, dispersion))
xbg = exb*xg 

nbyo = np.random.poisson(xbg)



response = nbyo
print(max(response))
df = pd.DataFrame({'Y': response,
                   'Constant': const,
                   'Var1': explanatory_var1, 
                   'Var2': explanatory_var2,
                   'Var3': explanatory_var3,
                   'Var4': explanatory_var4,
                   'Non_1': bad_1,
                    'Non_2': bad_2,       
                    'Non_3': bad_3,
                    'Non_4': bad_4,
                    'Non_5': bad_5})

df.to_csv('artificial_ZA.csv', index = False)
df = df.drop('Y', axis = 'columns')
df = df.drop(columns = ['Non_1', 'Non_2', 'Non_3', 'Non_4', 'Non_5'] )
res = sm.GLM(response, df, family=sm.families.NegativeBinomial()).fit()
print(res.summary())

print(2)

#df.to_csv('artificial_ZA.csv')


#gen_poisson_gp1 = sm.Poisson(y, X, p=2)
