import numpy as np
import pandas as pd
import pymc as pm
from scipy.optimize import minimize
from scipy.stats import nbinom
import bambi as bmb
import arviz as az
from scipy import stats
from scipy.special import gammaln
# Simulated data
np.random.seed(42)
n = 12000
x = np.random.rand(n)
x2 = np.random.uniform(.2, .8, n)
# True parameters
alpha_true = 7
beta_true = -2.0

phi_true = 2.0  # true dispersion parameter
# Negative binomial distribution parameters
r = 5  # number of successes
p = r / (r + np.exp(alpha_true + beta_true * x)/r)  # probability of success
mu = np.exp(alpha_true +beta_true*x)
g = stats.gamma.rvs(r, scale=mu / r, size=n)
y2=  stats.poisson.rvs(g)
import statsmodels.api as sm
y = nbinom.rvs(r, p, size=n)

data = pd.DataFrame({
    'x': x,
    'x2': x2,
    'y': y2,
})


# Step 1: Define the negative log-likelihood function
def neg_log_likelihood(params, x, y):
    alpha, beta, phi = params
    linear_predictor =  alpha +beta * x

    # Clip the linear predictor to avoid overflow
    # Adjust the threshold according to your specific needs
    linear_predictor_clipped = np.clip(linear_predictor, -600, 600)

    # Calculate mu
    phi2 = np.clip(phi, -10, 10)
    mu = np.exp(linear_predictor_clipped)*np.exp(phi2)
    #sum(mu)
    # Adjust the p parameter based on phi
    #p = r / (r + mu / phi)  # probability of success based on dispersion

   # alt = gammaln(y + phi) - gammaln(y + 1) - gammaln(phi) + y * np.log(mu) + alpha * np.log(phi) - (
               #         y + phi) * np.log(mu + phi)

    size = 1 / np.exp(phi2) * mu ** 0
    prob = size/ (size + mu)
    gg_alt = nbinom.pmf(y, 1 / np.exp(phi2), prob)
    ll_g = np.sum(np.log(gg_alt))
    print(ll_g)
    coeff = (gammaln(size + y) - gammaln(y + 1) -
             gammaln(size))
    llf = coeff + size * np.log(prob) + y * np.log(1 - prob)



    #ll_2 = np.sum(alt)
    #ll = np.sum(nbinom.logpmf(y, r, p))
    llsum = np.nanmax(np.sum(llf))
    #print(llsum)
    if llsum >0:
        llsum = -llsum
    print(llsum, 'd')
    return -llsum/n  # minimize the negative log-likelihood


# Fit using BFGS



# Function to fit the model using PyMC3
def fit_pymc3(x, y):
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=1, sigma=4)
        beta = pm.Normal('beta', mu=0, sigma=8)

        phi = pm.HalfCauchy('phi', 1)  # Dispersion parameter

        mu = pm.math.exp(alpha+beta * x)/phi
       # p = phi / (phi + mu)  # Adjust probability of success

        p = phi/(phi+mu)
        y_obs = pm.NegativeBinomial('y_obs', n=phi, p=p, observed=y)

        # Sample from the posterior
        trace = pm.sample(2000, tune=1000, return_inferencedata=False)

    return trace

def fit_py_by(x, y):
    with pm.Model() as model:
        alpha = pm.Normal('alpha',mu = 0, sigma = 4)
        beta = pm.Normal('beta', mu=0, sigma=8.6)
        #phi = pm.HalfNormal('phi', sigma=10)  # Dispersion parameter
        alpha1 = pm.Exponential('alpha1', .5)
        mu = pm.math.exp(alpha + beta * x)
       # p = r / (r + mu / phi)  # Adjust probability of success
       # y_obs = pm.NegativeBinomial('y_obs', n=r, p=p, observed=y)

        y_obs = pm.NegativeBinomial('Y_obs', alpha=alpha1, mu=mu, observed=y)
        # Sample from the posterior
        trace = pm.sample( return_inferencedata=False)
    return  trace

def bambi_d(data):
    model_additive = bmb.Model("y ~ 1 + x", data, family="negativebinomial")
    idata_additive = model_additive.fit()
    az.summary(idata_additive)
    print(az.summary(idata_additive))

# Main function to execute the fitting process
if __name__ == '__main__':

    bambi_d(data)
    trace = fit_pymc3(x, y)
    alpha_pymc3 = trace['alpha'].mean()
    beta_pymc3 = trace['beta'].mean()
    phi_pymc3 = trace['phi'].mean()
    print("\nPyMC3 Estimates:")
    print(f"Alpha: {alpha_pymc3}, Beta: {beta_pymc3}, Phi: {phi_pymc3}")
    trace_2 = fit_py_by(x, y)
    alpha_py = trace_2['alpha'].mean()
    beta_py = trace_2['beta'].mean()
    phi_py = trace_2['alpha1'].mean()
    print("\nPyMC3 Estimates Al:")
    print(f"Alpha: {alpha_py}, Beta: {beta_py}, Phi: {phi_py}")


    TEST = 0
    if TEST:
        #new = sm.NegativeBinomial(data['y'], sm.add_constant(data['x'])).fit()
        #print(new.summary())
        bambi_d(data)
        trace = fit_pymc3(x, y)

        # Get the posterior means for comparison
        alpha_pymc3 = trace['alpha'].mean()
        beta_pymc3 = trace['beta'].mean()
        phi_pymc3 = trace['phi'].mean()

        trace_2 = fit_py_by(x,y)
        alpha_py = trace_2['alpha'].mean()
        beta_py = trace_2['beta'].mean()
        phi_py = trace_2['alpha1'].mean()




    initial_params = [2.28, 4, np.log(.2)]
    result = minimize(neg_log_likelihood, initial_params, args=(x, y))
    beta_bfgs, bb, phi_bfgs = result.x
    new = sm.NegativeBinomial(data['y'], sm.add_constant(data[['x']])).fit()
    print(new.summary())
    #model = sm.GLM(data['y'], sm.add_constant(data[['x']]), family=sm.families.Poisson())
    #results = model.fit()
    #print(results.summary())
    # Output the results
    print("BFGS Estimates:")
    print(f" Beta: {beta_bfgs}, BB2: {bb},Phi: {np.exp(phi_bfgs)}")
    if TEST:

        print("\nPyMC3 Estimates:")
        print(f"Alpha: {alpha_pymc3}, Beta: {beta_pymc3}, Phi: {phi_pymc3}")

        print("\nPyMC ALte Estimates:")
        print(f"Alpha: {alpha_py}, Beta: {beta_py}, Phi: {phi_py}")

        model = sm.GLM(data['y'], data['x', 'x2'], family=sm.families.NegativeBinomial())
        results = model.fit()
        print(results.scale)
        # Print the summary of the model
        print(results.summary())
        new = sm.NegativeBinomial(data['y'], sm.add_constant(data['x'])).fit()
        print(new.summary())

        #model = sm.GLM(data['y'], sm.add_constant(data['x']), family=sm.families.Poisson())
        #results = model.fit()
        # Print the summary of the model
        #print(results.summary())