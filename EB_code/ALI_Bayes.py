import pymc as pm
import numpy as np
import  pandas as pd
from pytensor.tensor import gammaln


def run_mcmc():
    data = pd.read_csv('data/rural_int.csv')
    # Assuming N is defined and your data arrays are initialized
    N = len(data)
    y = data['crashes']
    lnF = np.log(data['FAADT'])
    x2 = data['DP01']
    x6 = data['DX32']
    x7 =data['segment_length']
    z1 = data['TMIN']
    z3 = data['TMAX']
    off = data['paved_shoulder']
    id_reg =pd.factorize(data['county'].values)
    #k = pm.Beta('k', alpha=1, beta=1, shape=8)  # Define k using a Beta distribution
    #t = pm.Gamma('t', alpha=0.1, beta=0.1, shape=8)  # Define t for each region
    #z = pm.Bernoulli('z', p=k[id_reg], shape=N)  # Define z as a Bernoulli variable based on k

    with pm.Model() as model:
        # Priors for region-specific parameters
        b1_reg_bar = pm.Normal('b1_reg_bar', mu=0, sigma=10)
        b1_reg_tau = pm.Gamma('b1_reg_tau', alpha=0.1, beta=0.1)
        b1_reg_var = 1 / b1_reg_tau
        b1_reg_sd = pm.math.sqrt(b1_reg_var)

        b2_reg_bar = pm.Normal('b2_reg_bar', mu=0, sigma=10)
        b2_reg_tau = pm.Gamma('b2_reg_tau', alpha=0.1, beta=0.1)
        b2_reg_var = 1 / b2_reg_tau
        b2_reg_sd = pm.math.sqrt(b2_reg_var)

        b3_reg_bar = pm.Normal('b3_reg_bar', mu=0, sigma=10)
        b3_reg_tau = pm.Gamma('b3_reg_tau', alpha=0.1, beta=0.1)
        b3_reg_var = 1 / b3_reg_tau
        b3_reg_sd = pm.math.sqrt(b3_reg_var)

        # Define region-specific parameters
        b1_reg = pm.Normal('b1_reg', mu=b1_reg_bar, sigma=b1_reg_sd, shape=8)
        b2_reg = pm.Normal('b2_reg', mu=b2_reg_bar, sigma=b2_reg_sd, shape=8)
        b3_reg = pm.Normal('b3_reg', mu=b3_reg_bar, sigma=b3_reg_sd, shape=8)

        # Other parameters
        b2 = pm.Normal('b2', mu=0, sigma=10)
        b3 = pm.Normal('b3', mu=0, sigma=10)
        b4 = pm.Normal('b4', mu=0, sigma=10)
        b5 = pm.Normal('b5', mu=0, sigma=10)
        print(1)
        phi = pm.Gamma('phi', alpha=0.1, beta=0.1)

        # Define k and t inside the model context
        k = pm.Beta('k', alpha=1, beta=1, shape=8)  # Define k using a Beta distribution
        t = pm.Gamma('t', alpha=0.1, beta=0.1, shape=8)  # Define t for each region

        # Likelihood
        mu = pm.math.exp(b1_reg[id_reg] * lnF + b2 * x2 + b3 * x6 + b4 * x7 +
                         b2_reg[id_reg] * z1 + b3_reg[id_reg] * z3 + b5 * off)
        print(mu)
        z = pm.Bernoulli('z', p=k[id_reg], shape=N)  # Define z as a Bernoulli variable based on
        eps = pm.Gamma('eps', alpha=1 + z, beta=t[id_reg])  # Define eps using gamma distribution based on z
        #z = pm.Bernoulli('z', p=k[id_reg], shape=N)  # Define z as a Bernoulli variable based on
        prob = phi / (phi + eps * mu)

        # Negative Binomial likelihood
        y_obs = pm.NegativeBinomial('y_obs', mu=prob, alpha=phi, observed=y)

        # Log-likelihood computation
        #LL = (gammaln(phi + y) - gammaln(phi) - gammaln(y + 1) +
         #     phi * (pm.math.log(phi) - pm.math.log(eps * mu + phi)) +
         #     y * (pm.math.log(eps * mu) - pm.math.log(eps * mu + phi)))

        # You can create a log-likelihood variable if needed
        #log_likelihood = pm.Deterministic('log_likelihood', LL)
        trace = pm.sample(2000, tune=1000,return_inferencedata=False)
        print(1)
    return  trace
    # Sample or inference can be done here

if __name__ == '__main__':
    print('loading')
    trace = run_mcmc()
    print(trace)
    print('cool')
