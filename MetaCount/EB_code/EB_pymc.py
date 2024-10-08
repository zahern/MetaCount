

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pandas as pd
from sklearn import preprocessing
import scipy
import scipy.stats as stats
import seaborn.apionly as sns
import statsmodels.api as sm
import theano.tensor as tt

data = pd.read_csv('radon.csv')

county_names = data.county.unique()
county_idx = data['county_code'].values





with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu_a = pm.Normal('mu_alpha', mu=0., sigma=1)
    sigma_a = pm.HalfCauchy('sigma_alpha', beta=1)
    mu_b = pm.Normal('mu_beta', mu=0., sigma=1)
    sigma_b = pm.HalfCauchy('sigma_beta', beta=1)

    # Intercept for each county, distributed around group mean mu_a
    a = pm.Normal('alpha', mu=mu_a, sigma=sigma_a, shape=len(data.county.unique()))
    # Intercept for each county, distributed around group mean mu_a
    b = pm.Normal('beta', mu=mu_b, sigma=sigma_b, shape=len(data.county.unique()))

    # Model error
    eps = pm.HalfCauchy('eps', beta=1)

    # Expected value
    radon_est = a[county_idx] + b[county_idx] * data.floor.values

    # Data likelihood
    y_like = pm.Normal('y_like', mu=radon_est, sigma=eps, observed=data.log_radon)


# random effects
le = preprocessing.LabelEncoder()
messages = pd.read_csv('data/hangout_chat_data.csv')
participants_idx = le.fit_transform(messages['prev_sender'])
participants = le.classes_
n_participants = len(participants)

with pm.Model() as model:
    intercept = pm.Normal('intercept', mu=0, sd=100, shape=n_participants)
    slope_message_length = pm.Normal('slope_message_length', mu=0, sd=100)
    slope_is_weekend = pm.Normal('slope_is_weekend', mu=0, sd=100)
    slope_num_participants = pm.Normal('slope_num_participants', mu=0, sd=100)

    mu = tt.exp(intercept[participants_idx]
                + slope_message_length * messages.message_length
                + slope_is_weekend * messages.is_weekend
                + slope_num_participants * messages.num_participants)

    y_est = pm.Poisson('y_est', mu=mu, observed=messages['time_delay_seconds'].values)

    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(200000, step, start=start, progressbar=True)


## negative binomial

COORDS = {"regressor": ["nomeds", "alcohol", "nomeds:alcohol"], "obs_idx": df.index}

with pm.Model(coords=COORDS) as m_sneeze_inter:
    a = pm.Normal("intercept", mu=0, sigma=5)
    b = pm.Normal("slopes", mu=0, sigma=1, dims="regressor")
    alpha = pm.Exponential("alpha", 0.5)

    M = pm.ConstantData("nomeds", df.nomeds.to_numpy(), dims="obs_idx")
    A = pm.ConstantData("alcohol", df.alcohol.to_numpy(), dims="obs_idx")
    S = pm.ConstantData("nsneeze", df.nsneeze.to_numpy(), dims="obs_idx")

    λ = pm.math.exp(a + b[0] * M + b[1] * A + b[2] * M * A)

    y = pm.NegativeBinomial("y", mu=λ, alpha=alpha, observed=S, dims="obs_idx")

    idata = pm.sample()