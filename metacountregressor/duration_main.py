#from main_hpc import *

import numpy as np
import pandas as pd

def generate_schedule_data(N=300, A=4, seed=0):

    np.random.seed(seed)

    rows = []

    for i in range(N):

        B = np.random.uniform(12, 18)  # variable daily budget

        beta = np.array([0.4, -0.2])
        sigma = 0.4

        total_time = 0.0

        durations = []

        for a in range(A):

            x1 = np.random.normal()
            x2 = np.random.normal()

            eta = beta[0]*x1 + beta[1]*x2

            T = np.exp(np.random.normal(eta, sigma))
            durations.append((T, x1, x2))

            total_time += T

        # Normalize to budget (so truth obeys constraint)
        scale = B / total_time

        for a, (T, x1, x2) in enumerate(durations):

            rows.append([
                i, a,
                T * scale,
                x1, x2,
                B
            ])

    df = pd.DataFrame(rows,
                      columns=["ID","ACT","DURATION","x1","x2","B"])

    return df


def prepare_data(df):

    X = df[["x1","x2"]].values
    y = df["DURATION"].values
    ids = df["ID"].values
    budgets = df.groupby("ID")["B"].first().values

    return X, y, ids, budgets

import jax
import jax.numpy as jnp

def lognormal_ll(y, eta, sigma):

    return (
        - jnp.log(y)
        - jnp.log(sigma)
        - 0.5 * ((jnp.log(y) - eta)/sigma)**2
    )
    
def ll_independent(params, X, y):

    beta = params[:2]
    sigma = jax.nn.softplus(params[2])

    eta = X @ beta

    ll = lognormal_ll(y, eta, sigma)

    return -jnp.sum(ll)


def ll_with_budget_penalty(params, X, y, ids, budgets, lambda_penalty=10.0):

    beta = params[:2]
    sigma = jax.nn.softplus(params[2])

    eta = X @ beta

    ll = lognormal_ll(y, eta, sigma)

    total_ll = jnp.sum(ll)

    # Budget penalty
    unique_ids = jnp.unique(ids)

    penalty = 0.0

    for i, uid in enumerate(unique_ids):

        mask = ids == uid

        total_time = jnp.sum(y[mask])
        B = budgets[i]

        penalty += (total_time - B)**2

    total_ll -= lambda_penalty * penalty

    return -total_ll


from scipy.optimize import minimize
from jax import grad

def estimate_model(objective, init_params):

    result = minimize(
        objective,
        init_params,
        jac=grad(objective),
        method="L-BFGS-B"
    )

    return result



import numpy as np
import jax
import jax.numpy as jnp

def predict_daily_schedule(params, df):

    beta = params[:2]
    sigma = jax.nn.softplus(params[2])

    X = df[["x1","x2"]].values
    ids = df["ID"].values

    # Step 1: unconstrained lognormal mean
    eta = X @ beta
    unconstrained = np.exp(np.array(eta + 0.5 * sigma**2))

    df["pred_unscaled"] = unconstrained

    predictions = []

    # Step 2: rescale per individual
    for i in df["ID"].unique():

        sub = df[df["ID"] == i]

        B = sub["B"].iloc[0]

        total_pred = sub["pred_unscaled"].sum()

        scaled = B * sub["pred_unscaled"] / total_pred

        predictions.extend(scaled)

    return np.array(predictions)

df = generate_schedule_data()

X, y, ids, budgets = prepare_data(df)

init = np.zeros(3)




df = generate_schedule_data()

X, y, ids, budgets = prepare_data(df)

init = np.zeros(3)

res = estimate_model(
    lambda p: ll_with_budget_penalty(p, X, y, ids, budgets, lambda_penalty=50),
    init
)

pred_durations = predict_daily_schedule(res.x, df)

df["predicted_duration"] = pred_durations