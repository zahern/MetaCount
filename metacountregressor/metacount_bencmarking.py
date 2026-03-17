# ================================================
# BENCHMARK EXPERIMENT:
# MetaCountRegressor vs Conventional Estimators
# ================================================

import numpy as np
import pandas as pd
import time
import itertools

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# MetaCountRegressor
from metacountregressor.solution import ObjectiveFunction
from metacountregressor.metaheuristics import harmony_search

# R interface
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
MASS = importr("MASS")


# ================================================
# Utility Functions
# ================================================

def mspe(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def run_poisson(X_train, y_train, X_test, y_test):
    start = time.time()
    
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)

    model = sm.GLM(
        y_train,
        X_train_sm,
        family=sm.families.Poisson()
    ).fit()

    preds = model.predict(X_test_sm)
    
    runtime = time.time() - start

    return {
        "AIC": model.aic,
        "BIC": model.bic,
        "LogLik": model.llf,
        "MSPE": mspe(y_test, preds),
        "Runtime": runtime
    }


def run_nb_statsmodels(X_train, y_train, X_test, y_test):
    start = time.time()

    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)

    model = sm.GLM(
        y_train,
        X_train_sm,
        family=sm.families.NegativeBinomial()
    ).fit()

    preds = model.predict(X_test_sm)
    
    runtime = time.time() - start

    return {
        "AIC": model.aic,
        "BIC": model.bic,
        "LogLik": model.llf,
        "MSPE": mspe(y_test, preds),
        "Runtime": runtime
    }


def run_stepwise_nb(X_train, y_train, X_test, y_test):
    start = time.time()

    best_aic = np.inf
    best_model = None

    for k in range(1, len(X_train.columns) + 1):
        for combo in itertools.combinations(X_train.columns, k):
            X_sub = sm.add_constant(X_train[list(combo)])
            model = sm.GLM(
                y_train,
                X_sub,
                family=sm.families.NegativeBinomial()
            ).fit()
            
            if model.aic < best_aic:
                best_aic = model.aic
                best_model = model
                best_combo = combo

    X_test_sub = sm.add_constant(X_test[list(best_combo)])
    preds = best_model.predict(X_test_sub)
    
    runtime = time.time() - start

    return {
        "AIC": best_model.aic,
        "BIC": best_model.bic,
        "LogLik": best_model.llf,
        "MSPE": mspe(y_test, preds),
        "Runtime": runtime
    }


def run_nb_r(X_train, y_train, X_test, y_test):
    start = time.time()

    df_train = X_train.copy()
    df_train["y"] = y_train

    r_df = pandas2ri.py2rpy(df_train)

    ro.globalenv["data"] = r_df
    formula = ro.Formula("y ~ .")

    model = MASS.glm_nb(formula, data=r_df)

    # Predictions
    df_test = X_test.copy()
    r_test = pandas2ri.py2rpy(df_test)
    preds = ro.r.predict(model, newdata=r_test, type="response")
    preds = np.array(preds)

    runtime = time.time() - start

    return {
        "AIC": np.array(ro.r.AIC(model))[0],
        "BIC": np.array(ro.r.BIC(model))[0],
        "LogLik": np.array(ro.r.logLik(model))[0],
        "MSPE": mspe(y_test, preds),
        "Runtime": runtime
    }


def run_metacount(X_train, y_train, X_test, y_test):
    start = time.time()

    obj_fun = ObjectiveFunction(X_train, y_train)
    results = harmony_search(obj_fun)

    best_model = results["best_model"]
    preds = best_model.predict(X_test)

    runtime = time.time() - start

    return {
        "AIC": best_model.aic,
        "BIC": best_model.bic,
        "LogLik": best_model.loglik,
        "MSPE": mspe(y_test, preds),
        "Runtime": runtime
    }


# ================================================
# MAIN EXPERIMENT LOOP
# ================================================

def run_experiment(X, y, seeds=[1,2,3,4,5]):

    all_results = []

    for seed in seeds:

        print(f"Running seed {seed}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed
        )

        results = {
            "Poisson": run_poisson(X_train, y_train, X_test, y_test),
            "NB_statsmodels": run_nb_statsmodels(X_train, y_train, X_test, y_test),
            "Stepwise_NB": run_stepwise_nb(X_train, y_train, X_test, y_test),
            "NB_R_MASS": run_nb_r(X_train, y_train, X_test, y_test),
            "MetaCountRegressor": run_metacount(X_train, y_train, X_test, y_test)
        }

        for method, metrics in results.items():
            metrics["Method"] = method
            metrics["Seed"] = seed
            all_results.append(metrics)

    df_results = pd.DataFrame(all_results)

    summary = df_results.groupby("Method").mean().reset_index()

    return df_results, summary


# ================================================
# EXAMPLE USAGE
# ================================================

if __name__ == "__main__":

    # Example synthetic dataset (replace with crash data)
    np.random.seed(42)
    n = 1000

    X = pd.DataFrame({
        "X1": np.random.normal(size=n),
        "X2": np.random.normal(size=n),
        "X3": np.random.normal(size=n),
        "X4": np.random.normal(size=n)
    })

    lambda_true = np.exp(0.5*X["X1"] - 0.3*X["X2"] + 0.2*X["X3"])
    y = np.random.poisson(lambda_true)

    full_results, summary_results = run_experiment(X, y)

    print("\n===== Average Results Across Seeds =====")
    print(summary_results)