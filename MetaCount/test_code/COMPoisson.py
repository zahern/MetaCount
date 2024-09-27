import numpy as np
import math

def com_log_sum(x, y):
    if x == - math.inf:
        return y

    elif y == - math.inf:
        return x

    elif x > y:
        return x + np.log(1 + np.exp(y - x))

    else:
        return y + np.log(1 + np.exp(x - y))


def com_compute_log_z(lambda_, nu, log_error=1e-6):
    # Perform argument checking
    if lambda_ <= 0 or nu < 0:
        raise Exception("Invalid arguments, only defined for lambda > 0, nu >= 0")

    if nu == 0:
        return - np.log(1 - lambda_)  # Geometric sum
    if nu == 1:
        return np.exp(- lambda_)  # Poisson normalizing constant

        # Initialize values
    j = 0

    llambda = np.log(lambda_)  # precalculate for speed
    lfact = 0  # log(factorial(0))
    z = j * llambda - nu * lfact
    # first term in sum
    z_last = - math.inf  # to ensure entering the loop

    # Continue until we have reached specified precision
    while abs(z - z_last) > log_error:
        z_last = z
        # For comparison in while statement
        j = j + 1
        # Next term in sum
        lfact = lfact + np.log(j)  # Calculate increment for log factorial
        z = com_log_sum(z, j * llambda - nu * lfact)
        # Log of current sum

    return z


def com_loglikelihood(x, lamda, nu):
    if lamda < 0 or nu < 0:
        return -math.inf

    logz = com_compute_log_z(lamda, nu)
    return x*np.log(lamda)-nu*np.log(math.factorial(x))-logz