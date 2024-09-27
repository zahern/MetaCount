import numpy as np
from scipy.stats import rv_discrete
from scipy.special import gamma, gammaln


class gpd_gen(rv_discrete):
    """
    A Lagrangian Generalised Poisson-Poisson distribution.

    ``eta`` is the branching ratio,
    ``mu`` is the intital population expectation.

    See Consul, P. C., & Famoye, F. (2006)
        Lagrangian probability distributions, chapter 9.

    Also https://github.com/scipy/scipy/blob/master/scipy/stats/_distn_infrastructure.py

    Statistics are computed using numerical integration by default.
      For speed you can redefine this using ``_stats``:
       - take shape parameters and return mu, mu2, g1, g2
       - If you can't compute one of these, return it as None
       - Can also be defined with a keyword argument ``moments``, which is a
         string composed of "m", "v", "s", and/or "k".
         Only the components appearing in string should be computed and
         returned in the order "m", "v", "s", or "k"  with missing values
         returned as None.
      Alternatively, you can override ``_munp``, which takes ``n`` and shape
      parameters and returns the n-th non-central moment of the distribution.
    """

    def _argcheck(self, mu, eta):
        """
        ``eta``, ``mu`` arguments are handled here
        """
        return mu >= 0.0 and eta >= 0.0 and eta <= 1.0

    def _rvs(self, mu, eta):
        """
        Simulate using branching processes.
        """
        # print("mu", mu, "eta", eta)
        # Always work with arrays for consistent iteration
        population = np.asarray(
            self._random_state.poisson(mu, self._size)
        )
        if population.shape == ():
            population = population.reshape(-1)
        offspring = population.copy()
        while np.any(offspring > 0):
            # probability dists are NOT ufuncs
            # print("offspring", offspring)
            offspring[:] = [
                self._random_state.poisson(m)
                for m in eta*offspring
            ]
            population += offspring
        return population

    def _pmf(self, k, mu, eta):
        """
        Warning, numerically unstable;
        I should probably exponentiate the log pmf
        """
        offset = mu + eta*k
        return mu*(offset**(k-1))/(gamma(k+1)*np.exp(offset))

    def _logpmf(self, k, mu, eta):
        offset = mu + eta*k
        return np.log(mu) + (k-1) * np.log(offset) - gammaln(k+1) - offset

    def _munp(self, n, mu, eta):
        """
        See Consul and Famoye Ch 9.3, or Consul and Shenton 1975 (30) or
        Janardan 1984.

        TODO: make sure floats are handled right

        TODO: construct noncentral moments from central ones,
        which are given in the above references;
        requires tedious calculations.
        """
        if n == 1:
            return mu/(1-eta)
        elif n == 2:
            return (mu/(1-eta))**2+mu/(1-eta)**3
gpd = gpd_gen(name='gpd')
