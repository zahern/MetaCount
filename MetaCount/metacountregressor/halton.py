import numpy as np
from scipy.optimize import minimize, rosen
from scipy.optimize import fmin_bfgs
from scipy.stats import poisson
from scipy.stats import t, norm, gamma, poisson, triang
from scipy.special import gammaln, factorial
import itertools
import pandas as pd
import warnings

class count_model(object):

    def __init__(self, nDraws, distribution = None, predictor = 'FREQ'):
        self.Ndraws = nDraws
        self.draws1 = None
        self.observations = None
        self.distribution = distribution
        self.predictor = predictor
        self._x_data = None



    def PoissonNegLogLikelihood(self, lam, y):
        """computers the negative log-likelihood for a poisson random variable"""
        prob = poisson.pmf(y, lam)
        prob = prob.reshape(self.observations, -1, order = 'F')
        log_lik = np.sum(np.log(prob.mean(axis=1)))
        '''return log_like at possitive to ensure minimisation'''
        return -log_lik






    def poisson_mle(self, data):
        """
            Compute the maximum likelihood estimate (mle) for a poisson distribution given data.

            Inputs:
            data - float or array.  Observed data.

            Outputs:
            lambda_mle - float.  The mle for poisson distribution.
            """
        mle = minimize(self.PoissonNegLogLikelihood, 1, args=(data))
        lambda_mle = mle.x[0]
        return lambda_mle


    def poissonRegressionNegLogLikelihood(self, b, X, y, Xr =None):
        """
        Computes the negative log-likelihood for a poisson regression.

        Inputs:
        b - array.  Coefficients for the poisson regression
        X - array.  Design matrix.
        y - array.  Observed outcomes.

        Outputs:
        log_lik - float.  Negative log likelihood for the poisson regression with coefficients b.

        """


        if Xr is not None:
            n, p = X.shape
            nr, pr = Xr.shape


            #if nfixed is not None:
            bf = b[0:p]
            print(self.Ndraws)
            eta = np.tile(np.dot(X, np.transpose(bf)), (1, self.Ndraws))

            if self.draws1 is None:
                draws1 = self.prepare_halton(pr, nr, self.Ndraws, self.distribution)
                self.draws1 = draws1
            else:
                draws1 = self.draws1

            br = b[p:p+pr]

            brstd = b[p+pr:p+pr+pr]


            beta = draws1*brstd+br

            datadraws = np.tile(Xr, (self.Ndraws, 1))
            het = eta + np.sum(datadraws*beta, axis=1)
            lam = np.exp(het)
            y = np.tile(y, (1, self.Ndraws))
            log_lik = self.PoissonNegLogLikelihood(lam, y)
            print(log_lik)
            return log_lik
        else:
            print('this shouldnt happen')
            n, p = X.shape
            eta = X @ b
            lam = np.exp(eta)
            log_lik = self.PoissonNegLogLikelihood(lam, y)

            return log_lik

    def fitPoissonRegression(self, X, y, Xr = None):
        """
        Fits a poisson regression given data and outcomes.

        Inputs:
        X - array.  Design matrix
        y - array.  Observed outcomes

        Outputs:
        betas_est - array.  Coefficients which maximize the negative log-liklihood.
        """
        if Xr is not None:
            n, p = X.shape
            _r, pr = Xr.shape
            paramNum = p+2*pr
            degF = n -paramNum

            b = np.random.normal(size=paramNum)



            betas_est = minimize(self.poissonRegressionNegLogLikelihood, b, args=(X, y, Xr), method='BFGS', tol = 1e-5, options={'gtol': 1e-5})
            #betas_est = fmin_bfgs(poissonRegressionNegLogLikelihood, b, args=(X, y, Xr), maxiter=10, full_output=False, retall=False).x

            stderr = np.sqrt(np.diag(betas_est.hess_inv))

            zvalues = np.nan_to_num(betas_est.x/stderr)
            zvalues = [z if z < 1e+5 else 1e+5 for z in zvalues]
            zvalues = [z if z > -1e+5 else -1e+5 for z in zvalues]
            #pvalues = 2*(1-t.cdf(np.abs(zvalues),df =degF)) #todo check
            pvalue_alt = norm.sf(np.abs(zvalues))*2

            return -betas_est.fun, betas_est.x, stderr, pvalue_alt
        else:
            _, p = X.shape
            b = np.random.normal(size= p)
            betas_est = minimize(self.poissonRegressionNegLogLikelihood, b, args=(X, y, Xr), method='BFGS', tol = 1e-5, options={'gtol': 1e-5}).x


    def tranformer(self, transform, idc):

        if transform == 0 or 1 or None:
            tr = self._x_data.iloc[:, idc]
        elif transform == 'ln':
            tr = np.log(self._x_data.iloc[:, idc])
        elif transform == 'exp':
            tr = np.log(self._x_data.iloc[:, idc])
        elif transform == 'sqrt':
            tr = np.power(self._x_data.iloc[:, idc], 0.5)
        else: #will be a number
            tr = np.power(self._x_data.iloc[:, idc], transform)

        if np.any(np.isfinite(tr)) == False:
            tr = self._x_data.iloc[:, idc]
            
        return tr



    def makePoissonRegressionPlot(self, alpha = None, alpha_rdm = None, transform = None):
        df = pd.read_csv("Ex-16-3.csv")
        self.observations = df.shape[0]
        ones = np.ones(self.observations)
        df.insert(loc=0, column='ONE', value=ones)
        self._x_data = df
        df_tf = df

        if transform is None:
            f =1
        if transform is not None:
            for idx, t in enumerate(transform):
                df_tf.iloc[:, idx] = self.tranformer(t, idx)


        select_data = self._x_data.columns.values.tolist()

        if alpha is None:
            alpha = np.zeros(32)
            alpha[1] = 1
            alpha[2] = 1

        select_subset_fixed = [x for x, z in zip(select_data, alpha) if z == 1]
        X = df_tf[select_subset_fixed].to_numpy()
        if alpha_rdm is not None:
            select_subset_rdm = [x for x, y in zip(select_data, alpha_rdm) if y == 1]
            Xr = df_tf[select_subset_rdm].to_numpy()
        else:
            Xr = None

        y = df[self.predictor].values

        x = df.AVEPRE.values.reshape(-1,1)
        X = np.c_[np.ones(x.shape[0]),x]
        xn = np.log(df.AADT.values)
        xnn = df.MEDWIDTH.values
        Xr = np.c_[xn, xnn]


        log_lik, betas = self.fitPoissonRegression(X, y, Xr)
        print(betas)



        return None




    def primes_from_2_to(self, n):
        """Prime number from 2 to n.
        From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
        :param int n: sup bound with ``n >= 6``.
        :return: primes in 2 <= p < n.
        :rtype: list
        """
        sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
        for i in range(1, int(n ** 0.5) // 3 + 1):
            if sieve[i]:
                k = 3 * i + 1 | 1
                sieve[k * k // 3::2 * k] = False
                sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
        return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


    def van_der_corput(self, n_sample, base=2):
        """Van der Corput sequence.
        :param int n_sample: number of element of the sequence.
        :param int base: base of the sequence.
        :return: sequence of Van der Corput.
        :rtype: list (n_samples,)
        """
        sequence = []
        for i in range(n_sample):
            n_th_number, denom = 0., 1.
            while i > 0:
                i, remainder = divmod(i, base)
                denom *= base
                n_th_number += remainder / denom
            sequence.append(n_th_number)

        return sequence





    def halton(self, dim, n_sample):
        """Halton sequence.
        :param int dim: dimension
        :param int n_sample: number of samples.
        :return: sequence of Halton.
        :rtype: array_like (n_samples, n_features)
        """
        big_number = 10
        while 'Not enought primes':
            base = self.primes_from_2_to(big_number)[:dim]
            if len(base) == dim:
                break
            big_number += 1000

        # Generate a sample using a Van der Corput sequence per dimension.
        sample = [self.van_der_corput(n_sample + 1, dim) for dim in base]
        sample = np.stack(sample, axis=-1)[1:]

        return sample


    def prepare_halton(self, dim, n_sample, draws, distribution, init_coeff = None):
        n_coef = 0
        sample = self.halton(dim, n_sample*draws)
        while n_coef < len(distribution):
            if distribution[n_coef] == 'normal': #normal based
                sample[:, n_coef] = norm.ppf(sample[:, n_coef])
            elif distribution[n_coef] == 'gamma':
                sample[:, n_coef] = gamma.ppf(sample[:, n_coef])
            elif distribution[n_coef] == 'triangular':
                sample[:, n_coef] = triang.ppf(sample[:, n_coef])
            elif distribution[n_coef] == 'uniform':  # Uniform
                sample[:, n_coef] = 2 * draws[:, n_coef] - 1
            n_coef+=1

        if init_coeff is None:
            betas = np.repeat(0.1, n_coef)
        else:
            betas = init_coeff
            if len(init_coeff) != n_coef:
                raise ValueError("The size of the init_coeff must be " + n_coef)

        return sample





s = count_model(500, ['normal', 'normal'])

s.makePoissonRegressionPlot()
#Ndraws=500      # set number of draws
#dimensions=2
#N = 275
#distribution = ['normal', 'normal']
#prepare_halton(dimensions, N, Ndraws, distribution)




