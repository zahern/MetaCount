import jax.numpy as jnp
from addicty import Dict
from jaxopt import ScipyMinimize, OptaxSolver
import math
import itertools
import pandas as pd
from texttable import Texttable
import latextable
from jax import lax
import jax
from solution import ObjectiveFunction
from _device_jax_cust import device as dev
import  logging
import jax.scipy.special as sp

from jax.scipy.special import logsumexp
# define the computation boundary limits
min_comp_val = 1e-160
max_comp_val = 1e+200
log_lik_min = -1e+200
log_lik_max = 1e+200
from pareto_file import Pareto, Solution
import jax.scipy as jsp
# Setup Limits, and Batches for custom GPU code
EXP_UPPER_LIMIT = jnp.float64(jnp.log(jnp.finfo(jnp.float64).max) - 50.0)
from copy import deepcopy
from functools import  partial
import numpy as np
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_disable_jit", True)
import optax
def _unpack_tuple(x): return x if len(x) > 1 else x[0]

class JAXMLE(ObjectiveFunction):
    """JAX-based Gaussian maximum likelihood estimation objective.
    Model: y ~ Normal(mu = a*x + b, sigma = exp(log_sigma))
    """

    def __init__(self, x_data, y_data, **kwargs):
        # Call parent initializer
        super().__init__(x_data, y_data, **kwargs)

        # Child class specific attributes
        self.model_name = kwargs.get("model_name", "JAXMLE")


    def nbinom_pmf(self, y, r, p):
        """Compute NB PMF fully in JAX (safe under jit/grad)."""
        # Clip p for numerical stability (avoid log(0))
        p = jnp.clip(p, 1e-9, 1.0 - 1e-9)

        log_coeff = jsp.special.gammaln(y + r) - jsp.special.gammaln(r) - jsp.special.gammaln(y + 1)
        log_pmf = log_coeff + r * jnp.log(p) + y * jnp.log1p(-p)

        return jnp.exp(log_pmf)


    def _penalty_dispersion(self, dispersion, b_gam, eVd, y, penalty=0.0, model_nature=None):
        """JAX-compatible version of _penalty_dispersion."""

        def handle_nb(_):
            # dispersion in {1, 4}
            # Apply penalty if b_gam <= 0
            cond = b_gam >= 100000

            add_penalty = jnp.where(cond, jnp.minimum(1.0, jnp.abs(b_gam)), 0.0)
            return penalty + add_penalty, b_gam

        def handle_case2(_):
            # dispersion == 2
            over_cond = b_gam >= 1116
            under_cond = b_gam <= -1111

            penalty_over = jnp.where(over_cond, 10.0 * b_gam, 0.0)
            penalty_under = jnp.where(under_cond, jnp.abs(b_gam) * 100.0, 0.0)
            total_penalty = penalty + penalty_over + penalty_under

            # elementwise replacement
            new_b_gam = jnp.where(under_cond, 0, b_gam)
            return 0.0, b_gam

        def handle_other(_):
            # dispersion not 1,2,4 → do nothing
            return penalty, b_gam

        # --- main dispersion conditional ---
        penalty, b_gam = lax.cond(
            jnp.logical_or(dispersion == 1, dispersion == 4),
            handle_nb,
            lambda _: lax.cond(
                dispersion == 2,
                handle_case2,
                handle_other,
                operand=None
            ),
            operand=None
        )

        # Ensure penalty is non-negative
        penalty = jnp.where(penalty < 0, 0.0, penalty)

        return penalty, b_gam




    def _build_design_matrix(self, mod):
        """
        Build the design matrix `XX` by combining `X`, `Xr`, `XG`, and `XH`.

        Parameters:
            mod: Dictionary containing data and parameters.

        Returns:
            Combined design matrix `XX`.
        """
        X, Xr, XG, XH = mod.get('X'), mod.get('Xr'), mod.get('XG'), mod.get('XH')
        arrays = [a for a in (X,XG, Xr, XH) if a is not None]

        if len(arrays) == 0:
            raise ValueError("No design matrices provided.")

        # Ensure same leading dimensions
        return jnp.concatenate(arrays, axis=2) if len(arrays) > 1 else arrays[0]


    def makeRegression(self, model_nature, layout=None, *args, **kwargs):
        """Fully JAX-compatible version of makeRegression."""

        # -- Preconfig
        model_nature = deepcopy(model_nature)
        dispersion = model_nature.get('dispersion', 0)

        df_tf = jnp.array(self._x_data)
        df_test = jnp.array(self._x_data_test) if self.is_multi else None
        transformations = list(model_nature.get('transformations', []))
        pvalues = None

        # -- Apply transformations immutably
        updated_transforms = []
        for idx, t in enumerate(transformations):
            transformed, new_t = self.transformer(t, idx, df_tf[:, :, idx])
            df_tf = df_tf.at[:, :, idx].set(transformed)
            updated_transforms.append(new_t)

            if self.is_multi:
                transformed_test, _ = self.transformer(t, idx, df_test[:, :, idx])
                df_test = df_test.at[:, :, idx].set(transformed_test)

            if jnp.max(transformed) >= 7700000:
                raise ValueError("Normalization required (encountered extreme value).")

        model_nature['transformations'] = updated_transforms

        # -- Standardize design matrices
        self.define_selfs_fixed_rdm_cor(model_nature)

        indices = self.get_named_indices(self.fixed_fit)
        indices5 = self.get_named_indices(self.hetro_fit)

        # --- Heterogeneity setup
        x_h_storage = []
        x_h_storage_test = []
        transform_hetro = []

        for _, j in model_nature.get('hetro_hold', {}).items():
            indices_hetro = self.get_named_indices(j)
            extracted = [model_nature['transformations'][k] for k in indices_hetro]
            transform_hetro.append(extracted)

            X_h = df_tf[:, :, indices_hetro]
            x_h_storage.append(X_h)

            if self.is_multi:
                X_h_test = df_test[:, :, indices_hetro]
                x_h_storage_test.append(X_h_test)

        model_nature['x_h_storage'] = x_h_storage
        model_nature['transform_hetro'] = transform_hetro
        model_nature['x_h_storage_test'] = x_h_storage_test

        # --- Group dummies (if present)
        if hasattr(self, 'group_dummies'):
            indices4 = (
                jnp.repeat(self.get_named_indices(self.grouped_rpm),
                           self.group_dummies.shape[2])
                if self.grouped_rpm != [] else jnp.array([], dtype=int)
            )

            X_set = df_tf[:, :, indices4]
            XG = (
                jnp.tile(self.group_dummies, len(self.grouped_rpm)) * X_set
                if X_set.shape[2] != 0 else None
            )

            if XG is not None:
                model_nature['XG'] = XG
        else:
            XG = None

        # --- Core matrices
        X = df_tf[:, :, indices]
        XH = df_tf[:, :, indices5]
        model_nature['X'] = X
        model_nature['XH'] = XH

        if self.is_multi:
            X_test = df_test[:, :, indices]
            XH_test = df_test[:, :, indices5]
            model_nature['X_test'] = X_test
            model_nature['XH_test'] = XH_test
        else:
            X_test = None

        # --- Sanity checks
        if jnp.any(~jnp.isfinite(X)):
            raise ValueError("Invalid value in design matrix X (Inf/NaN detected).")

        # --- Random-effect matrices
        indices2 = self.get_named_indices(self.rdm_fit)
        Xr = df_tf[:, :, indices2]

        if self.rdm_cor_fit is not None:
            indices3 = self.get_named_indices(self.rdm_cor_fit)
            Xr_cor = df_tf[:, :, indices3]
            Xr = jnp.concatenate((Xr, Xr_cor), axis=2)
            self.sanity_check(Xr)

        model_nature['Xr'] = Xr

        if self.is_multi:
            Xr_test = df_test[:, :, indices2]
            if self.rdm_cor_fit is not None:
                Xr_test_cor = df_test[:, :, indices3]
                Xr_test = jnp.concatenate((Xr_test, Xr_test_cor), axis=2)
            model_nature['Xr_test'] = Xr_test
        else:
            model_nature['Xr_test'] = None

        if (Xr.ndim <= 1) or (Xr.shape[0] <= 11) or jnp.any(~jnp.isfinite(Xr)):
            raise ValueError("Invalid or insufficient Xr dimensions.")

        if Xr.size == 0:
            Xr = None
            model_nature['Xr'] = None
            model_nature['Xr_test'] = None

        # --- Response vectors
        y = jnp.array(self._y_data)
        model_nature['y'] = y
        if self.is_multi:
            model_nature['y_test'] = jnp.array(self.y_data_test)

        # --- Fit the regression model (external function)
        if model_nature.get('dispersion') is not None:
            (obj_1, log_lik, betas, stderr, pvalues,
             zvalues, is_halton, is_delete) = self.fitRegression(model_nature)

            if obj_1 is None:
                obj_1 = Solution()

            obj_1.add_layout(layout)

            model_form_name = self.check_complexity(
                self.fixed_fit, self.rdm_fit, self.rdm_cor_fit, None,
                dispersion, is_halton, model_nature
            )

            obj_1.add_names(
                self.fixed_fit.copy(),
                self.rdm_fit.copy(),
                self.rdm_cor_fit.copy(),
                model_form_name, None, pvalues)

            if not isinstance(obj_1, dict):
                raise TypeError("obj_1 must be a dict-like structure.")

            # --- significance checks
            if self.is_quanitifiable_num(obj_1[self._obj_1]) and pvalues is not None:
                if not is_delete and is_halton:
                    if obj_1[self._obj_1] <= self.best_obj_1:
                        self.pvalue_sig_value = 0.1
                    try:
                        _, self.pvalue_exceed = self.get_pvalue_info_alt(
                            pvalues, self.coeff_names, self.pvalue_sig_value,
                            dispersion, is_halton, 0, 1)
                    except Exception:
                        self.pvalue_exceed = sum(a > self.pvalue_sig_value for a in pvalues)
                else:
                    self.pvalue_exceed = sum(a > self.pvalue_sig_value for a in pvalues)

                obj_1[self._obj_1] += self.pvalue_penalty * self.pvalue_exceed
                obj_1.add_objective(pval_exceed=self.pvalue_exceed)

                if obj_1[self._obj_1] <= self.best_obj_1:
                    self.best_obj_1 = obj_1[self._obj_1]
            else:
                self.significant = 3
        else:
            obj_1 = Solution()
            self.significant = 3
            print("Dispersion not implemented yet.")

        # --- If converged, store results
        if self.is_quanitifiable_num(obj_1[self._obj_1]) and pvalues is not None:
            self.bic = obj_1.get('bic', None)
            self.pvalues = pvalues
            self.coeff_ = betas
            self.stderr = stderr
            self.zvalues = zvalues
            self.log_lik = log_lik

            if self.significant == 0 and not self.test_flag:
                self.modify(self.fixed_fit, self.rdm_fit, self.rdm_cor_fit)
                return obj_1, model_nature

            elif self.significant == 1:
                self.grab_transforms = 1

            if not jnp.isfinite(obj_1[self._obj_1]) or obj_1[self._obj_1] <= 0:
                obj_1[self._obj_1] = 10 ** 100
        else:
            print("Did not converge.")
            obj_1[self._obj_1] = 10 ** 100
            self.significant = 3
            return obj_1, model_nature

        # --- Post-fit analysis
        if not self.test_flag:
            self.modify(self.fixed_fit, self.rdm_fit, self.rdm_cor_fit)

        if self.grab_transforms:
            if is_halton and self.significant == 1:
                if self.is_multi:
                    pareto_pop = self._pareto_population
                    dominant = self.pareto_printer.check_if_dominance(pareto_pop, obj_1)
                    efficient = self.pareto_printer.is_pareto_efficient(obj_1, pareto_pop)
                    if dominant or efficient:
                        self.please_print = 1
                        self.summary_alternative(
                            long_print=1, model=dispersion,
                            solution=obj_1, save_state=1)
                    elif obj_1.get('layout') is None:
                        print("No layout detected.")
                else:
                    self.summary_alternative(long_print=1, model=dispersion, solution=obj_1)
            else:
                if self.significant == 1 and obj_1.get('layout') is not None:
                    self.summary_alternative(
                        long_print=self.non_sig_prints, model=dispersion, solution=obj_1)

        return obj_1, model_nature

    def get_elasticities(self, betas, X, return_mean=True):
        """
        Compute elasticities for Poisson or NB model with log link.

        Parameters
        ----------
        betas : array-like, shape (K,)
            Estimated coefficients.
        X : array-like, shape (N, K)
            Explanatory variables matrix (same columns as in estimation).
        return_mean : bool
            If True, return average elasticity per variable.

        Returns
        -------
        E : (N, K) array of elasticities per observation
        E_mean : (K,) array of mean elasticities (if return_mean)
        """
        betas = jnp.array(betas)
        X = jnp.array(X)
        E = X * betas  # each entry i,k: elasticity of y_i wrt x_ik

        return jnp.mean(E, axis=0) if return_mean else E



    def nbinom_pmf_batched(self, y, mu, alpha):
        """
        Negative binomial PMF P(Y=y) = Γ(y+r)/(Γ(r)Γ(y+1)) * (p^r)*(1-p)^y
        where mean = μ, var = μ + αμ², r = 1/α, p = r/(r+μ)
        """
        r = 1.0 / jnp.maximum(alpha, 1e-10)
        p = r / (r + mu)
        log_p = (
                sp.gammaln(y + r)
                - sp.gammaln(r)
                - sp.gammaln(y + 1)
                + r * jnp.log(p)
                + y * jnp.log1p(-p)
        )
        return jnp.exp(log_p)



    def prob_obs_draws_all_at_once(self, eVi, y, disp, dispersion):
        """
        JAX-compatible version of probability computation across dispersion models.
        Args:
            eVi: expected values (λ or μ)
            y: observed dependent variable
            disp: dispersion parameter (α, φ, etc.)
            dispersion: int code
                0 = Poisson
                1 = Negative Binomial
                2 = Generalized Poisson
                3 = Double Poisson
                4 = Double Negbinom (Monli)
        Returns:
            Tuple of (sum probabilities, per-observation probabilities)
        """

        # Define wrapped functions for JAX dispatcher
        def poisson_case(args):
            y, eVi, disp = args
            # Poisson PMF: P(y|λ) = λ^y * e^-λ / y!
            log_p = y * jnp.log(eVi) - eVi - jax.scipy.special.gammaln(y + 1)
            return log_p

        def nb_case(args):
            y, eVi, disp = args
            # call custom JAX-safe nbinom batched version
            return self.nbinom_pmf_batched(y, eVi, disp)

        def nb_case_pt(args):
            from jax.scipy.special import gammaln
            y, mu, alpha = args
            # log-likelihood (see Hilbe 2011)
            theta = alpha
            log_p = (
                    gammaln(y + theta)
                    - gammaln(theta)
                    - gammaln(y + 1)
                    + theta * jnp.log(theta / (theta + mu))
                    + y * jnp.log(mu / (theta + mu))
            )

            ll = (
                    gammaln(y + 1 / alpha)
                    - gammaln(1 / alpha)
                    - gammaln(y + 1)
                    + (1 / alpha) * jnp.log(1 / (1 + alpha * mu))
                    + y * jnp.log((alpha * mu) / (1 + alpha * mu))
            )
            r = 1/alpha
            p = r / (r + mu)

            log_pmf = (
                    gammaln(y + r)
                    - gammaln(r)
                    - gammaln(y + 1)
                    + r * jnp.log(p)
                    + y * jnp.log(1 - p)
            )

            logging.info('try log_p')
            return log_p

        # Build a switchable table for the dispersion cases
        proba_r_log = lax.switch(
            dispersion,  # integer 0–4
            [
                partial(poisson_case, (y, eVi, disp)),
                partial(nb_case_pt, (y, eVi, disp))

            ]
        )
        proba_r = jnp.exp(proba_r_log)

        # Combine panels appropriately
        if self.panels is None:
            # Multiply across P panels → (N, R)
            proba_panel_prod = jnp.prod(proba_r, axis=1)
        else:
            proba_panel_prod = self._prob_product_across_panels(proba_r, self.panel_info)

        # Return per-observation per-draw probabilities and full tensor
        return proba_panel_prod, proba_r, proba_r_log

    def _loglik_gradient(self, betas, Xd, y, draws=None, Xf=None, Xr=None, batch_size=None, return_gradient=False,
                         return_gradient_n=False, dispersion=0, test_set=0, return_EV=False, verbose=0, corr_list=None,
                         zi_list=None, exog_infl=None, draws_grouped=None, Xgroup=None, model_nature=None, kwarg=None,
                         **kwargs):
        """Fixed and random parameters are handled separately to speed up the estimation and the results are concatenated.
        """
        try:
            return_gradient = kwargs.get('return_gradients', return_gradient)
        except:
            print('s')

        betas = jnp.nan_to_num(betas, 1)
        if test_set == 2:  # set the offset for test data or train data
            offset_n = int(
                len(self._offsets_test) * self.test_percentage / (self.val_percentage + self.test_percentage))
            offset = self._offsets_test[offset_n:, :, :]
        elif test_set == 1:
            offset_n = int(
                len(self._offsets_test) * self.test_percentage / (self.val_percentage + self.test_percentage))
            offset = self._offsets_test[:offset_n, :, :]
        else:
            offset = self._offsets.copy()
        penalty = 0.0

        # self.round_to_closest(betas, dispersion)



        penalty = self._penalty_betas(
            betas, dispersion, penalty, float(len(y) / 10.0))
        self.n_obs = len(y)  # feeds into gradient
        if not self._no_draws(draws, draws_grouped, model_nature):
            # TODO do i shuffle the draws
            if type(Xd) == dict:
                N, Kf, P = 0, 0, 0
                for key in Xd:
                    N += Xd[key].shape[0]
                    P += Xd[key].shape[1]
                    Kf += Xd[key].shape[2]
            else:
                self.naming_for_printing(betas, 1, dispersion, model_nature=model_nature)
                N, P, Kf = Xd.shape[0], Xd.shape[1], Xd.shape[2]
            betas = jnp.array(betas)
            Bf = betas[0:Kf]  # Fixed betas

            main_disper = self.get_dispersion_paramaters(
                betas, dispersion)

            eVd = self.eXB_calc(Bf, Xd, offset, main_disper, self.linear_regression)

            if return_EV:
                return eVd

            if self.is_dispersion(dispersion):
                penalty, main_disper = self._penalty_dispersion(dispersion, main_disper, eVd, y, penalty,
                                                                model_nature)
                # b_pen = self.custom_betas_to_penalise(betas, dispersion)
                # penalty =  self.regularise_l2(betas) + self.regularise_l1(betas)
                # penalty = self.custom_penalty(betas, penalty)

                betas = betas.at[-1].set(main_disper)

            # b_pen = self.custom_betas_to_penalise(betas, dispersion)
            penalty = self.regularise_l2(betas) + self.regularise_l1(betas)
            penalty = self.custom_penalty(betas, penalty)

            if self.linear_regression:
                # LINEAR MODEL PROCESS
                mse = self._linear_logliklihood(y, eVd, main_disper)
                # mse = np.mean((y - eVd) ** 2)
                return (-mse + penalty) * self.minimize_scaler

            ### GLM PROCESS ########
            llf_main = self.loglik_obs(
                y, eVd, dispersion, main_disper, None, betas)

            llf_main = jnp.clip(llf_main, log_lik_min, log_lik_max)

            loglik = jnp.sum(llf_main)
            loglik = jnp.clip(loglik, log_lik_min, log_lik_max)
            if self.power_up_ll:
                loglik += 2 * loglik
                print('am i powering up')

            # b_pen = self.custom_betas_to_penalise(betas, dispersion)
            penalty = self.regularise_l2(betas) + self.regularise_l1(betas)
            penalty = self.custom_penalty(betas, penalty)

            #jax.debug.print("Value inside fun: {v}", v=loglik)
            ploglik = (-loglik + penalty) * self.minimize_scaler
            jax.debug.print("Value inside fun: {v}", v=ploglik)
            fx_safe = jnp.where(jnp.isfinite(ploglik), ploglik, 1e10)
            fx_safe = jnp.sum(fx_safe)
            return fx_safe
        ### ELSE WE HAVE DRAW DO THE DRAWS CODE ####
        ## ELSE DRAWS ####
        #############################################
        self.n_obs = len(y) * self.Ndraws  # todo is this problematic
        penalty += self._penalty_betas(
            betas, dispersion, penalty, float(len(y) / 10.0))

        if kwarg is not None:
            betas = kwarg['fix_the_betas'] + betas
            # Kf =0
        betas = jnp.array(betas)
        betas = dev.to_cpu(betas)  # TODO fix mepotnetially problem
        self.naming_for_printing(betas, 0, dispersion, model_nature=model_nature)
        y = dev.to_cpu(y)
        if draws is not None and draws_grouped is not None:
            draws = jnp.concatenate((draws_grouped, draws), axis=1)
            Xr = jnp.concatenate((Xgroup, Xr), axis=2)
        elif draws is None and draws_grouped is not None:
            draws = draws_grouped
            Xr = Xgroup

            # print('todo check if this breaks the model the mode')
        N = Xd.shape[0]
        R = draws.shape[2] if draws is not None else self.Ndraws
        if Xf is None:
            Kf = 0
            Xf = jnp.zeros((N, 1, 0))
            Xf = Xf.astype('float')

            Xdf = dev.to_cpu(Xf)
        else:
            Xf = Xf.astype('float')
            Kf = Xf.shape[-1]

            #
            Xdf = Xf.reshape(N, self.P, Kf)  # Data for fixed parameters
            Xdf = dev.to_cpu(Xdf)

        if Xr is None:
            Kr = 0
            Xr = jnp.zeros((N, self.P, 0))
            Xr = Xr.astype('float')
            Xdr = dev.to_cpu(Xr)
        else:
            Xr = Xr.astype('float')
            Kr = Xr.shape[2] if Xr is not None else 0
            Xdr = Xr.reshape(N, self.P, Kr)  # Data for random parameters
            Xdr = dev.to_cpu(Xdr)

        if self.rdm_cor_fit is None:
            Kr_b = 0
            Kchol = Kr
            n_coeff = self.get_param_num(dispersion)
        else:
            Kr_b = Kr - len(self.rdm_cor_fit)
            Kchol = int((len(self.rdm_cor_fit) *
                         (len(self.rdm_cor_fit) + 1)) / 2)
            # if (Kchol +Kr) != (len(betas) -Kf-Kr -self.is_dispersion(dispersion)):
            # print('I think this is fine')
            n_coeff = self.get_param_num(dispersion)
            Kf_a, Kr_a, Kr_c, Kr_b_a, Kchol_a, Kh = self.get_num_params()
            if Kchol_a != Kchol:
                print('this should not happeb but why', Kr_b, 'kr)b is')

            if Kr_b != Kr_a:
                print('self.rdm_cor_fit', self.rdm_cor_fit)
                print('grouped_fit', self.rdm_grouped_fit)
                print('self.rdm_fit', self.rdm_fit)
                print('odd, check this this should never happen')

            self.sanity_check(Xr, self.grouped_rpm)

        if kwarg is not None:
            Bf = kwarg['fix_the_betas']
            Kf = 0
        else:
            if n_coeff != len(betas):
                raise Exception(

                )
            Bf = betas[0:Kf]  # Fixed betas
        TEST_ME = False
        if not TEST_ME:
            Bf, br, brstd, Br_rema = self.extract_parameters(betas, Kf, Kr, Kchol_a, Kr_b_a)

        Vdf = dev.cust_einsum('njk,k -> nj', Xdf, Bf)  # (N, P)
        if TEST_ME:
            br = betas[Kf:Kf + Kr]

        # i have an array of betas, Kf represents the first kf of the betas array
        # now return Bf where size of bf = kf

        # size of br needs to be Kr
        # Kr
        # now extract from betas, after all the Bf
        # cakk

        # the next array is brstd

        # size of brstd needs to be
        # Kchol_a + Krb_a
        # its grabbing from the

        if TEST_ME:
            brstd = betas[Kf + Kr:Kf + Kr + Kr_b + Kchol]

        # initialises size matrix
        proba = []  # Temp batching storage

        # todo implement batchesfor batch_start, batch_end in batches_idx(batch_size, n_samples=R):
        if draws is not None:

            # Utility for random parameters

            if len(self.none_handler(self.rdm_cor_fit)) == 0:
                # Br = self._transform_rand_betas(br, np.abs(
                #     brstd), draws_)  # Get random coefficients, old method
                # TODO

                Br = self._transform_rand_betas(br,
                                                brstd, draws)  # Get random coefficients
                self.naming_for_printing(betas, dispersion=dispersion, model_nature=model_nature)
                self.Br = Br.copy()

            else:
                self.naming_for_printing(betas, dispersion=dispersion, model_nature=model_nature)
                chol_mat, corr_matt, std_devs = self._chol_mat(
                    len(self.rdm_cor_fit), br, brstd, self.rdm_cor_fit)
                self.chol_mat = chol_mat.copy()
                Br = br[None, :, None] + \
                     jnp.matmul(chol_mat[:len(br), :len(br)], draws)
                self.Br = Br.copy()
        else:
            if 'draws_hetro' in model_nature:
                self.naming_for_printing(betas, dispersion=dispersion, model_nature=model_nature)
            Br = br.reshape((N, 0, self.Ndraws))
            draws = jnp.zeros((N, 0, self.Ndraws))

        if model_nature is not None:

            if 'draws_hetro' in model_nature:
                draws_hetro = model_nature['draws_hetro_test'] if test_set else model_nature['draws_hetro']

                Xdh = model_nature['XH_test'] if test_set else model_nature['XH']
                if test_set:
                    n_split = int(len(Xdh) * (self.test_percentage / (self.test_percentage + self.val_percentage)))
                    if test_set == 2:
                        Xdh = Xdh[n_split:, :, :]
                        draws_hetro = draws_hetro[n_split:, :, :]
                    else:
                        Xdh = Xdh[:n_split, :, :]
                        draws_hetro = draws_hetro[:n_split, :, :]
                KFH = Xdh.shape[2]
                KFHs = draws_hetro.shape[1]
                betas_hetro = betas[Kf + Kr + Kr_b + Kchol:Kf + Kr + Kr_b + Kchol + KFH]
                # betas_hetro = betas[:Xdh.shape[2]+1]
                betas_hetro_sd = betas[Kf + Kr + Kr_b + Kchol + KFH:Kf + Kr + Kr_b + Kchol + KFH + KFHs]

                if KFHs >= 1:
                    # print('now what how do i split')
                    x_i_h = model_nature.get('x_h_storage_test') if test_set else model_nature.get('x_h_storage')
                    if test_set == 2:
                        x_i_h = [i[n_split:, :, :] for i in x_i_h]

                    elif test_set == 1:
                        x_i_h = [i[:n_split, :, :] for i in x_i_h]
                    ddd = 0
                    Vdh = jnp.zeros((Xdh.shape[0], Xdh.shape[1], draws_hetro.shape[2]))
                    Vdh = Vdh.astype('float')
                    for j, i in enumerate(x_i_h):
                        bbb = i.shape[2]
                        bet_h_i = betas_hetro[ddd:bbb + ddd]
                        bet_sd_i = betas_hetro_sd[j, None]
                        ddd += bbb
                        Bh = self._transform_rand_betas(bet_h_i, bet_sd_i, draws_hetro[:, j, None, :])
                        # Bh = self._transform_rand_betas(bet_h_i, bet_sd_i, draws_hetro[:, j,None, :])
                        Vdh += dev.cust_einsum("njk,nkr -> njr", i, Bh)

                else:
                    print('this will not work cause i cant trul the right drawse')
                    Bh = self._transform_rand_betas(betas_hetro, betas_hetro_sd, draws_hetro)

                    Vdh = dev.cust_einsum("njk,nkr -> njr", Xdh, Bh)
            else:
                Vdh = jnp.zeros_like(Vdf[:, :, None])
                betas_hetro = None
                betas_hetro_sd = None

        else:
            Vdh = jnp.zeros_like(Vdf[:, :, None])
            betas_hetro = None
            betas_hetro_sd = None

        Vdr = dev.cust_einsum("njk,nkr -> njr", Xdr, Br)  # (N,P,R)
        if self.linear_regression:
            ### LINEAR MODEL WAY #######
            eVd = jnp.clip(
                Vdf[:, :, None] + Vdr + Vdh + dev.jnp.array(offset), None, None)
            main_disper = self.get_dispersion_paramaters(betas, dispersion)
            penalty, main_disper = self._penalty_dispersion(
                dispersion, main_disper, eVd, y, penalty, model_nature)
            error_term = jnp.random.normal(loc=0, scale=main_disper, size=eVd.shape)
            b_pen = self.custom_betas_to_penalise(betas, dispersion)
            penalty += self.regularise_l2(b_pen) + self.regularise_l1(b_pen)
            # penalty = 0
            penalty = self.custom_penalty(betas, penalty)
            # LINEAR MODEL PROCESS
            mse = self._linear_logliklihood(y, eVd, main_disper)
            # mse = jnp.mean((y - eVd) ** 2)

            return -mse + penalty

        ##### GLM WAY #####
        eVd = jnp.exp(jnp.clip(
            Vdf[:, :, None] + Vdr + Vdh, None, EXP_UPPER_LIMIT) + jnp.array(offset))
        if dispersion == 3:
            eVd = self.lam_transform(eVd, dispersion, betas[-1])

        if self.is_dispersion(dispersion):
            if not self.no_extra_param:
                if not isinstance(betas, jnp.ndarray):
                    betas = jnp.asarray(betas)
                penalty, new_val = self._penalty_dispersion(
                    dispersion, betas[-1], eVd, y, penalty, model_nature)
                betas = betas.at[-1].set(new_val)

        ''' 
        if dev._using_gpu:
            self.lam = eVd.get()
        else:
            self.lam = eVd
        '''
        if return_EV:
            return eVd.mean(axis=2)
            #    return eVd.mean(axis=(1, 2))

        Vdr, Br = None, None  # Release memory

        """"" Old way keep just in cass
        proba_n = jnp.zeros((N, R), dtype=jnp.float64) #setup storage for invidvidual probs



        for r in range(batch_start, batch_end):


            eVi = eVd[:, :, r] #was dev.tocpu
            proba_r = self.prob_obs_draws(eVi, y, betas[-1], dispersion) #TODO make sure betas[-1] handles

            proba_n[:, r] = proba_r


        proba_ = proba_n.sum(axis =1)

        """""
        main_disper = self.get_dispersion_paramaters(betas, dispersion)
        eps = 1e-12  # protect against log(0)
        # print(betas_last)
        proba_per_draw, proba_full, proba_log= self.prob_obs_draws_all_at_once(
            eVd, jnp.atleast_3d(y), main_disper, dispersion)
        proba_safe = jnp.clip(proba_per_draw, eps, 1.0)

        # self._prob_product_against_panels()
#        jax.debug.print(f"⚠️ Non‑finite loglik detected at step: {betas}", x=proba_safe)
        # print(top_stats)
        loglik_i_d= jnp.log(proba_safe)
        loglik_i = jnp.mean(loglik_i_d, axis = 1)
        #loglik_i = jsp.special.logsumexp(jnp.log(proba_safe), axis=1) - jnp.log(proba_safe.shape[1])
        loglik = jnp.sum(loglik_i)
        jax.debug.print("loglik = {}", loglik)
        #if not jnp.isfinite(loglik):
      #  jax.debug.print(f"⚠️ Non‑finite loglik detected at step: {betas}", x=loglik)
        loglik = jnp.clip(loglik, log_lik_min, log_lik_max)
        if self.power_up_ll:
            penalty += self.regularise_l2(betas)

        penalty += self.regularise_l2(betas) + self.regularise_l1(betas)
        a= (-loglik + penalty) * self.minimize_scaler
        a = jnp.nan_to_num(a, 10000000)
        return jnp.sum(a)

    def sanity_check_nb(self):
        y = jnp.arange(0, 6)
        lam = 2.0
        gamma = 0.5

        # Your manual version
        r = 1 / gamma
        p = gamma / (gamma + lam)

        nbinom_param_form = (sp.gamma(y + r) / (sp.gamma(r) * sp.gamma(y + 1))) * (p ** r) * ((1 - p) ** y)

        # Your mean-dispersion form
        def nbinom_pmf_batched(y, mu, alpha):
            r = 1 / alpha
            p = r / (r + mu)
            log_p = (
                    sp.gammaln(y + r)
                    - sp.gammaln(r)
                    - sp.gammaln(y + 1)
                    + r * jnp.log(p)
                    + y * jnp.log1p(-p)
            )
            return jnp.exp(log_p)

        nbinom_batch_form = nbinom_pmf_batched(y, lam, gamma)

        print(jnp.allclose(nbinom_param_form, nbinom_batch_form))

    def _chol_mat(self, correlationLength, br, Br_w, correlation):
        """
        JAX version of the Cholesky-covariance constructor
        for random parameters, compatible with JIT.

        Args:
            correlationLength: length (# of random effects)
            br: vector of non-correlated random param stds
            Br_w: concatenated parameter vector (std + corr terms)
            correlation: list of variables that are correlated
        Returns:
            chol_mat: lower-triangular Cholesky matrix
        """

        # Combine all variable names relevant to random parameters
        varnames = (
                self.none_handler(self.grouped_rpm)
                + self.none_handler(self.rdm_fit)
                + self.none_handler(self.rdm_cor_fit)
        )

        Kchol = int((len(correlation) * (len(correlation) + 1)) / 2)

        # Extract correlation (chol) and uncorrelated parts
        chol = Br_w[-Kchol:]
        br_w = Br_w[:-Kchol]

        chol_mat_temp = jnp.zeros((len(br), len(br)))

        rv_count = 0
        rv_count_all = 0
        corr_indices = []
        chol_count = 0

        # NOTE: small loop — fine outside jit; can be replaced with vmap if needed
        for ii, var in enumerate(varnames):
            is_correlated = var in self.none_handler(self.rdm_cor_fit)

            rv_val = chol[chol_count] if is_correlated else br_w[rv_count]
            chol_mat_temp = chol_mat_temp.at[rv_count_all, rv_count_all].set(rv_val)

            rv_count_all += 1
            if is_correlated:
                chol_count += 1
            else:
                rv_count += 1

            if var in self.rdm_cor_fit:
                corr_indices.append(rv_count_all - 1)

        # Determine correlation pairs
        if self.rdm_cor_fit is None:
            corr_pairs = list(itertools.combinations(self.Kr, 2))
        else:
            corr_pairs = list(itertools.combinations(corr_indices, 2))

        # Fill off-diagonal lower triangle with correlation coefficients
        for corr_pair in corr_pairs:
            chol_mat_temp = chol_mat_temp.at[corr_pair[::-1]].set(chol[chol_count])
            chol_count += 1

        # Compute omega = L * L^T
        omega = chol_mat_temp @ chol_mat_temp.T

        # Compute correlation matrix and standard deviations
        standard_devs = jnp.sqrt(jnp.abs(jnp.diag(omega)))
        outer_prod = jnp.outer(standard_devs, standard_devs)
        corr_mat = omega / outer_prod

        return chol_mat_temp, corr_mat, standard_devs

    def loglik_obs(self, y, eVd, dispersion, b_gam, l_pam=None, betas=None):
        """
        JAX-compatible version of loglik_obs.
        Works under jit/grad/vmap, avoids NumPy ops and Python control flow.
        """

        min_comp_val, max_comp_val = 1e-320, 1e12

        # --- handle dictionary input ---
        if isinstance(y, dict):
            keys = list(y.keys())
            y = jnp.concatenate([jnp.array(y[k])[:, None] for k in keys])
            weights = jnp.concatenate(
                [jnp.ones((len(y[k]), 1)) * self.weights[i] for i, k in enumerate(keys)]
            )
        else:
            y = jnp.atleast_2d(y)
            weights = jnp.ones_like(y)



        # --- select distribution type ---

        def case_poisson(_):
            # Poisson PMF in JAX
            from jax import scipy as jsp
            probaa_r = jnp.exp(
                y * jnp.log(eVd) - eVd - jsp.special.gammaln(y + 1)
            )
            proba_r  = jsp.stats.poisson.pmf(y, eVd)
            return proba_r

        def case_nb(_):
            return self._nonlog_nbin(y, eVd, b_gam)



        def case_not_implemented(_):
            raise ValueError("Unsupported dispersion model")

        # branch for dispersion numerics
        proba_r = lax.switch(
            dispersion,
            [case_poisson, case_nb],
            None
        )

        # panel combination — assumed already JAX-safe
        if self.panels is not None:
            proba_r = self._prob_product_across_panels(proba_r, self.panel_info)

        # --- Numeric stability ---
        proba_r = jnp.clip(proba_r, 0.0000000000000001, max_comp_val)
        # if clipped how to add peanalty to the below otherwise theres no issue
        loglik = jnp.log(proba_r)

        return loglik


    def regularise_l2(self, betas, backwards=False):
        l2_term = jnp.sum(jnp.square(betas))
        sign = -1 if backwards else 1
        return self.reg_penalty * sign * l2_term


    def regularise_l1(self, betas, backwards=False):
        l1_term = jnp.sum(jnp.abs(betas))
        sign = -1 if backwards else 1
        return self.reg_penalty * sign * l1_term


    def eXB_calc(self, params_main, Xd, offset, dispersion, linear=False):
        """JAX-safe version of eXB_calc."""

        def compute_eta_with_dispersion(_):
            sigma = dispersion
            eta = jnp.dot(Xd, params_main)[:, :, None]
            epsilon = jax.random.normal(
                jax.random.PRNGKey(0),  # In real code, pass key as param
                shape=eta.shape
            ) * sigma
            return eta + epsilon

        def compute_eta_no_dispersion(_):
            return jnp.dot(Xd, params_main)[:, :, None]

        # --- handle dispersion flag safely ---
        eta = lax.cond(
            jnp.any(dispersion != 0),
            compute_eta_with_dispersion,
            compute_eta_no_dispersion,
            operand=None,
        )

        eta = eta.astype('float64')

        # --- handle linear flag safely ---
        def return_linear(_):
            return eta

        def return_exp(_):
            eVd = jnp.exp(jnp.clip(eta, None, EXP_UPPER_LIMIT) + jnp.array(offset))
            return eVd

        result = lax.cond(
            linear,
            return_linear,
            return_exp,
            operand=None
        )

        return result


    def _minimize(self, loglik_fn, x, args, method, tol, options, bounds=None, hess_calc=None):
        # method = 'BFGS'
        if not isinstance(x, jnp.ndarray):
            x = jnp.asarray(x)
        def objective(params):
            return loglik_fn(params, *args)
        #the args need to be applied onto the loglik_fun not the function..

        return jax_minimize(objective, x, method='BFGS', tol=tol, options=options)


    def _nonlog_nbin(self, y, lam, gamma, Q=0):
        """generates non_loged probabilities
        Args:
            y (_type_): _description_
            lam (_type_): _description_
            gamma (_type_): _description_
            Q (int, optional): _description_. Defaults to 0.
        Returns:
            _type_: _description_
        """
        mu = lam
        theta = gamma
        log_p = (
                jsp.special.gammaln(y + theta)
                - jsp.special.gammaln(theta)
                - jsp.special.gammaln(y + 1)
                + theta * jnp.log(theta / (theta + mu))
                + y * jnp.log(mu / (theta + mu))
        )
        return jnp.exp(log_p)


        # if gamma <= 0.01: #min defined value for stable nb
        #  gamma = 0.01

        # g = stats.gamma.rvs(gamma, scale = lam/gamma, size = 1.0 / gamma * lam ** Q )

        # gg = stats.poisson.rvs(g)

        endog = y
        mu = lam
        ''''
        mu = lam*np.exp(gamma) #TODO check that this does not need to be multiplied
        alpha = np.exp(gamma)

        '''
        alpha = gamma
        size = 1.0 / alpha * mu ** Q
        r = 1 / gamma
        p = gamma / (gamma + lam)
        pmf = self.nbinom_pmf(y, r, p)
        # pmf = self.nbinom_pmf(y, lam, gamma)
        # p = lam/(lam+r)
        prob = size / (size + mu)

        # binom_coeff = math.comb(int(y +r - 1), y)
        # ff = binom_coeff * ((1 - p) ** r) * (p ** y)
        '''test'''

        '''
        size = 1 / np.exp(gamma) * mu ** 0
        prob = size / (size + mu)
        coeff = (gammaln(size + y) - gammaln(y + 1) -
             gammaln(size)) 
        llf = coeff + size * np.log(prob) + y * np.log(1 - prob)
        '''

        try:
            # print(np.shape(y),np.shape(size), np.shape(prob))
            # gg2 = self.negbinom_pmf(alpha_size, size/(size+mu), y)
            # import time
            # start_time = time.time()

            # Measure time for negbinom_pmf
            # start_time = time.time()
            # for _ in range(10000):

            # end_time = time.time()
            # print("Custom functieon time:", end_time - start_time)
            # start_time = time.time()
            # for _ in range(10000):
            '''
            gg = np.exp(
                gammaln(y + alpha) - gammaln(y + 1) - gammaln(alpha) + y * np.log(mu) + alpha * np.log(alpha) - (
                        y + alpha) * np.log(mu + alpha))
            gg[np.isnan(gg)] = 1
            '''
            gg_alt = self.nbinom_pmf(y, alpha, prob)
            # gg_alt_2 = (gammaln(size + y) - gammaln(y + 1) -
            # gammaln(size)) + size * np.log(prob) + y * np.log(1 - prob)
            # print('check theses')
            # gg = nbinom.pmf(y ,alpha, prob)
            # end_time = time.time()
            # print("Custom functieon time:", end_time - start_time)

        except Exception as e:
            print("Neg Binom error.")
        return pmf

    def _build_initial_params(self, num_coefficients, dispersion, XX, y):
        import statsmodels.api as sm

        """
        Build the initial parameter array for optimization.
        This runs in pure Python/NumPy, not inside jit.
        """

        # --- Run ordinary statsmodels GLM outside JAX ---
        y_np = np.asarray(y.squeeze())
        X_np = np.asarray(XX.squeeze())

        # Pick model type based on dispersion
        if dispersion == 0:
            fam = sm.families.Poisson()
        else:
            fam = sm.families.NegativeBinomial()

        '''
        model = sm.GLM(y_np, X_np, family=fam)
        result = model.fit()
        initial_params = result.params.copy()
        '''
        X_np = np.asarray(X_np, dtype=float)

        # Drop columns with any NaN or Inf
        bad_cols = np.where(~np.isfinite(X_np).all(axis=0))[0]
        if len(bad_cols) > 0:
            X_np = np.delete(X_np, bad_cols, axis=1)

        # Drop constant columns
        const_cols = np.where(np.ptp(X_np, axis=0) == 0)[0]
        if len(const_cols) > 0:
            X_np = np.delete(X_np, const_cols, axis=1)

        # CLEAN y
        y_np = np.asarray(y_np, dtype=float)
        if np.any(~np.isfinite(y_np)):
            y_np = np.nan_to_num(y_np, nan=0.0, posinf=1e6, neginf=0.0)

        # GUARANTEE non-negative for Poisson/NB
        y_np = np.clip(y_np, 0, None)

        # NOW FIT
        try:
            model = sm.GLM(y_np, X_np, offset = self._offsets.ravel(), family=fam)
            result = model.fit()
            initial_params = result.params.copy()

        except Exception as e:
            print("GLM failed:", str(e))
            # Safe fallback: return zeros or small random initials
            initial_params = np.zeros(X_np.shape[1])

        # --- Extend for dispersion ---
        if len(initial_params) < num_coefficients:
            pearson_residuals = result.resid_pearson
            alpha = (pearson_residuals ** 2).sum() / result.df_resid
            if alpha > 0:
                alpha = np.log(alpha)
            initial_params = np.concatenate(
                [initial_params.ravel(), np.array([alpha])]
            )

        # --- If still short, pad with small random jitter ---
        if len(initial_params) < num_coefficients:
            new_elements = np.random.uniform(
                -0.000001, 0.000000003, size=num_coefficients - len(initial_params)
            )
            # Insert new elements before the last element(s) depending on dispersion
            insertion_index = -dispersion if dispersion > 0 else len(initial_params)
            initial_params = np.insert(initial_params, insertion_index, new_elements)

        # --- Optional: zero out some block ---
        parma_sum = sum(self.get_num_params()[:2])
        stop_index = -dispersion if dispersion > 0 else len(initial_params)
        initial_params[parma_sum:stop_index] = 0.0001

        # --- Handle dispersion param ---
        if dispersion == 1:
            calculated_dispersion = self.poisson_mean_get_dispersion(
                initial_params[:-1], XX, y
            )
            initial_params[-1] = calculated_dispersion

        # Convert to JAX array for later optimization
        return jnp.asarray(initial_params)


    def fitRegression(self, mod, dispersion=0, maxiter=2000, batch_size=None, num_hess=False, **kwargs):
        """
        Fits a Poisson regression, NB regression (dispersion=1), or GP regression (dispersion=2).

        Parameters:
            mod: Dictionary containing data and parameters.
            dispersion: 0 for Poisson, 1 for NB, 2 for GP.
            maxiter: Maximum number of optimization iterations.
            batch_size: Batch size for certain methods (if applicable).
            num_hess: Whether to compute the numerical Hessian.

        Returns:
            obj_1, log_lik, betas, stderr, pvalues, zvalues, is_halton, is_delete
        """

        dispersion = mod.get('dispersion', dispersion)
        # Preprocessing
        tol = {'ftol': 1e-10, 'gtol': 1e-6, 'xtol': 1e-7}
        y, X, Xr, XG, XH = mod.get('y'), mod.get('X'), mod.get('Xr'), mod.get('XG'), mod.get('XH')

        # Validate input data
        if y is None or X is None:
            raise ValueError("Both `y` and `X` must be provided in the `mod` dictionary.")

        # Build the design matrix `XX` and test matrix `XX_test` if applicable
        XX = self._build_design_matrix(mod)
        XX_test = self._build_test_matrix(mod) if self.is_multi else None

        # Determine the number of coefficients
        num_coefficients = self._calculate_num_coefficients(mod, dispersion)

        # Build initial parameters and bounds
        initial_params = self._build_initial_params(num_coefficients, dispersion, XX, y)
        bounds = self._set_bounds(initial_params, dispersion)

        # Run optimization
        # initial_params = [2.82, 1.11]
        optimization_result = self._run_optimization(
            XX, y, dispersion, initial_params, bounds, tol, mod, maxiter=maxiter
        )

        # Post-process results
        log_lik, aic, bic, stderr, zvalues, pvalues, in_sample_mae, out_sample_mae, out_sample_val = self._postprocess_results(
            optimization_result, XX, XX_test, y, mod.get('y_test'), dispersion, mod
        )

        # Extract other outputs
        betas = optimization_result['x'] if optimization_result is not None else None
        is_halton = Xr is not None and Xr.size > 0  # Halton draws used if `Xr` is not empty

        # Determine `is_delete`
        is_delete = not (
                optimization_result is not None
                and 'fun' in optimization_result
                and not math.isnan(optimization_result['fun'])
                and not math.isinf(optimization_result['fun'])
        )

        betas_est = optimization_result

        # Post-fit metrics
        log_ll, aic, bic, stderr, zvalues, pvalue_alt, other_measures = self._post_fit_ll_aic_bic(
            betas_est, simple_fit=False, is_dispersion=dispersion
        )

        # Number of parameters
        paramNum = len(betas_est['x'])

        # Naming for printing (optional, for formatting or debugging purposes)
        self.convergence = not is_delete
        self.naming_for_printing(betas_est['x'], 0, dispersion, model_nature=mod)

        # Add metrics to solution object
        sol = Solution()  # Assuming Solution is the appropriate class to store results

        sol.add_objective(
            bic=bic,
            aic=aic,
            loglik=log_ll,
            TRAIN=in_sample_mae,
            TEST=out_sample_mae,
            VAL=out_sample_val,
            num_parm=paramNum,
            GOF=other_measures
        )



        return (
            sol,  # obj_1
            log_lik,
            betas,
            stderr,
            pvalues,
            zvalues,
            is_halton,
            is_delete
        )

    def _run_optimization(self, XX, y, dispersion, initial_params, bounds, tol, mod, maxiter):
        """
        Run the optimization process with draws logic and update the Solution object.

        Parameters:
            XX: Design matrix.
            y: Observed outcomes.
            dispersion: Dispersion parameter (0=Poisson, 1=NB, 2=GP).
            initial_params: Initial parameter array.
            bounds: List of bounds for each parameter.
            tol: Tolerance for the optimization process (dictionary with ftol and gtol).
            mod: Dictionary containing additional data.

        Returns:
            Solution object with updated objectives.
        """
        # Extract relevant data
        X, Xr, XG, XH = mod.get('X'), mod.get('Xr'), mod.get('XG'), mod.get('XH')
        distribution = mod.get('dist_fit')

        # Prepare draws
        draws = self._prepare_draws(Xr, distribution)
        draws_grouped = self._prepare_grouped_draws(XG, mod) if XG is not None else None
        mod = self._prepare_hetro(mod)
        # Optimization method and options
        method = self.method_ll if bounds is None else 'L-BFGS-B'

        # method = 'Nelder-Mead-BFGS'

        options = {'gtol': tol['gtol'], 'ftol': tol['ftol'], 'maxiter': maxiter}
        args = (
            X, y, draws, X, Xr, self.batch_size, self.grad_yes, self.hess_yes, dispersion, 0, False, 0,
            self.rdm_cor_fit, None, None, draws_grouped, XG, mod
        )
        # Run optimization
        optimization_result = self._minimize(
            self._loglik_gradient,
            initial_params,
            args=(
                X, y, draws, X, Xr, self.batch_size, self.grad_yes, self.hess_yes, dispersion, 0, False, 0,
                self.rdm_cor_fit, None, None, draws_grouped, XG, mod
            ),
            method=method,
            bounds=bounds,
            tol=tol.get('ftol', 1e-6),  # Use 'ftol' as the default tolerance
            options=options
        )

        # print(result.summary())

        # i want to compare this to stats model.s

        if optimization_result.message == 'NaN result encountered.':
            optimization_result = self._minimize(self._loglik_gradient,
                                                 initial_params,
                                                 args=(
                                                     X, y, draws, X, Xr, self.batch_size, self.grad_yes, self.hess_yes,
                                                     dispersion, 0, False, 0,
                                                     self.rdm_cor_fit, None, None, draws_grouped, XG, mod
                                                 ),
                                                 method='Nelder-Mead-BFGS',
                                                 bounds=bounds,
                                                 tol=tol.get('ftol', 1e-4),  # Use 'ftol' as the default tolerance
                                                 options=options
                                                 )

        if self.run_numerical_hessian:
            std_errors = self.bootstrap_std_dev(
                initial_params=optimization_result.x,
                XX=XX,
                y=y,
                dispersion=dispersion,
                bounds=bounds,
                tol=tol,
                mod=mod,
                n_bootstraps=5
            )
            self.stderr = std_errors

        # Run the bootstrap to calculate standard errors
        if self.run_bootstrap:
            std_errors = self.bootstrap_std_dev(
                initial_params=optimization_result.x,
                XX=XX,
                y=y,
                dispersion=dispersion,
                bounds=bounds,
                tol=tol,
                mod=mod,
                n_bootstraps=6
            )
            self.stderr = std_errors

        return optimization_result

    def convert_coefficients(self, params, dispersion):
        if isinstance(params, list):
            # convert all elements to float if possible
            params = [float(p) if not isinstance(p, (float, int)) else p for p in params]
        params = jnp.asarray(params)
        num_params = self.get_num_params()
        skip_count = sum(num_params[:2])
        remain_params = num_params[2:]
        params = params.at[skip_count:skip_count + remain_params[1]].set(
            jnp.abs(params[skip_count:skip_count + remain_params[1]])
        )
        return params

    def summary_alternative(self, long_print=0, model=0, solution=None, save_state=1):
        fmt = "{:19} {:13} {:13.10f} {:13.10f}{:13.10f} {:13.3g} {:3}"
        coeff_name_str_length = 19

        if self.coeff_ is None:
            print('The current model has not been estimated yet')
            return
        if self.coeff_names is None:
            raise Exception

        if self.pvalues is None:
            raise Exception

        if isinstance(self.pvalues, str):
            raise Exception



        if 'nb' in self.coeff_names and self.no_extra_param:
            self.pvalues = jnp.append(self.pvalues, 0)

        if self.please_print or save_state:

            if self.convergence is not None:
                print("-" * 80)

                print('Log-Likelihood: ', self.log_lik)
                print("-" * 80)

                if self.bic is not None:
                    print(f"{self._obj_1}: {self.round_with_padding(self.bic, 2)}")
                    print("-" * 80)

                if solution is not None:
                    if self.is_multi:
                        print(f"{self._obj_2}: {self.round_with_padding(solution[self._obj_2], 2)}")

            self.pvalues = [self.round_with_padding(
                x, 2) for x in self.pvalues]
            signif_list = self.pvalue_asterix_add(self.pvalues)
            if model == 1:
                # raise to the exponential

                    # transform if negative
                self.coeff_ = self.coeff_.at[-1].set(jnp.exp(self.coeff_[-1]))
                    # self.coeff_[-1] = 0.0000001

                if self.no_extra_param:
                    print('adding in nb parma')
                    exit()
                    self.coeff_ = jnp.append(self.coeff_, self.nb_parma)
                    self.stderr = jnp.append(self.stderr, 0.00001)
                    self.zvalues = jnp.append(self.zvalues, 50)

                # elif self.coeff_[-1] < 0.25:
                # print(self.coeff_[-1], 'Warning Check Dispersion')
                # print(f'dispession is para,aters {np.exp(self.coeff_[-1])}')
                # self.coeff_[-1] = np.exp(self.coeff_[-1])  # min possible value for negbinom

            self.coeff_ = self.convert_coefficients(self.coeff_, model)
            self.coeff_ = [self.round_with_padding(x, self.rounding_point) for x in self.coeff_]
            self.stderr = [self.round_with_padding(x, self.rounding_point) for x in self.stderr]
            self.zvalues = [self.round_with_padding(
                x, 2) for x in self.zvalues]

            table = Texttable()

            def pad_or_truncate(lst, target):
                lst = list(lst)
                diff = target - len(lst)
                if diff > 0:
                    lst.extend([""] * diff)
                elif diff < 0:
                    lst = lst[:target]
                return lst
            if long_print:
                target_len = len(self.coeff_names)



                self.print_transform = pad_or_truncate(self.print_transform, target_len)
                self.coeff_ = pad_or_truncate(self.coeff_, target_len)
                self.stderr = pad_or_truncate(self.stderr, target_len)
                self.zvalues = pad_or_truncate(self.zvalues, target_len)
                signif_list = pad_or_truncate(signif_list, target_len)

                latex_dict = {
                    'Effect': self.coeff_names,
                    r'$\tau$': self.print_transform,
                    'Coeff': self.coeff_,
                    'Std. Err': self.stderr,
                    'z-values': self.zvalues,
                    'Prob |z|>Z': signif_list
                }

                df = pd.DataFrame.from_dict(latex_dict)
                ncols = len(df.columns)
                table.set_cols_align(['l'] + ['c'] * (ncols - 1))
                table.set_cols_dtype(['t'] * ncols)

            else:
                if model in (1, 2):
                    self.coeff_[-1] = jnp.log(self.coeff_[-1])

                target_len = len(self.coeff_names)
                self.print_transform = pad_or_truncate(self.print_transform, target_len)
                signif_list = pad_or_truncate(signif_list, target_len)
                self.coeff_ = pad_or_truncate(self.coeff_, target_len)

                latex_dict = {
                    'Effect': self.coeff_names,
                    'Transformation': self.print_transform,
                    'Coefficient': self.coeff_,
                    'Prob |z|>Z': signif_list
                }

                df = pd.DataFrame.from_dict(latex_dict)
                ncols = len(df.columns)
                table.set_cols_align(['l'] + ['c'] * (ncols - 1))
                table.set_cols_dtype(['t'] * ncols)

            # Add the data safely
            rows = [df.columns.tolist()] + df.fillna("").values.tolist()
            table.add_rows(rows)

            if self.please_print or save_state:
                print(table.draw())
                # ---- Print stored Elasticity Summary if available ----


            if model is not None:
                caption_parts = []
                if self.algorithm is not None:
                    caption_parts.append(
                        f"{self._model_type_codes[model]} model found through the {self.algorithm} algorithm.")

                if self.bic is not None:
                    caption_parts.append(f"{self._obj_1}: {self.round_with_padding(self.bic, 2)}")

                if self.log_lik is not None:
                    caption_parts.append(f"Log-Likelihood: {self.round_with_padding(self.log_lik, 2)}")

                if solution is not None:
                    if self.is_multi:
                        caption_parts.append(f"{self._obj_2}: {self.round_with_padding(solution[self._obj_2], 2)}")

                caption = " ".join(caption_parts)
                # print(latextable.draw_latex(table, caption=caption, caption_above = True))
                if solution is None:
                    file_name = self.instance_name + "/sln" + \
                                "_with_BIC_" + str(self.bic) + ".tex"
                else:
                    file_name = self.instance_name + "/sln" + \
                                str(solution['sol_num']) + \
                                "_with_BIC_" + str(self.bic) + ".tex"

                if save_state:
                    # print(file_name)
                    if self.save_state:
                        self.save_to_file(latextable.draw_latex(
                            table, caption=caption, caption_above=True), file_name)

    def _get_obj1(self):

        arr = self._obj_2
        if hasattr(arr, "block_until_ready"):  # JAX array
            arr.block_until_ready()
        return str(arr)

    def _get_obj2(self):
        """Return obj_2 as a NumPy array or scalar."""

        arr = self._obj_2
        if hasattr(arr, "block_until_ready"):  # JAX array
            arr.block_until_ready()
        return str(arr)

    def update_gbl_best(self, obj_1):
        current_val = jnp.nan_to_num(
            jnp.asarray(obj_1[self._obj_1], dtype=jnp.float32),
            nan=jnp.inf, posinf=jnp.inf, neginf=-jnp.inf
        )
        gbl_best_val = jnp.nan_to_num(
            jnp.asarray(self.gbl_best, dtype=jnp.float32),
            nan=jnp.inf, posinf=jnp.inf, neginf=-jnp.inf
        )

        improved = current_val < gbl_best_val

        # Keep dtype consistent for 'significant'
        sig_dtype = jnp.result_type(self.significant)
        one = jnp.array(1, dtype=sig_dtype)

        new_gbl_best, new_significant = lax.cond(
            improved,
            lambda _: (current_val, one),
            lambda _: (gbl_best_val, self.significant),
            operand=None
        )
        # ✅ Convert results to NumPy before storing
        if hasattr(new_gbl_best, "block_until_ready"):
            new_gbl_best.block_until_ready()
        if hasattr(new_significant, "block_until_ready"):
            new_significant.block_until_ready()
        # Update attributes (allowed outside JIT)
        self.gbl_best = float(new_gbl_best)
        self.significant = int(new_significant)



    def _penalty_betas(self, betas, dispersion, penalty, penalty_ap=100.0):
        penalty_val = 0.1
        penalty_val_max = 5000

        # print('change_later')
        if dispersion != 0:
            a = betas[:-1]
        else:
            a = betas

        # for i in a:
        #  if abs(i) < penalty_val:
        #    penalty += np.nan_to_num(1/np.max((0.01, abs(i))), nan=10000)
        for i in a:
            penalty+=jnp.where(jnp.abs(i) > penalty_val_max, penalty_val_max, i)


        #if abs(i) < penalty_val:
        #    penalty += 5

        # penalty = 0
        return penalty


    def get_dispersion_paramaters(self, betas, dispersion):
        """JAX-safe version that returns a numeric dispersion value."""
        val = jnp.exp(betas[-1])
        clipped = jnp.clip(val, 0.001, 10)

        # Handle dispersion values:
        # - If dispersion == 0 → return 0.0 (numeric placeholder)
        # - If dispersion == 1 → return clipped dispersion
        # - else → return 0.0 (safe default)
        return lax.switch(
            dispersion,
            [
                lambda: jnp.array(0.0),  # case 0
                lambda: clipped  # case 1
            ],
        )

    def round_with_padding(self, value, round_digits):
        # If it's a string, return unchanged
        if isinstance(value, str):
            return value

        # Convert safely to float (covers np.float64, jnp.float64, ints, etc.)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)

        # Use plain NumPy round — this is not part of your model, just for printing
        rounded_value = np.round(numeric_value, round_digits)

        # Format with fixed decimal places (adds trailing zeros)
        return format(rounded_value, f".{round_digits}f")








def jax_minimize(fun, x0, method="BFGS", tol=1e-8, options=None, callback=None,
                 has_aux=False, value_and_grad=False, dtype=jnp.float64,
                 implicit_diff_solve=True, maxiter=25000):
    """
    Drop-in SciPy-like minimize wrapper using JAXopt ScipyMinimize.

    Args:
        fun: callable `fun(x, *args, **kwargs)` to minimize.
        x0: initial parameters (JAX array or PyTree).
        method: solver name, same as scipy.optimize.minimize.
        tol: tolerance for termination.
        options: optional dictionary of solver options.
        callback: a callable `(xk)` run after each iteration.
        has_aux: whether `fun` returns auxiliary outputs `(loss, aux)`.
        value_and_grad: whether to pass a precomputed (value, grad) function.
        jit: whether to JIT the computations.
        dtype: dtype for NumPy arrays (typically float64).
        implicit_diff_solve: linear system solver for implicit differentiation.
        maxiter: maximum number of iterations.

    Returns:
        A result-like dict similar to SciPy:
        {
            'x': final parameters (PyTree or array),
            'fun': final objective value,
            'jac': gradient at solution,
            'hess_inv': approximate inverse Hessian,
            'nit': number of iterations,
            'nfev': number of function evaluations,
            'njev': number of jacobian evaluations,
            'status': SciPy-like status code,
            'success': bool,
            'message': textual status message,
        }
    """
    logging.info('turn off options..')

    options = options or {}

    from jaxopt import implicit_diff, linear_solve
    method = 'addddam'
    
    # 1️⃣ handle Optax-based methods
    if method.lower() in {"lion", "adam", "sgd", "rmsprop", "adagrad", "adamw"}:
        logging.info(f"Using OptaxSolver with {method.upper()}")
        learning_rate = options.get("learning_rate", 1e-3)
        clip_value = options.get("clip_value", 0.3)
        weight_decay = options.get("weight_decay", 1e-4)





        # Optional learning rate schedule for stability
        scheduler = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=500,
            decay_rate=0.99,
            end_value=learning_rate * 0.1,
        )

        # Base optimizer definitions
        base_opts = {
            "lion": optax.lion(learning_rate=scheduler),
            "adam": optax.adam(learning_rate=1e-3),
            "adamw": optax.adamw(learning_rate=scheduler, weight_decay=weight_decay),
            "sgd": optax.sgd(learning_rate=scheduler),
            "rmsprop": optax.rmsprop(learning_rate=scheduler),
            "adagrad": optax.adagrad(learning_rate=scheduler),
        }

        # Select the optimizer method
        base_opt = base_opts[method.lower()]

        # Combine gradient clipping + optimizer
        optax_opt = optax.chain(
            optax.clip_by_global_norm(clip_value),
            base_opt,
        )

        # Create solver
        solver = OptaxSolver(
            opt=optax_opt,
            fun=fun,
            maxiter=maxiter,
            has_aux=has_aux,
            #implicit_diff_solve=linear_solve.solve_normal_cg,
            tol = tol
        )

        state = solver.init_state(x0)
        #prev_state

        for i in range(maxiter):
            x0, state = solver.update(x0, state)
            print(state.value)
            if callback:
                callback(x0)

        H = jax.hessian(fun)(x0)
        H_inv = jnp.linalg.inv(H + 1e-8 * jnp.eye(H.shape[0]))

        res = {
            "x": x0,
            "fun": state.value,
            "nit": int(state.iter_num),
            "hess_inv": H_inv,
            "status": 0,
            "success": True,
            "message": f"Optimization using {method.upper()} terminated after {state.iter_num} iterations"
        }
        print('cool')
        print(state.value)
        return Dict(res)
    from jaxopt import ScipyMinimize
    solver = ScipyMinimize(
        fun=fun,
        method='BFGS'
    )

    opt_step = solver.run(x0)

    params, info = opt_step.params, opt_step.state

    # mimic SciPy result interface
    res = {
        "x": params,
        "fun": info.fun_val,
        "hess_inv": info.hess_inv,
        "nit": int(info.iter_num),
        "nfev": int(info.num_fun_eval),
        "njev": int(info.num_jac_eval),
        "status": int(info.status),
        "success": bool(info.success),
    }

    # add convenience message
    msg = "Optimization terminated successfully." if res["success"] else "Optimization failed."
    res["message"] = msg

    print(info.fun_val)

    return Dict(res)