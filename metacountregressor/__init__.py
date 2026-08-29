from importlib import import_module
import warnings
try:
    from ._version import __version__
except ImportError:
    from _version import __version__

__all__ = [
    "__version__",
    # Builders
    "CMFExperimentBuilder",
    "ExperimentBuilder",
    "StructureEvaluatorLC",
    # Constraints & config
    "ModelConstraints",
    "SearchOutputConfig",
    # Family search problems
    "CMFFamilySearchProblem",
    "DurationSearchProblem",
    "LinearSearchProblem",
    "MultivariateSearchProblem",
    "BayesianModel",
    "BayesianModelError",
    "build_bayesian_model",
    "resolve_search_spec",
    # Data loaders
    "load_example16_3_model_data",
    "load_example16_3_raw_data",
    "load_example_crash_data",
    "load_example_duration_data",
    "load_example_linear_data",
    "load_example_panel_data",
    "load_example_platform_gap_duration_data",
    "load_example_platform_speed_data",
    # Help system
    "get_help",
    "load_book_latent_class_spec",
    "describe_book_latent_class_spec",
    "load_book_cmf_spec",
    "describe_book_cmf_spec",
    "load_book_nb_baseline_spec",
    "describe_book_nb_baseline_spec",
    # Helper utilities
    "extract_summary",
    "extract_search_best",
    "compare_models",
    # Synthetic data generation
    "make_synthetic_count_data",
    # Fit inspection
    "FitResult",
    "print_fit",
    # Bivariate copula NB (Ahmad, Gayah & Donnell 2023) – joint
    # negative-binomial models with Frank / Normal / Kimeldorf-Sampson
    # copulas for modelling crash counts and near-miss counts jointly.
    "BivariateCopulaNB",
    "BivariateCopulaFit",
    "bivariate_copula_loglik",
    "famoye_bivariate_nb_loglik",
    "marshall_olkin_nb_loglik",
    "frank_logpdf",
    "normal_logpdf",
    "kimeldorf_sampson_logpdf",
    "fit_copula_bivariate_nb",
    "fit_famoye_nb",
    "fit_marshall_olkin_nb",
    "compare_bivariate_copulas",
    # Multivariate count regression – jointly predict M activity counts
    # per person using NB (or Poisson) marginals + Gaussian / vine-Frank copula.
    "MultivariateCountRegressor",
    "MultivariateCountFit",
    "gaussian_copula_loglik",
    "vine_frank_copula_loglik",
    "fit_multivariate_activity_model",
    # Plotting & figures
    "generate_all_lc_figures",
    "plot_search_convergence",
    "plot_objective_trace",
    "plot_class_profiles",
    # CMF plotting & figures
    "generate_all_cmf_figures",
    "plot_cmf_search_convergence",
    "plot_cmf_obs_vs_pred",
    "plot_cmf_model_comparison",
    # Pavement deterioration (clusterwise log-log regression search)
    "PavementCLROptimizer",
    "PavementTemporalComparison",
    "log_transform_pavement",
    "fit_cluster_ols",
    "fit_cluster_ar1",
    "fit_cluster_random_walk",
    "fit_cluster_nur",
    "forecast_deterioration",
    "forecast_to_threshold",
    # JAX device / backend configuration (GPU support)
    "configure_jax",
    "device_summary",
    "get_device_info",
    # Pavement deterioration: continuous-outcome engine (functional forms,
    # Markov transitions, hazard/survival, GP forms, pipeline evaluator)
    "PavementConstraints",
    "PavementDeteriorationEvaluator",
    "PavementEvalResult",
    "MetacountRegressorBridge",
    "run_pavement_pipeline",
    "run_pavement_multiobjective_search",
    "run_pavement_sparse_agds",
    "PavementMultiObjectiveEvaluator",
    # pavement building blocks
    "fit_form",
    "build_design",
    "transform_col",
    "discretize",
    "fit_transitions",
    "forward_propagate",
    "markov_bic",
    "composite_bic",
    "fit_hazard",
    "GPSymbolicRegressor",
    "tree_to_lambda",
]

_EXPORTS = {
    "CMFExperimentBuilder": ("cmf_package", "CMFExperimentBuilder"),
    "CMFFamilySearchProblem": ("family_search", "CMFFamilySearchProblem"),
    "DurationSearchProblem": ("family_search", "DurationSearchProblem"),
    "ExperimentBuilder": ("experiment_package", "ExperimentBuilder"),
    "LinearSearchProblem": ("family_search", "LinearSearchProblem"),
    "MultivariateSearchProblem": ("family_search", "MultivariateSearchProblem"),
    "BayesianModel": ("bayesian_model", "BayesianModel"),
    "BayesianModelError": ("bayesian_model", "BayesianModelError"),
    "build_bayesian_model": ("bayesian_model", "build_bayesian_model"),
    "resolve_search_spec": ("bayesian_model", "resolve_search_spec"),
    "ModelConstraints": ("model_constraints", "ModelConstraints"),
    "SearchOutputConfig": ("output_config", "SearchOutputConfig"),
    "StructureEvaluatorLC": ("experiment_package", "StructureEvaluatorLC"),
    "load_example16_3_model_data": ("sample_data", "load_example16_3_model_data"),
    "load_example16_3_raw_data": ("sample_data", "load_example16_3_raw_data"),
    "load_example_crash_data": ("sample_data", "load_example_crash_data"),
    "load_example_duration_data": ("sample_data", "load_example_duration_data"),
    "load_example_linear_data": ("sample_data", "load_example_linear_data"),
    "load_example_panel_data": ("sample_data", "load_example_panel_data"),
    "load_example_platform_gap_duration_data": ("sample_data", "load_example_platform_gap_duration_data"),
    "load_example_platform_speed_data": ("sample_data", "load_example_platform_speed_data"),
    # Help system
    "get_help": ("help", "get_help"),
    "load_book_latent_class_spec": ("fitted_specifications", "load_book_latent_class_spec"),
    "describe_book_latent_class_spec": ("fitted_specifications", "describe_book_latent_class_spec"),
    "load_book_cmf_spec": ("fitted_specifications", "load_book_cmf_spec"),
    "describe_book_cmf_spec": ("fitted_specifications", "describe_book_cmf_spec"),
    "load_book_nb_baseline_spec": ("fitted_specifications", "load_book_nb_baseline_spec"),
    "describe_book_nb_baseline_spec": ("fitted_specifications", "describe_book_nb_baseline_spec"),
    # Helper utilities
    "extract_summary": ("experiment_package", "extract_summary"),
    "extract_search_best": ("experiment_package", "extract_search_best"),
    "compare_models": ("experiment_package", "compare_models"),
    # Synthetic data generation
    "make_synthetic_count_data": ("experiment_package", "make_synthetic_count_data"),
    # Fit inspection
    "FitResult": ("experiment_package", "FitResult"),
    "print_fit": ("experiment_package", "print_fit"),
    # Plotting
    "generate_all_lc_figures": ("lc_plotting", "generate_all_lc_figures"),
    "plot_search_convergence": ("lc_plotting", "plot_search_convergence"),
    "plot_objective_trace": ("lc_plotting", "plot_objective_trace"),
    "plot_class_profiles": ("lc_plotting", "plot_class_profiles"),
    # CMF
    "generate_all_cmf_figures": ("cmf_plotting", "generate_all_cmf_figures"),
    "plot_cmf_search_convergence": ("cmf_plotting", "plot_cmf_search_convergence"),
    "plot_cmf_obs_vs_pred": ("cmf_plotting", "plot_cmf_obs_vs_pred"),
    "plot_cmf_model_comparison": ("cmf_plotting", "plot_cmf_model_comparison"),
    # Multivariate count regression
    "MultivariateCountRegressor": ("multivariate_count_regressor", "MultivariateCountRegressor"),
    "MultivariateCountFit": ("multivariate_count_regressor", "MultivariateCountFit"),
    "gaussian_copula_loglik": ("multivariate_count_regressor", "gaussian_copula_loglik"),
    "vine_frank_copula_loglik": ("multivariate_count_regressor", "vine_frank_copula_loglik"),
    "fit_multivariate_activity_model": ("multivariate_count_regressor", "fit_multivariate_activity_model"),
    # Bivariate copula NB
    "BivariateCopulaNB": ("bivariate_copula", "BivariateCopulaNB"),
    "BivariateCopulaFit": ("bivariate_copula", "BivariateCopulaFit"),
    "bivariate_copula_loglik": ("bivariate_copula", "bivariate_copula_loglik"),
    "famoye_bivariate_nb_loglik": ("bivariate_copula", "famoye_bivariate_nb_loglik"),
    "marshall_olkin_nb_loglik": ("bivariate_copula", "marshall_olkin_nb_loglik"),
    "frank_logpdf": ("bivariate_copula", "frank_logpdf"),
    "normal_logpdf": ("bivariate_copula", "normal_logpdf"),
    "kimeldorf_sampson_logpdf": ("bivariate_copula", "kimeldorf_sampson_logpdf"),
    "fit_copula_bivariate_nb": ("bivariate_copula", "fit_copula_bivariate_nb"),
    "fit_famoye_nb": ("bivariate_copula", "fit_famoye_nb"),
    "fit_marshall_olkin_nb": ("bivariate_copula", "fit_marshall_olkin_nb"),
    "compare_bivariate_copulas": ("bivariate_copula", "compare_bivariate_copulas"),
    # Pavement deterioration
    "PavementCLROptimizer": ("pavement_clr", "PavementCLROptimizer"),
    "PavementTemporalComparison": ("pavement_clr", "PavementTemporalComparison"),
    "log_transform_pavement": ("pavement_clr", "log_transform_pavement"),
    "fit_cluster_ols": ("pavement_clr", "fit_cluster_ols"),
    "fit_cluster_ar1": ("pavement_clr", "fit_cluster_ar1"),
    "fit_cluster_random_walk": ("pavement_clr", "fit_cluster_random_walk"),
    "fit_cluster_nur": ("pavement_clr", "fit_cluster_nur"),
    "forecast_deterioration": ("pavement_clr", "forecast_deterioration"),
    "forecast_to_threshold": ("pavement_clr", "forecast_to_threshold"),
    # JAX device / backend configuration (GPU support)
    "configure_jax": ("_jax_config", "configure_jax"),
    "device_summary": ("_jax_config", "device_summary"),
    "get_device_info": ("_jax_config", "get_device_info"),
    # Pavement deterioration: continuous-outcome engine
    "PavementConstraints": ("pavement_pipeline", "Constraints"),
    "PavementDeteriorationEvaluator": (
        "pavement_pipeline", "PavementDeteriorationEvaluator"),
    "PavementEvalResult": ("pavement_pipeline", "EvalResult"),
    "MetacountRegressorBridge": ("pavement_pipeline", "MetacountRegressorBridge"),
    "run_pavement_pipeline": ("pavement_pipeline", "run_pipeline"),
    "run_pavement_multiobjective_search": (
        "pavement_search", "run_pavement_multiobjective_search"),
    "run_pavement_sparse_agds": (
        "pavement_search", "run_pavement_sparse_agds"),
    "PavementMultiObjectiveEvaluator": (
        "pavement_search", "PavementMultiObjectiveEvaluator"),
    # pavement building blocks
    "fit_form": ("pavement_forms", "fit_form"),
    "build_design": ("pavement_forms", "build_design"),
    "transform_col": ("pavement_forms", "transform_col"),
    "discretize": ("pavement_markov", "discretize"),
    "fit_transitions": ("pavement_markov", "fit_transitions"),
    "forward_propagate": ("pavement_markov", "forward_propagate"),
    "markov_bic": ("pavement_markov", "markov_bic"),
    "composite_bic": ("pavement_markov", "composite_bic"),
    "fit_hazard": ("pavement_hazard", "fit_hazard"),
    "GPSymbolicRegressor": ("pavement_gp", "GPSymbolicRegressor"),
    "tree_to_lambda": ("pavement_gp", "tree_to_lambda"),
}

_LEGACY_EXPORTS = {
    "DataProcessor": (
        "data_split_helper",
        "DataProcessor",
        "DataProcessor is deprecated; use ExperimentBuilder and JAX evaluators instead.",
    ),
    "ObjectiveFunction": (
        "solution",
        "ObjectiveFunction",
        "ObjectiveFunction is deprecated; use ExperimentBuilder and run_search().",
    ),
    "differential_evolution": (
        "metaheuristics",
        "differential_evolution",
        "Legacy metaheuristics are deprecated; use ExperimentBuilder.run(..., algo='de').",
    ),
    "harmony_search": (
        "metaheuristics",
        "harmony_search",
        "Legacy metaheuristics are deprecated; use ExperimentBuilder.run(..., algo='hs').",
    ),
    "simulated_annealing": (
        "metaheuristics",
        "simulated_annealing",
        "Legacy metaheuristics are deprecated; use ExperimentBuilder.run(..., algo='sa').",
    ),
    "sparse_ea_agds": (
        "metaheuristics",
        "sparse_ea_agds",
        "Legacy metaheuristics are deprecated; use ExperimentBuilder.run(..., algo='agds').",
    ),
}


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        target = _LEGACY_EXPORTS.get(name)
        if target is not None:
            warnings.warn(target[2], DeprecationWarning, stacklevel=2)
            module_name, attr_name, _ = target
            try:
                module = import_module(f".{module_name}", __name__)
            except ImportError:
                module = import_module(module_name)

            value = getattr(module, attr_name)
            globals()[name] = value
            return value

    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    try:
        module = import_module(f".{module_name}", __name__)
    except ImportError:
        module = import_module(module_name)

    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
