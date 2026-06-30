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
]

_EXPORTS = {
    "CMFExperimentBuilder": ("cmf_package", "CMFExperimentBuilder"),
    "CMFFamilySearchProblem": ("family_search", "CMFFamilySearchProblem"),
    "DurationSearchProblem": ("family_search", "DurationSearchProblem"),
    "ExperimentBuilder": ("experiment_package", "ExperimentBuilder"),
    "LinearSearchProblem": ("family_search", "LinearSearchProblem"),
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
