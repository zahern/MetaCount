from importlib import import_module
import warnings
try:
    from ._version import __version__
except ImportError:
    from _version import __version__

__all__ = [
    "__version__",
    # Core builders
    "CMFExperimentBuilder",
    "CMFFamilySearchProblem",
    "DurationSearchProblem",
    "ExperimentBuilder",
    "LinearSearchProblem",
    "ModelConstraints",
    "SearchOutputConfig",
    "SurvivalSearchProblem",
    "StructureEvaluatorLC",
    "RandomEffectsAFTFitter",
    "LogNormalRandomEffectsAFTFitter",
    "WeibullRandomEffectsAFTFitter",
    "LogLogisticRandomEffectsAFTFitter",
    # Data loaders
    "load_example16_3_model_data",
    "load_example16_3_raw_data",
    "load_example_crash_data",
    "load_example_duration_data",
    "load_example_linear_data",
    "load_example_panel_data",
    "load_example_platform_gap_duration_data",
    "load_example_platform_speed_data",
    # Help and templates
    "get_help",
    "get_templates",
    # Pre-fitted book specifications
    "describe_book_cmf_spec",
    "describe_book_latent_class_spec",
    "describe_book_nb_baseline_spec",
    "list_book_specifications",
    "load_book_cmf_spec",
    "load_book_latent_class_spec",
    "load_book_nb_baseline_spec",
    # Helper utilities
    "extract_summary",
    "extract_search_best",
    "compare_models",
]

_EXPORTS = {
    # Core builders
    "CMFExperimentBuilder": ("cmf_package", "CMFExperimentBuilder"),
    "CMFFamilySearchProblem": ("family_search", "CMFFamilySearchProblem"),
    "DurationSearchProblem": ("family_search", "DurationSearchProblem"),
    "ExperimentBuilder": ("experiment_package", "ExperimentBuilder"),
    "LinearSearchProblem": ("family_search", "LinearSearchProblem"),
    "ModelConstraints": ("model_constraints", "ModelConstraints"),
    "SearchOutputConfig": ("output_config", "SearchOutputConfig"),
    "SurvivalSearchProblem": ("survival_models", "SurvivalSearchProblem"),
    "StructureEvaluatorLC": ("experiment_package", "StructureEvaluatorLC"),
    "RandomEffectsAFTFitter": ("survival_models", "RandomEffectsAFTFitter"),
    "LogNormalRandomEffectsAFTFitter": ("survival_models", "LogNormalRandomEffectsAFTFitter"),
    "WeibullRandomEffectsAFTFitter": ("survival_models", "WeibullRandomEffectsAFTFitter"),
    "LogLogisticRandomEffectsAFTFitter": ("survival_models", "LogLogisticRandomEffectsAFTFitter"),
    # Data loaders
    "load_example16_3_model_data": ("sample_data", "load_example16_3_model_data"),
    "load_example16_3_raw_data": ("sample_data", "load_example16_3_raw_data"),
    "load_example_crash_data": ("sample_data", "load_example_crash_data"),
    "load_example_duration_data": ("sample_data", "load_example_duration_data"),
    "load_example_linear_data": ("sample_data", "load_example_linear_data"),
    "load_example_panel_data": ("sample_data", "load_example_panel_data"),
    "load_example_platform_gap_duration_data": ("sample_data", "load_example_platform_gap_duration_data"),
    "load_example_platform_speed_data": ("sample_data", "load_example_platform_speed_data"),
    # Help and templates
    "get_help": ("help_guide", "get_help"),
    "get_templates": ("help_guide", "get_templates"),
    # Pre-fitted book specifications
    "describe_book_cmf_spec": ("fitted_specifications", "describe_book_cmf_spec"),
    "describe_book_latent_class_spec": ("fitted_specifications", "describe_book_latent_class_spec"),
    "describe_book_nb_baseline_spec": ("fitted_specifications", "describe_book_nb_baseline_spec"),
    "list_book_specifications": ("fitted_specifications", "list_book_specifications"),
    "load_book_cmf_spec": ("fitted_specifications", "load_book_cmf_spec"),
    "load_book_latent_class_spec": ("fitted_specifications", "load_book_latent_class_spec"),
    "load_book_nb_baseline_spec": ("fitted_specifications", "load_book_nb_baseline_spec"),
    # Helper utilities
    "extract_summary": ("experiment_package", "extract_summary"),
    "extract_search_best": ("experiment_package", "extract_search_best"),
    "compare_models": ("experiment_package", "compare_models"),
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
