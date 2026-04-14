from importlib import import_module
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
    # Search algorithms
    "differential_evolution",
    "harmony_search",
    "simulated_annealing",
    # Family search problems
    "CMFFamilySearchProblem",
    "DurationSearchProblem",
    "LinearSearchProblem",
    # Data utilities
    "DataProcessor",
    "ObjectiveFunction",
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
]

_EXPORTS = {
    "CMFExperimentBuilder": ("cmf_package", "CMFExperimentBuilder"),
    "CMFFamilySearchProblem": ("family_search", "CMFFamilySearchProblem"),
    "DataProcessor": ("data_split_helper", "DataProcessor"),
    "DurationSearchProblem": ("family_search", "DurationSearchProblem"),
    "ExperimentBuilder": ("experiment_package", "ExperimentBuilder"),
    "LinearSearchProblem": ("family_search", "LinearSearchProblem"),
    "ModelConstraints": ("model_constraints", "ModelConstraints"),
    "ObjectiveFunction": ("solution", "ObjectiveFunction"),
    "SearchOutputConfig": ("output_config", "SearchOutputConfig"),
    "StructureEvaluatorLC": ("experiment_package", "StructureEvaluatorLC"),
    "differential_evolution": ("metaheuristics", "differential_evolution"),
    "harmony_search": ("metaheuristics", "harmony_search"),
    "load_example16_3_model_data": ("sample_data", "load_example16_3_model_data"),
    "load_example16_3_raw_data": ("sample_data", "load_example16_3_raw_data"),
    "load_example_crash_data": ("sample_data", "load_example_crash_data"),
    "load_example_duration_data": ("sample_data", "load_example_duration_data"),
    "load_example_linear_data": ("sample_data", "load_example_linear_data"),
    "load_example_panel_data": ("sample_data", "load_example_panel_data"),
    "load_example_platform_gap_duration_data": ("sample_data", "load_example_platform_gap_duration_data"),
    "load_example_platform_speed_data": ("sample_data", "load_example_platform_speed_data"),
    "simulated_annealing": ("metaheuristics", "simulated_annealing"),
    # Help system
    "get_help": ("help", "get_help"),
    "load_book_latent_class_spec": ("help", "load_book_latent_class_spec"),
    "describe_book_latent_class_spec": ("help", "describe_book_latent_class_spec"),
    "load_book_cmf_spec": ("help", "load_book_cmf_spec"),
    "describe_book_cmf_spec": ("help", "describe_book_cmf_spec"),
}


def __getattr__(name):
    target = _EXPORTS.get(name)
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
