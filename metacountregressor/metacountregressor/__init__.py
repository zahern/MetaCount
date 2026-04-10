from importlib import import_module
try:
    from ._version import __version__
except ImportError:
    from _version import __version__

__all__ = [
    "__version__",
    # Core builders
    "CMFExperimentBuilder",
    "CMFFamilySearchProblem",
    "DataProcessor",
    "DurationSearchProblem",
    "ExperimentBuilder",
    "LinearSearchProblem",
    "ModelConstraints",
    "ObjectiveFunction",
    "SearchOutputConfig",
    "StructureEvaluatorLC",
    # Data loaders
    "load_example16_3_model_data",
    "load_example16_3_raw_data",
    "load_example_crash_data",
    "load_example_duration_data",
    "load_example_linear_data",
    "load_example_panel_data",
    "load_example_platform_gap_duration_data",
    "load_example_platform_speed_data",
    # Search algorithms
    "differential_evolution",
    "harmony_search",
    "simulated_annealing",
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
]

_EXPORTS = {
    # Core builders
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
    # Data loaders
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
