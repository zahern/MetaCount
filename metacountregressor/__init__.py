from importlib import import_module

__all__ = [
    "CMFExperimentBuilder",
    "CMFFamilySearchProblem",
    "DataProcessor",
    "DurationSearchProblem",
    "ExperimentBuilder",
    "LinearSearchProblem",
    "ObjectiveFunction",
    "SearchOutputConfig",
    "load_example_crash_data",
    "load_example_duration_data",
    "load_example_panel_data",
    "StructureEvaluatorLC",
    "differential_evolution",
    "harmony_search",
    "simulated_annealing",
]

_EXPORTS = {
    "CMFExperimentBuilder": ("cmf_package", "CMFExperimentBuilder"),
    "CMFFamilySearchProblem": ("family_search", "CMFFamilySearchProblem"),
    "DataProcessor": ("data_split_helper", "DataProcessor"),
    "DurationSearchProblem": ("family_search", "DurationSearchProblem"),
    "ExperimentBuilder": ("experiment_package", "ExperimentBuilder"),
    "LinearSearchProblem": ("family_search", "LinearSearchProblem"),
    "ObjectiveFunction": ("solution", "ObjectiveFunction"),
    "SearchOutputConfig": ("output_config", "SearchOutputConfig"),
    "StructureEvaluatorLC": ("experiment_package", "StructureEvaluatorLC"),
    "differential_evolution": ("metaheuristics", "differential_evolution"),
    "harmony_search": ("metaheuristics", "harmony_search"),
    "load_example_crash_data": ("sample_data", "load_example_crash_data"),
    "load_example_duration_data": ("sample_data", "load_example_duration_data"),
    "load_example_panel_data": ("sample_data", "load_example_panel_data"),
    "simulated_annealing": ("metaheuristics", "simulated_annealing"),
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
