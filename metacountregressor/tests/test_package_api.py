import importlib


def test_public_package_imports():
    package = importlib.import_module("metacountregressor")
    submodule = importlib.import_module("metacountregressor.experiment_package")
    cmf_submodule = importlib.import_module("metacountregressor.cmf_package")
    family_submodule = importlib.import_module("metacountregressor.family_search")

    assert hasattr(package, "ExperimentBuilder")
    assert hasattr(package, "__version__")
    assert hasattr(package, "CMFExperimentBuilder")
    assert hasattr(package, "LinearSearchProblem")
    assert hasattr(package, "DurationSearchProblem")
    assert hasattr(package, "SearchOutputConfig")
    assert hasattr(package, "load_example16_3_raw_data")
    assert hasattr(package, "load_example16_3_model_data")
    assert hasattr(package, "load_example_crash_data")
    assert hasattr(package, "load_example_platform_speed_data")
    assert hasattr(package, "load_example_platform_gap_duration_data")
    assert submodule.ExperimentBuilder is package.ExperimentBuilder
    assert cmf_submodule.CMFExperimentBuilder is package.CMFExperimentBuilder
    assert family_submodule.LinearSearchProblem is package.LinearSearchProblem
    assert package.__version__ == "1.0.33"
