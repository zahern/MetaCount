import importlib


def test_public_package_imports():
    package = importlib.import_module("metacountregressor")
    submodule = importlib.import_module("metacountregressor.experiment_package")
    cmf_submodule = importlib.import_module("metacountregressor.cmf_package")
    family_submodule = importlib.import_module("metacountregressor.family_search")

    assert hasattr(package, "ExperimentBuilder")
    assert hasattr(package, "CMFExperimentBuilder")
    assert hasattr(package, "LinearSearchProblem")
    assert hasattr(package, "DurationSearchProblem")
    assert hasattr(package, "SearchOutputConfig")
    assert hasattr(package, "load_example_crash_data")
    assert submodule.ExperimentBuilder is package.ExperimentBuilder
    assert cmf_submodule.CMFExperimentBuilder is package.CMFExperimentBuilder
    assert family_submodule.LinearSearchProblem is package.LinearSearchProblem
