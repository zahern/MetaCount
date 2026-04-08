import importlib


def test_public_package_imports():
    package = importlib.import_module("metacountregressor")
    submodule = importlib.import_module("metacountregressor.experiment_package")
    cmf_submodule = importlib.import_module("metacountregressor.cmf_package")

    assert hasattr(package, "ExperimentBuilder")
    assert hasattr(package, "CMFExperimentBuilder")
    assert submodule.ExperimentBuilder is package.ExperimentBuilder
    assert cmf_submodule.CMFExperimentBuilder is package.CMFExperimentBuilder
