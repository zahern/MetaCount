import importlib
from pathlib import Path


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
    # Version must match the single source of truth in version.txt
    version_file = Path(__file__).resolve().parents[2] / "version.txt"
    assert package.__version__ == version_file.read_text().strip()
