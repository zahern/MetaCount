"""The copilot is part of the installed metacountregressor package."""

import importlib


def test_copilot_is_exported_from_package():
    package = importlib.import_module("metacountregressor")
    assert hasattr(package, "run_cli")
    assert hasattr(package, "ExperimentCopilot")
    assert callable(package.run_cli)
    assert callable(package.ExperimentCopilot)


def test_copilot_cli_entrypoint_exists():
    cli = importlib.import_module("metacountregressor.copilot.cli")
    assert callable(cli.main)
    assert callable(cli.run_cli)


def test_delphos_driver_is_importable():
    delphos = importlib.import_module("metacountregressor.copilot.delphos")
    assert hasattr(delphos, "DelphosSearch")
    assert hasattr(delphos, "MultiTaskTrainer")
    assert hasattr(delphos, "DelphosCountEnv")
