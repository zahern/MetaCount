"""Paths pasted with quotes should load normally."""

import pandas as pd

from metacountregressor.copilot.profile import load_data, profile_dataframe


def test_load_data_strips_surrounding_quotes(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"FREQ": [1, 2, 3], "X": [0.1, 0.2, 0.3]}).to_csv(path, index=False)

    for quoted in (f'"{path}"', f"'{path}'"):
        df = load_data(quoted)
        assert list(df.columns) == ["FREQ", "X"]
        assert len(df) == 3


def test_profile_accepts_quoted_path(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"FREQ": [1, 2, 3], "X": [0.1, 0.2, 0.3]}).to_csv(path, index=False)
    profile = profile_dataframe(load_data(f'"{path}"'), path=str(path))
    assert profile.n_rows == 3
