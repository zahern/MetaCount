"""Offset expressions are evaluated safely and auto-logged."""

import numpy as np
import pandas as pd
import pytest

from metacountregressor.copilot.derived import evaluate_expression, is_log_offset


def test_offset_expression_and_auto_log():
    df = pd.DataFrame(
        {
            "LENGTH": [0.64, 1.44],
            "AADT": [3347, 3347],
        }
    )
    exposure = evaluate_expression("LENGTH*AADT*365/1e8", df)
    logged = evaluate_expression("log(LENGTH*AADT*365/1e8)", df)
    np.testing.assert_allclose(logged, np.log(exposure))
    assert is_log_offset("log(LENGTH*AADT*365/1e8)")
    assert not is_log_offset("LENGTH*AADT*365/1e8")


def test_offset_expression_rejects_unknown_names():
    df = pd.DataFrame({"LENGTH": [1.0], "AADT": [2.0]})
    with pytest.raises(ValueError):
        evaluate_expression("LENGTH*NOT_A_COLUMN", df)
    with pytest.raises(ValueError):
        evaluate_expression("__import__('os').system('echo bad')", df)
