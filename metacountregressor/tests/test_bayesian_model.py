import numpy as np
import pandas as pd
import pytest

import metacountregressor.bayesian_model as bayesian_model
import metacountregressor as package
from metacountregressor.bayesian_model import (
    BayesianModelError,
    build_bayesian_model,
    resolve_search_spec,
)


class FakeCMFBuilder:
    aadt_col = "AADT"
    local_vars = ["local"]

    @staticmethod
    def _cmf_term_map():
        return {
            "AADT": "__cmf_log_aadt",
            "local": "__cmf_local__local",
        }


def test_resolve_saved_spec_preserves_selected_structure():
    spec = {
        "fixed_terms": ["x"],
        "rdm_terms": ["z:normal"],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": [],
        "zi_terms": [],
        "membership_terms": [],
        "dispersion": 1,
        "latent_classes": 1,
    }

    resolved = resolve_search_spec({"family": "count", "model_spec": spec})

    assert resolved["fixed_terms"] == ["x"]
    assert resolved["rdm_terms"] == ["z:normal"]
    assert resolved["model"] == "nbl"
    assert resolved["family"] == "count"


def test_negative_binomial_aliases_resolve_to_lindley_model():
    for model in ("nb", "nb2", "negative binomial", "negative-binomial-lindley", "nbl"):
        resolved = resolve_search_spec(
            {"family": "count", "model_spec": {"model": model}}
        )
        assert resolved["model"] == "nbl"


def test_resolve_saved_json_wrapper_reads_nested_result():
    resolved = resolve_search_spec(
        {
            "family": "linear",
            "metadata": {"source": "saved"},
            "result": {
                "model_spec": {
                    "fixed_terms": ["x"],
                    "dispersion": 0,
                    "latent_classes": 1,
                    "model": "gaussian",
                }
            },
        }
    )

    assert resolved["family"] == "linear"
    assert resolved["metadata"] == {"source": "saved"}


def test_duration_spec_defaults_to_lognormal():
    resolved = resolve_search_spec(
        {
            "family": "duration",
            "model_spec": {"fixed_terms": ["x"], "latent_classes": 1},
        }
    )

    assert resolved["family"] == "duration"
    assert resolved["model"] == "lognormal"


def test_public_bayesian_exports_are_lazy():
    assert package.BayesianModel is bayesian_model.BayesianModel
    assert package.build_bayesian_model is bayesian_model.build_bayesian_model


def test_legacy_cmf_result_rebuilds_log_aadt_interactions():
    result = {
        "selected_baseline": ["base"],
        "selected_local": ["local"],
        "rand_baseline": [False],
        "rand_local": [True],
        "model": "nb",
    }
    builder = FakeCMFBuilder()

    resolved = resolve_search_spec(result, builder=builder, family="cmf")
    frame = bayesian_model._prepare_frame(
        pd.DataFrame({"AADT": [1000.0, 2000.0], "base": [1.0, 2.0], "local": [3.0, 4.0]}),
        resolved,
        builder=builder,
    )

    assert resolved["fixed_terms"] == ["__cmf_log_aadt", "base"]
    assert resolved["rdm_terms"] == ["__cmf_local__local:normal"]
    np.testing.assert_allclose(frame["__cmf_log_aadt"], np.log([1000.0, 2000.0]))
    np.testing.assert_allclose(
        frame["__cmf_local__local"], frame["local"] * frame["__cmf_log_aadt"]
    )


def test_unsupported_family_is_explicitly_rejected_before_pymc_import():
    with pytest.raises(BayesianModelError, match="not silently approximated"):
        build_bayesian_model(
            {"family": "pavement", "model_spec": {}},
            df=pd.DataFrame({"y": [1.0]}),
            y_col="y",
        )

    with pytest.raises(BayesianModelError, match="not silently approximated"):
        build_bayesian_model(
            {"front": [], "front_records": []},
            df=pd.DataFrame({"y": [1.0]}),
            y_col="y",
        )


def test_valid_model_keeps_pymc_as_lazy_optional_dependency(monkeypatch):
    def missing_pymc():
        raise ImportError("pymc is intentionally unavailable")

    monkeypatch.setattr(bayesian_model, "_import_pymc", missing_pymc)
    with pytest.raises(ImportError, match="intentionally unavailable"):
        build_bayesian_model(
            {
                "model_spec": {
                    "fixed_terms": ["x"],
                    "dispersion": 0,
                    "latent_classes": 1,
                }
            },
            df=pd.DataFrame({"y": [0, 1], "x": [1.0, 2.0]}),
            y_col="y",
        )


def test_pymc_nbl_graph_has_finite_initial_logp():
    pytest.importorskip("pymc")
    compiled = build_bayesian_model(
        {
            "family": "count",
            "model_spec": {
                "fixed_terms": ["x"],
                "dispersion": 1,
                "latent_classes": 1,
            },
        },
        df=pd.DataFrame({"y": [0, 1, 2, 3], "x": [0.2, 0.4, 0.8, 1.0]}),
        y_col="y",
    )

    logp = compiled.model.compile_logp()(compiled.model.initial_point())

    assert np.isfinite(logp)
    assert "nbl_theta" in compiled.model.named_vars


def test_nbl_theta_prior_scale_must_be_positive(monkeypatch):
    pytest.importorskip("pymc")
    with pytest.raises(BayesianModelError, match="nbl_theta_scale"):
        build_bayesian_model(
            {
                "family": "count",
                "model_spec": {"fixed_terms": ["x"], "dispersion": 1},
            },
            df=pd.DataFrame({"y": [0, 1], "x": [0.2, 0.4]}),
            y_col="y",
            priors={"nbl_theta_scale": 0},
        )
