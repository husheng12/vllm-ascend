import pytest
import torch

from vllm_ascend.models.deepseek_v4.dspark_weight_verifier import (
    validate_loaded_params,
    validate_no_nan_parameters,
)


def test_validate_loaded_params_accepts_complete_set():
    report = validate_loaded_params({"a", "b"}, {"a", "b"})
    assert report.missing == set()
    assert report.unexpected == set()


def test_validate_loaded_params_rejects_missing_parameter():
    with pytest.raises(RuntimeError, match="missing"):
        validate_loaded_params({"a", "b"}, {"a"})


def test_validate_loaded_params_rejects_unexpected_parameter():
    with pytest.raises(RuntimeError, match="unexpected"):
        validate_loaded_params({"a"}, {"a", "extra"})


def test_validate_no_nan_parameters_rejects_partial_load():
    parameters = {
        "loaded": torch.nn.Parameter(torch.ones(2)),
        "missing": torch.nn.Parameter(torch.tensor([float("nan"), 1.0])),
    }
    with pytest.raises(RuntimeError, match="missing"):
        validate_no_nan_parameters(parameters.items(), {"loaded", "missing"})


def test_validate_no_nan_parameters_ignores_excluded_and_non_float():
    parameters = {
        "excluded": torch.nn.Parameter(torch.tensor([float("nan")])),
        "integer": torch.nn.Parameter(torch.tensor([1], dtype=torch.int64)),
    }
    report = validate_no_nan_parameters(parameters.items(), {"integer"})
    assert report.nan_parameters == set()
