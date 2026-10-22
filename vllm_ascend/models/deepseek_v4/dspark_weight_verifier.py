"""Optional runtime checks for DSpark draft checkpoint loading."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch


@dataclass(frozen=True)
class LoadedParamReport:
    expected: set[str]
    loaded: set[str]
    missing: set[str]
    unexpected: set[str]


@dataclass(frozen=True)
class NanParamReport:
    checked: set[str]
    nan_parameters: set[str]


def validate_loaded_params(
    expected: Iterable[str], loaded: Iterable[str]
) -> LoadedParamReport:
    expected_set = set(expected)
    loaded_set = set(loaded)
    missing = expected_set - loaded_set
    unexpected = loaded_set - expected_set
    report = LoadedParamReport(expected_set, loaded_set, missing, unexpected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)[:20]}")
        if unexpected:
            details.append(f"unexpected={sorted(unexpected)[:20]}")
        raise RuntimeError("DSpark draft weight verification failed: " + "; ".join(details))
    return report


def validate_no_nan_parameters(
    parameters: Iterable[tuple[str, torch.Tensor]],
    expected: Iterable[str],
) -> NanParamReport:
    expected_set = set(expected)
    checked: set[str] = set()
    nan_parameters: set[str] = set()
    for name, parameter in parameters:
        if name not in expected_set or not parameter.is_floating_point():
            continue
        checked.add(name)
        if bool(torch.isnan(parameter).any()):
            nan_parameters.add(name)
    report = NanParamReport(checked, nan_parameters)
    if nan_parameters:
        raise RuntimeError(
            "DSpark draft parameters were not fully loaded: "
            f"{sorted(nan_parameters)[:20]}"
        )
    return report


def fill_parameters_with_nan(
    parameters: Iterable[tuple[str, torch.Tensor]],
    expected: Iterable[str],
) -> None:
    expected_set = set(expected)
    with torch.no_grad():
        for name, parameter in parameters:
            if name in expected_set and parameter.is_floating_point():
                parameter.fill_(float("nan"))
