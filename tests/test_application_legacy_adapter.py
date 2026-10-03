"""Exact kernel-to-DTO parity for both complete workflows."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from cdp_generator import core
from cdp_generator.application import LegacyConcreteAnalysisRequest, run_legacy_concrete_analysis
from cdp_generator.application.adapters.legacy_concrete import CURVE_SPECS
from cdp_generator.application.requests import parse_strain_rates
from cdp_generator.application.services import temperature_cases


@pytest.mark.parametrize("mode", ["strain_rate", "temperature"])
def test_all_curve_arrays_and_properties(mode):
    request = LegacyConcreteAnalysisRequest(mode=mode)
    if mode == "temperature":
        raw = core.calculate_stress_strain_temp(28, 0.0022, 0.0035, 1, verbose=False)
    else:
        raw = core.calculate_stress_strain(28, 0.0022, 0.0035, 1, request.strain_rates)
    with patch.object(
        core, "calculate_stress_strain_temp", wraps=core.calculate_stress_strain_temp
    ) as spy:
        bundle = run_legacy_concrete_analysis(request)
        if mode == "temperature":
            assert spy.call_args.kwargs == {"verbose": False}
    expected_variables = (
        temperature_cases() if mode == "temperature" else list(request.strain_rates)
    )
    assert bundle.variables == expected_variables
    assert len(bundle.curves) == 6 * len(expected_variables) + 3
    for spec in CURVE_SPECS:
        group, source, xkey, ykey, _, _, _, _, reference = spec
        curves = [c for c in bundle.curves if c.group == group]
        assert len(curves) == (1 if reference else len(expected_variables))
        for i, curve in enumerate(curves):
            data = raw[source]
            x = (
                (data["strain temp"][i] if mode == "temperature" else data[xkey])
                if xkey == "strain"
                else data[xkey][i]
            )
            np.testing.assert_array_equal(curve.x, x)
            np.testing.assert_array_equal(curve.y, data[ykey] if reference else data[ykey][i])
            assert curve.metadata["variable_value"] == expected_variables[i]
    assert {p.name: p.value for p in bundle.properties} == raw["properties"]
    assert next(p.unit for p in bundle.properties if p.name == "l0") == "m"
    serialized = bundle.to_dict()
    assert json.loads(json.dumps(serialized, allow_nan=False)) == serialized
    assert bundle.to_json() == bundle.to_json()
    assert serialized["schema_version"] == "1.0"
    assert serialized["workflow"] == "legacy_concrete"
    assert serialized["metadata"]["curve_authority"] is None

    def check_json_types(value):
        assert type(value) in (dict, list, str, int, float, bool, type(None))
        if isinstance(value, dict):
            for child in value.values():
                check_json_types(child)
        if isinstance(value, list):
            for child in value:
                check_json_types(child)

    check_json_types(serialized)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "other"},
        {"f_cm": 0},
        {"l_ch": -1},
        {"e_c1": 0},
        {"e_clim": float("inf")},
        {"f_cm": float("nan")},
        {"strain_rates": ()},
        {"strain_rates": (-1,)},
        {"strain_rates": (float("inf"),)},
    ],
)
def test_invalid_requests(kwargs):
    with pytest.raises(ValueError):
        LegacyConcreteAnalysisRequest(**kwargs)


def test_rate_parser():
    assert parse_strain_rates("0, 2, 30, 100") == (0, 2, 30, 100)
    with pytest.raises(ValueError, match="comma-separated"):
        parse_strain_rates("oops")


def test_base_import_does_not_load_web_dependencies():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import cdp_generator, sys; assert 'plotly' not in sys.modules; assert 'streamlit' not in sys.modules",
        ],
        check=True,
    )
