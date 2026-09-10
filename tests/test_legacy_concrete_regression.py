"""Regression freeze generated from the pre-G0 starting implementation."""

import json
import math
from pathlib import Path

import numpy as np

from cdp_generator import (
    calculate_cdp_parameters,
    calculate_characteristic_length,
    calculate_concrete_strength_properties,
    calculate_elastic_modulus,
    calculate_fracture_energy,
    calculate_poisson_ratios,
    calculate_stress_strain,
    calculate_stress_strain_temp,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((REPO_ROOT / "qualification/legacy_v1_baseline.json").read_text())
E_C1 = BASELINE["inputs"]["e_c1"]
E_CLIM = BASELINE["inputs"]["e_clim"]
L_CH = BASELINE["inputs"]["l_ch_mm"]


def assert_nested_close(actual, expected, path="root"):
    """Compare nested legacy outputs tightly while tolerating libm-level float noise."""

    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), path
        for key in expected:
            assert_nested_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, expected_item in enumerate(expected):
            assert_nested_close(actual[index], expected_item, f"{path}[{index}]")
        return
    if isinstance(expected, int | float) and not isinstance(expected, bool):
        assert math.isclose(float(actual), float(expected), rel_tol=1e-13, abs_tol=1e-14), (
            path,
            actual,
            expected,
        )
        return
    assert actual == expected, path


def scalar_record(f_cm):
    strength = calculate_concrete_strength_properties(f_cm)
    elastic = calculate_elastic_modulus(f_cm)
    poisson = calculate_poisson_ratios(f_cm, elastic["E_c"], E_C1)
    cdp = calculate_cdp_parameters(
        f_cm,
        elastic["E_c"],
        E_C1,
        poisson["v_c0"],
        poisson["v_ce"],
    )
    fracture_energy = calculate_fracture_energy(f_cm)
    characteristic_length = calculate_characteristic_length(
        elastic["E_c"], fracture_energy, strength["f_ctm"]
    )
    return {
        "calculate_concrete_strength_properties": strength,
        "calculate_elastic_modulus": elastic,
        "calculate_poisson_ratios": poisson,
        "calculate_cdp_parameters": cdp,
        "calculate_fracture_energy": fracture_energy,
        "calculate_characteristic_length": characteristic_length,
    }


def test_scalar_legacy_snapshot_is_unchanged():
    for key, expected in BASELINE["scalar_cases"].items():
        assert_nested_close(scalar_record(float(key)), expected, "scalar_cases." + key)


def test_static_stress_strain_snapshot_is_unchanged():
    for key, expected in BASELINE["stress_strain_static"].items():
        actual = calculate_stress_strain(float(key), E_C1, E_CLIM, L_CH, [0])
        assert_nested_close(actual, expected, "stress_strain_static." + key)


def test_multi_rate_snapshot_is_unchanged():
    case = BASELINE["stress_strain_multi_rate"]
    actual = calculate_stress_strain(case["f_cm"], E_C1, E_CLIM, L_CH, case["strain_rates"])
    assert_nested_close(actual, case["result"], "stress_strain_multi_rate")


def test_temperature_snapshot_is_unchanged():
    case = BASELINE["stress_strain_temperature"]
    actual = calculate_stress_strain_temp(case["f_cm"], E_C1, E_CLIM, L_CH, verbose=False)
    assert_nested_close(actual, case["result"], "stress_strain_temperature")


def test_high_strength_legacy_branch_boundary_is_frozen():
    boundary = [
        np.nextafter(58.0, -math.inf).item(),
        58.0,
        np.nextafter(58.0, math.inf).item(),
    ]
    for f_cm in boundary:
        key = format(f_cm, ".17g")
        actual = calculate_concrete_strength_properties(f_cm)
        expected = BASELINE["high_strength_boundary"][key]
        assert_nested_close(actual, expected, "high_strength_boundary." + key)

    below = calculate_concrete_strength_properties(boundary[0])["f_ctm"]
    at = calculate_concrete_strength_properties(boundary[1])["f_ctm"]
    above = calculate_concrete_strength_properties(boundary[2])["f_ctm"]
    assert below == at
    assert above != at
