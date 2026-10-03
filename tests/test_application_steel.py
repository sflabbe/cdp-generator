import json
from dataclasses import asdict

import numpy as np
import pytest

from cdp_generator.application import (
    SteelAnalysisRequest,
    SteelInputError,
    available_steel_grades,
    available_steel_standards,
    parse_float_series,
    run_steel_analysis,
)
from cdp_generator.steel import calibrate_jc_from_spec, generate_jc_curves_multicase, get_steel_spec
from cdp_generator.steel.standards import list_available_standards

CASES = [
    {},
    {"standard": "ACI", "grade": "A615_60"},
    {"standard": "NCh", "grade": "A630-420H"},
    {"material_overrides": {"fy": 550, "fu": 650, "Agt": 8}},
    {
        "source": "custom",
        "grade": "TestSteel",
        "material_overrides": {"fy": 500, "fu": 620, "Agt": 10, "E": 200000, "nu": 0.3},
    },
    {
        "assume_rate_temp_neutral": False,
        "C_default": 0.02,
        "m_default": 1.0,
        "n_default": 0.25,
        "strain_rates": (0.001, 10),
        "temperatures": (20, 400),
    },
    {"output_kind": "engineering"},
]


@pytest.mark.parametrize("options", CASES, ids=["S1", "S2", "S3", "S4", "S5", "S6", "S7"])
def test_steel_direct_domain_parity(options, capsys):
    request = SteelAnalysisRequest(**{"strain_rates": (0.001,), "temperatures": (20,), **options})
    result = run_steel_analysis(request)
    assert capsys.readouterr().out == ""
    spec = get_steel_spec(
        "Custom" if request.source == "custom" else request.standard,
        request.grade,
        overrides=dict(request.material_overrides),
    )
    params = calibrate_jc_from_spec(
        spec,
        assume_rate_temp_neutral=request.assume_rate_temp_neutral,
        n_default=request.n_default,
        C_default=request.C_default,
        m_default=request.m_default,
        epsdot0=request.epsdot0,
        verbose=False,
    )
    raw = generate_jc_curves_multicase(
        params,
        spec.E,
        eps_max=request.eps_max,
        n_points=request.n_points,
        strain_rates=request.strain_rates,
        temperatures=request.temperatures,
        output_kind=request.output_kind,
    )
    assert result.material == asdict(spec)
    assert result.johnson_cook_parameters == asdict(params)
    for i, c in enumerate(raw["curves"]):
        total, plastic = result.curves[2 * i : 2 * i + 2]
        np.testing.assert_array_equal(total.x, c["strain"])
        np.testing.assert_array_equal(total.y, c["stress"])
        np.testing.assert_array_equal(plastic.x, c["plastic_strain"])
        np.testing.assert_array_equal(plastic.y, c["stress"])
        assert total.x_quantity == f"{request.output_kind.title()} strain"
        assert plastic.x_quantity == "True plastic strain"
        assert plastic.y_quantity == "Stress"
        assert plastic.metadata["stress_kind"] == request.output_kind
        assert total.metadata["case_id"] == c["case_id"]
    assert result.data_status == (
        "user_provided" if request.source == "custom" else "approximate_preset"
    )
    assert result.metadata["material_overrides"] == dict(request.material_overrides)
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False)) == result.to_dict()
    assert run_steel_analysis(request).to_json() == result.to_json()
    if request.assume_rate_temp_neutral:
        assert params.C == params.m == 0


def test_catalog_and_custom_ratio():
    catalog = list_available_standards()
    assert available_steel_standards() == tuple(catalog)
    for standard, grades in catalog.items():
        assert available_steel_grades(standard) == tuple(grades)
    result = run_steel_analysis(
        SteelAnalysisRequest(
            source="custom",
            grade="RatioSteel",
            material_overrides={"fy": 500, "fu_fy_ratio": 1.24, "Agt": 10},
        )
    )
    assert result.material["fu"] == 620


@pytest.mark.parametrize(
    "options",
    [
        {"strain_rates": ()},
        {"temperatures": (float("nan"),)},
        {"epsdot0": 0},
        {"eps_max": 0},
        {"n_points": 1},
        {"n_points": 3.5},
        {"n_points": True},
        {"output_kind": "unknown"},
        {"source": "unknown"},
        {"material_overrides": {"E": -1}},
        {"material_overrides": {"fy": None}},
        {"material_overrides": {"fu": 620, "fu_fy_ratio": 1.2}},
        {"material_overrides": {"unknown": 1}},
    ],
)
def test_structural_validation(options):
    with pytest.raises(SteelInputError):
        SteelAnalysisRequest(**options)


@pytest.mark.parametrize(
    "options",
    [
        {"grade": "unknown"},
        {"standard": "unknown"},
        {"source": "custom", "material_overrides": {}},
        {"material_overrides": {"T_melt": 20}},
        {"material_overrides": {"fu": 300}},
    ],
)
def test_domain_validation(options):
    with pytest.raises(SteelInputError):
        run_steel_analysis(SteelAnalysisRequest(**options))


@pytest.mark.parametrize("text", ["", "1,", "no", "nan", "inf", "1;2"])
def test_series_errors(text):
    with pytest.raises(SteelInputError):
        parse_float_series(text)


def test_immutable_input_and_unexpected_errors(monkeypatch):
    values = {"fy": 550.0}
    request = SteelAnalysisRequest(material_overrides=values)
    values["fy"] = 900
    assert request.material_overrides["fy"] == 550
    with pytest.raises(TypeError):
        request.material_overrides["fy"] = 1

    def defect(*args, **kwargs):
        raise RuntimeError("scientific defect")

    monkeypatch.setattr(
        "cdp_generator.application.steel_services.generate_jc_curves_multicase", defect
    )
    with pytest.raises(RuntimeError, match="scientific defect"):
        run_steel_analysis(request)
