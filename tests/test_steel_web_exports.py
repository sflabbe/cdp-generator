import json
from io import BytesIO
from math import isfinite
from pathlib import Path

import openpyxl
import pytest

from cdp_generator.application import (
    SteelAnalysisRequest,
    run_steel_analysis,
    steel_result_abaqus_text,
    steel_result_excel_bytes,
)
from cdp_generator.steel import calibrate_jc_from_spec, generate_jc_curves_multicase, get_steel_spec
from cdp_generator.steel.export import (
    build_abaqus_material_card_text,
    build_steel_excel_bytes,
    export_abaqus_material_card,
    export_steel_to_excel,
)


def cells(source):
    workbook = openpyxl.load_workbook(source)
    return {
        sheet.title: [list(row) for row in sheet.iter_rows(values_only=True)] for sheet in workbook
    }


def assert_workbook_baseline(actual, expected):
    """Allow only float roundoff against the historical Linux-generated fixture.

    Workbook structure, text and integer cells stay exact. Exports produced on
    the same platform are compared exactly separately below.
    """
    assert list(actual) == list(expected)
    for name, rows in expected.items():
        assert len(actual[name]) == len(rows), name
        for row_index, (values, baseline) in enumerate(zip(actual[name], rows, strict=True)):
            assert len(values) == len(baseline), (name, row_index)
            for column, (value, reference) in enumerate(zip(values, baseline, strict=True)):
                location = (name, row_index + 1, column + 1)
                if isinstance(reference, float):
                    assert isinstance(value, (int, float)) and not isinstance(value, bool), location
                    assert isfinite(value), location
                    assert value == pytest.approx(reference, rel=1e-12, abs=1e-12), location
                else:
                    assert value == reference, location


def test_baseline_allows_platform_roundoff():
    expected = {
        "Curves": [
            ["Stress [MPa]", "Strain [-]", None, 4],
            [570.4625314384795, 0.00808080808080808, "case", 4],
        ]
    }
    actual = {
        "Curves": [
            ["Stress [MPa]", "Strain [-]", None, 4],
            [570.4625314384796, 0.008080808080808081, "case", 4],
        ]
    }
    assert_workbook_baseline(actual, expected)


@pytest.mark.parametrize(
    "actual",
    [
        {"Other": [["Stress [MPa]", 500.0, 4]]},
        {"Curves": []},
        {"Curves": [["Stress [MPa]", 500.0]]},
        {"Curves": [["Stress [Pa]", 500.0, 4]]},
        {"Curves": [["Stress [MPa]", 500.000001, 4]]},
        {"Curves": [["Stress [MPa]", float("nan"), 4]]},
        {"Curves": [["Stress [MPa]", float("inf"), 4]]},
        {"Curves": [["Stress [MPa]", "500", 4]]},
        {"Curves": [["Stress [MPa]", 500.0, 5]]},
    ],
)
def test_baseline_rejects_real_content_changes(actual):
    with pytest.raises(AssertionError):
        assert_workbook_baseline(actual, {"Curves": [["Stress [MPa]", 500.0, 4]]})


@pytest.mark.parametrize("kind", ["true", "engineering"])
def test_workbook_baseline_and_filesystem_parity(tmp_path, kind):
    spec = get_steel_spec("EC2", "B500C")
    params = calibrate_jc_from_spec(spec)
    raw = generate_jc_curves_multicase(
        params, spec.E, strain_rates=[0.001, 1], temperatures=[20, 400], output_kind=kind
    )
    expected = json.loads(
        (Path(__file__).parent / "fixtures/steel_web_exports.json").read_text(encoding="utf-8")
    )[kind]
    path = tmp_path / "steel.xlsx"
    export_steel_to_excel(raw, str(path), verbose=False)
    filesystem_cells = cells(path)
    assert_workbook_baseline(filesystem_cells, expected)
    assert cells(BytesIO(build_steel_excel_bytes(raw))) == filesystem_cells
    result = run_steel_analysis(
        SteelAnalysisRequest(strain_rates=(0.001, 1), temperatures=(20, 400), output_kind=kind)
    )
    assert cells(BytesIO(steel_result_excel_bytes(result))) == filesystem_cells


@pytest.mark.parametrize("density", [None, 7.85e-9])
def test_abaqus_baseline_and_filesystem_parity(tmp_path, density):
    spec = get_steel_spec("EC2", "B500C")
    params = calibrate_jc_from_spec(spec)
    expected = json.loads(
        (Path(__file__).parent / "fixtures/steel_web_exports.json").read_text(encoding="utf-8")
    )["abaqus_density" if density else "abaqus"]
    text = build_abaqus_material_card_text(params, spec.E, spec.nu, density)
    path = tmp_path / "steel.inp"
    export_abaqus_material_card(params, spec.E, spec.nu, density, str(path))
    assert text == path.read_text() == expected
    if density is None:
        result = run_steel_analysis(SteelAnalysisRequest())
        assert steel_result_abaqus_text(result) == expected
