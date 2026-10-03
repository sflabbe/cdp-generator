import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from cdp_generator import core
from cdp_generator.application import (
    LegacyConcreteAnalysisRequest,
    result_excel_bytes,
    run_legacy_concrete_analysis,
)
from cdp_generator.application.services import temperature_cases
from cdp_generator.export import build_excel_bytes, export_to_excel


@pytest.mark.parametrize("mode", ["strain_rate", "temperature"])
def test_workbooks_against_original_export(tmp_path, mode):
    # Frozen pre-WEB-M1 exporter is an independent oracle; works without git.
    original_module = Path(__file__).parent / "fixtures/legacy_excel_export.txt"
    reference = tmp_path / "reference.xlsx"
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import runpy; from cdp_generator import core; from cdp_generator.application.services import temperature_cases; ns=runpy.run_path({str(original_module)!r}); mode={mode!r}; raw=core.calculate_stress_strain(28,.0022,.0035,1,[0,2,30,100]) if mode=='strain_rate' else core.calculate_stress_strain_temp(28,.0022,.0035,1,verbose=False); ns['export_to_excel'](raw,[0,2,30,100] if mode=='strain_rate' else temperature_cases(), mode, {str(reference)!r})",
        ],
        check=True,
    )
    request = LegacyConcreteAnalysisRequest(mode=mode)
    bundle = run_legacy_concrete_analysis(request)
    raw = (
        core.calculate_stress_strain(28, 0.0022, 0.0035, 1, request.strain_rates)
        if mode == "strain_rate"
        else core.calculate_stress_strain_temp(28, 0.0022, 0.0035, 1, verbose=False)
    )
    variables = list(request.strain_rates) if mode == "strain_rate" else temperature_cases()
    wrapped = tmp_path / "wrapped.xlsx"
    assert export_to_excel(raw, variables, mode, str(wrapped)) == str(wrapped)
    expected = pd.read_excel(reference, sheet_name=None)
    assert list(expected) == [
        "Compression Stress-Strain",
        "Compression Inl.Strain",
        "Compression Damage",
        "Tension Cracking",
        "Tension Cr.Strain",
        "Tension Damage",
        "Tension Cracking Power",
        "Tension Damage Power",
    ]
    for target in (
        wrapped,
        BytesIO(build_excel_bytes(raw, variables, mode)),
        BytesIO(result_excel_bytes(bundle)),
    ):
        actual = pd.read_excel(target, sheet_name=None)
        assert list(actual) == list(expected)
        for name in expected:
            pd.testing.assert_frame_equal(expected[name], actual[name], check_exact=True)
