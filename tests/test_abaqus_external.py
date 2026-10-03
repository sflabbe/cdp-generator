"""Real-solver ABAQUS-Q1 gate. Opt in with: uv run pytest -q -m abaqus_external

Behaviour: no resolvable solver -> SKIP; solver found and any case fails -> FAIL;
solver found and every case reaches its target state -> PASS. A solver that is found is never
skipped because a job failed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cdp_generator.application import abaqus_runner as runner

pytestmark = pytest.mark.abaqus_external
POST = Path(__file__).resolve().parents[1] / "qualification" / "abaqus" / "postprocess_odb.py"


def _require_solver() -> None:
    discovery = runner.discover_abaqus_command()
    if discovery.command is None:
        pytest.skip(f"Abaqus external qualification: NOT_AVAILABLE ({discovery.reason})")


def _failures(report: dict) -> dict[str, object]:
    return {
        case: {"state": rec["state"], "diagnostics": rec["diagnostics_excerpt"]}
        for case, rec in report["cases"].items()
    }


def test_abaqus_datacheck_all_decks():
    _require_solver()
    report = runner.run_abaqus_qualification(
        runner.QualificationOptions(datacheck_only=True, postprocess_script=POST)
    )
    assert report["overall_state"] == "PASS", _failures(report)
    assert {c["state"] for c in report["cases"].values()} == {"DATACHECK_PASS"}


def test_abaqus_full_static_analysis_and_odb_postprocess():
    _require_solver()
    report = runner.run_abaqus_qualification(runner.QualificationOptions(postprocess_script=POST))
    assert report["overall_state"] == "PASS", _failures(report)
    for case in ("static_compression", "static_tension"):
        record = report["cases"][case]
        assert record["state"] == "ANALYSIS_PASS"
        assert all(record["checks"].values()), record["checks"]
