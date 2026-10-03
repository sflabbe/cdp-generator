"""ABAQUS-Q1 decks and runner — pure tests; a fake launcher stands in for the solver."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cdp_generator.application import (
    AbaqusLegacyDependentRequest,
    AbaqusLegacyMaterialRequest,
    abaqus_legacy_dependent_material_text,
    abaqus_legacy_material_text,
    run_abaqus_legacy_dependent_material,
    run_abaqus_legacy_material,
)
from cdp_generator.application import abaqus_runner as runner
from cdp_generator.application.abaqus_decks import (
    QUALIFICATION_MATERIAL_NAME,
    build_abaqus_dependent_datacheck_deck,
    build_abaqus_static_compression_qualification_deck,
    build_abaqus_static_tension_qualification_deck,
)

ROOT = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).parent / "fixtures" / "fake_abaqus.py"
SCRIPT = ROOT / "scripts" / "qualify_abaqus.py"
POST = ROOT / "qualification" / "abaqus" / "postprocess_odb.py"


def _fake_command() -> str:
    return f'"{sys.executable}" "{FAKE}"'


def _material_block(deck_text: str) -> str:
    start = deck_text.index("** ---- material card")
    end = deck_text.index("** ---- end of material card")
    body = deck_text[start:end].splitlines()[1:]
    return "\n".join(body) + "\n"


# --------------------------------------------------------------------------- decks


def test_static_decks_embed_the_canonical_material_card_unchanged():
    static = run_abaqus_legacy_material(AbaqusLegacyMaterialRequest())
    card = abaqus_legacy_material_text(static, QUALIFICATION_MATERIAL_NAME)
    for deck in (
        build_abaqus_static_compression_qualification_deck(static),
        build_abaqus_static_tension_qualification_deck(static),
    ):
        assert _material_block(deck.text) == card
        assert deck.text.count("*MATERIAL,") == 1
        assert "*STABILIZE" not in deck.text.upper() and "*DENSITY" not in deck.text
        assert deck.text.endswith("*END STEP\n") and "\r" not in deck.text
    again = build_abaqus_static_tension_qualification_deck(static)
    assert again.text == build_abaqus_static_tension_qualification_deck(static).text


def test_dependent_datacheck_decks_embed_dependent_cards():
    rate = run_abaqus_legacy_dependent_material(
        AbaqusLegacyDependentRequest(mode="strain_rate", strain_rates=(0.0, 2.0))
    )
    temp = run_abaqus_legacy_dependent_material(AbaqusLegacyDependentRequest(mode="temperature"))
    rate_deck = build_abaqus_dependent_datacheck_deck(rate)
    temp_deck = build_abaqus_dependent_datacheck_deck(temp)
    assert _material_block(rate_deck.text) == abaqus_legacy_dependent_material_text(
        rate, QUALIFICATION_MATERIAL_NAME
    )
    assert _material_block(temp_deck.text) == abaqus_legacy_dependent_material_text(
        temp, QUALIFICATION_MATERIAL_NAME
    )
    assert "*INITIAL CONDITIONS, TYPE=TEMPERATURE\nALLN, 20\n" in temp_deck.text
    assert "*INITIAL CONDITIONS" not in rate_deck.text
    assert rate_deck.case == "rate_datacheck" and temp_deck.case == "temperature_datacheck"


def test_uniaxial_boundary_conditions_do_not_confine_laterally():
    static = run_abaqus_legacy_material(AbaqusLegacyMaterialRequest())
    deck = build_abaqus_static_compression_qualification_deck(static)
    lines = deck.text.splitlines()
    nsets = {}
    for i, line in enumerate(lines):
        if line.startswith("*NSET, NSET="):
            nsets[line.split("=")[1]] = [int(v) for v in lines[i + 1].split(",")]
    nodes = {}
    for line in lines[lines.index("*NODE, NSET=ALLN") + 1 :]:
        if line.startswith("*"):
            break
        idx, *xyz = (float(v) for v in line.split(","))
        nodes[int(idx)] = xyz
    assert all(nodes[n][0] == 0.0 for n in nsets["X0"])
    assert all(nodes[n][0] == 1.0 for n in nsets["X1"])
    assert all(nodes[n][1] == 0.0 for n in nsets["Y0"])
    assert all(nodes[n][2] == 0.0 for n in nsets["Z0"])
    first_boundary = lines.index("*BOUNDARY")
    assert lines[first_boundary + 1 : first_boundary + 4] == ["X0, 1, 1", "Y0, 2, 2", "Z0, 3, 3"]
    # Only symmetry planes are constrained: nodes on y=1 / z=1 have no lateral constraint.
    assert not (set(nsets["Y0"]) & {n for n, c in nodes.items() if c[1] == 1.0})
    assert "*ELEMENT, TYPE=C3D8, ELSET=EALL" in deck.text
    load = lines[lines.index("*STATIC") + 3]
    assert load.startswith("X1, 1, 1, -")
    assert deck.input_summary["applied_nominal_strain"] == pytest.approx(-3 * 0.0022)


def test_tension_deck_targets_softening_branch():
    static = run_abaqus_legacy_material(AbaqusLegacyMaterialRequest())
    deck = build_abaqus_static_tension_qualification_deck(static)
    u = deck.input_summary["applied_displacement_mm"]
    w_max = static.tension_stiffening[-1]["cracking_displacement_mm"]
    peak = static.tension_stiffening[0]["stress_mpa"]
    assert u == pytest.approx(peak / static.elastic["E_mpa"] + 0.25 * w_max)
    assert deck.input_summary["damage_enabled"] is True
    for variable in ("S", "PEEQT", "DAMAGET", "DAMAGEC", "PEEQ"):
        assert variable in deck.text


def test_deck_inventory_and_hashes_are_deterministic():
    a = runner.qualification_decks()
    b = runner.qualification_decks()
    assert [d.case for d in a] == [
        "static_compression",
        "static_tension",
        "rate_datacheck",
        "temperature_datacheck",
    ]
    assert [d.sha256 for d in a] == [d.sha256 for d in b]
    assert all(runner.sanitize_job_name(d.job_name) == d.job_name for d in a)


# --------------------------------------------------------------------------- discovery


@pytest.mark.parametrize("name", ["job; rm -rf /", "a b", "", "../x", "x" * 65, "é"])
def test_job_names_are_sanitized(name):
    with pytest.raises(ValueError, match="Unsafe"):
        runner.sanitize_job_name(name)


def test_discovery_order_and_unavailable(monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)
    found = runner.discover_abaqus_command(None, {})
    assert found.command is None and "solver unavailable" in found.reason
    bad = runner.discover_abaqus_command(None, {"ABAQUS_COMMAND": "definitely-not-abaqus-x"})
    assert bad.command is None and "ABAQUS_COMMAND" in bad.reason
    env_cmd = runner.discover_abaqus_command(None, {"ABAQUS_COMMAND": _fake_command()})
    assert env_cmd.command is not None and env_cmd.command.source == "ABAQUS_COMMAND"
    explicit = runner.discover_abaqus_command(_fake_command(), {"ABAQUS_COMMAND": "nope"})
    assert explicit.command is not None and explicit.command.source == "--command"
    assert str(ROOT) not in explicit.command.display


def test_discovery_uses_path_lookup_without_guessing(monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda name: f"/opt/x/{name}")
    monkeypatch.setattr(runner.Path, "is_file", lambda self: False)
    found = runner.discover_abaqus_command(None, {})
    assert found.command is not None
    assert found.command.argv == ("/opt/x/abaqus",)
    assert found.command.display == "abaqus"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Abaqus 2025\nLicense ...", "2025"),
        ("Abaqus Release 2024 HF3", "2024 HF3"),
        ("Abaqus 6.14-2", "6.14-2"),
        ("Release 2023", "2023"),
        ("nothing useful", "unknown"),
    ],
)
def test_version_parsing(text, expected):
    assert runner.parse_abaqus_version(text) == expected


# --------------------------------------------------------------------------- states


def test_not_available_is_never_pass(monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)
    report = runner.run_abaqus_qualification(runner.QualificationOptions(environ={}))
    assert report["overall_state"] == "NOT_AVAILABLE"
    assert report["exit_code"] == runner.EXIT_NOT_AVAILABLE
    assert {c["state"] for c in report["cases"].values()} == {"NOT_AVAILABLE"}
    assert all(c["deck_sha256"] for c in report["cases"].values())
    json.loads(runner.report_json(report))


def _run_fake(monkeypatch, mode, tmp_path, **kwargs):
    monkeypatch.setenv("FAKE_ABAQUS_MODE", mode)
    options = runner.QualificationOptions(
        command=_fake_command(),
        environ={},
        timeout_s=kwargs.pop("timeout_s", 60.0),
        workdir=tmp_path / "work",
        **kwargs,
    )
    return runner.run_abaqus_qualification(options)


def test_fake_solver_full_pass(monkeypatch, tmp_path):
    report = _run_fake(monkeypatch, "pass", tmp_path)
    states = {k: v["state"] for k, v in report["cases"].items()}
    assert states == {
        "static_compression": "ANALYSIS_PASS",
        "static_tension": "ANALYSIS_PASS",
        "rate_datacheck": "DATACHECK_PASS",
        "temperature_datacheck": "DATACHECK_PASS",
    }
    assert report["overall_state"] == "PASS" and report["exit_code"] == 0
    assert report["detected_version"] == "2099"
    comp = report["cases"]["static_compression"]
    assert comp["checks"]["plasticity_activates"] and comp["checks"]["damage_activates"]
    assert [p["phase"] for p in comp["phases"]] == ["datacheck", "analysis", "postprocess"]
    assert ".odb" in comp["artifacts_present"]
    text = runner.report_json(report)
    assert str(tmp_path) not in text  # no private absolute paths
    assert len(text) < 60_000  # logs are not serialized wholesale


def test_fake_solver_datacheck_only(monkeypatch, tmp_path):
    report = _run_fake(monkeypatch, "pass", tmp_path, datacheck_only=True)
    assert {c["state"] for c in report["cases"].values()} == {"DATACHECK_PASS"}
    assert report["scope"] == "datacheck_only" and report["overall_state"] == "PASS"


def test_fake_solver_datacheck_failure(monkeypatch, tmp_path):
    report = _run_fake(monkeypatch, "datacheck_error", tmp_path)
    rate = report["cases"]["rate_datacheck"]
    assert rate["state"] == "DATACHECK_FAIL" and rate["return_code"] == 1
    assert any("***ERROR" in line for line in rate["diagnostics_excerpt"])
    assert report["cases"]["static_compression"]["state"] == "ANALYSIS_PASS"
    assert report["overall_state"] == "FAIL" and report["exit_code"] == runner.EXIT_FAIL


def test_fake_solver_error_marker_fails_even_with_zero_return_code(monkeypatch, tmp_path):
    report = _run_fake(monkeypatch, "analysis_error", tmp_path)
    tension = report["cases"]["static_tension"]
    assert tension["state"] == "ANALYSIS_FAIL" and tension["return_code"] == 0
    assert report["exit_code"] == runner.EXIT_FAIL


def test_fake_solver_timeout_is_analysis_fail(monkeypatch, tmp_path):
    report = _run_fake(monkeypatch, "timeout", tmp_path, timeout_s=2.0)
    comp = report["cases"]["static_compression"]
    assert comp["state"] == "ANALYSIS_FAIL"
    assert any("timed out" in line for line in comp["diagnostics_excerpt"])
    assert report["overall_state"] == "FAIL"


def test_fake_solver_postprocess_failure(monkeypatch, tmp_path):
    report = _run_fake(monkeypatch, "post_fail", tmp_path)
    assert report["cases"]["static_compression"]["state"] == "POSTPROCESS_FAIL"
    assert report["exit_code"] == runner.EXIT_FAIL


def test_fake_solver_response_check_failure(monkeypatch, tmp_path):
    report = _run_fake(monkeypatch, "bad_response", tmp_path)
    tension = report["cases"]["static_tension"]
    assert tension["state"] == "ANALYSIS_FAIL"
    assert tension["checks"]["softens_after_peak"] is False
    assert "response check failed: softens_after_peak" in tension["diagnostics_excerpt"]


def test_temp_workdir_removed_unless_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_ABAQUS_MODE", "pass")
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "auto"))
    report = runner.run_abaqus_qualification(
        runner.QualificationOptions(command=_fake_command(), environ={}, datacheck_only=True)
    )
    assert report["overall_state"] == "PASS"
    assert not (tmp_path / "auto").exists()


# --------------------------------------------------------------------------- evaluation


def test_evaluation_rejects_invalid_extract_and_non_finite_values():
    summary = {"table_peak_stress_mpa": 28.0, "damage_enabled": True}
    assert runner.evaluate_odb_extract({}, "compression", summary)["passed"] is False
    frames = [{"step_time": 1.0, "S11": -27.0, "PEEQ": None, "DAMAGEC": float("nan")}]
    result = runner.evaluate_odb_extract(
        {"schema_version": "abaqus_odb_extract.v1", "frames": frames}, "compression", summary
    )
    assert result["passed"] is False
    assert result["checks"]["DAMAGEC_finite"] is False
    assert result["checks"]["plasticity_activates"] is False


def test_evaluation_requires_final_time_and_sign():
    summary = {"table_peak_stress_mpa": 2.2, "damage_enabled": False}
    frames = [{"step_time": 0.5, "S11": -1.0, "PEEQT": 0.0}]
    result = runner.evaluate_odb_extract(
        {"schema_version": "abaqus_odb_extract.v1", "frames": frames}, "tension", summary
    )
    assert result["checks"]["reached_final_step_time"] is False
    assert result["checks"]["stress_is_tensile"] is False


# --------------------------------------------------------------------------- CLI / isolation


def _cli(*args, env=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        check=False,
    )


def test_cli_not_available_exit_code_and_report(tmp_path):
    env = {"PATH": str(tmp_path), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    report_path = tmp_path / "report.json"
    proc = _cli("--json-report", str(report_path), env=env)
    assert proc.returncode == runner.EXIT_NOT_AVAILABLE, proc.stderr
    assert "Abaqus external qualification: NOT_AVAILABLE" in proc.stdout
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall_state"] == "NOT_AVAILABLE"


def test_cli_fake_solver_pass_and_write_decks(tmp_path):
    env = dict(__import__("os").environ, FAKE_ABAQUS_MODE="pass")
    proc = _cli("--command", _fake_command(), "--datacheck-only", env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Abaqus external qualification: PASS" in proc.stdout
    decks = tmp_path / "decks"
    proc = _cli("--write-decks", str(decks))
    assert proc.returncode == 0
    assert sorted(p.name for p in decks.iterdir()) == [
        "q1_rate_datacheck.inp",
        "q1_static_compression.inp",
        "q1_static_tension.inp",
        "q1_temperature_datacheck.inp",
    ]
    assert _cli("--timeout", "-1").returncode == runner.EXIT_USAGE


def test_postprocessor_is_abaqus_only_and_never_imported():
    tree = ast.parse(POST.read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert {"odbAccess", "abaqusConstants"} <= imported
    assert importlib.util.find_spec("odbAccess") is None
    package_sources = "\n".join(
        p.read_text(encoding="utf-8") for p in (ROOT / "cdp_generator").rglob("*.py")
    )
    assert "import odbAccess" not in package_sources
    assert "from odbAccess" not in package_sources
    for path in (ROOT / "cdp_generator").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.keyword) and node.arg == "shell":
                pytest.fail(f"subprocess shell= keyword used in {path.name}")
