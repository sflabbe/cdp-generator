"""Optional external Abaqus qualification runner (ABAQUS-Q1).

Standard-library only. Nothing here imports Abaqus; the solver is reached exclusively through
``subprocess`` argument lists (never ``shell=True``). The only Abaqus Python script ever executed
is the repository-owned ``qualification/abaqus/postprocess_odb.py``.

States are deliberately explicit: an unavailable solver is ``NOT_AVAILABLE``, never a pass, and a
generated deck is never reported as solver-qualified.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .abaqus_decks import (
    QualificationDeck,
    build_abaqus_dependent_datacheck_deck,
    build_abaqus_static_compression_qualification_deck,
    build_abaqus_static_tension_qualification_deck,
)
from .abaqus_dependent import AbaqusLegacyDependentRequest, run_abaqus_legacy_dependent_material
from .abaqus_legacy import AbaqusLegacyMaterialRequest, run_abaqus_legacy_material

REPORT_SCHEMA_VERSION = "abaqus_qualification_report.v1"
ODB_EXTRACT_SCHEMA_VERSION = "abaqus_odb_extract.v1"
ABAQUS_COMMAND_ENV = "ABAQUS_COMMAND"
DEFAULT_TIMEOUT_S = 600.0
VERSION_TIMEOUT_S = 120.0
EXCERPT_MAX_LINES = 20
EXCERPT_MAX_CHARS = 300
POSTPROCESS_SCRIPT = (
    Path(__file__).resolve().parents[2] / "qualification" / "abaqus" / "postprocess_odb.py"
)
_JOB_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
_ERROR_MARKERS = (
    "***ERROR",
    "*** ERROR",
    "ABAQUS/ANALYSIS EXITED WITH ERROR",
    "ANALYSIS INPUT FILE PROCESSOR EXITED WITH AN ERROR",
)
_SUCCESS_MARKER = "THE ANALYSIS HAS COMPLETED SUCCESSFULLY"
_TEXT_OUTPUTS = (".dat", ".msg", ".sta", ".log")

#: Documented exit codes of ``scripts/qualify_abaqus.py``.
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_NOT_AVAILABLE = 3


class QualificationState(StrEnum):
    NOT_AVAILABLE = "NOT_AVAILABLE"
    DATACHECK_PASS = "DATACHECK_PASS"
    DATACHECK_FAIL = "DATACHECK_FAIL"
    ANALYSIS_PASS = "ANALYSIS_PASS"
    ANALYSIS_FAIL = "ANALYSIS_FAIL"
    POSTPROCESS_FAIL = "POSTPROCESS_FAIL"


class OverallState(StrEnum):
    NOT_AVAILABLE = "NOT_AVAILABLE"
    PASS = "PASS"
    FAIL = "FAIL"


def sanitize_job_name(name: str) -> str:
    """Accept only conservative job names; never build commands from free text."""

    if not isinstance(name, str) or not _JOB_NAME.fullmatch(name) or len(name) > 64:
        raise ValueError(f"Unsafe Abaqus job name: {name!r}; allowed pattern [A-Za-z0-9_-]+.")
    return name


@dataclass(frozen=True)
class SolverCommand:
    """Resolved launcher argument prefix, e.g. ``["/opt/abaqus/Commands/abaqus"]``."""

    argv: tuple[str, ...]
    source: str  # "ABAQUS_COMMAND" | "--command" | "PATH"

    @property
    def display(self) -> str:
        # Avoid serializing absolute private machine paths into reports.
        return " ".join(
            Path(part).name if os.sep in part or "/" in part else part for part in self.argv
        )


@dataclass(frozen=True)
class Discovery:
    command: SolverCommand | None
    reason: str


def _split_command(text: str) -> list[str]:
    parts = shlex.split(text, posix=os.name != "nt")
    return [part[1:-1] if len(part) >= 2 and part[0] == part[-1] == '"' else part for part in parts]


def _resolve(argv: list[str]) -> list[str] | None:
    if not argv:
        return None
    first = argv[0]
    candidate = Path(first)
    if candidate.is_file():
        return [str(candidate), *argv[1:]]
    found = shutil.which(first)
    return [found, *argv[1:]] if found else None


def discover_abaqus_command(
    explicit: str | None = None,
    environ: dict[str, str] | None = None,
) -> Discovery:
    """Deterministic discovery: explicit option, then ``ABAQUS_COMMAND``, then ``abaqus`` on PATH.

    Installation directories are never guessed and PATH is never mutated.
    """

    env = os.environ if environ is None else environ
    for text, source in (
        (explicit, "--command"),
        (env.get(ABAQUS_COMMAND_ENV), ABAQUS_COMMAND_ENV),
    ):
        if text is not None and text.strip():
            argv = _resolve(_split_command(text))
            if argv is None:
                return Discovery(None, f"{source} is set but not executable/resolvable")
            return Discovery(SolverCommand(tuple(argv), source), f"resolved from {source}")
    argv = _resolve(["abaqus"])
    if argv is None:
        return Discovery(None, "solver unavailable: ABAQUS_COMMAND unset and 'abaqus' not on PATH")
    return Discovery(SolverCommand(tuple(argv), "PATH"), "resolved 'abaqus' from PATH")


@dataclass
class ProcessOutcome:
    return_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    launch_error: str | None = None


Runner = Callable[[Sequence[str], Path, float], ProcessOutcome]


def run_process(argv: Sequence[str], cwd: Path, timeout: float) -> ProcessOutcome:
    start = time.monotonic()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        return ProcessOutcome(None, out, "", True, time.monotonic() - start)
    except OSError as exc:
        return ProcessOutcome(None, "", "", False, time.monotonic() - start, str(exc))
    return ProcessOutcome(
        completed.returncode,
        completed.stdout,
        completed.stderr,
        False,
        time.monotonic() - start,
    )


_VERSION_PATTERNS = (
    re.compile(r"Abaqus\s+(?:Release\s+)?(\d{4}(?:\s*(?:HF|FP\.CFA\.|\.)\s*[\w.-]+)?)", re.I),
    re.compile(r"Abaqus\s+(?:Release\s+)?(\d+\.\d+(?:-\d+)?)", re.I),
    re.compile(r"Release\s+(\d{4}[\w.-]*|\d+\.\d+(?:-\d+)?)", re.I),
)


def parse_abaqus_version(text: str) -> str:
    for pattern in _VERSION_PATTERNS:
        match = pattern.search(text)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return "unknown"


def detect_version(command: SolverCommand, runner: Runner, workdir: Path) -> str:
    outcome = runner([*command.argv, "information=release"], workdir, VERSION_TIMEOUT_S)
    if outcome.launch_error or outcome.timed_out:
        return "unknown"
    return parse_abaqus_version(outcome.stdout + "\n" + outcome.stderr)


def _error_lines(text: str) -> list[str]:
    return [
        line.strip()[:EXCERPT_MAX_CHARS]
        for line in text.splitlines()
        if any(marker in line.upper() for marker in _ERROR_MARKERS)
    ]


def _job_text_outputs(casedir: Path, job: str) -> str:
    chunks = []
    for suffix in _TEXT_OUTPUTS:
        path = casedir / f"{job}{suffix}"
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _artifacts(casedir: Path, job: str) -> list[str]:
    return sorted(
        suffix
        for suffix in (
            ".inp",
            ".dat",
            ".msg",
            ".sta",
            ".log",
            ".odb",
            ".prt",
            ".com",
            ".odb_extract.json",
        )
        if (casedir / f"{job}{suffix}").exists()
    )


def _excerpt(lines: list[str]) -> list[str]:
    return lines[:EXCERPT_MAX_LINES]


def _phase(
    *,
    command: SolverCommand,
    runner: Runner,
    casedir: Path,
    job: str,
    phase: str,
    timeout: float,
) -> dict[str, Any]:
    argv = [*command.argv, f"job={job}", f"input={job}.inp", phase, "interactive"]
    outcome = runner(argv, casedir, timeout)
    outputs = _job_text_outputs(casedir, job)
    errors = _error_lines(outputs + "\n" + outcome.stdout + "\n" + outcome.stderr)
    passed = (
        outcome.launch_error is None
        and not outcome.timed_out
        and outcome.return_code == 0
        and not errors
    )
    if phase == "analysis" and passed:
        completed_marker = (
            _SUCCESS_MARKER in outputs.upper()
            or _SUCCESS_MARKER in (outcome.stdout + outcome.stderr).upper()
        )
        odb = (casedir / f"{job}.odb").exists()
        passed = completed_marker and odb
        if not completed_marker:
            errors.append("analysis completion marker not found in .sta/.msg/.log/stdout")
        if not odb:
            errors.append(f"{job}.odb not produced")
    if outcome.timed_out:
        errors.insert(0, f"{phase} timed out after {timeout:g} s")
    if outcome.launch_error:
        errors.insert(0, f"launch error: {outcome.launch_error[:EXCERPT_MAX_CHARS]}")
    if outcome.return_code not in (0, None):
        errors.insert(0, f"{phase} return code {outcome.return_code}")
    return {
        "phase": phase,
        "passed": passed,
        "return_code": outcome.return_code,
        "timed_out": outcome.timed_out,
        "duration_s": round(outcome.duration_s, 3),
        "diagnostics": _excerpt(errors),
    }


# --------------------------------------------------------------------------- response checks


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _series(frames: list[dict[str, Any]], key: str) -> list[float] | None:
    values = [frame.get(key) for frame in frames]
    if all(v is None for v in values):
        return None
    return [float(v) if v is not None and _finite(v) else math.nan for v in values]


def evaluate_odb_extract(
    extract: dict[str, Any], loading: str, input_summary: dict[str, Any]
) -> dict[str, Any]:
    """Pure acceptance checks on the normal-Python JSON written by ``postprocess_odb.py``.

    The checks establish executable compatibility (completion, finite outputs, sign/order,
    activation of plasticity/damage in the intended branch). They are deliberately not a
    pointwise table-reproduction gate. Landmark ratios are reported for information.
    """

    frames = list(extract.get("frames") or [])
    checks: dict[str, Any] = {}
    info: dict[str, Any] = {}
    if extract.get("schema_version") != ODB_EXTRACT_SCHEMA_VERSION or not frames:
        return {"passed": False, "checks": {"extract_valid": False}, "info": info}
    checks["extract_valid"] = True
    final_time = frames[-1].get("step_time")
    checks["reached_final_step_time"] = _finite(final_time) and float(final_time) >= 1.0 - 1e-6
    s11 = _series(frames, "S11") or []
    checks["stress_output_present_and_finite"] = bool(s11) and all(math.isfinite(v) for v in s11)
    for key in ("E11", "PEEQ", "PEEQT", "DAMAGEC", "DAMAGET", "RF1_X1", "U1_REFNODE"):
        series = _series(frames, key)
        if series is not None:
            checks[f"{key}_finite"] = all(math.isfinite(v) for v in series)
    peak_table = float(input_summary["table_peak_stress_mpa"])
    damage_enabled = bool(input_summary.get("damage_enabled"))
    if loading == "compression":
        peak = -min(s11) if s11 else math.nan
        info["peak_compressive_stress_mpa"] = peak
        info["peak_ratio_to_table"] = peak / peak_table if peak_table else math.nan
        checks["stress_is_compressive"] = bool(s11) and min(s11) < 0.0
        checks["peak_magnitude_order"] = 0.5 <= info["peak_ratio_to_table"] <= 1.5
        peeq = _series(frames, "PEEQ")
        checks["plasticity_activates"] = peeq is not None and peeq[-1] > 0.0
        if damage_enabled:
            dc = _series(frames, "DAMAGEC")
            checks["damage_nonnegative"] = dc is not None and min(dc) >= 0.0
            checks["damage_activates"] = dc is not None and dc[-1] > 0.0
    elif loading == "tension":
        peak = max(s11) if s11 else math.nan
        info["peak_tensile_stress_mpa"] = peak
        info["peak_ratio_to_table"] = peak / peak_table if peak_table else math.nan
        info["final_tensile_stress_mpa"] = s11[-1] if s11 else math.nan
        checks["stress_is_tensile"] = bool(s11) and peak > 0.0
        checks["peak_magnitude_order"] = 0.5 <= info["peak_ratio_to_table"] <= 1.5
        checks["softens_after_peak"] = bool(s11) and s11[-1] < 0.9 * peak
        peeqt = _series(frames, "PEEQT")
        checks["cracking_activates"] = peeqt is not None and peeqt[-1] > 0.0
        if damage_enabled:
            dt = _series(frames, "DAMAGET")
            checks["damage_nonnegative"] = dt is not None and min(dt) >= 0.0
            checks["damage_activates"] = dt is not None and dt[-1] > 0.0
    else:
        raise ValueError(f"Unknown loading {loading!r}")
    info = {k: (v if _finite(v) else None) for k, v in info.items()}
    return {"passed": all(bool(v) for v in checks.values()), "checks": checks, "info": info}


# --------------------------------------------------------------------------- orchestration


@dataclass
class QualificationOptions:
    command: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    datacheck_only: bool = False
    keep_workdir: bool = False
    workdir: Path | None = None
    environ: dict[str, str] | None = None
    runner: Runner = field(default=run_process)
    postprocess_script: Path = POSTPROCESS_SCRIPT


def qualification_decks() -> list[QualificationDeck]:
    """Inventory of decks: static full-analysis targets plus dependent datacheck decks.

    Static decks use the M4 default material with both damage branches enabled. Dependent
    decks use the default legacy cases with ``damage_policy="omit"`` (the dependent default).
    """

    static = run_abaqus_legacy_material(AbaqusLegacyMaterialRequest())
    rate = run_abaqus_legacy_dependent_material(
        AbaqusLegacyDependentRequest(mode="strain_rate", strain_rates=(0.0, 2.0, 30.0, 100.0))
    )
    temperature = run_abaqus_legacy_dependent_material(
        AbaqusLegacyDependentRequest(mode="temperature")
    )
    return [
        build_abaqus_static_compression_qualification_deck(static),
        build_abaqus_static_tension_qualification_deck(static),
        build_abaqus_dependent_datacheck_deck(rate),
        build_abaqus_dependent_datacheck_deck(temperature),
    ]


def _case_record(deck: QualificationDeck) -> dict[str, Any]:
    return {
        "state": None,
        "return_code": None,
        "deck_sha256": deck.sha256,
        "job_name": deck.job_name,
        "input_summary": deck.input_summary,
        "phases": [],
        "checks": {},
        "info": {},
        "artifacts_present": [],
        "diagnostics_excerpt": [],
    }


def _run_case(
    deck: QualificationDeck,
    command: SolverCommand,
    options: QualificationOptions,
    root: Path,
) -> dict[str, Any]:
    job = sanitize_job_name(deck.job_name)
    casedir = root / sanitize_job_name(deck.case)
    casedir.mkdir(parents=True, exist_ok=False)
    (casedir / f"{job}.inp").write_bytes(deck.text.encode("utf-8"))
    record = _case_record(deck)

    datacheck = _phase(
        command=command,
        runner=options.runner,
        casedir=casedir,
        job=job,
        phase="datacheck",
        timeout=options.timeout_s,
    )
    record["phases"].append(datacheck)
    record["return_code"] = datacheck["return_code"]
    full = deck.loading in {"compression", "tension"} and not options.datacheck_only
    if not datacheck["passed"]:
        record["state"] = QualificationState.DATACHECK_FAIL.value
        record["diagnostics_excerpt"] = datacheck["diagnostics"]
    elif not full:
        record["state"] = QualificationState.DATACHECK_PASS.value
    else:
        analysis = _phase(
            command=command,
            runner=options.runner,
            casedir=casedir,
            job=job,
            phase="analysis",
            timeout=options.timeout_s,
        )
        record["phases"].append(analysis)
        record["return_code"] = analysis["return_code"]
        if not analysis["passed"]:
            record["state"] = QualificationState.ANALYSIS_FAIL.value
            record["diagnostics_excerpt"] = analysis["diagnostics"]
        else:
            record.update(_postprocess(deck, command, options, casedir, job))
    record["artifacts_present"] = _artifacts(casedir, job)
    return record


def _postprocess(
    deck: QualificationDeck,
    command: SolverCommand,
    options: QualificationOptions,
    casedir: Path,
    job: str,
) -> dict[str, Any]:
    script = options.postprocess_script
    out_json = casedir / f"{job}.odb_extract.json"
    argv = [*command.argv, "python", str(script), f"{job}.odb", out_json.name]
    outcome = options.runner(argv, casedir, options.timeout_s)
    phase: dict[str, Any] = {
        "phase": "postprocess",
        "return_code": outcome.return_code,
        "timed_out": outcome.timed_out,
        "duration_s": round(outcome.duration_s, 3),
    }
    problems: list[str] = []
    if outcome.launch_error or outcome.timed_out or outcome.return_code != 0:
        problems.append(
            f"postprocess failed (rc={outcome.return_code}, timed_out={outcome.timed_out})"
        )
        problems += [line[:EXCERPT_MAX_CHARS] for line in outcome.stderr.splitlines()[-5:]]
    extract: dict[str, Any] | None = None
    if not problems:
        try:
            extract = json.loads(out_json.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"ODB extract JSON unreadable: {type(exc).__name__}")
    if extract is None or problems:
        phase["passed"] = False
        phase["diagnostics"] = _excerpt(problems)
        return {
            "state": QualificationState.POSTPROCESS_FAIL.value,
            "phases_extra": [phase],
            "diagnostics_excerpt": _excerpt(problems),
        }
    evaluation = evaluate_odb_extract(extract, deck.loading, deck.input_summary)
    phase["passed"] = True
    phase["diagnostics"] = []
    state = (
        QualificationState.ANALYSIS_PASS
        if evaluation["passed"]
        else QualificationState.ANALYSIS_FAIL
    )
    failed = [name for name, ok in evaluation["checks"].items() if not ok]
    return {
        "state": state.value,
        "phases_extra": [phase],
        "checks": evaluation["checks"],
        "info": evaluation["info"],
        "available_variables": sorted(extract.get("available_variables") or []),
        "diagnostics_excerpt": [f"response check failed: {name}" for name in failed],
    }


def _target_state(deck: QualificationDeck, datacheck_only: bool) -> QualificationState:
    if deck.loading in {"compression", "tension"} and not datacheck_only:
        return QualificationState.ANALYSIS_PASS
    return QualificationState.DATACHECK_PASS


def run_abaqus_qualification(options: QualificationOptions | None = None) -> dict[str, Any]:
    """Discover the solver, generate decks, datacheck/analyse/postprocess; return the report."""

    options = options or QualificationOptions()
    decks = qualification_decks()
    discovery = discover_abaqus_command(options.command, options.environ)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "abaqus_command": discovery.command.display if discovery.command else None,
        "abaqus_command_source": discovery.command.source if discovery.command else None,
        "discovery": discovery.reason,
        "detected_version": None,
        "platform": f"{platform.system()} {platform.machine()}".strip(),
        "scope": "datacheck_only" if options.datacheck_only else "full",
        "timeout_s": options.timeout_s,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": {},
        "overall_state": None,
        "exit_code": None,
        "claims": (
            "A PASS establishes only that the generated keywords are accepted, the material "
            "initializes, the single-element loading paths execute and the requested state "
            "variables behave consistently. It does not establish experimental, structural, "
            "normative or multiaxial validity."
        ),
    }
    if discovery.command is None:
        for deck in decks:
            record = _case_record(deck)
            record["state"] = QualificationState.NOT_AVAILABLE.value
            report["cases"][deck.case] = record
        report["detected_version"] = "unknown"
        report["overall_state"] = OverallState.NOT_AVAILABLE.value
        report["exit_code"] = EXIT_NOT_AVAILABLE
        return report

    created = options.workdir is None
    root = (
        Path(tempfile.mkdtemp(prefix="cdp_abaqus_q1_"))
        if options.workdir is None
        else Path(options.workdir)
    )
    root.mkdir(parents=True, exist_ok=True)
    try:
        report["detected_version"] = detect_version(discovery.command, options.runner, root)
        for deck in decks:
            record = _run_case(deck, discovery.command, options, root)
            record["phases"].extend(record.pop("phases_extra", []))
            report["cases"][deck.case] = record
        passed = all(
            report["cases"][deck.case]["state"] == _target_state(deck, options.datacheck_only).value
            for deck in decks
        )
        report["overall_state"] = (OverallState.PASS if passed else OverallState.FAIL).value
        report["exit_code"] = EXIT_PASS if passed else EXIT_FAIL
        if options.keep_workdir:
            report["workdir_kept"] = True
    finally:
        if created and not options.keep_workdir:
            shutil.rmtree(root, ignore_errors=True)
    return report


def report_json(report: dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
