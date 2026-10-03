"""Optional external Abaqus qualification gate (ABAQUS-Q1).

Usage::

    uv run python scripts/qualify_abaqus.py [--command CMD] [--timeout S] [--datacheck-only]
                                            [--keep-workdir] [--workdir DIR]
                                            [--json-report PATH] [--write-decks DIR]

Solver discovery: ``--command``, then the ``ABAQUS_COMMAND`` environment variable, then
``abaqus`` on PATH. Installation directories are never guessed.

Exit codes:

* 0 — PASS: every required case reached its target state;
* 1 — FAIL: Abaqus was found but at least one case failed (datacheck/analysis/postprocess);
* 2 — usage error;
* 3 — NOT_AVAILABLE: no Abaqus solver could be resolved (this is never a pass).

``--write-decks DIR`` only writes the generated qualification decks and exits 0; it does not
contact a solver and does not claim qualification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdp_generator.application.abaqus_runner import (  # noqa: E402
    DEFAULT_TIMEOUT_S,
    EXIT_NOT_AVAILABLE,
    EXIT_PASS,
    EXIT_USAGE,
    QualificationOptions,
    qualification_decks,
    report_json,
    run_abaqus_qualification,
)

POSTPROCESS = ROOT / "qualification" / "abaqus" / "postprocess_odb.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--command", help="Abaqus launcher (overrides ABAQUS_COMMAND).")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Per-phase timeout in seconds (default %(default)s).",
    )
    parser.add_argument(
        "--datacheck-only", action="store_true", help="Only run datacheck for every deck."
    )
    parser.add_argument(
        "--keep-workdir", action="store_true", help="Keep the solver working directory."
    )
    parser.add_argument(
        "--workdir", type=Path, help="Use this (new or empty) directory instead of a temporary one."
    )
    parser.add_argument("--json-report", type=Path, help="Write the JSON report to PATH.")
    parser.add_argument(
        "--write-decks",
        type=Path,
        help="Write the generated decks to DIR and exit without a solver.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_USAGE if exc.code else EXIT_PASS
    if args.timeout is None or not args.timeout > 0:
        print("--timeout must be > 0", file=sys.stderr)
        return EXIT_USAGE
    if args.write_decks is not None:
        args.write_decks.mkdir(parents=True, exist_ok=True)
        for deck in qualification_decks():
            path = args.write_decks / f"{deck.job_name}.inp"
            path.write_bytes(deck.text.encode("utf-8"))
            print(f"{deck.case}: {path.name} sha256={deck.sha256}")
        print("Decks written; no solver was run (not a qualification result).")
        return EXIT_PASS
    if args.workdir is not None and args.workdir.exists() and any(args.workdir.iterdir()):
        print("--workdir must be new or empty", file=sys.stderr)
        return EXIT_USAGE

    report = run_abaqus_qualification(
        QualificationOptions(
            command=args.command,
            timeout_s=args.timeout,
            datacheck_only=args.datacheck_only,
            keep_workdir=args.keep_workdir or args.workdir is not None,
            workdir=args.workdir,
            postprocess_script=POSTPROCESS,
        )
    )
    text = report_json(report)
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_bytes(text.encode("utf-8"))
    print(f"Abaqus command: {report['abaqus_command'] or '-'} ({report['discovery']})")
    print(f"Detected version: {report['detected_version']}")
    for case, record in report["cases"].items():
        print(f"  {case:24s} {record['state']}")
        for line in record["diagnostics_excerpt"][:5]:
            print(f"      {line}")
    if report["exit_code"] == EXIT_NOT_AVAILABLE:
        print("Abaqus external qualification: NOT_AVAILABLE")
    else:
        print(f"Abaqus external qualification: {report['overall_state']}")
    return int(report["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
