"""Deterministic stand-in for the Abaqus launcher used ONLY by runner unit tests.

It mimics the documented command forms (``information=release``, ``job=.. input=..
datacheck|analysis interactive`` and ``python <script> <odb> <out>``) and writes the text
artifacts the runner inspects. Behaviour is selected with FAKE_ABAQUS_MODE. It is not Abaqus
and nothing produced by it is a qualification result.
"""

import json
import os
import sys
import time

MODE = os.environ.get("FAKE_ABAQUS_MODE", "pass")


def _write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _frames(job):
    frames = []
    n = 20
    for i in range(n + 1):
        t = i / n
        if "compression" in job:
            peak = 28.0
            s = -peak * min(1.0, 2.0 * t) * (1.0 if t <= 0.5 else (1.0 - 0.5 * (t - 0.5)))
            frames.append(
                {
                    "step_time": t,
                    "S11": s,
                    "E11": -0.0066 * t,
                    "PE11": -0.003 * t,
                    "PEEQ": max(0.0, t - 0.2) * 0.004,
                    "PEEQT": 0.0,
                    "DAMAGEC": max(0.0, t - 0.5) * 0.4,
                    "DAMAGET": 0.0,
                    "RF1_X1": s,
                    "U1_REFNODE": -0.0066 * t,
                }
            )
        else:
            peak = 2.21
            soft = MODE == "bad_response"
            s = peak * (2.0 * t if t <= 0.5 else (1.0 if soft else 1.0 - 1.6 * (t - 0.5)))
            frames.append(
                {
                    "step_time": t,
                    "S11": s,
                    "E11": 0.075 * t,
                    "PE11": 0.0,
                    "PEEQ": 0.0,
                    "PEEQT": max(0.0, t - 0.5) * 0.07,
                    "DAMAGEC": 0.0,
                    "DAMAGET": max(0.0, t - 0.5) * 1.5,
                    "RF1_X1": s,
                    "U1_REFNODE": 0.075 * t,
                }
            )
    return frames


def main(argv):
    if argv == ["information=release"]:
        print("Abaqus 2099 (fake test launcher)")
        return 0
    if argv and argv[0] == "python":
        _, _script, _odb, out = argv
        if MODE == "post_fail":
            sys.stderr.write("Traceback: fake odbAccess failure\n")
            return 1
        job = _odb[: -len(".odb")]
        doc = {
            "schema_version": "abaqus_odb_extract.v1",
            "step": "LOAD",
            "frame_count": 21,
            "available_variables": ["S11", "PEEQ"],
            "frames": _frames(job),
        }
        _write(out, json.dumps(doc))
        return 0
    args = dict(a.split("=", 1) for a in argv if "=" in a)
    flags = [a for a in argv if "=" not in a]
    job = args["job"]
    assert os.path.isfile(args["input"]), "input deck missing"
    phase = "datacheck" if "datacheck" in flags else "analysis"
    if MODE == "timeout" and phase == "analysis":
        time.sleep(30)
    if MODE == "datacheck_error" and "rate" in job:
        _write(job + ".dat", "***ERROR: fake rate-dependent table rejected\n")
        return 1
    if MODE == "analysis_error" and phase == "analysis" and "tension" in job:
        _write(job + ".msg", "***ERROR: TOO MANY ATTEMPTS MADE FOR THIS INCREMENT\n")
        _write(job + ".sta", "THE ANALYSIS HAS NOT BEEN COMPLETED\n")
        return 0  # launcher return codes vary; text markers must still fail the phase
    _write(job + ".dat", "fake data file\n")
    if phase == "analysis":
        _write(job + ".sta", " THE ANALYSIS HAS COMPLETED SUCCESSFULLY\n")
        _write(job + ".odb", "fake odb")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
