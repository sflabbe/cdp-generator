# Abaqus external qualification gate (ABAQUS-Q1)

Optional, solver-backed qualification of the generated Abaqus CDP material cards. The ordinary
test suite never needs Abaqus; this gate is opt-in.

```bash
uv run python scripts/qualify_abaqus.py                 # datacheck + static analyses + ODB
uv run python scripts/qualify_abaqus.py --datacheck-only
uv run python scripts/qualify_abaqus.py --json-report build/abaqus_q1_report.json
uv run python scripts/qualify_abaqus.py --write-decks build/q1_decks   # decks only, no solver
uv run pytest -q -m abaqus_external                     # same gate through pytest
```

## Solver discovery

1. `--command "<launcher>"`
2. `ABAQUS_COMMAND` environment variable, e.g. `abaqus`, `abq2025`,
   `"C:\SIMULIA\Commands\abq2025.bat"`
3. `abaqus` resolved on PATH

Installation directories are never guessed and PATH is never modified. All solver calls use
argument lists (no `shell=True`); job names are restricted to `[A-Za-z0-9_-]+`; every job runs
in an isolated temporary directory that is deleted unless `--keep-workdir`/`--workdir` is used.

## States and exit codes

| Case state | Meaning |
| --- | --- |
| `NOT_AVAILABLE` | no solver could be resolved — **never a pass** |
| `DATACHECK_PASS` / `DATACHECK_FAIL` | `abaqus job=<job> input=<job>.inp datacheck interactive` |
| `ANALYSIS_PASS` / `ANALYSIS_FAIL` | `... analysis interactive`, completion marker, `.odb`, response checks; timeouts are `ANALYSIS_FAIL` |
| `POSTPROCESS_FAIL` | `abaqus python postprocess_odb.py <job>.odb <json>` failed or wrote no readable JSON |

A phase fails on a non-zero return code, on a timeout, or on an explicit `***ERROR` marker in
`.dat/.msg/.sta/.log`/launcher output (launcher return codes vary between installations).

| Exit code | Overall |
| --- | --- |
| 0 | PASS — every case reached its target state |
| 1 | FAIL — solver found, at least one case failed |
| 2 | usage error |
| 3 | NOT_AVAILABLE |

## Decks

| Case | Material | Gate |
| --- | --- | --- |
| `static_compression` | M4 static default, both damage branches | datacheck + full analysis + ODB checks |
| `static_tension` | M4 static default, both damage branches | datacheck + full analysis + ODB checks |
| `rate_datacheck` | M5 dependent, rates 0/2/30/100 1/s, damage omitted | datacheck |
| `temperature_datacheck` | M5 dependent, 20…1100 °C, damage omitted, initial T = 20 °C | datacheck |

Each deck is one fully integrated `C3D8` brick of 1 mm (N-mm-MPa-s). BCs are symmetry planes
only (`U1=0` at x=0, `U2=0` at y=0, `U3=0` at z=0); the faces y=1 and z=1 are free, so the
state is uniaxial stress without lateral confinement. Loading is a prescribed `U1` on x=1 in
one Abaqus/Standard `*STATIC` step (max increment 0.01). Compression goes to 3·e_c1 nominal
strain; tension goes to `f_t/E0 + 0.25·w_max`, i.e. into the softening branch. No
stabilization, extra viscosity or density is added: the material card is embedded byte-for-byte
as produced by the canonical builder. Field output: `S, E, PE, PEEQ, PEEQT, DAMAGEC, DAMAGET,
U, RF` every increment.

## ODB postprocessing

`postprocess_odb.py` runs only under `abaqus python` (Python 2.7 or 3 interpreters). It imports
`odbAccess` and is never imported by the package or tests. It writes
`abaqus_odb_extract.v1` JSON: per frame step time, element-mean `S11`, `E11`, `PE11`, `PEEQ`,
`PEEQT`, `DAMAGEC`, `DAMAGET`, summed `RF1` on x=1 and `U1` of node 7; missing variables are
`null`.

## Acceptance checks (normal Python)

Compression: final step time reached; outputs finite; stress compressive; peak |S11| within
0.5–1.5 × table peak; `PEEQ` > 0 at the end; `DAMAGEC` ≥ 0 and > 0 at the end when damage is
exported.

Tension: final step time reached; outputs finite; positive peak within 0.5–1.5 × table
cracking stress; final stress < 0.9 × peak (softening); `PEEQT` > 0; `DAMAGET` ≥ 0 and > 0 at the
end when damage is exported.

Peak/table ratios are reported as information, not as a pointwise reproduction gate.

## What a PASS does and does not establish

It establishes that the generated keywords are accepted, the material initializes, the two
single-element loading paths execute, and the requested state variables behave consistently.

It does **not** establish experimental validity, structural-model validity, general
convergence, normative EC2/fib calibration, or accuracy for arbitrary multiaxial states. The
dependent decks are datacheck-only in M5: their nonlinear response is not solver-validated.

## Known harness limitations

- On timeout the launcher process is killed; some installations spawn detached solver
  processes that may need manual cleanup.
- Error detection relies on documented `***ERROR` markers plus return codes; exotic launcher
  wrappers may need `--keep-workdir` for manual inspection.
- Version detection uses `abaqus information=release`; unparseable output is reported as
  `unknown` and does not fail qualification.
