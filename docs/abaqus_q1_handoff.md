# ABAQUS-Q1 / WEB-M5 handoff

## Status

Implemented against the integrated WEB-M4 snapshot `ade662488cd865a67e30f5dd7a632490770b659d`
(ZIP without Git metadata; delivered as one patch). No commit, push or tag was created.

```text
legacy kernel
  → static (M4, unchanged) / rate-dependent / temperature-dependent Abaqus backend
  → semantic + documented Abaqus validity checks (per family)
  → complete minimal qualification decks
  → optional Abaqus datacheck / analysis / ODB postprocess
  → machine-readable qualification report
```

**Abaqus external qualification: NOT_AVAILABLE** — no Abaqus installation exists in the
implementation environment. Nothing in this handoff claims a solver pass.

## Public APIs added (`cdp_generator.application`)

| API | Purpose |
| --- | --- |
| `AbaqusLegacyDependentRequest` | typed request: `mode` (`strain_rate`/`temperature`), legacy inputs, `strain_rates`, `tension_law`, `damage_policy` (`omit` default / `reference_damage`), REF LENGTH, eccentricity, viscosity |
| `AbaqusLegacyDependentResult` | strict JSON `abaqus_legacy_dependent_material_result.v1`, workflow `legacy_abaqus_cdp_dependent` |
| `run_abaqus_legacy_dependent_material` | build + validate every family |
| `AbaqusLegacyDependentValidationError` | subclass of `AbaqusLegacyValidationError`; `.failures` = mode, value, unit, branch, failed condition |
| `abaqus_legacy_dependent_material_text` / `build_abaqus_legacy_dependent_material_text` | deterministic keyword text |
| `rate_mapping_audit_rows`, `temperature_elastic_audit_rows` | read-only audit tables |
| `AbaqusDamagePolicy`, `AbaqusDependentMode` | enums |

Additional modules: `application/abaqus_decks.py` (pure deck generators
`build_abaqus_static_compression_qualification_deck`,
`build_abaqus_static_tension_qualification_deck`, `build_abaqus_dependent_datacheck_deck`),
`application/abaqus_runner.py` (stdlib-only discovery/execution/evaluation/report).
Kernel helpers: `strain_rate.legacy_cracking_displacement_rate`,
`core.legacy_temperature_base_properties`, `core.calculate_temperature_elastic_states`.
`abaqus_legacy.abaqus_table_checks` is the non-raising core of the unchanged public
`validate_abaqus_legacy_material_tables`.

The M4 static API, JSON schema and `.inp` text are unchanged (the default card is frozen by
`tests/fixtures/abaqus_m4_static_default.inp`, generated from the baseline and compared
byte-identically against baseline output before freezing).

## Decisions

**Rate axis.** Compression: the legacy curve-family control rate is the Abaqus
compression-hardening rate coordinate (`LEGACY_RATE_AXIS_MAPPING`, explicitly not a
reconstructed inelastic strain rate). Tension: `w_dot = strain_rate · l_ch` [mm/s] through the
single shared helper, which `apply_fracture_energy_rate_effects` now also calls
(`LEGACY_CRACK_OPENING_RATE_MAPPING`). Rates must be strictly increasing for export (Abaqus
ascending-order rule).

**Temperature elasticity.** `*ELASTIC` rows `E(T), nu, T`. `E(T)` is `E_c1_temp`, the exact
modulus `calculate_stress_strain_temp` passes to `calculate_inelastic_compression` (verified by
a spy test and a bit-exact family reconstruction). `nu` = legacy `v_ce`, constant
(`LEGACY_CONSTANT_NU_ASSUMPTION`). Documented legacy consequences: `E(20 °C) = 20363.6 MPa ≠
static E0 = 26171.1 MPa`; `E(T)` is constant from 600 °C.

**Damage.** `omit` is the dependent default. `reference_damage` reuses the kernel's own
first-family damage with the M4 normalization, 2 columns, no rate/temperature column, and is
validated against every family with that family's modulus. Failures refuse the material.

## Damage-policy results for default legacy cases

| Case | `omit` | `reference_damage` |
| --- | --- | --- |
| rates 0, 2, 30, 100 1/s (both tension laws) | PASS | rejected: 30 1/s, compression, `compression_plastic_strain_monotonic` |
| rates 0, 2 1/s | PASS | PASS (damage tables identical to M4 static) |
| temperatures 20…1100 °C (both tension laws) | PASS | rejected: compression branch at every T ≥ 100 °C |

These outcomes are frozen by tests; the validator was not weakened.

## Qualification deck inventory (`--write-decks`)

| Case | Job | Lines | SHA-256 |
| --- | --- | --- | --- |
| static_compression | `q1_static_compression` | 132 | `ff22fbda304edd1616003ad7658a37166c16927965e0cf652f06b6aef3bd104f` |
| static_tension | `q1_static_tension` | 132 | `f40fb13c4cafa21b8e3e6922c79c2277486d0404695b853cb0e69431a5a00681` |
| rate_datacheck | `q1_rate_datacheck` | 204 | `0e77e322fc09098de603b95f2778b8f2b83dfe0608bf8912351fe08af5ad50f9` |
| temperature_datacheck | `q1_temperature_datacheck` | 1021 | `2964730ed6ed8b978e8a726e9cd1ab5c45c2a2faabd97d9d3fc06b083d2e95b4` |

Design (C3D8 unit brick, symmetry-plane BCs, displacement control, Abaqus/Standard, no
stabilization) and acceptance checks: `qualification/abaqus/README.md`.

## Internal qualification (Linux sandbox, Python 3.13, `uv run --locked --extra web`)

```text
git diff --check                                   PASS
scripts/normalize_frozen_text.py                   Frozen artifacts verified; 0 CRLF checkout conversions repaired.
ruff check .                                       All checks passed!
mypy cdp_generator                                 Success: no issues found in 67 source files
pytest -q                                          688 passed, 2 deselected
uv lock --check                                    PASS
uv build --wheel                                   PASS (new application modules included)
imports with streamlit/plotly/odbAccess blocked    PASS
headless streamlit /_stcore/health                 ok
pytest -q -m abaqus_external                       2 skipped (NOT_AVAILABLE)
scripts/qualify_abaqus.py                          NOT_AVAILABLE, exit 3
```

The 2 deselected tests are the `abaqus_external` solver tests (`addopts = -m 'not
abaqus_external'`). Focused groups: M4 static + M4 web 18 passed; dependent rate 26,
temperature 10, damage 7, keyword text 4; deck generation 5; runner (fake launcher) 27; M5
AppTests 6; legacy regression/Excel parity/Windows portability 10.

Native Windows `scripts/qualify_web.ps1` was **not** run for M5 in this environment. The runner
avoids `shell=True`, strips quotes from `ABAQUS_COMMAND` on Windows, and keyword fixtures are
compared with explicit CRLF→LF normalization; `.inp` fixtures are pinned `eol=lf` in
`.gitattributes`.

## Known limitations

- No real Abaqus run: decks and keyword text are validated against the documented contract
  only. Run `uv run python scripts/qualify_abaqus.py` on a licensed machine.
- Dependent decks are datacheck-only; no nonlinear rate/temperature solver validation.
- Default `reference_damage` is rejected for the default rate set and for all temperature
  families; users must choose `omit` there.
- Legacy temperature E(T) differs from the static E0 at 20 °C and stays constant from 600 °C —
  inherited from the legacy kernel, documented, not repaired.
- Combined rate × temperature surfaces, rate- or temperature-dependent damage laws,
  Abaqus/Explicit harnesses and multi-element campaigns are out of scope.
- A one-element PASS would prove executable compatibility only — not experimental,
  structural, normative or multiaxial validity.
