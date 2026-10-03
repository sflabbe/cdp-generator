# WEB-M2 — authority-aware physical concrete and CDPM2

## Delivered behavior

WEB-6 and WEB-7 implemented as one additive vertical slice. Streamlit retains
Legacy curves as the default workflow, with the existing rate/temperature,
Plotly, properties, raw tables and JSON/XLSX behavior. The new workflow builds
verified class-based physical materials, exposes resolution and full provenance,
assesses CDPM2 readiness, accepts explicit constitutive overrides, resolves all
20 semantic fields only when READY, and constructs the existing 24-slot backend
payload after a valid separate LCHAR is supplied.

No scientific formulas, calibration values, backend mapping or readiness rules
were copied into presentation/application code. No domain/core/frozen test,
manifest, hash or Windows qualification change was needed. No dependencies or
subpackages were added; the existing package list and locked web extra include
the new modules. No commit, push or tag was performed.

## Public application APIs added

- `AuthorityConcreteRequest`, `Cdpm2ConversionRequest`: frozen requests with
  defensively frozen mappings and explicit `characteristic_length_mm` context.
- `AuthorityConcreteResult`, `PhysicalPropertyRecord`, `Cdpm2ConversionResult`:
  distinct deterministic JSON contracts (`authority_concrete_result.v1` and
  `cdpm2_conversion_result.v1`); strict finite JSON via `to_dict` / `to_json`.
- `build_authority_concrete(request)`: canonical v2 physical material plus full
  records, requested inputs and effective domain profile policy.
- `run_cdpm2_conversion(request)`: domain readiness always; semantic and backend
  sections absent when blocked; semantic output can exist without runtime context.
- `available_physical_profiles`, `available_concrete_classes`,
  `profile_parameter_specs`, `physical_profile_label`, `ProfileParameterSpec`:
  small frontend catalog using the domain registry, options, bounds and defaults.
- `cdpm2_override_fields`, `cdpm2_parameter_units`, `parse_cdpm2_overrides`:
  domain-owned override names/units with controlled JSON validation and rejection
  of conflicting common/advanced inputs.
- `AuthorityInputError`: expected structural/domain-input failure; unexpected
  resolver/backend defects are not swallowed.

Canonical material, semantic and backend payloads are preserved field-for-field
from their domain `to_dict` methods. Supplemental backend slot labels use the
frozen domain constant. Negative-source provenance explaining an unresolved
property is retained exactly when provided by the domain; its value remains
`None`, with no invented producing equation. Truly absent provenance stays absent.

## UI and downloads

Workflow radio selector: Legacy curves / Authority-aware material / CDPM2.
Profile selection is reactive so class/parameter catalogs update; calculations
use a form. Modern and legacy successful state keys are independent. The modern
summary identifies displayed profile/class, requested/effective physical inputs
and constitutive overrides, even when form inputs subsequently change.

Physical properties, Provenance, CDPM2, Backend and Raw / Export tabs expose all
values and resolution/source categories textually. Common E and G_Ft overrides
start unchecked and blank. Advanced JSON permits the existing domain catalog.
No guessed fracture energy or E_secant fallback is used. Backend runtime input
is available only after semantic resolution; cm(1:24) and LCHAR [mm] are shown
separately. A new material assessment clears its old backend payload. Invalid
inputs keep the previous successful result.

Downloads: `Concrete-Material.json`, `CDP-M2-Result.json`, `CDPM2-Semantic.json`
when READY, and `CDPM2-legacy24.json` after valid runtime context. No new XLSX
format is introduced. The scientific sources are physical EC2/fib authority,
Grassl 2013 static calibration, and backend compatibility, kept separate.
Modern physical profiles do not qualify or generate EC2/fib stress-strain curves.

## Acceptance matrix — all cases passed

| Case | Profile / class / inputs | Readiness | Semantic/backend with valid LCHAR |
| --- | --- | --- | --- |
| A | fib MC2010 C30, defaults | READY | Both present |
| B | EC2 2004 C30/37, defaults | COMPOSITION_REQUIRED | Both absent |
| C | EC2 2004 C30/37, G_Ft=0.15 | READY | Both present |
| D | EC2 2023 C30/37, age 28, defaults | COMPOSITION_REQUIRED | Both absent |
| E | EC2 2023 C30/37, age 56, defaults | UNRESOLVED_PHYSICAL_INPUT | Both absent |
| F | EC2 2023 age 56, only G_Ft=0.15 | UNRESOLVED_PHYSICAL_INPUT | Both absent |
| G | EC2 2023 age 56, only E=41000 | COMPOSITION_REQUIRED | Both absent |
| H | EC2 2023 age 56, E=41000 and G_Ft=0.15 | READY | Both present |

Case E preserves both exact domain blockers: `E_initial unresolved; E_secant
fallback forbidden` and `G_Ft composition required`. Instrumented tests prove
`Concrete.to_cdpm2` is never called while blocked. READY cases compare all
semantic values/provenance and all 24 backend slots directly against the domain.
These inputs are qualification examples, not recommended project assumptions.

## Exact files added or modified

- `README.md`
- `cdp_generator/application/__init__.py`
- `cdp_generator/application/authority_catalog.py`
- `cdp_generator/application/authority_requests.py`
- `cdp_generator/application/authority_results.py`
- `cdp_generator/application/authority_services.py`
- `cdp_generator/web/app.py`
- `cdp_generator/web/authority.py`
- `docs/web_architecture.md`
- `docs/web_m2_handoff.md`
- `tests/test_application_authority_concrete.py`
- `tests/test_application_cdpm2.py`
- `tests/test_web_authority_cdpm2.py`

## Validation performed in Linux / Python 3.13.15

| Command | Result |
| --- | --- |
| `uv sync --locked --extra web` | PASS, existing 61-package lock reused |
| `git diff --check` | PASS |
| `uv run --locked --extra web ruff check .` | PASS |
| `uv run --locked --extra web mypy cdp_generator` | PASS, 50 source files |
| `uv run --locked --extra web pytest -q` | PASS, 501 tests |
| `uv run --locked --extra web pytest -q tests/test_application_authority_concrete.py tests/test_application_cdpm2.py tests/test_web_authority_cdpm2.py` | PASS, 41 tests |
| `uv run --locked --extra web pytest -q tests/test_application_legacy_adapter.py tests/test_plotly_visualization.py tests/test_web_excel_parity.py tests/test_streamlit_smoke.py tests/test_legacy_concrete_regression.py` | PASS, 25 tests |
| `uv run --locked --extra web pytest -q tests/test_cdpm2_integration.py` | PASS, 24 tests |
| `uv build --wheel --out-dir /tmp/cdp-web-m2-dist` | PASS; wheel inspected for every added/modified package module |

Five new Streamlit AppTest cases exercise fib READY/semantic/backend/download
availability, EC2 2004 blocked then explicit G_Ft READY, EC2 2023 high age with
both blockers then explicit E/G_Ft READY, invalid advanced JSON retaining state,
and enabled blank override error handling. They also check independent legacy
state, retained result on reruns, and rejected invalid LCHAR. No manual browser
download clicks or pixel inspection are claimed.

A subprocess ran `python -m streamlit run cdp_generator/web/app.py
--server.headless=true --server.port=8766 --browser.gatherUsageStats=false`, received
`ok` from `/_stcore/health`, then terminated cleanly. A fresh import-hook check
blocked Plotly and Streamlit imports and successfully imported both the base
package and the application API. Protected domain/core files, qualification
artifacts, existing tests, `.gitattributes` and both Windows scripts were compared
byte-for-byte with the supplied ZIP state and remained unchanged.

The incremental mypy cache encountered a malformed local SQLite database; it
was removed, and the standard mypy command passed without changing checks/code.

Windows-native `.\scripts\qualify_web.ps1` could not be run in this Linux
execution environment. Prior M1 Windows evidence is not claimed as M2 Windows
qualification. Run the preserved script on the user's Windows machine.

## Deferred scope

Explicit secondary physical fracture-energy composition UX is deferred. Required
composition blockers and constitutive G_Ft override are supported. Comparison
workspace, steel UI, modern curves, new mechanics, Kratos execution, persistence,
authentication, TypeScript/React/FastAPI and a large redesign remain out of scope.
No unresolved Linux qualification failure remains.

## Apply and run

```bash
git apply --check /path/to/web_m2_authority_cdpm2_vertical_slice.patch
git apply /path/to/web_m2_authority_cdpm2_vertical_slice.patch
uv sync --locked --extra web
uv run --locked --extra web streamlit run cdp_generator/web/app.py
```

Windows qualification:

```powershell
.\scripts\qualify_web.ps1
```
