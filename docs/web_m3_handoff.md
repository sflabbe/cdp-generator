# WEB-M3 implementation handoff

## Integration basis and outcome

Patch: `web_m3_steel_comparison_ux_closeout.patch`, against the supplied ZIP for
expected integrated main `7059eb2b1cf10c58305b86f6757a01b55e803ac1` (WEB-M2).
The ZIP contains no Git history; patch applicability is checked against its exact
files. Do not reapply M1/M2. No commit, push or tag was performed.

WEB-8 comparison, WEB-9 steel, WEB-10 UX and applicable WEB-11/12 qualification
are implemented together. Final routes retain existing labels to preserve M1/M2:

| Workflow | Result and actions |
| --- | --- |
| Legacy curves | Historical concrete calculation, nine Plotly groups, JSON/XLSX, named snapshot |
| Authority-aware material / CDPM2 | Physical provenance, readiness, READY semantics, optional backend, JSON, named snapshot |
| Steel Johnson-Cook | Approximate preset/custom material, existing calibration/curves, Plotly, JSON/XLSX/experimental ABAQUS, named snapshot |
| Compare | Select family/cases; compatible overlays or descriptive tables; rename/remove/clear |

## Public application APIs

All below are exported from `cdp_generator.application`, with no web dependency:

- `SteelAnalysisRequest`, `SteelAnalysisResult`, `SteelInputError`.
- `available_steel_standards()`, `available_steel_grades(standard)`.
- `parse_float_series(text)`.
- `run_steel_analysis(request)`.
- `steel_result_excel_bytes(result)`, `steel_result_abaqus_text(result)`.
- `ComparisonCase`, `ComparisonInputError`, `make_comparison_case(result, label)`.
- `add_comparison_case(cases, case)`, `rename_comparison_case(cases, case_id, label)`.
- `curve_semantic_signature(curve)`, `compatible_curve_groups(cases)`.
- `authority_comparison_tables(cases)`, `steel_comparison_table(cases)`.

Module-level helpers: `application.comparison.case_curves` and
`comparison_curves`; `steel.export.build_steel_excel_bytes` and
`build_abaqus_material_card_text`; `visualization.steel_plotly.build_steel_figures`
and `visualization.comparison_plotly.comparison_figure`.

Steel material overrides are a frozen mapping. Custom requests use `source="custom"`
and domain `get_steel_spec("Custom", ...)` with explicit strengths/ductility.
The result contains material, calibration, eight JC fields, analysis, canonical
curves, metadata, warnings and export capabilities. Strict JSON is deterministic
and rejects nonfinite output. Preset overrides are recorded explicitly.

## Acceptance evidence

| Case | Result |
| --- | --- |
| S1 EC2 B500C neutral | Resolved spec, automatic calibration and arrays exactly equal direct steel calls; C=m=0 |
| S2 ACI A615_60 | Exact direct domain parity |
| S3 NCh A630-420H | Exact direct domain parity |
| S4 B500C fy=550, fu=650, Agt=8 | Exact direct override parity; approximate status retained |
| S5 Custom TestSteel fy=500, fu=620, Agt=10, E=200000, nu=.3 | Direct custom parity; user_provided status |
| S6 Explicit n=.25, C=.02, m=1; rates .001/10, temperatures 20/400 | Exact multicase curve parity |
| S7 Engineering output | Engineering total quantities; true plastic strain; stress representation metadata preserved |
| Custom fu/fy | Explicit ratio resolves via existing domain; no inferred missing strength input |
| XLSX true/engineering | Sheet names and every cell equal a fixture generated from original pre-M3 exporter, filesystem writer and in-memory/application builders |
| ABAQUS with/without density | Text equal original fixture, filesystem writer and in-memory builder |
| Legacy/steel overlays | Trace arrays and labels preserved; mismatched quantities or units rejected |
| True/engineering steel | Total and plastic groups rejected when stress representations differ; no conversion |
| Cross-family | Explicitly identical concrete/steel arrays and axes still rejected by family guard |
| Authority tables | None/unit/resolution preserved; READY and blocked coexist; 20 semantic fields per READY case only when at least two READY cases |
| Case management | Four per family; duplicate labels/IDs rejected; rename stable ID; snapshots independent |
| AppTest | Preset/custom/ratio, active calibration, outputs/downloads, controlled parser/domain/blank errors, retained results, workflow switching, all three comparison families, rename/remove/clear |

Built-in steel database values remain APPROXIMATE and are visibly described as:
"Built-in approximate preset — verify against the applicable standard / project data."
Custom input is user-provided, not verified. No normative citations are invented.
Automatic n is a current-model heuristic, not experimental fitting. ABAQUS is an
experimental template requiring version-specific verification. Comparison is
strictly descriptive, with no rankings/recommendations. LCHAR/backend slots are
not mixed into material/semantic comparison.

## Qualification executed on Linux

| Command / check | Result |
| --- | --- |
| `uv sync --locked --extra web` | Locked 61-package environment; no new dependencies |
| `git diff --check` | PASS |
| `uv run --locked --extra web ruff check .` | All checks passed |
| `uv run --locked --extra web mypy cdp_generator` | Success, 61 source files |
| `uv run --locked --extra web pytest -q` | 556 passed, 17.58 s |
| Five new M3 test modules | 55 passed, 6.75 s |
| M1 adapter/Plotly/Excel/Streamlit/legacy regression | 25 passed, 7.79 s |
| M2 authority/application/CDPM2/Streamlit | 41 passed, 3.10 s |
| Existing steel scientific `tests/test_steel.py` | 16 passed, .71 s |
| CDPM2 integration and backend adapter | 85 passed, 1.15 s |
| `uv build --wheel --out-dir /tmp/cdp-web-m3-dist` | Wheel built; all 24 top-level application/visualization/web Python modules present |
| Base/application import hook blocking Streamlit/Plotly | PASS; neither imported eagerly |
| Headless Streamlit port 8767, `/_stcore/health` | HTTP 200 `ok`; process terminated cleanly |
| Interactive `cdp-steel` scripted preset path | Calculation + filesystem XLSX/ABAQUS outputs passed |
| Existing Matplotlib `plot_steel_results(..., show=False)` with Agg | PASS |

Mypy's existing local SQLite cache failed internally on reuse; moving the local
cache aside and rerunning the same command passed without configuration changes.
`cdp-steel` is an interactive command, not an argparse CLI; its real interactive
path was qualified with scripted stdin. Manual browser clicks/downloads were not
performed; export content, download presence and workflows are automated checks.

Native Windows qualification was subsequently executed with the supported
`.\scripts\qualify_web.ps1` gate: **566 passed in 9.94 s**, with **0 CRLF
conversions repaired**. UTF-8, frozen LF normalization and fail-fast checks remained
intact. This handoff note was amended after that native Windows qualification.

## Exact files

Modified:

```text
README.md
cdp_generator/application/__init__.py
cdp_generator/steel/export.py
cdp_generator/web/app.py
cdp_generator/web/authority.py
docs/web_architecture.md
```

Added:

```text
cdp_generator/application/comparison.py
cdp_generator/application/steel_catalog.py
cdp_generator/application/steel_requests.py
cdp_generator/application/steel_results.py
cdp_generator/application/steel_services.py
cdp_generator/visualization/comparison_plotly.py
cdp_generator/visualization/steel_plotly.py
cdp_generator/web/comparison.py
cdp_generator/web/legacy.py
cdp_generator/web/shared.py
cdp_generator/web/steel.py
docs/web_m3_handoff.md
tests/fixtures/steel_web_exports.json
tests/test_application_comparison.py
tests/test_application_steel.py
tests/test_steel_plotly.py
tests/test_steel_web_exports.py
tests/test_web_m3.py
```

All existing scientific/domain modules, original tests/fixtures, frozen artifacts,
Windows scripts, `.gitattributes`, `pyproject.toml` and `uv.lock` retain original
bytes, except the additive steel export sharing described above. No model
formulas, curve behavior or backend mappings changed. The new XLSX fixture is
independent evidence from the original exporter, not an updated scientific baseline.

## Limits and TS-GATE

Session cases are ephemeral, limited to four per family, and cannot be shared
across restarts. Engineering/true comparisons with distinct stress representation
require compatible cases; no numerical conversion is introduced. Historical steel
workbook interpolation is intentionally preserved. Existing steel-core Ruff/mypy
exclusions remain technical debt, not a reason to rewrite Johnson-Cook here.
Optional secondary physical fracture-energy composition remains deferred from M2.

The planned Python/Streamlit implementation scope is complete with Linux gates
passing and the later native Windows gate recorded above. The next architectural
decision is **keep Streamlit vs FastAPI + React/TypeScript**.
The existing versioned application contracts/services are suitable for
application → FastAPI → OpenAPI/generated TS client → React, without forcing a
migration now. Concrete triggers are authentication/users, persistent projects,
shareable deep URLs, complex client interactions, large responsive UI needs,
background jobs, or embedding in a broader product. Without those requirements,
Streamlit remains a valid internal/scientific frontend.

No FastAPI/TS placeholders, database, authentication, persistent storage, workers,
Kratos execution, new concrete/CDPM2 mechanics, normative steel verification,
experimental-data fitting, optimization or ranking were implemented.
