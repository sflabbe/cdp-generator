# Legacy concrete web boundary (WEB-0–WEB-5)

`kernel → application DTO → visualization/UI`. Future FastAPI and TypeScript
frontends must consume the same versioned `ResultBundle` representation.

| Concept | Owner / meaning |
| --- | --- |
| Physical property | Kernel properties such as modulus, strength, fracture energy; modern `Concrete.from_class` profiles separately describe authority-aware material definitions. |
| Constitutive parameter | Legacy CDP dilation angle, Kc, fbfc; modern backend conversion has its own `cdpm2_readiness` / `to_cdpm2` gates. |
| Calculated curve | Historical core stress, strain and damage arrays, copied without resampling by the legacy adapter. |
| Runtime/mesh context | `l_ch` in mm is analysis input; legacy property `l0` is in metres. Neither is an authority certification. |
| Provenance | Bundle declares `legacy_v1`, with no EC2/fib curve authority. |
| UI-only state | Current widgets, selected raw curve, last successful bundle in session state. |

**A legacy curve must never acquire EC2/fib authority merely because some material
properties came from an authority-aware profile.** The legacy workflow uses the existing
legacy curve generator exclusively. Authority-aware materials and constitutive
backend readiness/conversion use the separate WEB-M2 workflow described below.

Application requests validate finite positive scalar inputs and non-negative,
non-empty rates; they do not introduce material applicability ranges. The kernel
owns temperature cases and all physics. Adapter metadata identifies each case,
and damage preserves the first case reference. In rate mode even the historical
shared compression strain array (returned by the final kernel iteration) is
preserved. No curve grids are corrected or reinterpreted.

`ResultBundle.to_dict()` and `.to_json()` provide strict finite JSON schema 1.0.
Non-finite kernel output is rejected instead of silently changed. Plotly consumes
semantic curves, never historical keys. XLSX uses an internal reverse adapter
and the shared memory workbook builder to preserve the eight historical sheets.
The old file export wraps that builder; JSON includes all nine plotted groups.

Web dependencies are optional. Base package imports do not import Plotly or
Streamlit. Install `[web]` and qualify with `uv run --locked --extra web ...`.


## WEB-M2 additive material and constitutive boundary

`AuthorityConcreteRequest → Concrete.from_class → AuthorityConcreteResult`.
The material DTO contains canonical v2 domain serialization plus property
records with units, resolution and full provenance. Domain negative-source
notes for unresolved/composition-required values are retained; absent values
remain `None`, and absent provenance remains absent. Requested profile inputs
and effective policy inputs are separate. Catalog classes, parameter options,
bounds, override names and units come from domain registries/constants; defaults
without public constants are queried through the public material builder.

`Cdpm2ConversionRequest → cdpm2_readiness → [READY only] to_cdpm2`.
Readiness and every blocker come directly from the domain. The semantic JSON is
the unmodified domain parameter payload; overrides retain user-constitutive
provenance, and derived/default values retain Grassl/OOFEM provenance. WEB-M2
itself generated no curves or borrowed M1 curves; WEB-M4 adds only canonical
views of the resolved CDPM2 tensile-softening semantic law. Non-READY is a normal
result containing no semantic parameters or backend.

A valid optional `characteristic_length_mm` adds the backend through the existing
`adapt_cdpm2_legacy_backend`. Backend JSON equals the canonical domain payload;
slot names are separate presentation metadata from the frozen domain constant.
LCHAR stays separate from physical properties, the 20 semantic fields and the
24-slot tuple. The UI gathers runtime context only after semantic resolution.

Modern UI state uses `authority_request` / `authority_result`, independent of
M1 `last_result`. Forms retain the displayed result until successful submission.
Rebuilding a material clears any old backend context; invalid requests retain
the previous successful result. JSON downloads mirror these displayed DTOs.
Only expected input errors are presented; unexpected resolver/backend defects
remain diagnosable. Constitutive `G_Ft` override remains supported without
relabeling authority; WEB-M4 additionally implements explicit secondary fib
MC2010 fracture-energy composition when the primary physical profile leaves it
unresolved.


## WEB-M3 steel boundary and session comparison

`SteelAnalysisRequest → get_steel_spec → calibrate_jc_from_spec →
generate_jc_curves_multicase → SteelAnalysisResult`. Application code copies
arrays into canonical `CurveSeries` without equations, conversion or resampling.
Catalog helpers query `list_available_standards`. Requests freeze overrides and
series, validate finite structural inputs, and leave scientific validation to
the domain. Only expected `ValueError` validation is converted to
`SteelInputError`; unexpected errors remain diagnosable. Normal calls are silent.

Steel results carry `steel_analysis_result.v1`, resolved material/metadata,
`approximate_preset` or `user_provided`, visible disclaimer, calibration settings,
all eight JC parameters, analysis configuration, curves and export capabilities.
Automatic n is the existing heuristic; neutral C/m remains the default. Total
strain/stress quantities follow output kind. Plastic strain is always true;
its generic Stress ordinate is qualified by `stress_kind` metadata so engineering
and true stress are not confused in comparison.

`steel.export` shares its exact historical workbook writer between file and
in-memory exports. Workbook interpolation remains historical exporter behavior;
it is not applied to canonical result arrays. The ABAQUS builder shares the
historical template with its filesystem wrapper and retains EXPERIMENTAL status.
Application download helpers reconstruct exporter inputs from stored result
arrays and parameters, never regenerate scientific curves.

`ComparisonCase` stores an immutable JSON text snapshot plus stable UUID, label,
family, source workflow and schema. Parsed payload access returns a fresh object.
Adding another result never mutates previous cases. Duplicate IDs/labels and
more than four cases per family are controlled errors. Rename preserves ID and
snapshot. UI remove/clear operations affect only explicitly selected cases/family.

Shared application guards first require one family, then intersect available
curve groups and require identical x/y quantities, x/y units and `stress_kind`.
All traces in all selected cases must agree. Plotly receives original arrays and
prefixes trace names with case labels. Cross-family overlays fail even if every
axis semantic and unit is identical. No true/engineering conversions are made.
Authority comparison projects descriptive long-form tables preserving `None`,
units and resolution, including mixed readiness states and all blockers. Only
READY cases participate in semantic tables, with at least two such cases. Backend
slots and mesh context are excluded. Steel tables expose 13 material/JC fields
with units and source status. No ranking or suitability calculations exist.

`web.app` is a thin router. `legacy`, `authority`, `steel`, `comparison` own
presentation; `shared` owns the consistent sidebar Add to Compare interaction.
Result keys are `last_result`, `authority_request`/`authority_result`,
`steel_result`, and `comparison_cases`. Forms calculate only on explicit
submission. Results survive invalid input and workflow switches. Current widgets
and displayed successful result summaries remain distinct. All exports use the
displayed successful result, not unsubmitted controls. No physics lives in web
or visualization modules. Application imports do not import Streamlit/Plotly.

The next architectural choice is TS-GATE, not implementation of another Python
workflow. Versioned application DTOs/services can feed FastAPI, generated OpenAPI
clients and React if product requirements later justify that work. There are no
FastAPI placeholders, authentication, persistent projects, database or jobs here.
The existing steel-core Ruff/mypy exclusions remain separate modernization debt.


## WEB-M4 concrete backend successor

### Secondary fib fracture-energy composition

`Cdpm2ConversionRequest.fracture_energy_policy` is frontend-independent and accepts
`profile_only` or `fib_mc2010_if_missing`. The latter never forces fib over an
already resolved primary fracture energy. When the primary profile reports
`COMPOSED_REQUIRED`, the service applies the verified MC2010 Eq. (5.1-9) estimator
to the primary material's resolved `f_cm` and passes a separate
`Cdpm2FractureEnergyComposition` into the existing domain readiness/resolution
path. The primary `ConcretePhysicalProperties` object is not mutated. An explicit
constitutive `G_Ft` override and secondary fib composition are rejected as a
source conflict at the request boundary.

The MC2010 fracture-energy equation now has one source of truth in
`estimate_fib_mc2010_fracture_energy`; the verified fib profile itself calls the
same helper, preserving its previous values and provenance. Result JSON exposes
the optional composition as a separate physical-authority object.

### CDPM2 softening views

READY semantic parameters are converted to canonical `CurveSeries` before any
visualization. The crack-opening view contains exactly `(0,f_t)`, `(w_f1,f_t1)`
and `(w_f,0)`. Application code integrates the two line segments and raises a
scientific/programming error unless the area equals `G_Ft` within tight floating
tolerance. If valid LCHAR runtime context exists, a second canonical view divides
only the three crack openings by LCHAR. It is explicitly a **regularized
tensile-softening view**, not an integrated uniaxial CDPM2 stress-strain response.
Plotly only copies canonical curve data; no constitutive equations live in the
visualization layer.

### Full legacy Abaqus CDP material

The Legacy curves workflow contains an additive **Abaqus CDP** tab backed by
`AbaqusLegacyMaterialRequest → run_abaqus_legacy_material →
AbaqusLegacyMaterialResult`. The service always regenerates a static reference
legacy case (`strain_rate=0`) from the displayed base inputs; rate/temperature
legacy visualizations are not exported as dependent Abaqus tables.

The full material preserves the existing scalar legacy `dilation_angle`, `fbfc`
and `Kc`; the old `Concrete.to_abaqus_cdp()` API is unchanged. Linear elastic
`E`/`nu` and every stress/damage curve retain legacy implementation provenance.
Abaqus-documented eccentricity 0.1 and viscosity 0.0 are backend defaults unless
explicitly overridden. Tension exports the existing bilinear or power-law
stress-crack-opening array using `TYPE=DISPLACEMENT`; `TYPE=GFI` is deliberately
not used because it would replace the existing nonlinear law with Abaqus' linear
GFI loss-of-strength assumption.

The adapter normalizes only its Abaqus representation: first compression damage
is exactly zero and damage above 0.99 is capped, with changed rows recorded as
`ABAQUS_BACKEND_NORMALIZATION`. Historical kernel arrays are untouched. Abaqus
`REF LENGTH` is a separate `damage_conversion_reference_length_mm` backend
setting and is never inferred from legacy `l_ch`. Before `.inp` export the
application validates the documented compression plastic-strain and tension
plastic-displacement conversions for finite, nonnegative, monotonic output and
exact coordinate alignment. Invalid material representations produce typed errors
and no executable card.

The deterministic JSON is the audit artifact; the deterministic `.inp` material
card is the execution artifact. No Abaqus Python dependency, CAE object creation,
job execution, Kratos or OOFEM dependency is introduced.

## ABAQUS-Q1 / WEB-M5 dependent Abaqus tables and solver gate

The **Abaqus CDP** tab of the Legacy curves workflow now has an **Abaqus export mode**
selector: *Static reference* (the unchanged M4 renderer), and — depending on the displayed
legacy result — *Strain-rate dependent* or *Temperature dependent*. Dependent exports use the
rates/temperature cases of the displayed result; nothing is re-typed.

```text
LegacyConcreteAnalysisRequest (displayed result)
    → AbaqusLegacyDependentRequest(mode, …, damage_policy="omit" | "reference_damage")
    → run_abaqus_legacy_dependent_material
        ├─ core.calculate_stress_strain / calculate_stress_strain_temp   (exact families)
        ├─ strain_rate.legacy_cracking_displacement_rate                 (shared w_dot)
        ├─ core.calculate_temperature_elastic_states                     (E(T), constant nu)
        ├─ run_abaqus_legacy_material (M4)                               (scalars, E0, nu)
        └─ abaqus_table_checks per family                                (shared M4 checks)
    → AbaqusLegacyDependentResult (JSON v1) → build_abaqus_legacy_dependent_material_text
```

`AbaqusLegacyDependentValidationError` carries structured failures; the web layer shows them
and keeps the last valid dependent result. Plotly figures (`build_abaqus_dependent_figures`)
draw one trace per exported family and a single reused reference-damage curve; nothing is
synthesized in the web layer.

Qualification decks (`application/abaqus_decks.py`) and the stdlib-only runner
(`application/abaqus_runner.py`) are frontend-independent and are never imported by Streamlit.
Solver execution belongs to `scripts/qualify_abaqus.py` and the `abaqus_external` pytest
marker; the web UI only points to that command and has no button that launches Abaqus.
