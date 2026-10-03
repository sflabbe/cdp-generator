# WEB-M4 / CONCRETE-BACKENDS-M1 handoff

## Status

WEB-M4 is implemented against the supplied WEB-M3 repository snapshot. No commit, push, or tag
was created. The supplied ZIP did not contain Git metadata, so the deliverable is a clean patch
against the exact extracted snapshot.

The implementation closes the three requested concrete-backend gaps:

- explicit secondary fib Model Code 2010 fracture-energy composition for CDPM2 when the primary
  physical profile leaves `G_F` unresolved;
- exact, frontend-independent CDPM2 bilinear tensile-softening plot data, including the optional
  LCHAR regularized view and an application-layer energy invariant;
- a full static-reference legacy Abaqus CDP material backend with machine-readable provenance,
  backend-only damage normalization, documented plastic strain/displacement validation,
  deterministic JSON and `.inp` export, and Streamlit presentation.

The stale WEB-M3 Windows statement was also amended to reflect the native qualification supplied
with the integrated M3 handoff: `scripts/qualify_web.ps1` completed with 566 passed in 9.94 s and
0 CRLF conversions repaired.

## fib MC2010 fracture-energy composition

`estimate_fib_mc2010_fracture_energy(f_cm)` is now the single implementation of the existing
repository-normalized MC2010 estimate. `FibMc2010Profile.build()` calls that helper directly, so
the verified fib profile and the secondary-composition path cannot drift to separate formulas.

`Cdpm2ConversionRequest.fracture_energy_policy` accepts `profile_only` or
`fib_mc2010_if_missing`. The latter composes a `Cdpm2FractureEnergyComposition` only when the
primary profile does not already provide usable fracture energy. It never mutates the primary
physical material. An explicit constitutive `G_Ft` override combined with the fib composition
policy is rejected at the application boundary rather than silently assigning precedence.

The application result serializes the secondary authority separately as `strategy`, `value`,
`unit`, and physical provenance. The mapped CDPM2 `G_Ft` retains `PHYSICAL_SOURCE` provenance;
it does not become a `USER_CONSTITUTIVE_OVERRIDE`.

Acceptance evidence from the supplied snapshot:

| Case | Result |
| --- | --- |
| EC2:2004 C30/37 + `profile_only` | `COMPOSITION_REQUIRED`; `G_Ft composition required` |
| EC2:2004 C30/37 + fib-if-missing | `READY`; secondary composition present |
| EC2:2023 C30/37, 28 d + fib-if-missing | `READY` |
| EC2:2023 C30/37, 56 d + fib-if-missing | `UNRESOLVED_PHYSICAL_INPUT`; only `E_initial unresolved; E_secant fallback forbidden` |
| EC2:2023 C30/37, 56 d + fib-if-missing + E override | `READY` |
| fib MC2010 C30 + fib-if-missing | `READY`; no redundant composition |
| fib-if-missing + explicit `G_Ft` override | controlled `AuthorityInputError` |

For EC2:2004 C30/37 the composed value in this snapshot is
`G_Ft = 0.14050245330952899 N/mm`.

## CDPM2 tensile-softening views

The application builds canonical `CurveSeries` data from the already-resolved Grassl 2013
semantic parameters. No constitutive integration was added to the web layer.

The crack-opening curve is exactly:

- x = `[0, w_f1, w_f]`;
- y = `[f_t, f_t1, 0]`;
- x quantity = crack opening [mm];
- y quantity = tensile stress [MPa].

The application computes the polyline area and requires it to equal `G_Ft` within tight floating
point tolerance. Failure raises `Cdpm2SofteningInvariantError`; the check is not delegated to
Plotly. For the EC2:2004 C30/37 + fib-composition case:

- points x = `[0.0, 0.03233879926589872, 0.2155919951059915]` mm;
- points y = `[2.896468153816889, 0.8689404461450667, 0.0]` MPa;
- integrated area = `0.14050245330952899 N/mm`;
- target `G_Ft = 0.14050245330952899 N/mm`.

With `LCHAR = 12.5 mm`, the second view uses x =
`[0.0, 0.0025871039412718976, 0.01724735960847932]` and the same stresses. The Streamlit copy
explicitly labels this as a regularized tensile-softening view, not an integrated uniaxial CDPM2
stress-strain response.

## Full legacy Abaqus CDP backend

`AbaqusLegacyMaterialRequest` builds only the static reference case from the historical kernel
with `strain_rates=[0.0]`. The existing rate/temperature legacy analysis remains unchanged and is
not exported as dependent Abaqus damage tables in M4.

The material backend keeps the existing repository calibration for `dilation_angle`, `fbfc`, and
`Kc`; existing `Concrete.to_abaqus_cdp()` behavior remains untouched. E and nu come from the
legacy physical/kernel path. Abaqus-documented defaults are introduced only for fields that the
legacy scalar API did not calculate: eccentricity 0.1 and viscosity 0.0. Explicit overrides of
those backend settings receive `USER_BACKEND_OVERRIDE` provenance.

The historical nonlinear tension curves are exported as
`*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT`. They are not recast as `TYPE=GFI`, because that
would change the constitutive softening law. Abaqus `REF LENGTH` is exposed independently as
`damage_conversion_reference_length_mm`, default 1.0 mm in the repository N-mm-MPa convention;
it is not mapped from legacy `l_ch`.

Backend-only normalization preserves the legacy arrays outside the adapter. For the default
28 MPa material, the two normalizations are:

- compression damage row 0: `1.6555682445074105e-05 -> 0.0`;
- tension damage final row: `1.0 -> 0.99`.

The validator applies the documented Abaqus conversion relationships. In the default material it
reports PASS with compression plastic-strain minimum `0.0`, minimum increment
`0.0005588425472520774`, tension plastic-displacement minimum `0.0`, and minimum increment
`0.015814851123600378`. The result has 16 compression-hardening rows, 20 tension-stiffening rows,
and four canonical export curves when both damage branches are enabled.

The deterministic `.inp` order is material, elasticity, CDP scalar data, compression hardening,
optional compression damage, displacement-based tension stiffening, and optional displacement-
based tension damage. Full provenance and validation remain in the JSON audit artifact. No Abaqus
solver execution is claimed.

## Abaqus research implemented

The implementation follows the indexed Abaqus 2025 documentation for Concrete Damaged Plasticity,
`*CONCRETE DAMAGED PLASTICITY`, compression hardening/damage, tension damage, and tension
stiffening. In particular: compression hardening is stress versus inelastic strain; the first
compression damage point is normalized to zero; damage is kept below one with the requested 0.99
backend cap; damage coordinates are aligned with their hardening/stiffening counterparts; and the
plastic strain/displacement validation uses the documented conversion relations and explicit
`REF LENGTH` semantics.

See `docs/research/abaqus_cdp_backend.md` for the implementation-focused source record and links.
The online documentation was researched, but Abaqus itself was not installed or executed in this
environment.

## Qualification executed here

The following executable checks completed successfully with the dependencies already installed in
the environment:

```text
pytest -q tests/test_web_m4_fib_cdpm2.py
11 passed in 0.42s

pytest -q tests/test_abaqus_legacy_full.py
15 passed in 0.40s

pytest -q tests/test_cdpm2_integration.py tests/test_application_cdpm2.py \
  tests/test_cdpm2_backend_adapter.py tests/test_cdpm2_mapping.py \
  tests/test_cdpm2_schema.py tests/test_cdpm2_authority_contract.py
251 passed in 0.49s

pytest -q tests/test_legacy_concrete_regression.py
5 passed in 0.26s

pytest -q tests/test_web_m4_fib_cdpm2.py tests/test_abaqus_legacy_full.py \
  tests/test_application_cdpm2.py tests/test_fib_mc2010_profile.py
88 passed in 0.46s

pytest -q tests --ignore=tests/test_streamlit_smoke.py \
  --ignore=tests/test_web_authority_cdpm2.py --ignore=tests/test_web_m3.py \
  --ignore=tests/test_web_m4.py
578 passed in 6.66s

python -m compileall -q cdp_generator tests
PASS

base/application import check with imports of streamlit and plotly deliberately blocked
PASS
```

The environment could not complete the locked WEB qualification because network access is disabled
and the uv cache is incomplete. `UV_OFFLINE=1 uv run --locked --extra web ...` was attempted for
Ruff, mypy, and pytest; dependency resolution stopped before those tools ran because required
artifacts such as `matplotlib==3.10.9`, `plotly==6.9.0`, or `rpds-py==2026.6.3` were not cached.
`UV_OFFLINE=1 uv build --wheel` likewise could not resolve uncached build-system dependencies.

A direct full `pytest -q` also cannot collect the four AppTest modules because this execution
environment has no `streamlit` installation. The same absence blocks the requested headless
Streamlit health check. This is an environment qualification gap, not a passing result; no green
claim is made for Ruff, mypy, wheel, AppTest, headless Streamlit, or native Windows M4 execution.
The M4 Streamlit tests are nevertheless included in `tests/test_web_m4.py` for execution in the
project's normal locked web environment.

## Known limitations / deferred scope

M4 intentionally does not implement a uniaxial/multiaxial CDPM2 integrator, CDPM2 compression
response plots, Kratos/OOFEM execution, Abaqus rate- or temperature-dependent material tables,
Abaqus CAE Python objects/jobs, or new EC2/fib constitutive curves. The legacy Abaqus backend is a
compatibility calibration and must be checked against project calibration, experiments, and the
actual Abaqus version used.
