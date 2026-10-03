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
properties came from an authority-aware profile.** This UI uses the existing
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
provenance, and derived/default values retain Grassl/OOFEM provenance. No curves
are generated or borrowed from M1. Non-READY is a normal result containing no
semantic parameters or backend.

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
remain diagnosable. Explicit secondary physical fracture-energy composition is
deferred; constitutive `G_Ft` override is supported without relabeling authority.
