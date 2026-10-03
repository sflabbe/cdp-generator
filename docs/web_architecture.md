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
backend readiness/conversion remain separate successor workflows.

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
