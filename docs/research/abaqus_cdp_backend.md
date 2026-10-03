# Abaqus CDP backend research basis — WEB-M4

This note records the implementation-relevant Abaqus findings used by the full
legacy compatibility backend. It is not a claim that the generated material was
executed or qualified inside Abaqus; no proprietary solver run is part of WEB-M4.

## Primary documentation inspected

- SIMULIA/Abaqus 2025, **Concrete Damaged Plasticity**:
  https://docs.software.vt.edu/abaqusv2025/English/SIMACAEMATRefMap/simamat-c-concretedamaged.htm
- `*CONCRETE DAMAGED PLASTICITY`:
  https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-concretedamagedplasticity.htm
- `*CONCRETE COMPRESSION HARDENING`:
  https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-concretecompressionhardening.htm
- `*CONCRETE COMPRESSION DAMAGE`:
  https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-concretecompressiondamage.htm
- `*CONCRETE TENSION DAMAGE`:
  https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-concretetensiondamage.htm
- ConcreteTensionStiffening object / keyword semantics:
  https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKERRefMap/simaker-c-concretetensionstiffeningpyc.htm

The indexed online manual used here is Abaqus 2025. The implementation does not
claim Abaqus 2025/2026 execution or solver convergence.

## Required CDP material components

Abaqus CDP requires linear isotropic elasticity together with
`*CONCRETE DAMAGED PLASTICITY`, `*CONCRETE COMPRESSION HARDENING` and
`*CONCRETE TENSION STIFFENING`. Tensile and compressive damage are optional
stiffness-degradation definitions. The legacy full backend therefore exports the
required blocks and conditionally emits the two damage blocks.

## Scalar parameters and defaults

`*CONCRETE DAMAGED PLASTICITY` takes dilation angle, flow-potential
eccentricity, the biaxial/uniaxial initial compressive yield-stress ratio,
`Kc`, and viscosity. Abaqus documents defaults of 0.1 for eccentricity, 1.16
for the biaxial ratio, 2/3 for `Kc`, and 0 for viscosity. WEB-M4 does **not**
replace the repository's historical `dilation_angle`, `fbfc` or `Kc`: those
remain the exact `calculate_cdp_parameters` legacy calibration. Only quantities
that legacy never calculated use documented backend defaults: eccentricity 0.1
and viscosity 0.0, unless the user explicitly overrides them.

## Compression hardening and damage

Compression hardening is supplied as compressive yield stress versus inelastic
(crushing) strain, with the first inelastic strain equal to zero. Compression
damage uses the same inelastic-strain coordinates. Abaqus requires the first
compression-damage point at each condition to have both crushing strain and
damage equal to zero. The historical generator produces a very small nonzero
first damage in representative cases, so the adapter alone normalizes that
first damage value to exactly 0.0 and records a backend-normalization event.

## Tension: STRAIN, DISPLACEMENT and GFI

Abaqus supports postcracking tension input as cracking strain, cracking
displacement, or failure stress plus fracture energy (`GFI`). The `GFI` form
assumes a linear strength loss. The repository's historical bilinear and
power-law laws already contain explicit nonlinear stress versus crack-opening
arrays. Exporting those arrays as `GFI` would therefore change the constitutive
law. WEB-M4 exports them as:

```text
*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT
```

and, when enabled, uses matching cracking-displacement coordinates for:

```text
*CONCRETE TENSION DAMAGE, TYPE=DISPLACEMENT
```

No resampling is performed.

## Damage cap and coordinate alignment

The Abaqus material guide recommends avoiding damage values above 0.99 and
strongly recommends that damage tables use the same cracking/inelastic
coordinates as the corresponding stiffening/hardening tables. Historical tensile
damage reaches 1.0 at zero stress. WEB-M4 caps damage at 0.99 **only in the
Abaqus export representation**, leaves the historical arrays unchanged, and
records every changed row. Compression damage receives the same cap if needed.
Coordinates remain exactly aligned with the source legacy arrays.

## REF LENGTH is distinct from legacy `l_ch`

For displacement-based tensile damage Abaqus converts cracking displacement to
plastic displacement using a specimen/reference length `l0`. The keyword
parameter is `REF LENGTH`; Abaqus defaults it to one unit length and recommends
specifying it explicitly to make the unit convention clear.

WEB-M4 exposes `damage_conversion_reference_length_mm`, default 1.0 in the
repository N-mm-MPa convention. It is deliberately **not** inferred from the
historical crack-band `l_ch` input and is not inferred from the legacy
`calculate_characteristic_length(...)` property. The JSON/UI state this
separation explicitly.

## Plastic strain/displacement validation

Abaqus converts user tables internally. WEB-M4 independently checks the same
relationships before offering an executable `.inp` material card.

Compression:

```text
epsilon_c_pl = epsilon_c_in - d_c/(1-d_c) * sigma_c/E0
```

Tension with displacement input:

```text
u_t_pl = u_t_ck - d_t/(1-d_t) * sigma_t * REF_LENGTH/E0
```

The resulting sequences must be finite, nonnegative within a tight numerical
tolerance, and nondecreasing. The first compression point resolves to zero.
Coordinate alignment and damage bounds are checked as well. A hard violation
raises a typed validation error; the exporter does not hide or repair a broken
plastic-strain/displacement sequence.

## Calibration boundary

The full material is a **legacy compatibility calibration** combining the
repository's historical physical/curve implementation with Abaqus documented
backend requirements/defaults. Old comments such as `CEB-90` or `FIB2010` are
not promoted to verified normative provenance. Every scalar/table identifies
whether its source is legacy implementation, Abaqus documentation default,
backend normalization, or explicit user backend override.

Rate- and temperature-dependent full Abaqus tables are deferred. The current
backend always builds the static reference legacy case (`strain_rate=0`) from
`calculate_stress_strain(...)`; the existing legacy rate/temperature plots remain
analysis visualizations only.
