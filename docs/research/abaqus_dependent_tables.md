# Abaqus rate- and temperature-dependent CDP tables — ABAQUS-Q1 / WEB-M5

This note records how the existing legacy rate and temperature curve families are mapped to
Abaqus dependency columns. Abaqus documentation is the authority for **table structure,
keyword semantics and backend requirements only**. The numerical values remain the
repository's historical `LEGACY_IMPLEMENTATION`; nothing here verifies the legacy
rate/temperature laws as normative Abaqus, EC2, CEB or fib laws.

## Documentation consulted (Abaqus 2025 online manual)

| Topic | Page |
| --- | --- |
| Compression hardening data columns | `SIMACAEKEYRefMap/simakey-r-concretecompressionhardening.htm` |
| Tension stiffening (`TYPE=DISPLACEMENT`) columns | `SIMACAEKERRefMap/simaker-c-concretetensionstiffeningpyc.htm` |
| Compression / tension damage | `simakey-r-concretecompressiondamage.htm`, `simakey-r-concretetensiondamage.htm` |
| CDP rate sensitivity, conversion formulas | `SIMACAEMATRefMap/simamat-c-concretedamaged.htm` |
| Isotropic elasticity with temperature | `simaker-c-elasticpyc.htm`, `simamat-c-linearelastic.htm` |
| Ordering of dependent tabular data | `SIMACAEMATRefMap/simamat-c-materialdata.htm` |
| Execution procedure (`datacheck`, `analysis`, `information=`) | `SIMACAEEXCRefMap/simaexc-c-analysisproc.htm` |
| Abaqus Python / `odbAccess` | `SIMACAECMDRefMap/simacmd-c-intpytintrointerpreter.htm` |

All under `https://docs.software.vt.edu/abaqusv2025/English/`.

Findings used:

- `*CONCRETE COMPRESSION HARDENING` data: yield stress, inelastic (crushing) strain, inelastic
  strain rate, temperature, field variables. "The first point at each value of temperature must
  have a crushing strain of 0.0".
- `*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT` data: remaining stress, cracking
  displacement, cracking displacement rate, temperature, field variables.
- `*CONCRETE COMPRESSION DAMAGE` / `*CONCRETE TENSION DAMAGE`: damage, coordinate,
  temperature, field variables — **no strain-rate column**.
- Rate sensitivity of CDP is specified only through the hardening and stiffening tables.
- Material data with several dependencies are given at fixed values of the other variables in
  **ascending** order of the dependency variable; properties are constant outside the given
  range.
- `*ELASTIC` (isotropic) data: E, nu, temperature.

The exported keyword text was checked against these column orders. No Abaqus installation
was available while implementing, so the text is **not** solver-qualified by this note; see the
external gate in `qualification/abaqus/README.md`.

## Strain-rate branch

### Compression rate coordinate — `LEGACY_RATE_AXIS_MAPPING`

The legacy kernel parameterizes curve families by one control `strain_rate` [1/s]. It does not
compute a pointwise inelastic-strain-rate history. The adapter therefore uses the legacy
control rate as the Abaqus compression-hardening rate coordinate and labels it explicitly:

> Legacy curve-family control rate is used as the Abaqus compression-hardening rate dependency
> coordinate. It is not reconstructed from a loading history.

### Tension rate coordinate — `LEGACY_CRACK_OPENING_RATE_MAPPING`

The existing fracture-energy rate model already maps `w_dot = strain_rate * l_ch`. M5 moves
that product into a single helper, `strain_rate.legacy_cracking_displacement_rate(strain_rate,
l_ch)`, which `apply_fracture_energy_rate_effects` now calls; the Abaqus adapter calls the same
helper. Units under the repository convention: 1/s × mm = mm/s. Fracture-energy results are
bit-identical (tests reproduce the historical inline formula exactly, and the frozen legacy
regression baseline still passes).

### Families and ordering

Each requested rate yields the exact kernel arrays: `inelastic stress[i]`, `inelastic
strain[i]`, `crack opening[i]` and the selected bilinear/power-law stress. The kernel itself
already interpolates every family onto the first family's grid; the adapter performs no
further interpolation or resampling. Because Abaqus requires ascending dependency values, the
dependent request requires strictly increasing, unique rates (the legacy analysis itself still
accepts any order).

### Elasticity

Abaqus linear elasticity is rate independent. The rate material exports the static legacy
`E0, nu` of the M4 backend.

## Temperature branch

### Elastic consistency — `LEGACY_TEMPERATURE_SECANT_MODULUS`

The legacy temperature kernel builds each family's inelastic strain as
`eps - sigma / E_c1_temp` in `calculate_inelastic_compression(...)`. For Abaqus to recover the
same total strain (`eps_in + sigma/E`), the elastic modulus at that temperature must be the
same `E_c1_temp`. The nomenclature `E_c1_temp` (secant) vs `E_ci_temp` (tangent) was resolved by
inspecting the kernel: `calculate_stress_strain_temp` passes `E_c1_temp` to
`calculate_inelastic_compression`. `core.calculate_temperature_elastic_states(f_cm, e_c1)`
orchestrates the existing `calculate_concrete_strength_properties`,
`calculate_elastic_modulus`, `get_eurocode_temperature_table` and `apply_temperature_effects`
(through the extracted `legacy_temperature_base_properties`, which the kernel also uses) and
returns that modulus per temperature. A spy test captures the modulus the kernel actually passes
and asserts exact equality; a second test reconstructs every temperature family bit-for-bit.

Consequences that are properties of the legacy kernel, recorded rather than repaired:

- At 20 °C, `E(T) = 20363.6 MPa` for the default `f_cm = 28 MPa, e_c1 = 0.0022`, which differs
  from the static M4 `E0 = E_c = 26171.1 MPa`.
- `E_c1_temp` is independent of the strength reduction factor; from 600 °C upward it is
  constant (`2036.4 MPa` for the default material).

### Poisson ratio — `LEGACY_CONSTANT_NU_ASSUMPTION`

No `nu(T)` exists in the kernel. The legacy room/reference elastic `v_ce` is held constant at
every temperature and labelled as such in JSON, `.inp` comments and UI.

### Hardening and stiffening

Rows are `stress, inelastic_strain, 0, T` and `stress, crack_opening, 0, T`. The rate column is
fixed to zero: M5 does not implement combined rate × temperature surfaces.

## Damage policy

| Policy | Emitted keywords | Semantics |
| --- | --- | --- |
| `omit` (default) | none | No damage keyword; dependent hardening/stiffening only. |
| `reference_damage` | 2-column compression and `TYPE=DISPLACEMENT` tension damage | The legacy kernel's own first-family (reference) damage, normalized with the unchanged M4 backend rules (first compression damage 0, cap 0.99). No rate/temperature column. |

For rates beginning at 0 the reference tables are identical to the M4 static tables (tested).
For the temperature branch the reference is the temperature kernel's own 20 °C damage on its own
40-point grid, so it is not byte-identical to the M4 table, which uses the 20-point
rate-kernel grid.

### Family-wise compatibility validation

With `reference_damage`, every family is checked with the documented conversions

```text
eps_pl = eps_in - d_c/(1-d_c) * sigma_c / E0
u_pl   = u_ck  - d_t/(1-d_t) * sigma_t * REF_LENGTH / E0
```

using **that family's** stress and that family's modulus (`E0` for rates, `E(T)` for
temperatures). Plastic strain/displacement must be finite, non-negative and non-decreasing. Any
failure refuses the material (and therefore `.inp` generation) with
`AbaqusLegacyDependentValidationError`, whose `failures` list names mode, dependency value,
branch and failed condition. Damage is never dropped silently for a failing family.

Truthful outcomes for the default legacy inputs (`f_cm=28, e_c1=0.0022, e_clim=0.0035, l_ch=1,
REF LENGTH=1`):

| Case | `omit` | `reference_damage` |
| --- | --- | --- |
| rates 0, 2, 30, 100 1/s (bilinear or power law) | PASS | REJECTED — 30 1/s, compression, `compression_plastic_strain_monotonic` |
| rates 0, 2 1/s | PASS | PASS |
| temperatures 20…1100 °C (bilinear or power law) | PASS | REJECTED — compression branch at every T ≥ 100 °C (non-monotonic; negative from 200 °C to 1000 °C) |

The validator was not weakened to make the optional policy green.
