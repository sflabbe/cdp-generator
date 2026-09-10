# Concrete authority architecture — G0

## Status

G0 establishes an authority split while preserving the repository's pre-existing concrete behavior. The current implementation remains available through the original top-level functions and CLI. No G0 change certifies the existing mixed formulas as an authoritative implementation of fib Model Code 2010 or Eurocode 2.

The compatibility identity for existing physical-property behavior is **`legacy_v1`**. Its meaning is deliberately narrow: reproduce the implementation that existed at the G0 starting baseline.

## Domain separation

The new namespace makes three domains explicit:

```text
physical concrete
    |
    +-- standard/profile layer
            |
            +--> ABAQUS CDP backend      (abaqus_cdp_legacy in G0)
            |
            +--> Grassl CDPM2 backend    (reserved for G2)

FE / integration context
    +--> characteristic length, mesh diagnostics, regularization context
```

`ConcretePhysicalProperties` contains intrinsic/input physical quantities such as strengths, elastic moduli, Poisson ratio, shear modulus, fracture energy, and compression-strain inputs. It does **not** contain ABAQUS-CDP scalars such as dilation angle, `Kc`, or `fb/fc`.

`AbaqusCdpParameters` contains those ABAQUS-CDP-specific scalars. They are produced only by an explicit constitutive conversion (`Concrete.to_abaqus_cdp()`), not by the physical schema itself.

Characteristic element length (`l_ch`) and the legacy characteristic-length diagnostic (`l0`) are not intrinsic material properties. They belong to FE/integration and mesh-regularization context and therefore are intentionally absent from `ConcretePhysicalProperties`. The legacy output dictionary is unchanged for backward compatibility.

## Provenance

Every generated physical field in the new schema carries immutable `PropertyProvenance` metadata. G0 provenance uses `source_id="legacy_v1"` and `LEGACY_IMPLEMENTATION` or `DERIVED` origin kinds rather than inventing normative citations. Statistical basis is explicit where meaningful, for example `MEAN` for `f_cm`/`f_ctm` and `CHARACTERISTIC` for `f_ck`.

The provenance model reserves distinct origin kinds for verified standards, literature calibrations, user overrides, and derived values. Future gates can therefore replace individual authorities without changing the physical schema or conflating them with constitutive-model calibration.

## G0 facade

```python
from cdp_generator.concrete import Concrete

concrete = Concrete.from_mean_strength(
    f_cm=38.0,
    e_c1=0.0022,
    e_clim=0.0035,
    profile="legacy_v1",
)

physical = concrete.physical
abaqus = concrete.to_abaqus_cdp(calibration="abaqus_cdp_legacy")
```

`Concrete.from_class(...)` is deliberately a `NotImplementedError` stub in G0. Concrete-class and verified standard resolution are G1 work. `Concrete.to_cdpm2(...)` is likewise a G2 stub and implements no Grassl formulas.

## Reserved future identities

Physical standard/profile identities reserved for G1 are:

- `fib_mc2010`
- `ec2_2023`
- `ec2_2004`

The constitutive backend identity reserved for G2 is `cdpm2_grassl_2013`.

The architecture is intended to allow future mixed provenance such as EC2 physical properties with fib fracture energy and a CDPM2 calibration. G0 does not implement those combinations or formulas.

## Qualification strategy

`qualification/legacy_v1_baseline.json` was generated from the G0 starting implementation before the authority layer was added. It freezes representative scalar properties, full static stress-strain outputs for six strengths, a multi-rate case, a temperature case, and the legacy tensile-strength branch immediately below, at, and above `f_cm = 58 MPa`.

G0 tests compare both the legacy API against that frozen baseline and the new adapters directly against the existing functions. This makes G0 an architectural seam rather than a formula migration.
