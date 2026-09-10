# Standards authority dossier for G1

## Purpose and scope

This document is the human-readable companion to
`qualification/standards/authority_matrix.json`. It records the authority and
field semantics that must be preserved when G1-B later implements verified
physical concrete profiles.

G1-A is an authority gate, not a formula implementation gate. The reserved
profiles `fib_mc2010`, `ec2_2004`, and `ec2_2023` remain non-operational here.
No statement in legacy code, comments, function names, or historical labels is
accepted as normative evidence merely because it says “FIB2010”, “Eurocode”,
or “EC2”. `legacy_v1` remains compatibility authority only.

## Authority hierarchy

The authority hierarchy is:

1. Primary normative authority: the final approved fib Model Code 2010
   publication and the applicable editions of EN 1992-1-1.
2. Supporting authoritative dissemination: material issued by CEN/JRC or
   equivalent official bodies that reproduces or explains the relevant
   second-generation Eurocode content.
3. Development oracle: StructuralCodes, used only to independently cross-check
   equations, branches, symbols, and expected values.

StructuralCodes is **not** normative authority and is **not** a runtime
repository dependency. A future implementation must remain valid without it.

For fib Model Code 2010, Bulletins 65/66 are the final approved Model Code 2010
publication sequence and are distinct from the earlier draft Bulletins 55/56.
The definitive 2013 fib/Ernst & Sohn publication is the discrepancy authority
for final-edition equation and table locators.

For EN 1992-1-1:2004, the G1 profile scope includes the effect of AC:2010 and
A1:2014 on the physical-property clauses under review. AC:2010 changes the
printed cap condition for `epsilon_c1` in Table 3.1 from `< 2.8` to `<= 2.8`;
this does not change the numerical `min(..., 2.8)` result. A1:2014 does not
modify the scoped Section 3.1 physical-property clauses.

For EN 1992-1-1:2023, the relevant physical-property authority is principally
Section 5 and Table 5.1. Supporting JRC material confirms the reference-age and
class-family decisions recorded below.

## Authority status vocabulary

Each field in the machine-readable matrix is assigned one of four states:

- `direct`: the governing standard explicitly defines or tabulates the
  quantity.
- `derived`: the repository quantity is not itself a directly tabulated field,
  but its mapping or derivation is authorized and its dependencies are known.
- `unresolved`: the evidence or required context is insufficient for automatic
  population.
- `composed-required`: the selected profile does not supply the property; a
  separate, explicit authority must be composed in later.

`unresolved` and `composed-required` are valid physical-schema states. They are
not invitations to borrow silently from another profile.

## Canonical repository units and sign conventions

The physical schema uses MPa for stress/modulus, N/mm for fracture energy, and
dimensionless strain. Unit conversion is provenance-relevant whenever the
source unit differs.

Compression landmarks are stored as **positive magnitudes** in the repository.
Where a source publishes compression as negative strain, the sign conversion
must be represented as machine-readable normalization/derivation metadata. A
free-text note alone is not sufficient provenance for the conversion.

Characteristic element length (`l_ch`, `l0`, or equivalent) remains outside
intrinsic physical material properties.

## Field semantics by profile

### Strengths and tensile characteristic values

All three profiles directly supply characteristic compressive strength `f_ck`,
mean compressive strength `f_cm`, and mean tensile strength `f_ctm` within
their authorized ranges.

The cross-standard physical schema should use the neutral names
`f_ctk_lower` and `f_ctk_upper`.

- MC2010 maps `f_ctk,min` and `f_ctk,max` to those fields. G1 must not invent
  percentile aliases for MC2010 merely because the numerical factors resemble
  EC2 fractiles.
- EC2:2004 maps `f_ctk,0.05` and `f_ctk,0.95` directly. EC2-facing APIs may
  expose `f_ctk_005` and `f_ctk_095` aliases.
- EC2:2023 likewise maps `f_ctk,0.05` and `f_ctk,0.95` directly and may expose
  the EC2-specific aliases.

For EC2:2023, the `fctm` branch boundary is authoritative at `f_ck = 50 MPa`:
`f_ck <= 50 MPa` belongs to the lower-strength branch and `f_ck > 50 MPa` to
the high-strength branch. This is a standard-profile decision and must not be
inferred from the separate G0 legacy sentinel expressed around `f_cm = 58 MPa`.

### `E_initial`

Repository field `E_initial` means the **initial/tangent elastic modulus
appropriate to the physical material state and reference-age policy selected
by the profile**. It is not a generic synonym for whichever elastic modulus a
standard happens to supply.

For MC2010, `E_initial` maps directly to `E_ci`, the mean initial/tangent
modulus defined in the Model Code.

For EC2:2004, `E_cm` is the normative secant modulus, not `E_initial`. The
selected general/Sargin stress-strain relation authorizes a tangent modulus
`E_c`; the standard permits `1.05 E_cm` as an approximation. Consequently the
repository mapping to `E_initial` is `DERIVED`, with `E_cm` as a dependency,
and must not be described as a directly tabulated initial modulus.

For EC2:2023, `E_cm` is again a secant modulus. `E_c,28` is the tangent modulus
at 28 days and may be approximated as `1.05 E_cm,28`; Annex B/testing may
provide a more accurate route. At `reference_age_days == 28`, an explicitly
selected profile policy may therefore derive `E_initial` through that
standard-authorized relation. At `reference_age_days > 28`, the implementation
must not substitute the reference-age `E_cm` for `E_cm,28`. If the information
needed to reconstruct the 28-day tangent state is absent, `E_initial` remains
`None`/unresolved in the base profile.

### `E_secant`

`E_secant` has profile-specific authority but a common repository semantic: a
secant elastic modulus, not an initial tangent modulus.

- MC2010 maps to its reduced/secant `E_c` from the Model Code elasticity
  section, not to the Sargin origin-to-peak secant quantity.
- EC2:2004 maps to `E_cm` from Section 3.1/Table 3.1.
- EC2:2023 maps to `E_cm`, explicitly defined as the secant modulus between
  `sigma_c = 0` and `sigma_c = 0.4 f_cm`.

Aggregate/stiffness policy is edition-specific. MC2010 exposes `alpha_E` by
aggregate type, with quartzite as a reference value rather than a universal
hidden material truth. EC2:2004 Table 3.1 values are for quartzite and gives
aggregate-dependent modifiers. EC2:2023 uses `k_E`; `9500` is the quartzite
assumption used by the development oracle, while the standard range is
5000–13000 and National Annex values may apply.

### Poisson ratio and shear modulus

`poisson_elastic` represents the uncracked elastic Poisson-ratio policy. The
authority matrix records `nu = 0.20` for the elastic/uncracked state for the
three profiles, with the source qualifications retained in machine-readable
notes and constraints.

The current G0/v1 field `shear_modulus` is **not** treated as a directly
standard-specified shear modulus. G1-A selects the preferred v2 design:

```text
shear_modulus
    -> shear_modulus_secant_equivalent
```

Its repository definition is:

```text
G = E_secant / [2 (1 + poisson_elastic)]
```

and its semantic identity is **derived isotropic secant-equivalent shear
modulus**. Provenance must carry:

```text
source_kind = DERIVED
derived_from = ("E_secant", "poisson_elastic")
```

The rename is intended to prevent callers from confusing this derived quantity
with a directly specified or tangent shear modulus.

### Compression strain landmarks

The physical schema uses landmarks of the general nonlinear/Sargin material
relation, not the simplified design-law strains.

- MC2010 uses `epsilon_c1` and `epsilon_c,lim` from the Sargin/general
  nonlinear relation and the associated table. Published compression signs
  are normalized to positive repository magnitudes with provenance.
- EC2:2004 uses `epsilon_c1` and `epsilon_cu1` associated with the general
  nonlinear relation in Section 3.1.5/Table 3.1.
- EC2:2023 uses `epsilon_c1` and `epsilon_cu1` from the corresponding general
  nonlinear relation (Eqs. 5.9 and 5.10 in the authority matrix).

Simplified parabolic-rectangular or bilinear design-law strains are distinct
quantities and must not silently populate these physical fields.

### Fracture energy

The fracture-energy policy is deliberately asymmetric:

- `fib_mc2010`: `fracture_energy` is `DIRECT` from `G_F`.
- `ec2_2004`: `fracture_energy` is `COMPOSED_REQUIRED`.
- `ec2_2023`: `fracture_energy` is `COMPOSED_REQUIRED`.

No implicit rule of the form “EC2 fracture energy = fib formula” is permitted.
If an EC2 physical profile later uses MC2010 fracture energy, that must be an
explicit cross-standard composition with its own provenance chain.

## Reference-age policy

MC2010 and EC2:2004 reference properties in the present G1 class/profile scope
are 28-day quantities unless the selected time-development context explicitly
states otherwise.

EC2:2023 makes reference age a required profile context. The normal reference
age is 28 days, but the project may specify `t_ref` from 28 to 91 days. G1-B
therefore needs `reference_age_days` in the physical schema and must preserve
age dependencies in provenance. In particular, a profile evaluated at
`t_ref > 28 d` must not erase the distinction between reference-age `E_cm` and
28-day tangent `E_c,28`.

## Class identity authority

Class identity authority is separate from property-formula authority. Knowing a
numerical cylinder `f_ck` does not authorize synthesizing an EN class string or
a cube strength.

The class registry in `authority_matrix.json` is therefore explicit and is the
future parser authority:

- `fib_mc2010`: `C12, C16, C20, C25, C30, C35, C40, C45, C50, C55, C60,
  C70, C80, C90, C100, C110, C120`. MC2010's canonical repository syntax is
  `C<number>`; its table separately authorizes the cylinder/cube pair.
- `ec2_2004`: canonical EN `Cxx/yy` identities through the edition-supported
  `C90/105` baseline represented in the registry. The base Eurocode/National
  Annex `Cmax` qualification remains explicit rather than inferred.
- `ec2_2023`: canonical EN classes from `C12/15` through `C100/115`.

The future `Concrete.from_class(...)` implementation should consume the
explicit registry rather than compute cube strength algorithmically.

## Schema implications for G1-B

G1-A concludes that `ConcretePhysicalProperties.v2` is required. This is not a
cosmetic version bump: truthful standard profiles need to represent normative
absence and new semantics that v1 cannot express.

Proposed v2 changes are:

- add `f_ctk_lower` and `f_ctk_upper`;
- add `reference_age_days`;
- allow `E_initial` to be optional/unresolved;
- allow `fracture_energy` to be optional/unresolved;
- rename `shear_modulus` to `shear_modulus_secant_equivalent`;
- retain `E_secant`, `poisson_elastic`, and the compression landmarks with the
  explicit semantics in this dossier;
- keep compression strains as positive magnitudes;
- keep FE characteristic length outside intrinsic physical material data.

`PropertyProvenance` also requires a machine-readable immutable dependency
relation such as:

```text
derived_from: tuple[str, ...]
```

or an equivalent stable identifier structure. Free-text `notes` alone cannot
reliably encode derived modulus relations, shear derivation, unit/sign
normalization, or cross-standard composition chains.

Because the physical field set, optionality, names, age context, and provenance
semantics change, the serialized concrete material definition should receive a
schema-version bump alongside physical schema v2.

EC2-specific aliases `f_ctk_005` and `f_ctk_095` may exist at an EC2-facing API
boundary. The cross-standard physical schema remains `f_ctk_lower` /
`f_ctk_upper`; those percentile names must not be projected onto MC2010.

## Development-oracle policy

StructuralCodes may be used in G1-B tests as an independent development oracle
for formulas, branch thresholds, and representative values. It remains outside
runtime dependencies and outside normative provenance. Any disagreement
between StructuralCodes and the primary standard must be adjudicated in favor
of the primary authority, with the discrepancy recorded rather than hidden.

## Remaining authority question

No remaining question blocks G1-A closure. One implementation question is
explicitly deferred: how a later age-development layer should compute the
required tangent modulus when EC2:2023 uses `t_ref > 28 d`. G1-B base profiles
must leave `E_initial` unresolved in that situation unless an explicit
age-development context supplies the necessary information.

G1-B must not begin until this dossier and the machine-readable matrix have
passed repository validation. G1-B remains a schema/profile implementation
step; CDPM2 remains outside its scope.
