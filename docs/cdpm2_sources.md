# CDPM2 G2-A authority dossier

## Scope and gate

This dossier closes **G2-A — CDPM2 Grassl 2013 authority + exact parameter-contract adjudication**. It does not implement a constitutive backend, does not activate `Concrete.to_cdpm2(...)`, and does not change the frozen G1 physical-concrete layer.

The frozen architecture invariant is:

```text
NORMATIVE PHYSICAL AUTHORITY
!=
CONSTITUTIVE MODEL AUTHORITY
!=
BACKEND / SOLVER INTERFACE AUTHORITY
```

Starting `cdp-generator` authority:

- HEAD `ea6af160cb3765214c948f7ea1371d20983f524c` — `Qualify verified standards integration`
- tree `9ff1597cd99eb2e09e52f94deb36a39f825359ea`
- all closed G1 artifacts retain the SHA256 values recorded in `qualification/cdpm2/authority_matrix.json`.

## Source hierarchy actually inspected

### Tier 1 — governing scientific model

Peter Grassl, Dimitrios Xenos, Ulrika Nyström, Rasmus Rempling and Kent Gylltoft, **“CDPM2: A damage-plasticity approach to modelling the failure of concrete”**, *International Journal of Solids and Structures* 50(24), 3805–3816 (2013), DOI `10.1016/j.ijsolstr.2013.07.008`.

- DOI: https://doi.org/10.1016/j.ijsolstr.2013.07.008
- author/arXiv full text inspected: https://arxiv.org/html/1307.6998v1
- author publication page: https://petergrassl.com/publications/graxennys13/

This source governs model semantics and equations. The effective stress law uses an elastic tensor built from Young's modulus `E` and Poisson ratio `nu`; the nominal law uses separate tensile and compressive damage variables. Section 5 identifies `E`, `nu`, `f_t`, `f_c` and `G_Ft` as the physical constants adjusted to each concrete.

### Corrigendum

Inspected: https://petergrassl.com/tempFiles/CDPM2corrigendum.pdf

The corrigendum explicitly says the governing equations in the 2013 paper are correct. It corrects the reported `A_s` for the Kupfer uniaxial/biaxial comparison from `1.5` to **`7`** and replaces affected comparison figures. Therefore `A_s=7` is **dataset-specific correction evidence, not a universal model default**.

### Tier 1b — inherited CDPM1 theory

Grassl & Jirásek, **“Damage-plastic model for concrete failure”**, IJSS 43 (2006), 7166–7196, is used only where the 2013 paper explicitly inherits definitions/calibration. In particular, the 2013 paper states that its hardening ductility definition is identical to CDPM1 and reuses the parameter set `A_h=0.08`, `B_h=0.003`, `C_h=2`, `D_h=1e-6`.

### Tier 2 — Grassl/OOfem implementation authority

Grassl Group CDPM2/OOFEM page: https://petergrassl.com/research/damage-plasticity/cdpm2-oofem/

The 2022 development manual was inspected: https://petergrassl.com/tempFiles/manualOOFEMDevelCDPM2.pdf

It exposes the implementation input vocabulary and defaults, including bilinear `stype=1`, `ft1=0.3`, `wf1=0.15`, `efc=100e-6`, `ecc=0.525`, `kinit=0.3`, `Asoft=15`, `dilation=0.85`, `hp=0.01`, `isoflag=0`, `rateflag=0`, plus `helem`, density, thermal dilation, `yieldtol`, and `newtoniter`. The same manual explicitly recommends `hp=0.5` when rate dependence is enabled; G2 static V1 does not enable that extension.

The current Grassl-group OOFEM implementation comparator was inspected at:

- repository `githubgrasp/oofem`
- commit `0eff9ee96297199a5f4cac2969bbc7fcecacd4de`
- `src/sm/Materials/ConcreteMaterials/concretedpm2.C`

The current source has drifted from the 2022 input-manual defaults in at least two interface details: it initializes `yieldHardPrimePeak`/`hp` to `0.5`, and its current `damageFlag` numbering/default differs from the older manual. These are recorded as implementation-interface drift, not as changes to the 2013 governing equations. Static V1 keeps `H_p=0.01` because that value is explicitly stated by Grassl et al. 2013 and by the 2022 static manual, and it represents damage semantically as `two_damage_variables` rather than exposing either OOFEM numeric flag convention.

OOFEM is used to adjudicate implementation defaults and input semantics, not to redefine the scientific equations or G1 physical authority.

### Tier 3 — Grassl LS-DYNA/UMAT comparator

Grassl Group page: https://petergrassl.com/research/damage-plasticity/cdpm-lsdyna/

The author-maintained UMAT repository was inspected at:

- repository `githubgrasp/cdpm2`
- commit `95fb5a4caeaabaab4c48dc46a38989a002becb6f`
- `cdpm2umat.f`

The UMAT confirms the 24-slot vocabulary and implementation defaults including `A_s=15`, `B_s=1`, `epsilon_fc=1e-4`, `w_f1=0.15 w_f`, `f_t1=0.3 f_t`, static rate flags zero and the legacy sentinel behavior. `B_s` is an implementation extension: it is not a 2013-paper input parameter and is not exposed by the OOFEM manual.

### Tier 4 — downstream compatibility target

Actual `sflabbe/cdpm2-kratos` authority at G2-A start:

- HEAD `b98bef2c753bb57e4890ff385d8eea15c185e0ef` — `Implement selected C4 solver tangent`
- tree `27ba50e0031ee46db32f7d500b7aaf8f16229089`

Inspected at that commit:

- `docs/cdpm2/c0_slot_inventory.md`
- `docs/cdpm2/c1_parameter_map.md`
- `docs/cdpm2/c1_constitutive_contract.md`
- `docs/cdpm2/c1_algorithm_flow.md`
- `docs/cdpm2/c3_numerical_adjudication.md`
- `docs/cdpm2/c8_independent_evidence.md`
- `legacy/source/candidate/src/cdpm2_legacy.f`
- `legacy/source/candidate/docs/MATERIAL_PARAMETERS.md`

`cdpm2-kratos` is **downstream parity/interface authority only**. Its 24 slots do not define the semantic public material API.

## Three-layer contract

### Layer A — semantic CDPM2 constitutive material

The future effective `Cdpm2Grassl2013Parameters` contract is frozen by `semantic_parameter_contract.json`.

Public/effective semantic fields are:

```text
model_id = cdpm2_grassl_2013
calibration_id = grassl_2013_static_default_v1

E
nu
f_t
f_c
G_Ft
w_f        # derived effective value
w_f1       # derived effective value
f_t1       # derived effective value
eccentricity
q_h0
H_p
D_f
A_h
B_h
C_h
D_h
A_s
epsilon_fc
tensile_softening_type = bilinear
damage_formulation = two_damage_variables
```

`B_s` is deliberately **not** a public V1 field. The downstream/LS-DYNA extension is fixed to `1`, which collapses the generalized implementation form back to the 2013 paper's Eq. (56).

### Layer B — backend adapter

The future adapter maps the semantic contract to the exact downstream `cm(1:24)` interface. All 24 slots and their dispositions are frozen in `backend_compatibility_contract.json`.

Legacy sentinels are **not** public semantics. The future adapter resolves explicit effective values first and serializes those values to the downstream slots.

### Layer C — runtime / FE / solver context

These are excluded from the semantic material object:

- `LCHAR/helem/element size`
- all 27 history/state slots
- `FAILFLG` deletion policy
- `PRINTFLAG`
- `yieldtol`, `newtoniter`
- finite-difference tangent steps and adaptive depth
- density and thermal expansion
- rate-control flags/parameters in static V1

`LCHAR` is mandatory runtime crack-band context for tensile damage and is passed separately from the 24 material slots. The 2013 paper adjusts tensile softening with element size but keeps the compressive damage evolution independent of element size; no compressive mesh-objectivity claim is introduced here.

## G1 -> CDPM2 physical mapping adjudication

### `E <- E_initial`

**Decision: `E_initial`, never `E_secant`.**

Grassl 2013 Eq. (2) defines `D_e` from the elastic Young modulus, and the cyclic response returns to compression with the **original Young modulus of the undamaged material**. That semantics is tangent/initial elastic stiffness. G1 intentionally distinguishes `E_initial` from `E_secant`.

Consequences:

- `fib_mc2010`: `E_initial = E_ci` is directly available.
- `ec2_2004`: the qualified G1 tangent approximation is available as derived `E_initial`.
- `ec2_2023 @ 28 d`: qualified `E_initial` is available as derived `E_c,28` approximation.
- `ec2_2023 @ t_ref > 28 d`: `E_initial` is intentionally `UNRESOLVED`; automatic CDPM2 conversion is blocked. **No fallback to `E_secant` is allowed.**

### `nu <- poisson_elastic`

The CDPM2 elastic tensor uses the initial elastic Poisson ratio. G1 `poisson_elastic` is the matching uncracked elastic quantity. Damaged/plastic effective ratios are not substituted.

### `f_t <- f_ctm`

Grassl's `f_t` is a constitutive uniaxial tensile strength fitted to material tests. The generic standards mapping therefore uses the **mean** tensile strength `f_ctm`, not characteristic lower/upper fractiles intended for reliability/design semantics.

### `f_c <- f_cm`

Grassl's `f_c` is the uniaxial compressive strength used to normalize the yield surface and fitted to measured concrete. The generic mapping therefore uses the **mean** compressive strength `f_cm`, not class characteristic `f_ck`.

### `G_Ft <- fracture_energy`

The 2013 paper treats tensile fracture energy as a physical constant directly related to softening parameters.

- `fib_mc2010`: G1 `fracture_energy` is available directly -> potentially `READY`.
- `ec2_2004`: `fracture_energy=None / COMPOSED_REQUIRED` -> conversion requires explicit composition or override.
- `ec2_2023`: same composition requirement.

No EC2 profile may silently call the MC2010 fracture-energy relation. Future composition must be explicit and provenance-bearing, or a constitutive `G_Ft` override must be explicitly identified as a user constitutive override.

### `legacy_v1`

**Decision: not accepted automatically by the verified static CDPM2 V1 mapping.**

`legacy_v1` contains numerically usable values, but its authority role is historical compatibility, not verified normative physical authority. A later, explicitly named legacy-compatibility conversion may be added, preserving legacy provenance; it is not part of `grassl_2013_static_default_v1`.

## Bilinear tensile softening

Grassl 2013 Eq. (59):

```text
G_Ft = f_t*w_f1/2 + f_t1*w_f/2
```

For the canonical ratios:

```text
w_f1 / w_f = 0.15
f_t1 / f_t = 0.30
```

the exact coefficient is:

```text
G_Ft = 0.225 * f_t * w_f
w_f = G_Ft / (0.225 * f_t)
w_f1 = 0.15 * w_f
f_t1 = 0.30 * f_t
```

The public contract records the exact `0.225` relation; it does not hard-code the paper's rounded reciprocal `4.444`.

Static V1 fixes bilinear softening. Linear and exponential variants are deferred extensions. Downstream `TYPE=3` remains unsupported/quarantined and is not normalized into a semantic option.

## Eccentricity policy

The effective `eccentricity` is a CDPM2 constitutive parameter, not a physical-standard property.

For `grassl_2013_static_default_v1`, the baseline follows Grassl 2013 Eq. (60):

```text
f_bc = 1.16 * f_c
epsilon = (f_t/f_bc) * (f_bc^2 - f_c^2) / (f_c^2 - f_t^2)
e = (1 + epsilon) / (2 - epsilon)
```

The OOFEM implementation default `0.525` is retained as a documented alternative implementation-default policy, not silently substituted for the literature calibration policy. A future explicit constitutive override is allowed with constitutive-override provenance.

## Hardening and flow defaults

Static baseline:

```text
q_h0 = 0.3
H_p  = 0.01
D_f  = 0.85
A_h  = 0.08
B_h  = 0.003
C_h  = 2
D_h  = 1e-6
```

These are constitutive calibration/default values, not G1 physical-standard properties. `E_h` and `F_h` are internal derived helpers from Eqs. (35)-(36), not public inputs.

## Damage ductility `A_s` and `B_s`

Grassl 2013 Eq. (56) is:

```text
x_s = 1 + (A_s - 1) R_s
```

`A_s` is a model parameter and is dataset-sensitive. The 2013 examples use different values. The corrigendum changes only the affected Kupfer comparison to `A_s=7`.

The author-maintained OOFEM manual and LS-DYNA UMAT use an implementation default `A_s=15`. G2 static V1 adopts **15 as a named implementation-default baseline**, while preserving it as user-selectable constitutive calibration; it is not labelled a universal 2013-paper default.

The LS-DYNA/downstream `B_s` exponent is not a 2013-paper public parameter. Static V1 fixes `B_s=1`, recovering Eq. (56), and does not expose it as a public semantic field.

## Compression softening `epsilon_fc`

`epsilon_fc` is the positive inelastic-strain threshold in the exponential compressive softening law (2013 Eq. (55)). It is not G1 `epsilon_c1` or `epsilon_cu1`, and no such mapping is authorized.

The static baseline is `epsilon_fc=1e-4`, supported by the repeated 2013 compression comparisons and the OOFEM/LS-DYNA implementation defaults. It remains a constitutive calibration field that may be explicitly overridden later.

The paper's compressive energy scale is:

```text
G_Fc = f_c * epsilon_fc * l_c * A_s
```

This is not G1 tensile `fracture_energy`. The downstream candidate does not pass `LCHAR` into its compression damage routine; G2-A makes no claim of explicit compressive crack-band regularization.

## Damage-model variant

Canonical model identity is two separate damage variables `omega_t` and `omega_c`.

Static V1 therefore fixes:

```text
damage_formulation = two_damage_variables
```

For the frozen downstream candidate, the numeric adapter value is:

```text
DAMAGEFLAG = 0
```

because source branch 0 actually implements the two-variable spectral tension/compression reconstruction. The candidate's bundled manual labels conflict with source behavior; this is an adapter quirk and is recorded, not “corrected” in cdp-generator.

## Rate, failure, print and solver controls

Static V1 is rate-independent:

```text
ERATETYPE = 0
SRATETYPE = 0
```

Rate dependence is a deferred extension. `FAILFLG=0` disables candidate element/IP failure semantics for the static material contract. `PRINTFLAG` is nonmechanical and fixed to a neutral caller value; the legacy oracle may alter a private copy for banner behavior.

OOFEM `yieldtol/newtoniter` and downstream FD/substepping controls remain solver/integration configuration, never fields of `Cdpm2Grassl2013Parameters`.

## Canonical units and signs

Repository semantic units are frozen as:

```text
E, f_t, f_c, f_t1    MPa
nu and calibration   dimensionless
G_Ft                  N/mm
w_f, w_f1, LCHAR      mm
epsilon_fc            dimensionless
```

`f_t`, `f_c` and `epsilon_fc` are stored as positive magnitudes. The downstream adapter requires a consistent unit system and must not introduce hidden conversions.

## Conversion readiness

Machine-readable statuses are:

```text
READY
UNRESOLVED_PHYSICAL_INPUT
COMPOSITION_REQUIRED
NOT_AUTHORIZED_PHYSICAL_SOURCE
UNSUPPORTED_CONFIGURATION
```

Summary:

| G1 source | E | nu | f_t | f_c | G_Ft | automatic static readiness |
|---|---|---|---|---|---|---|
| `legacy_v1` | not authorized | not authorized | not authorized | not authorized | not authorized | `NOT_AUTHORIZED_PHYSICAL_SOURCE` |
| `fib_mc2010` | `E_initial` direct | direct | `f_ctm` direct | `f_cm` direct | direct | `READY` |
| `ec2_2004` | `E_initial` derived | direct | direct | direct | composition required | `COMPOSITION_REQUIRED` |
| `ec2_2023 @ 28 d` | `E_initial` derived | direct | direct | direct | composition required | `COMPOSITION_REQUIRED` |
| `ec2_2023 @ >28 d` | unresolved | direct | direct | direct | composition required | `UNRESOLVED_PHYSICAL_INPUT` + `COMPOSITION_REQUIRED` |

## Exact 24-slot adapter disposition

The complete machine-readable map is `qualification/cdpm2/backend_compatibility_contract.json`. The critical fixed static slots are:

```text
14 ERATETYPE = 0
15 TYPE       = 1       # bilinear
16 BS         = 1       # compatibility extension; recovers 2013 Eq. 56
20 SRATETYPE  = 0
21 FAILFLG    = 0
23 DAMAGEFLAG = 0       # candidate two-damage-variable branch
24 PRINTFLAG  = 0       # nonmechanical caller value
```

Slots 1–13, 17–19 and 22 receive explicit effective semantic values. No public sentinel values are emitted.

`LCHAR` is **not** slot 25. It remains a separate runtime argument. The downstream 27-slot state array likewise remains runtime state and is never part of the parameter object.

## Public override policy and provenance

A future API may permit explicit overrides of mapped physical inputs and calibratable constitutive values, but the provenance domain must remain explicit:

```text
PHYSICAL_SOURCE
GRASSL_2013_DIRECT
GRASSL_2013_DERIVED
GRASSL_2013_DEFAULT_CALIBRATION
OOFEM_IMPLEMENTATION_DEFAULT
DOWNSTREAM_ADAPTER_FIXED
USER_CONSTITUTIVE_OVERRIDE
```

An override of `E`, `f_t`, `f_c` or `G_Ft` is a constitutive user override; it must never retain an EC2/fib source identity as though the standard produced that overridden value.

## Deferred extensions

Not part of static V1:

- strain-rate / dynamic increase factors
- rate-energy scaling
- linear or exponential tensile-softening public variants
- single/isotropic/no-damage variants
- element deletion / nonzero `FAILFLG`
- fire/thermal coupling
- density and thermal expansion in the CDPM2 semantic object
- broad solver tolerance configuration
- alternative backend-specific calibrations
- legacy_v1 compatibility conversion

## G2 roadmap after G2-A

Recommended bounded sequence:

```text
G2-A  authority + semantic contract + backend adapter contract       CLOSED here
G2-B  CDPM2 schema/provenance/configuration foundation              NO mechanics
G2-C  physical -> semantic CDPM2 mapping + static default policy    NO constitutive integration
G2-D  exact 24-slot adapter + LCHAR runtime boundary qualification
G2-Q  end-to-end material-factory qualification
```

`cdp-generator` must not duplicate the constitutive mechanics already owned downstream by `cdpm2-kratos`.
