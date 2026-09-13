"""G1 physical-to-semantic CDPM2 Grassl 2013 mapping and calibration resolution."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from ...provenance import PropertyProvenance, PropertyResolutionStatus, SourceKind
from ...schema import ConcretePhysicalProperties
from .configuration import (
    Cdpm2ConversionReadiness,
    Cdpm2Grassl2013Configuration,
    Cdpm2ReadinessAssessment,
)
from .provenance import Cdpm2ParameterProvenance, Cdpm2SourceKind
from .schema import (
    CDPM2_PARAMETER_UNITS,
    CDPM2_STATIC_CALIBRATION_ID,
    Cdpm2DamageFormulation,
    Cdpm2Grassl2013Parameters,
    Cdpm2TensileSofteningType,
)

CDPM2_FRACTURE_ENERGY_COMPOSITION_SCHEMA_VERSION = "cdpm2_fracture_energy_composition.v1"

_AUTHORIZED_PHYSICAL_PROFILES = frozenset({"fib_mc2010", "ec2_2004", "ec2_2023"})

_MAPPED_FIELDS: dict[str, str] = {
    "E": "E_initial",
    "nu": "poisson_elastic",
    "f_t": "f_ctm",
    "f_c": "f_cm",
    "G_Ft": "fracture_energy",
}

_DEFAULT_CALIBRATION_VALUES: dict[str, float] = {
    "q_h0": 0.3,
    "H_p": 0.01,
    "D_f": 0.85,
    "A_h": 0.08,
    "B_h": 0.003,
    "C_h": 2.0,
    "D_h": 1e-6,
    "A_s": 15.0,
    "epsilon_fc": 1e-4,
}

_GRASSL_DEFAULT_LOCATORS: dict[str, str] = {
    "q_h0": "Eq. (30); Sec. 5",
    "H_p": "Eqs. (30)-(31); Sec. 5",
    "D_f": "Eqs. (27)-(29); Sec. 5",
    "A_h": "Eq. (33); Sec. 5; inherited calibration from Grassl & Jirasek 2006",
    "B_h": "Eq. (33); Sec. 5; inherited calibration from Grassl & Jirasek 2006",
    "C_h": "Eq. (33); Sec. 5; inherited calibration from Grassl & Jirasek 2006",
    "D_h": "Eq. (33); Sec. 5; inherited calibration from Grassl & Jirasek 2006",
}

# Stable blocker strings are intentionally small API contracts for callers/tests.
BLOCKER_NOT_AUTHORIZED = "physical profile is not authorized for verified static CDPM2 mapping"
BLOCKER_GFT_CONFLICT = (
    "G_Ft physical composition conflicts with explicit constitutive G_Ft override"
)
BLOCKER_GFT_REDUNDANT_COMPOSITION = (
    "secondary fracture-energy composition cannot replace an already resolved physical value"
)
BLOCKER_E_UNRESOLVED = "E_initial unresolved; E_secant fallback forbidden"
BLOCKER_NU_UNRESOLVED = "poisson_elastic unresolved"
BLOCKER_FT_UNRESOLVED = "f_ctm unresolved"
BLOCKER_FC_UNRESOLVED = "f_cm unresolved"
BLOCKER_GFT_UNRESOLVED = "G_Ft physical input unresolved"
BLOCKER_GFT_COMPOSITION_REQUIRED = "G_Ft composition required"
BLOCKER_ECCENTRICITY = "default eccentricity cannot be derived from final f_t/f_c"


@dataclass(frozen=True, slots=True)
class Cdpm2FractureEnergyComposition:
    """Explicit secondary physical authority supplying tensile fracture energy."""

    value: float
    provenance: PropertyProvenance

    schema_version = CDPM2_FRACTURE_ENERGY_COMPOSITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise TypeError("fracture-energy composition value must be numeric")
        value = float(self.value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("fracture-energy composition value must be finite and > 0")
        if not isinstance(self.provenance, PropertyProvenance):
            raise TypeError("fracture-energy composition provenance must be PropertyProvenance")
        if self.provenance.units != "N/mm":
            raise ValueError("fracture-energy composition provenance units must be 'N/mm'")
        if self.provenance.overridden:
            raise ValueError("fracture-energy composition provenance must not be overridden")
        if self.provenance.source_kind not in {
            SourceKind.STANDARD,
            SourceKind.LITERATURE,
            SourceKind.DERIVED,
        }:
            raise ValueError(
                "fracture-energy composition requires STANDARD, LITERATURE, or DERIVED "
                "physical provenance"
            )
        if self.provenance.source_kind is SourceKind.DERIVED and not self.provenance.derived_from:
            raise ValueError("derived fracture-energy composition provenance needs dependencies")
        object.__setattr__(self, "value", value)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic explicit physical-composition serialization."""

        return {
            "schema_version": self.schema_version,
            "value": self.value,
            "provenance": self.provenance.to_dict(),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize deterministically."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class Cdpm2ConversionNotReadyError(ValueError):
    """Raised when a valid CDPM2 mapping request is not ready to resolve."""

    def __init__(self, assessment: Cdpm2ReadinessAssessment) -> None:
        self.assessment = assessment
        blockers = "; ".join(assessment.blockers)
        super().__init__(f"CDPM2 conversion is {assessment.state.value}: {blockers}")


def _configuration_or_default(
    configuration: Cdpm2Grassl2013Configuration | None,
) -> Cdpm2Grassl2013Configuration:
    if configuration is None:
        return Cdpm2Grassl2013Configuration()
    if not isinstance(configuration, Cdpm2Grassl2013Configuration):
        raise TypeError("configuration must be Cdpm2Grassl2013Configuration or None")
    return configuration


def _physical_field_is_usable(physical: ConcretePhysicalProperties, field: str) -> bool:
    value = physical.values_dict()[field]
    status = physical.resolution[field]
    return (
        value is not None
        and status in {PropertyResolutionStatus.DIRECT, PropertyResolutionStatus.DERIVED}
        and field in physical.provenance
    )


def _preview_value(
    *,
    target_field: str,
    physical: ConcretePhysicalProperties,
    configuration: Cdpm2Grassl2013Configuration,
) -> float | None:
    if target_field in configuration.overrides:
        return float(configuration.overrides[target_field])
    source_field = _MAPPED_FIELDS[target_field]
    if not _physical_field_is_usable(physical, source_field):
        return None
    value = physical.values_dict()[source_field]
    if value is None:  # narrowed by _physical_field_is_usable; defensive for typing.
        return None
    return float(value)


def _derive_eccentricity(f_t: float, f_c: float) -> float:
    """Resolve the frozen Grassl 2013 Eq. (60) static-calibration eccentricity."""

    f_bc = 1.16 * f_c
    denominator = f_c**2 - f_t**2
    if not math.isfinite(denominator) or denominator == 0.0:
        raise ValueError(BLOCKER_ECCENTRICITY)
    epsilon = (f_t / f_bc) * (f_bc**2 - f_c**2) / denominator
    second_denominator = 2.0 - epsilon
    if not math.isfinite(epsilon) or second_denominator == 0.0:
        raise ValueError(BLOCKER_ECCENTRICITY)
    eccentricity = (1.0 + epsilon) / second_denominator
    if not math.isfinite(eccentricity) or eccentricity <= 0.0:
        raise ValueError(BLOCKER_ECCENTRICITY)
    return eccentricity


def assess_cdpm2_grassl_2013_readiness(
    *,
    physical_profile: str,
    physical: ConcretePhysicalProperties,
    configuration: Cdpm2Grassl2013Configuration | None = None,
    fracture_energy_composition: Cdpm2FractureEnergyComposition | None = None,
) -> Cdpm2ReadinessAssessment:
    """Assess whether a verified G1 physical concrete can resolve static CDPM2 semantics."""

    if not isinstance(physical_profile, str) or not physical_profile:
        raise ValueError("physical_profile must be a non-empty string")
    if not isinstance(physical, ConcretePhysicalProperties):
        raise TypeError("physical must be ConcretePhysicalProperties")
    config = _configuration_or_default(configuration)
    if fracture_energy_composition is not None and not isinstance(
        fracture_energy_composition, Cdpm2FractureEnergyComposition
    ):
        raise TypeError(
            "fracture_energy_composition must be Cdpm2FractureEnergyComposition or None"
        )

    if physical_profile not in _AUTHORIZED_PHYSICAL_PROFILES:
        return Cdpm2ReadinessAssessment(
            Cdpm2ConversionReadiness.NOT_AUTHORIZED_PHYSICAL_SOURCE,
            (BLOCKER_NOT_AUTHORIZED,),
        )

    blockers: list[str] = []
    has_gft_override = "G_Ft" in config.overrides
    physical_has_gft = _physical_field_is_usable(physical, "fracture_energy")

    if fracture_energy_composition is not None and has_gft_override:
        blockers.append(BLOCKER_GFT_CONFLICT)
    elif fracture_energy_composition is not None and physical_has_gft:
        blockers.append(BLOCKER_GFT_REDUNDANT_COMPOSITION)

    unresolved_blockers: list[str] = []
    for target, source, blocker in (
        ("E", "E_initial", BLOCKER_E_UNRESOLVED),
        ("nu", "poisson_elastic", BLOCKER_NU_UNRESOLVED),
        ("f_t", "f_ctm", BLOCKER_FT_UNRESOLVED),
        ("f_c", "f_cm", BLOCKER_FC_UNRESOLVED),
    ):
        if target not in config.overrides and not _physical_field_is_usable(physical, source):
            unresolved_blockers.append(blocker)

    gft_blocker: str | None = None
    if not has_gft_override and fracture_energy_composition is None and not physical_has_gft:
        gft_status = physical.resolution["fracture_energy"]
        if gft_status is PropertyResolutionStatus.COMPOSED_REQUIRED:
            gft_blocker = BLOCKER_GFT_COMPOSITION_REQUIRED
        else:
            unresolved_blockers.append(BLOCKER_GFT_UNRESOLVED)

    if "eccentricity" not in config.overrides:
        f_t = _preview_value(target_field="f_t", physical=physical, configuration=config)
        f_c = _preview_value(target_field="f_c", physical=physical, configuration=config)
        if f_t is not None and f_c is not None:
            try:
                _derive_eccentricity(f_t, f_c)
            except ValueError:
                blockers.append(BLOCKER_ECCENTRICITY)

    # Deterministic precedence: request/configuration conflicts, then unresolved
    # required physical inputs, then fracture-energy composition, then READY.
    if blockers:
        all_blockers = tuple(
            blockers + unresolved_blockers + ([gft_blocker] if gft_blocker else [])
        )
        return Cdpm2ReadinessAssessment(
            Cdpm2ConversionReadiness.UNSUPPORTED_CONFIGURATION,
            all_blockers,
        )
    if unresolved_blockers:
        all_blockers = tuple(unresolved_blockers + ([gft_blocker] if gft_blocker else []))
        return Cdpm2ReadinessAssessment(
            Cdpm2ConversionReadiness.UNRESOLVED_PHYSICAL_INPUT,
            all_blockers,
        )
    if gft_blocker is not None:
        return Cdpm2ReadinessAssessment(
            Cdpm2ConversionReadiness.COMPOSITION_REQUIRED,
            (gft_blocker,),
        )
    return Cdpm2ReadinessAssessment(Cdpm2ConversionReadiness.READY)


def _mapped_physical_provenance(
    *, target_field: str, source_field: str, physical: ConcretePhysicalProperties
) -> Cdpm2ParameterProvenance:
    source = physical.provenance[source_field]
    return Cdpm2ParameterProvenance(
        source_id=source.source_id,
        source_kind=Cdpm2SourceKind.PHYSICAL_SOURCE,
        edition=source.edition,
        equation_or_section=source.equation_or_section,
        units=CDPM2_PARAMETER_UNITS[target_field],
        notes=(
            f"Mapped from G1 physical field {source_field!r}; physical authority is preserved "
            "separately from CDPM2 constitutive-model authority."
        ),
        overridden=False,
        derived_from=(source_field,),
    )


def _composition_provenance(
    composition: Cdpm2FractureEnergyComposition,
) -> Cdpm2ParameterProvenance:
    source = composition.provenance
    return Cdpm2ParameterProvenance(
        source_id=source.source_id,
        source_kind=Cdpm2SourceKind.PHYSICAL_SOURCE,
        edition=source.edition,
        equation_or_section=source.equation_or_section,
        units="N/mm",
        notes="Explicit secondary physical fracture-energy authority supplying CDPM2 G_Ft.",
        overridden=False,
        derived_from=("fracture_energy",),
    )


def _override_provenance(field: str, replaced_role: str) -> Cdpm2ParameterProvenance:
    return Cdpm2ParameterProvenance(
        source_id="user_constitutive_override",
        source_kind=Cdpm2SourceKind.USER_CONSTITUTIVE_OVERRIDE,
        edition=None,
        equation_or_section=None,
        units=CDPM2_PARAMETER_UNITS[field],
        notes=f"Explicit constitutive override replacing {replaced_role}.",
        overridden=True,
    )


def _grassl_default_provenance(field: str) -> Cdpm2ParameterProvenance:
    return Cdpm2ParameterProvenance(
        source_id="grassl_et_al_2013_cdpm2",
        source_kind=Cdpm2SourceKind.GRASSL_2013_DEFAULT_CALIBRATION,
        edition="2013",
        equation_or_section=_GRASSL_DEFAULT_LOCATORS[field],
        units=CDPM2_PARAMETER_UNITS[field],
        notes=f"Frozen {CDPM2_STATIC_CALIBRATION_ID} default calibration.",
    )


def _oofem_default_provenance(field: str) -> Cdpm2ParameterProvenance:
    locator = {
        "A_s": "ConcreteDPM2 input manual: Asoft default",
        "epsilon_fc": "ConcreteDPM2 input manual: efc default",
    }[field]
    note = {
        "A_s": (
            "A_s=15 is an implementation baseline, not a universal Grassl-paper constant; "
            "the corrigendum A_s=7 remains dataset-specific."
        ),
        "epsilon_fc": (
            "epsilon_fc=1e-4 is the OOFEM implementation baseline and is also corroborated "
            "by Grassl 2013 comparison calibrations."
        ),
    }[field]
    return Cdpm2ParameterProvenance(
        source_id="grassl_oofem_cdpm2_manual_2022",
        source_kind=Cdpm2SourceKind.OOFEM_IMPLEMENTATION_DEFAULT,
        edition="2022-05-18",
        equation_or_section=locator,
        units=CDPM2_PARAMETER_UNITS[field],
        notes=note,
    )


def _grassl_derived_provenance(
    *, field: str, locator: str, derived_from: tuple[str, ...]
) -> Cdpm2ParameterProvenance:
    return Cdpm2ParameterProvenance(
        source_id="grassl_et_al_2013_cdpm2",
        source_kind=Cdpm2SourceKind.GRASSL_2013_DERIVED,
        edition="2013",
        equation_or_section=locator,
        units=CDPM2_PARAMETER_UNITS[field],
        notes=f"Frozen {CDPM2_STATIC_CALIBRATION_ID} derived semantic value.",
        derived_from=derived_from,
    )


def _grassl_direct_policy_provenance(field: str) -> Cdpm2ParameterProvenance:
    locator = {
        "tensile_softening_type": "Sec. 2.3.3 and Sec. 5 bilinear study",
        "damage_formulation": "Eq. (1); Sec. 2.1",
    }[field]
    return Cdpm2ParameterProvenance(
        source_id="grassl_et_al_2013_cdpm2",
        source_kind=Cdpm2SourceKind.GRASSL_2013_DIRECT,
        edition="2013",
        equation_or_section=locator,
        units="enum",
        notes="Frozen static-V1 scientific/model identity.",
    )


def resolve_cdpm2_grassl_2013(
    *,
    physical_profile: str,
    physical: ConcretePhysicalProperties,
    configuration: Cdpm2Grassl2013Configuration | None = None,
    fracture_energy_composition: Cdpm2FractureEnergyComposition | None = None,
) -> Cdpm2Grassl2013Parameters:
    """Resolve a READY verified G1 concrete into complete static CDPM2 semantics."""

    config = _configuration_or_default(configuration)
    assessment = assess_cdpm2_grassl_2013_readiness(
        physical_profile=physical_profile,
        physical=physical,
        configuration=config,
        fracture_energy_composition=fracture_energy_composition,
    )
    if assessment.state is not Cdpm2ConversionReadiness.READY:
        raise Cdpm2ConversionNotReadyError(assessment)

    values: dict[str, float] = {}
    provenance: dict[str, Cdpm2ParameterProvenance] = {}

    for target in ("E", "nu", "f_t", "f_c"):
        source = _MAPPED_FIELDS[target]
        if target in config.overrides:
            values[target] = float(config.overrides[target])
            provenance[target] = _override_provenance(target, f"mapped G1 field {source}")
        else:
            physical_value = physical.values_dict()[source]
            assert physical_value is not None  # readiness proved this.
            values[target] = float(physical_value)
            provenance[target] = _mapped_physical_provenance(
                target_field=target, source_field=source, physical=physical
            )

    if "G_Ft" in config.overrides:
        values["G_Ft"] = float(config.overrides["G_Ft"])
        provenance["G_Ft"] = _override_provenance(
            "G_Ft", "mapped or explicitly composed physical fracture-energy authority"
        )
    elif _physical_field_is_usable(physical, "fracture_energy"):
        fracture_energy = physical.fracture_energy
        assert fracture_energy is not None
        values["G_Ft"] = float(fracture_energy)
        provenance["G_Ft"] = _mapped_physical_provenance(
            target_field="G_Ft", source_field="fracture_energy", physical=physical
        )
    else:
        assert fracture_energy_composition is not None  # readiness proved this.
        values["G_Ft"] = fracture_energy_composition.value
        provenance["G_Ft"] = _composition_provenance(fracture_energy_composition)

    for field, default in _DEFAULT_CALIBRATION_VALUES.items():
        if field in config.overrides:
            values[field] = float(config.overrides[field])
            provenance[field] = _override_provenance(
                field, f"{CDPM2_STATIC_CALIBRATION_ID} calibration baseline"
            )
        else:
            values[field] = default
            if field in _GRASSL_DEFAULT_LOCATORS:
                provenance[field] = _grassl_default_provenance(field)
            else:
                provenance[field] = _oofem_default_provenance(field)

    if "eccentricity" in config.overrides:
        values["eccentricity"] = float(config.overrides["eccentricity"])
        provenance["eccentricity"] = _override_provenance(
            "eccentricity", "Grassl 2013 Eq. (60) static-calibration baseline"
        )
    else:
        values["eccentricity"] = _derive_eccentricity(values["f_t"], values["f_c"])
        provenance["eccentricity"] = _grassl_derived_provenance(
            field="eccentricity",
            locator="Eq. (60)",
            derived_from=("f_t", "f_c"),
        )

    values["w_f"] = values["G_Ft"] / (0.225 * values["f_t"])
    values["w_f1"] = 0.15 * values["w_f"]
    values["f_t1"] = 0.30 * values["f_t"]
    provenance["w_f"] = _grassl_derived_provenance(
        field="w_f", locator="Eq. (59)", derived_from=("G_Ft", "f_t")
    )
    provenance["w_f1"] = _grassl_derived_provenance(
        field="w_f1",
        locator="Sec. 5; Eq. (59) baseline ratio",
        derived_from=("w_f",),
    )
    provenance["f_t1"] = _grassl_derived_provenance(
        field="f_t1",
        locator="Sec. 2.3.3 and Sec. 5 baseline ratio",
        derived_from=("f_t",),
    )

    provenance["tensile_softening_type"] = _grassl_direct_policy_provenance(
        "tensile_softening_type"
    )
    provenance["damage_formulation"] = _grassl_direct_policy_provenance("damage_formulation")

    return Cdpm2Grassl2013Parameters(
        E=values["E"],
        nu=values["nu"],
        f_t=values["f_t"],
        f_c=values["f_c"],
        G_Ft=values["G_Ft"],
        w_f=values["w_f"],
        w_f1=values["w_f1"],
        f_t1=values["f_t1"],
        eccentricity=values["eccentricity"],
        q_h0=values["q_h0"],
        H_p=values["H_p"],
        D_f=values["D_f"],
        A_h=values["A_h"],
        B_h=values["B_h"],
        C_h=values["C_h"],
        D_h=values["D_h"],
        A_s=values["A_s"],
        epsilon_fc=values["epsilon_fc"],
        tensile_softening_type=Cdpm2TensileSofteningType.BILINEAR,
        damage_formulation=Cdpm2DamageFormulation.TWO_DAMAGE_VARIABLES,
        provenance=provenance,
    )
