"""Strict immutable semantic parameter schema for static CDPM2 Grassl 2013."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar

from .provenance import Cdpm2ParameterProvenance, Cdpm2SourceKind

CDPM2_MODEL_ID = "cdpm2_grassl_2013"
CDPM2_STATIC_CALIBRATION_ID = "grassl_2013_static_default_v1"
CDPM2_PARAMETERS_SCHEMA_VERSION = "cdpm2_grassl_2013_parameters.v1"


class Cdpm2TensileSofteningType(StrEnum):
    """Static-V1 tensile softening policy."""

    BILINEAR = "bilinear"


class Cdpm2DamageFormulation(StrEnum):
    """Static-V1 scientific damage-model identity."""

    TWO_DAMAGE_VARIABLES = "two_damage_variables"


CDPM2_PARAMETER_FIELDS: tuple[str, ...] = (
    "E",
    "nu",
    "f_t",
    "f_c",
    "G_Ft",
    "w_f",
    "w_f1",
    "f_t1",
    "eccentricity",
    "q_h0",
    "H_p",
    "D_f",
    "A_h",
    "B_h",
    "C_h",
    "D_h",
    "A_s",
    "epsilon_fc",
    "tensile_softening_type",
    "damage_formulation",
)

CDPM2_PARAMETER_UNITS: Mapping[str, str] = MappingProxyType(
    {
        "E": "MPa",
        "nu": "dimensionless",
        "f_t": "MPa",
        "f_c": "MPa",
        "G_Ft": "N/mm",
        "w_f": "mm",
        "w_f1": "mm",
        "f_t1": "MPa",
        "eccentricity": "dimensionless",
        "q_h0": "dimensionless",
        "H_p": "dimensionless",
        "D_f": "dimensionless",
        "A_h": "dimensionless",
        "B_h": "dimensionless",
        "C_h": "dimensionless",
        "D_h": "dimensionless",
        "A_s": "dimensionless",
        "epsilon_fc": "dimensionless",
        "tensile_softening_type": "enum",
        "damage_formulation": "enum",
    }
)

_DERIVED_REQUIREMENTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "w_f": ("G_Ft", "f_t"),
        "w_f1": ("w_f",),
        "f_t1": ("f_t",),
    }
)


def validate_cdpm2_numeric_field(field: str, value: object) -> float:
    """Validate and canonicalize one numeric static-V1 semantic field."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be a finite numeric value, not {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")

    if field == "nu":
        if not 0.0 <= numeric < 0.5:
            raise ValueError("nu must satisfy 0 <= nu < 0.5")
    elif field == "q_h0":
        if not 0.0 < numeric <= 1.0:
            raise ValueError("q_h0 must satisfy 0 < q_h0 <= 1")
    elif field in {"H_p", "D_h"}:
        if numeric < 0.0:
            raise ValueError(f"{field} must be >= 0")
    elif field == "D_f":
        if numeric <= 0.5:
            raise ValueError("D_f must be > 0.5")
    elif field in {
        "E",
        "f_t",
        "f_c",
        "G_Ft",
        "w_f",
        "w_f1",
        "f_t1",
        "eccentricity",
        "A_h",
        "B_h",
        "C_h",
        "A_s",
        "epsilon_fc",
    }:
        if numeric <= 0.0:
            raise ValueError(f"{field} must be > 0")
    else:
        raise KeyError(f"Unknown numeric CDPM2 semantic field: {field!r}")
    return numeric


@dataclass(frozen=True, slots=True)
class Cdpm2Grassl2013Parameters:
    """Fully resolved, backend-agnostic static-V1 CDPM2 parameter set."""

    E: float
    nu: float
    f_t: float
    f_c: float
    G_Ft: float
    w_f: float
    w_f1: float
    f_t1: float
    eccentricity: float
    q_h0: float
    H_p: float
    D_f: float
    A_h: float
    B_h: float
    C_h: float
    D_h: float
    A_s: float
    epsilon_fc: float
    tensile_softening_type: Cdpm2TensileSofteningType
    damage_formulation: Cdpm2DamageFormulation
    provenance: Mapping[str, Cdpm2ParameterProvenance]
    calibration_id: str = CDPM2_STATIC_CALIBRATION_ID

    model_id: ClassVar[str] = CDPM2_MODEL_ID
    schema_version: ClassVar[str] = CDPM2_PARAMETERS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.calibration_id != CDPM2_STATIC_CALIBRATION_ID:
            raise ValueError(f"Unknown CDPM2 calibration: {self.calibration_id!r}")

        for field in CDPM2_PARAMETER_FIELDS[:18]:
            object.__setattr__(
                self, field, validate_cdpm2_numeric_field(field, getattr(self, field))
            )

        if not isinstance(self.tensile_softening_type, Cdpm2TensileSofteningType):
            raise TypeError("tensile_softening_type must be Cdpm2TensileSofteningType")
        if self.tensile_softening_type is not Cdpm2TensileSofteningType.BILINEAR:
            raise ValueError("static V1 supports only bilinear tensile softening")
        if not isinstance(self.damage_formulation, Cdpm2DamageFormulation):
            raise TypeError("damage_formulation must be Cdpm2DamageFormulation")
        if self.damage_formulation is not Cdpm2DamageFormulation.TWO_DAMAGE_VARIABLES:
            raise ValueError("static V1 supports only two_damage_variables")

        expected_w_f = self.G_Ft / (0.225 * self.f_t)
        if not math.isclose(self.w_f, expected_w_f, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("w_f must equal G_Ft/(0.225*f_t) for static V1")
        if not math.isclose(self.w_f1, 0.15 * self.w_f, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("w_f1 must equal 0.15*w_f for static V1")
        if not math.isclose(self.f_t1, 0.30 * self.f_t, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("f_t1 must equal 0.30*f_t for static V1")

        provenance = dict(self.provenance)
        expected_fields = set(CDPM2_PARAMETER_FIELDS)
        provenance_fields = set(provenance)
        missing = sorted(expected_fields - provenance_fields)
        extra = sorted(provenance_fields - expected_fields)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing={','.join(missing)}")
            if extra:
                details.append(f"extra={','.join(extra)}")
            raise ValueError("CDPM2 provenance must match effective fields: " + "; ".join(details))

        for field in CDPM2_PARAMETER_FIELDS:
            field_provenance = provenance[field]
            expected_units = CDPM2_PARAMETER_UNITS[field]
            if field_provenance.units != expected_units:
                raise ValueError(
                    f"Provenance units for {field} must be {expected_units!r}, "
                    f"got {field_provenance.units!r}"
                )

        for field, dependencies in _DERIVED_REQUIREMENTS.items():
            field_provenance = provenance[field]
            if field_provenance.source_kind is not Cdpm2SourceKind.GRASSL_2013_DERIVED:
                raise ValueError(f"{field} must use GRASSL_2013_DERIVED provenance")
            if field_provenance.derived_from != dependencies:
                raise ValueError(f"{field} derived_from must be {dependencies!r}")
            if field_provenance.overridden:
                raise ValueError(f"{field} cannot be overridden in static V1")

        for field in ("tensile_softening_type", "damage_formulation"):
            field_provenance = provenance[field]
            if field_provenance.source_kind is not Cdpm2SourceKind.GRASSL_2013_DIRECT:
                raise ValueError(f"{field} must use GRASSL_2013_DIRECT model-policy provenance")
            if field_provenance.derived_from:
                raise ValueError(f"{field} must not claim derived_from dependencies")

        object.__setattr__(self, "provenance", MappingProxyType(provenance))

    def values_dict(self) -> dict[str, float | str]:
        """Return all 20 effective fields in frozen semantic order."""

        return {
            "E": self.E,
            "nu": self.nu,
            "f_t": self.f_t,
            "f_c": self.f_c,
            "G_Ft": self.G_Ft,
            "w_f": self.w_f,
            "w_f1": self.w_f1,
            "f_t1": self.f_t1,
            "eccentricity": self.eccentricity,
            "q_h0": self.q_h0,
            "H_p": self.H_p,
            "D_f": self.D_f,
            "A_h": self.A_h,
            "B_h": self.B_h,
            "C_h": self.C_h,
            "D_h": self.D_h,
            "A_s": self.A_s,
            "epsilon_fc": self.epsilon_fc,
            "tensile_softening_type": self.tensile_softening_type.value,
            "damage_formulation": self.damage_formulation.value,
        }

    def provenance_dict(self) -> dict[str, dict[str, Any]]:
        """Return complete constitutive provenance in semantic field order."""

        return {field: self.provenance[field].to_dict() for field in CDPM2_PARAMETER_FIELDS}

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic standalone CDPM2 parameter serialization."""

        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "calibration_id": self.calibration_id,
            "values": self.values_dict(),
            "provenance": self.provenance_dict(),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize deterministically for artifacts and downstream consumers."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
