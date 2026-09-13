"""Pure static-V1 backend adapter for the frozen 24-slot CDPM2 contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar

from .schema import (
    Cdpm2DamageFormulation,
    Cdpm2Grassl2013Parameters,
    Cdpm2TensileSofteningType,
)

CDPM2_BACKEND_INPUT_SCHEMA_VERSION = "cdpm2_legacy24_backend_input.v1"
CDPM2_BACKEND_ID = "cdpm2_kratos_downstream_b98bef2"
CDPM2_LEGACY_SLOT_COUNT = 24
CDPM2_CHARACTERISTIC_LENGTH_UNITS = "mm"

CDPM2_LEGACY_SLOT_NAMES: tuple[str, ...] = (
    "E",
    "NU",
    "ECC",
    "QH0",
    "FT",
    "FC",
    "HP",
    "AH",
    "BH",
    "CH",
    "DH",
    "AS",
    "DF",
    "ERATETYPE",
    "TYPE",
    "BS",
    "WF",
    "WF1",
    "FT1",
    "SRATETYPE",
    "FAILFLG",
    "EFC",
    "DAMAGEFLAG",
    "PRINTFLAG",
)

CDPM2_LEGACY_SEMANTIC_SLOT_MAP: Mapping[int, str] = MappingProxyType(
    {
        1: "E",
        2: "nu",
        3: "eccentricity",
        4: "q_h0",
        5: "f_t",
        6: "f_c",
        7: "H_p",
        8: "A_h",
        9: "B_h",
        10: "C_h",
        11: "D_h",
        12: "A_s",
        13: "D_f",
        17: "w_f",
        18: "w_f1",
        19: "f_t1",
        22: "epsilon_fc",
    }
)

CDPM2_LEGACY_FIXED_SLOTS: Mapping[int, float] = MappingProxyType(
    {
        14: 0.0,  # ERATETYPE
        15: 1.0,  # TYPE = bilinear
        16: 1.0,  # BS = frozen paper-compatible baseline
        20: 0.0,  # SRATETYPE
        21: 0.0,  # FAILFLG
        23: 0.0,  # DAMAGEFLAG = two-variable spectral branch
        24: 0.0,  # PRINTFLAG
    }
)


def _validate_finite_numeric(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a finite numeric value")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _validate_characteristic_length(value: object) -> float:
    numeric = _validate_finite_numeric("characteristic_length", value)
    if numeric <= 0.0:
        raise ValueError("characteristic_length must be strictly positive")
    return numeric


@dataclass(frozen=True, slots=True)
class Cdpm2Legacy24BackendInput:
    """Immutable compatibility payload: exact ``cm(1:24)`` plus explicit LCHAR.

    ``cm`` uses the repository's existing consistent unit system.  The separate
    ``characteristic_length`` value is LCHAR in millimetres and is runtime mesh
    context, not a semantic material parameter.
    """

    cm: tuple[float, ...]
    characteristic_length: float

    schema_version: ClassVar[str] = CDPM2_BACKEND_INPUT_SCHEMA_VERSION
    backend_id: ClassVar[str] = CDPM2_BACKEND_ID
    characteristic_length_units: ClassVar[str] = CDPM2_CHARACTERISTIC_LENGTH_UNITS

    def __post_init__(self) -> None:
        cm = tuple(
            _validate_finite_numeric(f"cm[{index}]", value) for index, value in enumerate(self.cm)
        )
        if len(cm) != CDPM2_LEGACY_SLOT_COUNT:
            raise ValueError(f"cm must contain exactly {CDPM2_LEGACY_SLOT_COUNT} slots")
        object.__setattr__(self, "cm", cm)
        object.__setattr__(
            self,
            "characteristic_length",
            _validate_characteristic_length(self.characteristic_length),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe backend compatibility data."""

        return {
            "schema_version": self.schema_version,
            "backend_id": self.backend_id,
            "cm": list(self.cm),
            "characteristic_length": self.characteristic_length,
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize deterministically without qualification/runtime metadata."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def adapt_cdpm2_legacy_backend(
    parameters: Cdpm2Grassl2013Parameters,
    *,
    characteristic_length: float,
) -> Cdpm2Legacy24BackendInput:
    """Serialize resolved static-V1 semantics into frozen ``cm(1:24)`` + LCHAR.

    No physical mapping, calibration/default resolution, unit conversion, or
    constitutive mechanics occurs here.  The semantic object is already resolved.
    """

    if not isinstance(parameters, Cdpm2Grassl2013Parameters):
        raise TypeError("parameters must be Cdpm2Grassl2013Parameters")
    if parameters.tensile_softening_type is not Cdpm2TensileSofteningType.BILINEAR:
        raise ValueError("legacy24 static-V1 adapter requires bilinear tensile softening")
    if parameters.damage_formulation is not Cdpm2DamageFormulation.TWO_DAMAGE_VARIABLES:
        raise ValueError("legacy24 static-V1 adapter requires two_damage_variables")

    lchar = _validate_characteristic_length(characteristic_length)

    cm = (
        parameters.E,
        parameters.nu,
        parameters.eccentricity,
        parameters.q_h0,
        parameters.f_t,
        parameters.f_c,
        parameters.H_p,
        parameters.A_h,
        parameters.B_h,
        parameters.C_h,
        parameters.D_h,
        parameters.A_s,
        parameters.D_f,
        0.0,  # ERATETYPE
        1.0,  # TYPE = bilinear
        1.0,  # BS = frozen paper-compatible baseline
        parameters.w_f,
        parameters.w_f1,
        parameters.f_t1,
        0.0,  # SRATETYPE
        0.0,  # FAILFLG
        parameters.epsilon_fc,
        0.0,  # DAMAGEFLAG = canonical two-variable spectral branch
        0.0,  # PRINTFLAG
    )
    return Cdpm2Legacy24BackendInput(cm=cm, characteristic_length=lchar)
