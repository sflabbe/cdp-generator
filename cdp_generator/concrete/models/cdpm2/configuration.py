"""Requested calibration configuration and readiness value types for CDPM2."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar

from .schema import (
    CDPM2_MODEL_ID,
    CDPM2_STATIC_CALIBRATION_ID,
    validate_cdpm2_numeric_field,
)

CDPM2_CONFIGURATION_SCHEMA_VERSION = "cdpm2_grassl_2013_configuration.v1"
CDPM2_READINESS_SCHEMA_VERSION = "cdpm2_conversion_readiness.v1"

CDPM2_OVERRIDE_FIELDS: tuple[str, ...] = (
    "E",
    "nu",
    "f_t",
    "f_c",
    "G_Ft",
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
)
_ALLOWED_OVERRIDE_FIELDS = frozenset(CDPM2_OVERRIDE_FIELDS)


class Cdpm2ConversionReadiness(StrEnum):
    """Frozen G2-A conversion-readiness vocabulary."""

    READY = "READY"
    UNRESOLVED_PHYSICAL_INPUT = "UNRESOLVED_PHYSICAL_INPUT"
    COMPOSITION_REQUIRED = "COMPOSITION_REQUIRED"
    NOT_AUTHORIZED_PHYSICAL_SOURCE = "NOT_AUTHORIZED_PHYSICAL_SOURCE"
    UNSUPPORTED_CONFIGURATION = "UNSUPPORTED_CONFIGURATION"


@dataclass(frozen=True, slots=True)
class Cdpm2ReadinessAssessment:
    """Generic readiness result; profile-specific assessment belongs to G2-C."""

    state: Cdpm2ConversionReadiness
    blockers: tuple[str, ...] = ()

    schema_version: ClassVar[str] = CDPM2_READINESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.state, Cdpm2ConversionReadiness):
            raise TypeError("state must be Cdpm2ConversionReadiness")
        blockers = tuple(self.blockers)
        if any(not isinstance(blocker, str) or not blocker.strip() for blocker in blockers):
            raise ValueError("readiness blockers must be non-empty strings")
        if len(set(blockers)) != len(blockers):
            raise ValueError("readiness blockers must be unique")
        if self.state is Cdpm2ConversionReadiness.READY and blockers:
            raise ValueError("READY assessment cannot contain blockers")
        if self.state is not Cdpm2ConversionReadiness.READY and not blockers:
            raise ValueError("non-READY assessment requires at least one blocker")
        object.__setattr__(self, "blockers", blockers)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic generic readiness serialization."""

        return {
            "schema_version": self.schema_version,
            "state": self.state.value,
            "blockers": list(self.blockers),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize readiness deterministically."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True, slots=True)
class Cdpm2Grassl2013Configuration:
    """Requested constitutive overrides for the frozen static calibration.

    This object records caller intent only. It does not resolve defaults, derive
    parameters, inspect G1 physical concrete, or apply overrides.
    """

    overrides: Mapping[str, float] = field(default_factory=dict)
    calibration_id: str = CDPM2_STATIC_CALIBRATION_ID

    model_id: ClassVar[str] = CDPM2_MODEL_ID
    schema_version: ClassVar[str] = CDPM2_CONFIGURATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.calibration_id != CDPM2_STATIC_CALIBRATION_ID:
            raise ValueError(f"Unknown CDPM2 calibration: {self.calibration_id!r}")

        frozen: dict[str, float] = {}
        for key, value in self.overrides.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("CDPM2 override keys must be non-empty strings")
            if key not in _ALLOWED_OVERRIDE_FIELDS:
                raise ValueError(f"Unsupported CDPM2 static-V1 override: {key!r}")
            frozen[key] = validate_cdpm2_numeric_field(key, value)
        object.__setattr__(self, "overrides", MappingProxyType(frozen))

    def overrides_dict(self) -> dict[str, float]:
        """Return explicit caller overrides sorted by key."""

        return {key: self.overrides[key] for key in sorted(self.overrides)}

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic request/configuration serialization."""

        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "calibration_id": self.calibration_id,
            "overrides": self.overrides_dict(),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize deterministically without resolving defaults."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
