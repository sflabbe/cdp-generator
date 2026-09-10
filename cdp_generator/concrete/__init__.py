"""Authority-split concrete facade introduced by G0.

The existing top-level concrete functions remain supported unchanged. This
parallel namespace provides an immutable physical schema, provenance, and
explicit constitutive-model conversion without moving legacy implementation
files.
"""

import json
from dataclasses import dataclass
from typing import Any, Never, Self

from .models import AbaqusCdpParameters, LegacyAbaqusCdpBackend
from .profiles import (
    LEGACY_ABAQUS_CALIBRATION,
    LEGACY_PHYSICAL_PROFILE,
    RESERVED_CONSTITUTIVE_PROFILES,
    RESERVED_PHYSICAL_PROFILES,
    get_physical_profile,
)
from .provenance import PropertyProvenance, SourceKind, StatisticalBasis
from .schema import ConcretePhysicalProperties


@dataclass(frozen=True, slots=True)
class Concrete:
    """High-level concrete definition with an explicit physical profile."""

    physical_profile: str
    physical: ConcretePhysicalProperties

    @classmethod
    def from_mean_strength(
        cls,
        f_cm: float,
        e_c1: float,
        e_clim: float,
        profile: str = LEGACY_PHYSICAL_PROFILE,
    ) -> Self:
        """Create physical properties from mean strength using an explicit profile."""

        implementation = get_physical_profile(profile)
        physical = implementation.build(f_cm=f_cm, e_c1=e_c1, e_clim=e_clim)
        return cls(physical_profile=profile, physical=physical)

    @classmethod
    def from_class(cls, concrete_class: str, profile: str = "ec2_2023") -> Never:
        """Reserved G1 API for concrete-class/standard resolution."""

        raise NotImplementedError(
            "Concrete.from_class() is reserved for G1 verified-standard resolution; "
            f"requested class={concrete_class!r}, profile={profile!r}"
        )

    def to_abaqus_cdp(self, calibration: str = LEGACY_ABAQUS_CALIBRATION) -> AbaqusCdpParameters:
        """Convert physical concrete to explicit ABAQUS-CDP model parameters."""

        if calibration == LEGACY_ABAQUS_CALIBRATION:
            return LegacyAbaqusCdpBackend.from_physical(self.physical)
        raise ValueError(f"Unknown ABAQUS-CDP calibration: {calibration!r}")

    def to_cdpm2(self, calibration: str = "cdpm2_grassl_2013") -> Never:
        """Reserved G2 conversion seam; no CDPM2 formulas are implemented in G0."""

        if calibration in RESERVED_CONSTITUTIVE_PROFILES:
            raise NotImplementedError(
                f"CDPM2 backend {calibration!r} is reserved for G2 and is not implemented in G0"
            )
        raise ValueError(f"Unknown CDPM2 calibration: {calibration!r}")

    def to_dict(self, abaqus_cdp: AbaqusCdpParameters | None = None) -> dict[str, Any]:
        """Return a deterministic JSON-serializable material definition.

        Constitutive data is included only when explicitly supplied, preserving
        the distinction between physical properties and model parameters.
        """

        constitutive_models: dict[str, Any] = {}
        constitutive_provenance: dict[str, Any] = {}
        if abaqus_cdp is not None:
            constitutive_models["abaqus_cdp"] = abaqus_cdp.values_dict()
            constitutive_provenance["abaqus_cdp"] = abaqus_cdp.provenance_dict()

        return {
            "schema_version": "concrete_material_definition.v1",
            "physical_profile": self.physical_profile,
            "physical": self.physical.values_dict(),
            "constitutive_models": constitutive_models,
            "provenance": {
                "physical": self.physical.provenance_dict(),
                "constitutive_models": constitutive_provenance,
            },
        }

    def to_json(
        self,
        abaqus_cdp: AbaqusCdpParameters | None = None,
        indent: int | None = 2,
    ) -> str:
        """Serialize deterministically for qualification and downstream tools."""

        return json.dumps(self.to_dict(abaqus_cdp=abaqus_cdp), indent=indent, sort_keys=True)


__all__ = [
    "LEGACY_ABAQUS_CALIBRATION",
    "LEGACY_PHYSICAL_PROFILE",
    "RESERVED_CONSTITUTIVE_PROFILES",
    "RESERVED_PHYSICAL_PROFILES",
    "AbaqusCdpParameters",
    "Concrete",
    "ConcretePhysicalProperties",
    "LegacyAbaqusCdpBackend",
    "PropertyProvenance",
    "SourceKind",
    "StatisticalBasis",
]
