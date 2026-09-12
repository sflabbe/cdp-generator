"""Authority-split concrete facade introduced by G0 and prepared for G1."""

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Never, Self

from .class_registry import (
    ConcreteClassEntry,
    ConcreteClassError,
    CrossProfileConcreteClassError,
    MalformedConcreteClassError,
    UnknownPhysicalProfileError,
    UnsupportedConcreteClassError,
    class_entries,
    parse_concrete_class,
)
from .configuration import ProfileConfiguration
from .models import AbaqusCdpParameters, LegacyAbaqusCdpBackend
from .profiles import (
    EC2_2004_PHYSICAL_PROFILE,
    EC2_2023_PHYSICAL_PROFILE,
    FIB_MC2010_PHYSICAL_PROFILE,
    LEGACY_ABAQUS_CALIBRATION,
    LEGACY_PHYSICAL_PROFILE,
    RESERVED_CONSTITUTIVE_PROFILES,
    RESERVED_PHYSICAL_PROFILES,
    get_class_physical_profile,
    get_physical_profile,
)
from .provenance import (
    NormalizationKind,
    PropertyNormalization,
    PropertyProvenance,
    PropertyResolutionStatus,
    SourceKind,
    StatisticalBasis,
)
from .schema import ConcretePhysicalProperties

type MaterialSerializationVersion = Literal["v1", "v2"]


@dataclass(frozen=True, slots=True)
class Concrete:
    """High-level concrete definition with an explicit physical profile."""

    physical_profile: str
    physical: ConcretePhysicalProperties
    profile_parameters: ProfileConfiguration = field(default_factory=ProfileConfiguration)

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
    def from_class(
        cls,
        concrete_class: str,
        profile: str = "ec2_2023",
        profile_parameters: ProfileConfiguration | None = None,
    ) -> Self:
        """Create a verified named-class physical profile when implemented."""

        class_entry = parse_concrete_class(profile, concrete_class)
        implementation = get_class_physical_profile(profile)
        physical, effective_parameters = implementation.build(class_entry, profile_parameters)
        return cls(
            physical_profile=profile,
            physical=physical,
            profile_parameters=effective_parameters,
        )

    def to_abaqus_cdp(self, calibration: str = LEGACY_ABAQUS_CALIBRATION) -> AbaqusCdpParameters:
        """Convert physical concrete to explicit ABAQUS-CDP model parameters."""

        if calibration == LEGACY_ABAQUS_CALIBRATION:
            return LegacyAbaqusCdpBackend.from_physical(self.physical)
        raise ValueError(f"Unknown ABAQUS-CDP calibration: {calibration!r}")

    def to_cdpm2(self, calibration: str = "cdpm2_grassl_2013") -> Never:
        """Reserved G2 conversion seam; no CDPM2 formulas are implemented in G1."""

        if calibration in RESERVED_CONSTITUTIVE_PROFILES:
            raise NotImplementedError(
                f"CDPM2 backend {calibration!r} is reserved for G2 and is not implemented in G1"
            )
        raise ValueError(f"Unknown CDPM2 calibration: {calibration!r}")

    def _to_dict_v1(self, abaqus_cdp: AbaqusCdpParameters | None) -> dict[str, Any]:
        constitutive_models: dict[str, Any] = {}
        constitutive_provenance: dict[str, Any] = {}
        if abaqus_cdp is not None:
            constitutive_models["abaqus_cdp"] = abaqus_cdp.values_dict()
            constitutive_provenance["abaqus_cdp"] = {
                field: abaqus_cdp.provenance[field].to_legacy_v1_dict()
                for field in abaqus_cdp.values_dict()
            }

        return {
            "schema_version": "concrete_material_definition.v1",
            "physical_profile": self.physical_profile,
            "physical": self.physical.legacy_v1_values_dict(),
            "constitutive_models": constitutive_models,
            "provenance": {
                "physical": self.physical.legacy_v1_provenance_dict(),
                "constitutive_models": constitutive_provenance,
            },
        }

    def _to_dict_v2(self, abaqus_cdp: AbaqusCdpParameters | None) -> dict[str, Any]:
        constitutive_models: dict[str, Any] = {}
        if abaqus_cdp is not None:
            constitutive_models["abaqus_cdp"] = abaqus_cdp.to_dict()

        return {
            "schema_version": "concrete_material_definition.v2",
            "physical_profile": self.physical_profile,
            "profile_parameters": self.profile_parameters.to_dict(),
            "physical": self.physical.to_dict(),
            "constitutive_models": constitutive_models,
        }

    def to_dict(
        self,
        abaqus_cdp: AbaqusCdpParameters | None = None,
        schema_version: MaterialSerializationVersion | None = None,
    ) -> dict[str, Any]:
        """Return deterministic material serialization with explicit version policy.

        Legacy profile callers retain the historical v1 payload by default.  The
        canonical v2 representation is available explicitly now and will become
        the natural default for future verified profiles.
        """

        selected_version = schema_version
        if selected_version is None:
            selected_version = "v1" if self.physical_profile == LEGACY_PHYSICAL_PROFILE else "v2"

        if selected_version not in ("v1", "v2"):
            raise ValueError(
                f"Unknown concrete material serialization version: {selected_version!r}"
            )
        if selected_version == "v1":
            if self.physical_profile != LEGACY_PHYSICAL_PROFILE:
                raise ValueError(
                    "v1 material serialization is reserved for legacy_v1 compatibility"
                )
            return self._to_dict_v1(abaqus_cdp)
        return self._to_dict_v2(abaqus_cdp)

    def to_json(
        self,
        abaqus_cdp: AbaqusCdpParameters | None = None,
        indent: int | None = 2,
        schema_version: MaterialSerializationVersion | None = None,
    ) -> str:
        """Serialize deterministically for qualification and downstream tools."""

        return json.dumps(
            self.to_dict(abaqus_cdp=abaqus_cdp, schema_version=schema_version),
            indent=indent,
            sort_keys=True,
        )


__all__ = [
    "EC2_2004_PHYSICAL_PROFILE",
    "EC2_2023_PHYSICAL_PROFILE",
    "FIB_MC2010_PHYSICAL_PROFILE",
    "LEGACY_ABAQUS_CALIBRATION",
    "LEGACY_PHYSICAL_PROFILE",
    "RESERVED_CONSTITUTIVE_PROFILES",
    "RESERVED_PHYSICAL_PROFILES",
    "AbaqusCdpParameters",
    "Concrete",
    "ConcreteClassEntry",
    "ConcreteClassError",
    "ConcretePhysicalProperties",
    "CrossProfileConcreteClassError",
    "LegacyAbaqusCdpBackend",
    "MalformedConcreteClassError",
    "NormalizationKind",
    "ProfileConfiguration",
    "PropertyNormalization",
    "PropertyProvenance",
    "PropertyResolutionStatus",
    "SourceKind",
    "StatisticalBasis",
    "UnknownPhysicalProfileError",
    "UnsupportedConcreteClassError",
    "class_entries",
    "parse_concrete_class",
]
