"""Immutable physical-concrete schema used by the authority-split architecture."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .provenance import PropertyProvenance, PropertyResolutionStatus, SourceKind

PHYSICAL_SCHEMA_VERSION = "concrete_physical_properties.v2"
LEGACY_PHYSICAL_SCHEMA_VERSION = "concrete_physical_properties.v1"

PHYSICAL_UNITS: dict[str, str] = {
    "f_cm": "MPa",
    "f_ck": "MPa",
    "f_ctm": "MPa",
    "f_ctk_lower": "MPa",
    "f_ctk_upper": "MPa",
    "E_initial": "MPa",
    "E_secant": "MPa",
    "poisson_elastic": "dimensionless",
    "shear_modulus_secant_equivalent": "MPa",
    "fracture_energy": "N/mm",
    "strain_peak_compression": "dimensionless",
    "strain_limit_compression": "dimensionless",
    "reference_age_days": "days",
}

LEGACY_V1_FIELDS: tuple[str, ...] = (
    "f_cm",
    "f_ck",
    "f_ctm",
    "E_initial",
    "E_secant",
    "poisson_elastic",
    "shear_modulus",
    "fracture_energy",
    "strain_peak_compression",
    "strain_limit_compression",
)


@dataclass(frozen=True, slots=True)
class ConcretePhysicalProperties:
    """Physical concrete quantities, independent of a constitutive backend.

    This is the canonical v2 schema.  Stress/modulus values use MPa, fracture
    energy uses N/mm, strain is dimensionless, and reference age uses days.
    Compression strains are stored as positive magnitudes.  Characteristic
    element length is intentionally absent from intrinsic material data.
    """

    f_cm: float
    f_ck: float
    f_ctm: float
    f_ctk_lower: float | None
    f_ctk_upper: float | None
    E_initial: float | None
    E_secant: float
    poisson_elastic: float
    shear_modulus_secant_equivalent: float
    fracture_energy: float | None
    strain_peak_compression: float | None
    strain_limit_compression: float | None
    reference_age_days: float | None
    provenance: Mapping[str, PropertyProvenance]
    resolution: Mapping[str, PropertyResolutionStatus]

    def __post_init__(self) -> None:
        provenance = dict(self.provenance)
        resolution = dict(self.resolution)
        values = self.values_dict()

        missing_resolution = [field for field in values if field not in resolution]
        extra_resolution = [field for field in resolution if field not in values]
        if missing_resolution or extra_resolution:
            detail = []
            if missing_resolution:
                detail.append(f"missing={','.join(missing_resolution)}")
            if extra_resolution:
                detail.append(f"extra={','.join(extra_resolution)}")
            raise ValueError(f"Physical resolution map must match v2 fields: {'; '.join(detail)}")

        unknown_provenance = [field for field in provenance if field not in values]
        if unknown_provenance:
            raise ValueError(
                "Provenance contains unknown physical fields: " + ", ".join(unknown_provenance)
            )

        for field, value in values.items():
            status = resolution[field]
            field_provenance = provenance.get(field)

            if value is None:
                if status not in {
                    PropertyResolutionStatus.UNRESOLVED,
                    PropertyResolutionStatus.COMPOSED_REQUIRED,
                }:
                    raise ValueError(
                        f"Absent physical field {field} must be UNRESOLVED or COMPOSED_REQUIRED"
                    )
                if (
                    field_provenance is not None
                    and field_provenance.equation_or_section is not None
                ):
                    raise ValueError(
                        f"Absent physical field {field} must not claim a producing equation/section"
                    )
                continue

            if status not in {
                PropertyResolutionStatus.DIRECT,
                PropertyResolutionStatus.DERIVED,
            }:
                raise ValueError(f"Populated physical field {field} must be DIRECT or DERIVED")
            if field_provenance is None:
                raise ValueError(f"Missing provenance for populated physical field: {field}")
            if field_provenance.units != PHYSICAL_UNITS[field]:
                raise ValueError(
                    f"Provenance units for {field} must be {PHYSICAL_UNITS[field]!r}, "
                    f"got {field_provenance.units!r}"
                )
            if status is PropertyResolutionStatus.DERIVED:
                if field_provenance.source_kind is not SourceKind.DERIVED:
                    raise ValueError(f"Derived field {field} must use SourceKind.DERIVED")
                if not field_provenance.derived_from:
                    raise ValueError(
                        f"Derived field {field} must identify derived_from dependencies"
                    )
            elif field_provenance.source_kind is SourceKind.DERIVED:
                raise ValueError(f"Direct field {field} cannot use SourceKind.DERIVED")

        for field in ("strain_peak_compression", "strain_limit_compression"):
            value = values[field]
            if value is not None and value < 0:
                raise ValueError(f"{field} must use the repository positive-magnitude convention")

        if self.reference_age_days is not None and self.reference_age_days <= 0:
            raise ValueError("reference_age_days must be positive when supplied")

        object.__setattr__(self, "provenance", MappingProxyType(provenance))
        object.__setattr__(self, "resolution", MappingProxyType(resolution))

    @property
    def shear_modulus(self) -> float:
        """Compatibility alias for the G0/v1 field name."""

        return self.shear_modulus_secant_equivalent

    def values_dict(self) -> dict[str, float | None]:
        """Return canonical v2 physical values in stable schema order."""

        return {
            "f_cm": self.f_cm,
            "f_ck": self.f_ck,
            "f_ctm": self.f_ctm,
            "f_ctk_lower": self.f_ctk_lower,
            "f_ctk_upper": self.f_ctk_upper,
            "E_initial": self.E_initial,
            "E_secant": self.E_secant,
            "poisson_elastic": self.poisson_elastic,
            "shear_modulus_secant_equivalent": self.shear_modulus_secant_equivalent,
            "fracture_energy": self.fracture_energy,
            "strain_peak_compression": self.strain_peak_compression,
            "strain_limit_compression": self.strain_limit_compression,
            "reference_age_days": self.reference_age_days,
        }

    def legacy_v1_values_dict(self) -> dict[str, float | None]:
        """Adapt canonical v2 storage to the historical v1 serialized field names."""

        return {
            "f_cm": self.f_cm,
            "f_ck": self.f_ck,
            "f_ctm": self.f_ctm,
            "E_initial": self.E_initial,
            "E_secant": self.E_secant,
            "poisson_elastic": self.poisson_elastic,
            "shear_modulus": self.shear_modulus_secant_equivalent,
            "fracture_energy": self.fracture_energy,
            "strain_peak_compression": self.strain_peak_compression,
            "strain_limit_compression": self.strain_limit_compression,
        }

    def provenance_dict(self) -> dict[str, dict[str, Any]]:
        """Return available field provenance in canonical v2 order."""

        return {
            field: self.provenance[field].to_dict()
            for field in self.values_dict()
            if field in self.provenance
        }

    def legacy_v1_provenance_dict(self) -> dict[str, dict[str, Any]]:
        """Return historical v1 provenance keys while preserving v2 metadata."""

        result: dict[str, dict[str, Any]] = {}
        for field in LEGACY_V1_FIELDS:
            canonical = "shear_modulus_secant_equivalent" if field == "shear_modulus" else field
            if canonical in self.provenance:
                result[field] = self.provenance[canonical].to_legacy_v1_dict()
        return result

    def resolution_dict(self) -> dict[str, str]:
        """Return all v2 resolution states in canonical field order."""

        return {field: self.resolution[field].value for field in self.values_dict()}

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic canonical v2 physical schema data."""

        return {
            "schema_version": PHYSICAL_SCHEMA_VERSION,
            "values": self.values_dict(),
            "resolution": self.resolution_dict(),
            "provenance": self.provenance_dict(),
        }
