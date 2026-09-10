"""Immutable physical-concrete schema used by the authority-split architecture."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .provenance import PropertyProvenance

PHYSICAL_SCHEMA_VERSION = "concrete_physical_properties.v1"

PHYSICAL_UNITS: dict[str, str] = {
    "f_cm": "MPa",
    "f_ck": "MPa",
    "f_ctm": "MPa",
    "E_initial": "MPa",
    "E_secant": "MPa",
    "poisson_elastic": "dimensionless",
    "shear_modulus": "MPa",
    "fracture_energy": "N/mm",
    "strain_peak_compression": "dimensionless",
    "strain_limit_compression": "dimensionless",
}


@dataclass(frozen=True, slots=True)
class ConcretePhysicalProperties:
    """Physical concrete quantities, independent of a constitutive backend.

    Stress and modulus values use MPa, fracture energy uses N/mm, and strains
    are dimensionless. Characteristic element length is intentionally absent:
    it belongs to FE/integration context rather than intrinsic material data.
    """

    f_cm: float
    f_ck: float
    f_ctm: float
    E_initial: float
    E_secant: float
    poisson_elastic: float
    shear_modulus: float
    fracture_energy: float
    strain_peak_compression: float | None
    strain_limit_compression: float | None
    provenance: Mapping[str, PropertyProvenance]

    def __post_init__(self) -> None:
        frozen = dict(self.provenance)
        required = [
            "f_cm",
            "f_ck",
            "f_ctm",
            "E_initial",
            "E_secant",
            "poisson_elastic",
            "shear_modulus",
            "fracture_energy",
        ]
        if self.strain_peak_compression is not None:
            required.append("strain_peak_compression")
        if self.strain_limit_compression is not None:
            required.append("strain_limit_compression")

        missing = [field for field in required if field not in frozen]
        if missing:
            raise ValueError(f"Missing provenance for physical fields: {', '.join(missing)}")

        for field in required:
            provenance = frozen[field]
            if not provenance.source_id.strip():
                raise ValueError(f"Physical provenance source_id must be non-empty: {field}")
            expected_units = PHYSICAL_UNITS[field]
            if provenance.units != expected_units:
                raise ValueError(
                    f"Provenance units for {field} must be {expected_units!r}, "
                    f"got {provenance.units!r}"
                )

        object.__setattr__(self, "provenance", MappingProxyType(frozen))

    def values_dict(self) -> dict[str, float | None]:
        """Return only physical values, in stable schema order."""

        return {
            "f_cm": self.f_cm,
            "f_ck": self.f_ck,
            "f_ctm": self.f_ctm,
            "E_initial": self.E_initial,
            "E_secant": self.E_secant,
            "poisson_elastic": self.poisson_elastic,
            "shear_modulus": self.shear_modulus,
            "fracture_energy": self.fracture_energy,
            "strain_peak_compression": self.strain_peak_compression,
            "strain_limit_compression": self.strain_limit_compression,
        }

    def provenance_dict(self) -> dict[str, dict[str, Any]]:
        """Return field provenance in deterministic order."""

        return {
            field: self.provenance[field].to_dict()
            for field in self.values_dict()
            if field in self.provenance
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable physical schema."""

        return {
            "schema_version": PHYSICAL_SCHEMA_VERSION,
            "values": self.values_dict(),
            "provenance": self.provenance_dict(),
        }
