"""Legacy ABAQUS Concrete Damage Plasticity scalar-parameter backend."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ...material_properties import calculate_cdp_parameters, calculate_poisson_ratios
from ..provenance import PropertyProvenance, SourceKind, StatisticalBasis
from ..schema import ConcretePhysicalProperties

CALIBRATION_ID = "abaqus_cdp_legacy"


@dataclass(frozen=True, slots=True)
class AbaqusCdpParameters:
    """ABAQUS-CDP constitutive-model scalar parameters.

    These are not generic physical concrete properties.  They belong to the
    ABAQUS CDP constitutive model/calibration domain.
    """

    dilation_angle: float
    fbfc: float
    Kc: float
    provenance: Mapping[str, PropertyProvenance]

    def __post_init__(self) -> None:
        frozen = dict(self.provenance)
        required = ("dilation_angle", "fbfc", "Kc")
        missing = [field for field in required if field not in frozen]
        if missing:
            raise ValueError(f"Missing provenance for ABAQUS-CDP fields: {', '.join(missing)}")
        object.__setattr__(self, "provenance", MappingProxyType(frozen))

    def values_dict(self) -> dict[str, float]:
        return {
            "dilation_angle": self.dilation_angle,
            "fbfc": self.fbfc,
            "Kc": self.Kc,
        }

    def provenance_dict(self) -> dict[str, dict[str, Any]]:
        return {field: self.provenance[field].to_legacy_v1_dict() for field in self.values_dict()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "abaqus_cdp_parameters.v1",
            "calibration": CALIBRATION_ID,
            "values": self.values_dict(),
            "provenance": self.provenance_dict(),
        }


class LegacyAbaqusCdpBackend:
    """Adapter around the existing ``calculate_cdp_parameters`` function."""

    calibration_id = CALIBRATION_ID

    @staticmethod
    def from_physical(physical: ConcretePhysicalProperties) -> AbaqusCdpParameters:
        e_c1 = physical.strain_peak_compression
        if e_c1 is None:
            raise ValueError("abaqus_cdp_legacy requires strain_peak_compression")

        poisson = calculate_poisson_ratios(physical.f_cm, physical.E_secant, e_c1)
        legacy = calculate_cdp_parameters(
            physical.f_cm,
            physical.E_secant,
            e_c1,
            poisson["v_c0"],
            poisson["v_ce"],
        )

        def provenance(field: str, units: str) -> PropertyProvenance:
            return PropertyProvenance(
                source_id="legacy_v1",
                source_kind=SourceKind.LEGACY_IMPLEMENTATION,
                edition=None,
                equation_or_section=f"calculate_cdp_parameters:{field}",
                units=units,
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                notes=(
                    "Existing ABAQUS-CDP calibration wrapped by abaqus_cdp_legacy; "
                    "no normative standard authority is asserted in G0."
                ),
                overridden=False,
            )

        return AbaqusCdpParameters(
            dilation_angle=legacy["dilation_angle"],
            fbfc=legacy["fbfc"],
            Kc=legacy["K_c"],
            provenance={
                "dilation_angle": provenance("dilation_angle", "degrees"),
                "fbfc": provenance("fbfc", "dimensionless"),
                "Kc": provenance("K_c", "dimensionless"),
            },
        )
