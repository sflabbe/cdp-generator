"""Compatibility physical profile wrapping the pre-G0 concrete implementation."""

from ...material_properties import (
    calculate_concrete_strength_properties,
    calculate_elastic_modulus,
    calculate_fracture_energy,
    calculate_poisson_ratios,
)
from ..provenance import (
    PropertyProvenance,
    PropertyResolutionStatus,
    SourceKind,
    StatisticalBasis,
)
from ..schema import ConcretePhysicalProperties

PROFILE_ID = "legacy_v1"


def _legacy_provenance(
    units: str,
    statistical_basis: StatisticalBasis,
    equation_or_section: str,
    notes: str,
    source_kind: SourceKind = SourceKind.LEGACY_IMPLEMENTATION,
) -> PropertyProvenance:
    return PropertyProvenance(
        source_id=PROFILE_ID,
        source_kind=source_kind,
        edition=None,
        equation_or_section=equation_or_section,
        units=units,
        statistical_basis=statistical_basis,
        notes=notes,
        overridden=False,
    )


class LegacyV1Profile:
    """Adapter that reproduces the repository's pre-G0 physical behavior.

    This profile is a compatibility authority only.  It is not a certification
    that the wrapped formulas exactly implement fib Model Code or Eurocode.
    """

    profile_id = PROFILE_ID

    @staticmethod
    def build(f_cm: float, e_c1: float, e_clim: float) -> ConcretePhysicalProperties:
        strength = calculate_concrete_strength_properties(f_cm)
        elastic = calculate_elastic_modulus(f_cm)
        poisson = calculate_poisson_ratios(f_cm, elastic["E_c"], e_c1)
        fracture_energy = calculate_fracture_energy(f_cm)
        shear_modulus = elastic["E_c"] / (2 * (1 + poisson["v_ce"]))

        provenance: dict[str, PropertyProvenance] = {
            "f_cm": _legacy_provenance(
                "MPa",
                StatisticalBasis.MEAN,
                "input:f_cm",
                ("Caller-supplied mean strength retained by the legacy compatibility profile."),
            ),
            "f_ck": _legacy_provenance(
                "MPa",
                StatisticalBasis.CHARACTERISTIC,
                "calculate_concrete_strength_properties",
                ("Wrapped legacy implementation; normative authority is not asserted in G0."),
            ),
            "f_ctm": _legacy_provenance(
                "MPa",
                StatisticalBasis.MEAN,
                "calculate_concrete_strength_properties",
                ("Wrapped legacy implementation, including its existing high-strength branch."),
            ),
            "E_initial": _legacy_provenance(
                "MPa",
                StatisticalBasis.NOT_APPLICABLE,
                "calculate_elastic_modulus:E_ci",
                "Legacy tangent/initial modulus value.",
            ),
            "E_secant": _legacy_provenance(
                "MPa",
                StatisticalBasis.NOT_APPLICABLE,
                "calculate_elastic_modulus:E_c",
                "Legacy secant modulus value.",
            ),
            "poisson_elastic": _legacy_provenance(
                "dimensionless",
                StatisticalBasis.NOT_APPLICABLE,
                "calculate_poisson_ratios:v_ce",
                "Legacy elastic Poisson ratio value.",
            ),
            "shear_modulus_secant_equivalent": PropertyProvenance(
                source_id=PROFILE_ID,
                source_kind=SourceKind.DERIVED,
                edition=None,
                equation_or_section="G=E_c/(2*(1+v_ce))",
                units="MPa",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                notes="Derived from legacy_v1 E_secant and poisson_elastic.",
                overridden=False,
                derived_from=("E_secant", "poisson_elastic"),
            ),
            "fracture_energy": _legacy_provenance(
                "N/mm",
                StatisticalBasis.NOT_APPLICABLE,
                "calculate_fracture_energy",
                "Wrapped legacy fracture-energy implementation.",
            ),
            "strain_peak_compression": _legacy_provenance(
                "dimensionless",
                StatisticalBasis.NOT_APPLICABLE,
                "input:e_c1",
                "Caller-supplied strain retained by the legacy compatibility profile.",
            ),
            "strain_limit_compression": _legacy_provenance(
                "dimensionless",
                StatisticalBasis.NOT_APPLICABLE,
                "input:e_clim",
                "Caller-supplied strain retained by the legacy compatibility profile.",
            ),
        }

        return ConcretePhysicalProperties(
            f_cm=f_cm,
            f_ck=strength["f_ck"],
            f_ctm=strength["f_ctm"],
            f_ctk_lower=None,
            f_ctk_upper=None,
            E_initial=elastic["E_ci"],
            E_secant=elastic["E_c"],
            poisson_elastic=poisson["v_ce"],
            shear_modulus_secant_equivalent=shear_modulus,
            fracture_energy=fracture_energy,
            strain_peak_compression=e_c1,
            strain_limit_compression=e_clim,
            reference_age_days=None,
            provenance=provenance,
            resolution={
                "f_cm": PropertyResolutionStatus.DIRECT,
                "f_ck": PropertyResolutionStatus.DIRECT,
                "f_ctm": PropertyResolutionStatus.DIRECT,
                "f_ctk_lower": PropertyResolutionStatus.UNRESOLVED,
                "f_ctk_upper": PropertyResolutionStatus.UNRESOLVED,
                "E_initial": PropertyResolutionStatus.DIRECT,
                "E_secant": PropertyResolutionStatus.DIRECT,
                "poisson_elastic": PropertyResolutionStatus.DIRECT,
                "shear_modulus_secant_equivalent": PropertyResolutionStatus.DERIVED,
                "fracture_energy": PropertyResolutionStatus.DIRECT,
                "strain_peak_compression": PropertyResolutionStatus.DIRECT,
                "strain_limit_compression": PropertyResolutionStatus.DIRECT,
                "reference_age_days": PropertyResolutionStatus.UNRESOLVED,
            },
        )
