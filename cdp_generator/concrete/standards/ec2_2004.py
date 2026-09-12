"""Verified EN 1992-1-1:2004 physical concrete profile.

This module implements only the G1-A-adjudicated ambient, normal-weight,
28-day named-class material properties. It deliberately excludes time
development, fire design, fracture-energy composition, EC2:2023, and
constitutive-model calibration.
"""

import math
from types import MappingProxyType

from ..class_registry import ConcreteClassEntry
from ..configuration import ProfileConfiguration
from ..provenance import (
    NormalizationKind,
    PropertyNormalization,
    PropertyProvenance,
    PropertyResolutionStatus,
    SourceKind,
    StatisticalBasis,
)
from ..schema import ConcretePhysicalProperties

PROFILE_ID = "ec2_2004"
SOURCE_ID = "en_1992_1_1_2004"
SOURCE_EDITION = "2004 + AC:2010"
REFERENCE_AGE_DAYS = 28.0

_POISSON_ELASTIC = 0.20
_TANGENT_APPROXIMATION_FACTOR = 1.05

AGGREGATE_ECM_FACTOR = MappingProxyType(
    {
        "basalt": 1.2,
        "quartzite": 1.0,
        "limestone": 0.9,
        "sandstone": 0.7,
    }
)

_STRAIN_UNIT_NORMALIZATION = PropertyNormalization(
    kind=NormalizationKind.UNIT_CONVERSION,
    source_convention="compression strain expressed in per-mille",
    repository_convention="dimensionless strain; divide source magnitude by 1000",
)


def _standard_provenance(
    *,
    units: str,
    statistical_basis: StatisticalBasis,
    equation_or_section: str,
    notes: str,
    derived_from: tuple[str, ...] = (),
    normalizations: tuple[PropertyNormalization, ...] = (),
) -> PropertyProvenance:
    return PropertyProvenance(
        source_id=SOURCE_ID,
        source_kind=SourceKind.STANDARD,
        edition=SOURCE_EDITION,
        equation_or_section=equation_or_section,
        units=units,
        statistical_basis=statistical_basis,
        notes=notes,
        overridden=False,
        derived_from=derived_from,
        normalizations=normalizations,
    )


def _derived_provenance(
    *,
    source_id: str,
    equation_or_section: str | None,
    statistical_basis: StatisticalBasis,
    notes: str,
    derived_from: tuple[str, ...],
) -> PropertyProvenance:
    return PropertyProvenance(
        source_id=source_id,
        source_kind=SourceKind.DERIVED,
        edition=SOURCE_EDITION,
        equation_or_section=equation_or_section,
        units="MPa",
        statistical_basis=statistical_basis,
        notes=notes,
        overridden=False,
        derived_from=derived_from,
    )


def _resolve_configuration(
    profile_parameters: ProfileConfiguration | None,
) -> ProfileConfiguration:
    """Resolve the narrow G1-B3 EC2:2004 configuration."""

    parameters = {} if profile_parameters is None else dict(profile_parameters.parameters)
    allowed = {"aggregate_type", "reference_age_days"}
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unsupported ec2_2004 profile parameter(s): {names}")

    aggregate_type = parameters.get("aggregate_type", "quartzite")
    if not isinstance(aggregate_type, str) or aggregate_type not in AGGREGATE_ECM_FACTOR:
        allowed_names = ", ".join(AGGREGATE_ECM_FACTOR)
        raise ValueError(
            f"ec2_2004 aggregate_type must be one of {allowed_names}; got {aggregate_type!r}"
        )

    requested_age = parameters.get("reference_age_days", 28)
    if isinstance(requested_age, bool) or not isinstance(requested_age, int | float):
        raise TypeError("ec2_2004 reference_age_days must be numeric when supplied")
    if float(requested_age) != REFERENCE_AGE_DAYS:
        raise ValueError(
            "ec2_2004 G1-B3 implements only the 28-day named-class reference state; "
            f"got reference_age_days={requested_age!r}"
        )

    return ProfileConfiguration(
        {
            "aggregate_type": aggregate_type,
            "Ecm_factor": AGGREGATE_ECM_FACTOR[aggregate_type],
            "reference_age_days": 28,
        }
    )


class Ec2_2004Profile:
    """Verified EC2:2004 named-class physical profile for normal-weight concrete."""

    profile_id = PROFILE_ID

    @staticmethod
    def build(
        class_entry: ConcreteClassEntry,
        profile_parameters: ProfileConfiguration | None = None,
    ) -> tuple[ConcretePhysicalProperties, ProfileConfiguration]:
        """Build one 28-day named-class EC2:2004 physical definition."""

        if class_entry.profile != PROFILE_ID:
            raise ValueError(
                f"Ec2_2004Profile requires a {PROFILE_ID!r} class entry, "
                f"got {class_entry.profile!r}"
            )

        effective_configuration = _resolve_configuration(profile_parameters)
        aggregate_type = effective_configuration.parameters["aggregate_type"]
        ecm_factor = effective_configuration.parameters["Ecm_factor"]
        if not isinstance(aggregate_type, str) or not isinstance(ecm_factor, float | int):
            raise TypeError("resolved ec2_2004 aggregate configuration is invalid")

        f_ck = class_entry.f_ck_cylinder_mpa
        f_cm = f_ck + 8.0

        if f_ck <= 50.0:
            f_ctm = 0.30 * f_ck ** (2.0 / 3.0)
            f_ctm_branch = "Table 3.1 power-law branch for f_ck <= 50 MPa"
        else:
            f_ctm = 2.12 * math.log(1.0 + f_cm / 10.0)
            f_ctm_branch = "Table 3.1 logarithmic branch for f_ck > 50 MPa"

        f_ctk_lower = 0.7 * f_ctm
        f_ctk_upper = 1.3 * f_ctm

        e_secant_base = 22_000.0 * (f_cm / 10.0) ** 0.3
        e_secant = e_secant_base * float(ecm_factor)
        e_initial = _TANGENT_APPROXIMATION_FACTOR * e_secant

        poisson_elastic = _POISSON_ELASTIC
        shear_modulus = e_secant / (2.0 * (1.0 + poisson_elastic))

        epsilon_c1_per_mille = min(0.7 * f_cm**0.31, 2.8)
        if f_ck < 50.0:
            epsilon_cu1_per_mille = 3.5
            epsilon_cu1_branch = "Table 3.1 constant branch for f_ck < 50 MPa"
        else:
            epsilon_cu1_per_mille = 2.8 + 27.0 * ((98.0 - f_cm) / 100.0) ** 4
            epsilon_cu1_branch = "Table 3.1 high-strength expression for f_ck >= 50 MPa"

        provenance: dict[str, PropertyProvenance] = {
            "f_ck": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.CHARACTERISTIC,
                equation_or_section="§3.1.2(1)-(3); Table 3.1",
                notes=(
                    "Characteristic 5% cylinder compressive strength from the explicit "
                    "EC2:2004 class-registry entry."
                ),
            ),
            "f_cm": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.MEAN,
                equation_or_section="Table 3.1",
                notes="Mean cylinder compressive strength; f_cm = f_ck + 8 MPa.",
                derived_from=("f_ck",),
            ),
            "f_ctm": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.MEAN,
                equation_or_section="Table 3.1",
                notes=f"Mean axial tensile strength; {f_ctm_branch}.",
                derived_from=("f_ck", "f_cm"),
            ),
            "f_ctk_lower": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.CHARACTERISTIC,
                equation_or_section="Table 3.1",
                notes="EC2 characteristic tensile strength f_ctk,0.05; explicit 5% fractile.",
                derived_from=("f_ctm",),
            ),
            "f_ctk_upper": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.CHARACTERISTIC,
                equation_or_section="Table 3.1",
                notes="EC2 characteristic tensile strength f_ctk,0.95; explicit 95% fractile.",
                derived_from=("f_ctm",),
            ),
            "E_initial": _derived_provenance(
                source_id=SOURCE_ID,
                equation_or_section="§3.1.4(2)",
                statistical_basis=StatisticalBasis.MEAN,
                notes=(
                    "Repository E_initial maps the EC2 tangent modulus E_c approximation "
                    "E_c = 1.05 E_cm; it is not a directly tabulated initial modulus."
                ),
                derived_from=("E_secant",),
            ),
            "E_secant": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.MEAN,
                equation_or_section="§3.1.3(2); Table 3.1",
                notes=(
                    "EC2 E_cm is the secant modulus from sigma_c=0 to 0.4 f_cm. "
                    f"aggregate_type={aggregate_type}; Ecm_factor={float(ecm_factor):g}."
                ),
                derived_from=("f_cm",),
            ),
            "poisson_elastic": _standard_provenance(
                units="dimensionless",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                equation_or_section="§3.1.3(4)",
                notes=(
                    "EC2 uncracked elastic Poisson ratio nu = 0.20; the separate cracked "
                    "concrete allowance nu = 0 is not the repository poisson_elastic state."
                ),
            ),
            "shear_modulus_secant_equivalent": _derived_provenance(
                source_id=PROFILE_ID,
                equation_or_section=None,
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                notes=(
                    "Repository-derived isotropic secant-equivalent shear modulus; "
                    "not claimed as a directly specified EC2 quantity."
                ),
                derived_from=("E_secant", "poisson_elastic"),
            ),
            "fracture_energy": PropertyProvenance(
                source_id=SOURCE_ID,
                source_kind=SourceKind.STANDARD,
                edition=SOURCE_EDITION,
                equation_or_section=None,
                units="N/mm",
                statistical_basis=StatisticalBasis.UNSPECIFIED,
                notes=(
                    "EN 1992-1-1:2004 Section 3 does not supply the repository "
                    "plain-concrete fracture-energy property; explicit composition with a "
                    "second authority is required."
                ),
                overridden=False,
            ),
            "strain_peak_compression": _standard_provenance(
                units="dimensionless",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                equation_or_section="§3.1.5; Table 3.1; Eq. (3.14) context",
                notes=(
                    "General nonlinear/Sargin peak strain epsilon_c1. AC:2010 corrects the "
                    "Table 3.1 cap notation to <= 2.8 per-mille; source and repository both "
                    "use positive compression magnitudes."
                ),
                derived_from=("f_cm",),
                normalizations=(_STRAIN_UNIT_NORMALIZATION,),
            ),
            "strain_limit_compression": _standard_provenance(
                units="dimensionless",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                equation_or_section="§3.1.5; Table 3.1; Eq. (3.14) context",
                notes=f"General nonlinear nominal ultimate strain epsilon_cu1; {epsilon_cu1_branch}.",
                derived_from=("f_ck", "f_cm"),
                normalizations=(_STRAIN_UNIT_NORMALIZATION,),
            ),
            "reference_age_days": _standard_provenance(
                units="days",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                equation_or_section="§3.1.2(2); Table 3.1",
                notes=(
                    "G1-B3 named-class reference properties are fixed to 28 days; arbitrary "
                    "time development is outside this profile slice."
                ),
            ),
        }

        resolution = {
            field: PropertyResolutionStatus.DIRECT
            for field in (
                "f_cm",
                "f_ck",
                "f_ctm",
                "f_ctk_lower",
                "f_ctk_upper",
                "E_secant",
                "poisson_elastic",
                "strain_peak_compression",
                "strain_limit_compression",
                "reference_age_days",
            )
        }
        resolution["E_initial"] = PropertyResolutionStatus.DERIVED
        resolution["shear_modulus_secant_equivalent"] = PropertyResolutionStatus.DERIVED
        resolution["fracture_energy"] = PropertyResolutionStatus.COMPOSED_REQUIRED

        physical = ConcretePhysicalProperties(
            f_cm=f_cm,
            f_ck=f_ck,
            f_ctm=f_ctm,
            f_ctk_lower=f_ctk_lower,
            f_ctk_upper=f_ctk_upper,
            E_initial=e_initial,
            E_secant=e_secant,
            poisson_elastic=poisson_elastic,
            shear_modulus_secant_equivalent=shear_modulus,
            fracture_energy=None,
            strain_peak_compression=epsilon_c1_per_mille / 1000.0,
            strain_limit_compression=epsilon_cu1_per_mille / 1000.0,
            reference_age_days=REFERENCE_AGE_DAYS,
            provenance=provenance,
            resolution=resolution,
        )
        return physical, effective_configuration
