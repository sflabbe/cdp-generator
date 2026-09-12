"""Verified EN 1992-1-1:2023 physical concrete profile.

This module implements only the G1-A-adjudicated ambient normal-concrete
named-class reference state. It preserves the distinction between properties at
``t_ref`` and the specifically 28-day tangent modulus ``E_c,28``. It does not
implement strength-age evolution, Annex-B ageing, fire design, fracture-energy
composition, or constitutive-model calibration.
"""

import math

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

PROFILE_ID = "ec2_2023"
SOURCE_ID = "en_1992_1_1_2023"
SOURCE_EDITION = "2023"
DEFAULT_REFERENCE_AGE_DAYS = 28.0
MIN_REFERENCE_AGE_DAYS = 28.0
MAX_REFERENCE_AGE_DAYS = 91.0
DEFAULT_K_E = 9500.0
MIN_K_E = 5000.0
MAX_K_E = 13000.0

_POISSON_ELASTIC = 0.20
_TANGENT_APPROXIMATION_FACTOR = 1.05

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
    overridden: bool = False,
) -> PropertyProvenance:
    return PropertyProvenance(
        source_id=SOURCE_ID,
        source_kind=SourceKind.STANDARD,
        edition=SOURCE_EDITION,
        equation_or_section=equation_or_section,
        units=units,
        statistical_basis=statistical_basis,
        notes=notes,
        overridden=overridden,
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


def _validated_numeric_parameter(
    *,
    profile_parameter: str,
    value: object,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"ec2_2023 {profile_parameter} must be numeric when supplied")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"ec2_2023 {profile_parameter} must be finite")
    if not minimum <= numeric <= maximum:
        raise ValueError(
            f"ec2_2023 {profile_parameter} must be within [{minimum:g}, {maximum:g}]; got {value!r}"
        )
    return numeric


def _resolve_configuration(
    profile_parameters: ProfileConfiguration | None,
) -> ProfileConfiguration:
    """Resolve the narrow G1-B4 EC2:2023 reference-state configuration."""

    parameters = {} if profile_parameters is None else dict(profile_parameters.parameters)
    allowed = {"k_E", "reference_age_days"}
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unsupported ec2_2023 profile parameter(s): {names}")

    explicit_k_e = "k_E" in parameters
    k_e = _validated_numeric_parameter(
        profile_parameter="k_E",
        value=parameters.get("k_E", DEFAULT_K_E),
        minimum=MIN_K_E,
        maximum=MAX_K_E,
    )
    reference_age_days = _validated_numeric_parameter(
        profile_parameter="reference_age_days",
        value=parameters.get("reference_age_days", DEFAULT_REFERENCE_AGE_DAYS),
        minimum=MIN_REFERENCE_AGE_DAYS,
        maximum=MAX_REFERENCE_AGE_DAYS,
    )

    return ProfileConfiguration(
        {
            "k_E": k_e,
            "k_E_basis": (
                "explicit_standard_parameter" if explicit_k_e else "quartzite_assumption"
            ),
            "reference_age_days": reference_age_days,
        }
    )


class Ec2_2023Profile:
    """Verified EC2:2023 named-class physical profile for normal concrete."""

    profile_id = PROFILE_ID

    @staticmethod
    def build(
        class_entry: ConcreteClassEntry,
        profile_parameters: ProfileConfiguration | None = None,
    ) -> tuple[ConcretePhysicalProperties, ProfileConfiguration]:
        """Build one named-class EC2:2023 reference-state physical definition."""

        if class_entry.profile != PROFILE_ID:
            raise ValueError(
                f"Ec2_2023Profile requires a {PROFILE_ID!r} class entry, "
                f"got {class_entry.profile!r}"
            )

        raw_parameters = {} if profile_parameters is None else dict(profile_parameters.parameters)
        explicit_k_e = "k_E" in raw_parameters
        effective_configuration = _resolve_configuration(profile_parameters)
        k_e = effective_configuration.parameters["k_E"]
        k_e_basis = effective_configuration.parameters["k_E_basis"]
        reference_age_days = effective_configuration.parameters["reference_age_days"]
        if (
            not isinstance(k_e, float | int)
            or not isinstance(k_e_basis, str)
            or not isinstance(reference_age_days, float | int)
        ):
            raise TypeError("resolved ec2_2023 profile configuration is invalid")

        k_e_float = float(k_e)
        reference_age = float(reference_age_days)
        age_note = f"selected t_ref={reference_age:g} days"

        f_ck = class_entry.f_ck_cylinder_mpa
        f_cm = f_ck + 8.0

        if f_ck <= 50.0:
            f_ctm = 0.30 * f_ck ** (2.0 / 3.0)
            f_ctm_branch = "Table 5.1 lower branch for f_ck <= 50 MPa"
        else:
            f_ctm = 1.10 * f_ck ** (1.0 / 3.0)
            f_ctm_branch = "Table 5.1 high branch for f_ck > 50 MPa"

        f_ctk_lower = 0.7 * f_ctm
        f_ctk_upper = 1.3 * f_ctm

        e_secant = k_e_float * f_cm ** (1.0 / 3.0)
        if reference_age == DEFAULT_REFERENCE_AGE_DAYS:
            e_initial: float | None = _TANGENT_APPROXIMATION_FACTOR * e_secant
            e_initial_resolution = PropertyResolutionStatus.DERIVED
            e_initial_provenance = _derived_provenance(
                source_id=SOURCE_ID,
                equation_or_section="§5.1.5(1), Eq. (5.2) context",
                statistical_basis=StatisticalBasis.MEAN,
                notes=(
                    "Repository E_initial maps the specifically 28-day tangent modulus E_c,28; "
                    "at t_ref=28 days the standard-authorized approximation is "
                    "E_c,28 = 1.05 E_cm."
                ),
                derived_from=("E_secant", "reference_age_days"),
            )
        else:
            e_initial = None
            e_initial_resolution = PropertyResolutionStatus.UNRESOLVED
            e_initial_provenance = _derived_provenance(
                source_id=SOURCE_ID,
                equation_or_section=None,
                statistical_basis=StatisticalBasis.MEAN,
                notes=(
                    "Repository E_initial maps E_c,28, but this object is defined at "
                    f"t_ref={reference_age:g} days. Its E_secant is E_cm at the selected "
                    "reference state; reconstructing the 28-day tangent state requires "
                    "additional age-development context outside G1-B4."
                ),
                derived_from=("E_secant", "reference_age_days"),
            )

        poisson_elastic = _POISSON_ELASTIC
        shear_modulus = e_secant / (2.0 * (1.0 + poisson_elastic))

        epsilon_c1_uncapped = 0.7 * f_cm ** (1.0 / 3.0)
        epsilon_c1_per_mille = min(epsilon_c1_uncapped, 2.8)
        epsilon_c1_state = "capped at 2.8 per-mille" if epsilon_c1_uncapped >= 2.8 else "uncapped"

        epsilon_cu1_uncapped = 2.8 + 14.0 * (1.0 - f_cm / 108.0) ** 4
        epsilon_cu1_per_mille = min(epsilon_cu1_uncapped, 3.5)
        epsilon_cu1_state = "capped at 3.5 per-mille" if epsilon_cu1_uncapped >= 3.5 else "uncapped"

        provenance: dict[str, PropertyProvenance] = {
            "f_ck": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.CHARACTERISTIC,
                equation_or_section="§5.1.3(1)-(3); Table 5.1",
                notes=(
                    "Characteristic 5% cylinder compressive strength from the explicit "
                    f"EC2:2023 class-registry entry at {age_note}."
                ),
            ),
            "f_cm": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.MEAN,
                equation_or_section="Table 5.1",
                notes=f"Mean cylinder compressive strength at {age_note}; f_cm = f_ck + 8 MPa.",
                derived_from=("f_ck",),
            ),
            "f_ctm": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.MEAN,
                equation_or_section="Table 5.1",
                notes=f"Mean axial tensile strength at {age_note}; {f_ctm_branch}.",
                derived_from=("f_ck",),
            ),
            "f_ctk_lower": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.CHARACTERISTIC,
                equation_or_section="Table 5.1",
                notes=f"EC2 f_ctk,0.05 at {age_note}; explicit 5% fractile.",
                derived_from=("f_ctm",),
            ),
            "f_ctk_upper": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.CHARACTERISTIC,
                equation_or_section="Table 5.1",
                notes=f"EC2 f_ctk,0.95 at {age_note}; explicit 95% fractile.",
                derived_from=("f_ctm",),
            ),
            "E_initial": e_initial_provenance,
            "E_secant": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.MEAN,
                equation_or_section="§5.1.4(2), Eq. (5.1)",
                notes=(
                    "EC2:2023 E_cm is the secant modulus from sigma_c=0 to 0.4 f_cm; "
                    f"k_E={k_e_float:g}, basis={k_e_basis}, {age_note}."
                ),
                derived_from=("f_cm", "reference_age_days"),
                overridden=explicit_k_e,
            ),
            "poisson_elastic": _standard_provenance(
                units="dimensionless",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                equation_or_section="§5.1.4(3)",
                notes=(
                    "EC2:2023 uncracked elastic Poisson ratio nu = 0.20; the separate "
                    "allowance nu = 0 for concrete cracked in tension is not the repository "
                    "poisson_elastic state."
                ),
            ),
            "shear_modulus_secant_equivalent": _derived_provenance(
                source_id=PROFILE_ID,
                equation_or_section=None,
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                notes=(
                    "Repository-derived isotropic secant-equivalent shear modulus; "
                    "not claimed as a directly specified EC2:2023 quantity."
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
                    "EN 1992-1-1:2023 Section 5 does not supply the repository plain-concrete "
                    "fracture-energy quantity; explicit composition with another authority "
                    "is required."
                ),
                overridden=False,
            ),
            "strain_peak_compression": _standard_provenance(
                units="dimensionless",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                equation_or_section="§5.1.6(3), Eq. (5.9)",
                notes=(
                    "General nonlinear/Sargin peak strain epsilon_c1; "
                    f"{epsilon_c1_state}; {age_note}. Source and repository both use "
                    "positive compression magnitudes."
                ),
                derived_from=("f_cm",),
                normalizations=(_STRAIN_UNIT_NORMALIZATION,),
            ),
            "strain_limit_compression": _standard_provenance(
                units="dimensionless",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                equation_or_section="§5.1.6(3), Eq. (5.10)",
                notes=(
                    "General nonlinear/Sargin nominal ultimate strain epsilon_cu1; "
                    f"{epsilon_cu1_state}; {age_note}. Source and repository both use "
                    "positive compression magnitudes."
                ),
                derived_from=("f_cm",),
                normalizations=(_STRAIN_UNIT_NORMALIZATION,),
            ),
            "reference_age_days": _standard_provenance(
                units="days",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                equation_or_section="§5.1.3(1)-(3); Table 5.1",
                notes=(
                    "EC2:2023 reference age is 28 days generally and may be project-specified "
                    f"from 28 to 91 days; selected t_ref={reference_age:g} days."
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
        resolution["E_initial"] = e_initial_resolution
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
            reference_age_days=reference_age,
            provenance=provenance,
            resolution=resolution,
        )
        return physical, effective_configuration
