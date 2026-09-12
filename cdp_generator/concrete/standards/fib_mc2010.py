"""Verified fib Model Code 2010 physical concrete profile.

This module implements only the authority-adjudicated named-grade physical
properties for ordinary normal-weight concrete.  It deliberately does not
provide time-development models, arbitrary strength interpolation, EC2
profiles, or constitutive-model calibration.
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

PROFILE_ID = "fib_mc2010"
SOURCE_ID = "fib_mc2010_2013"
SOURCE_EDITION = "2013"
REFERENCE_AGE_DAYS = 28.0

_E_C0_MPA = 21_500.0
_POISSON_ELASTIC = 0.20

AGGREGATE_ALPHA_E = MappingProxyType(
    {
        "basalt": 1.2,
        "quartzite": 1.0,
        "limestone": 0.9,
        "sandstone": 0.7,
    }
)

# Table 5.1-8 values are source-negative per-mille values.  The production
# schema stores positive dimensionless magnitudes, so build() normalizes both
# sign and units explicitly and records those steps in provenance.
_COMPRESSION_STRAINS_PER_MILLE = MappingProxyType(
    {
        "C12": (-1.9, -3.5),
        "C16": (-2.0, -3.5),
        "C20": (-2.1, -3.5),
        "C25": (-2.2, -3.5),
        "C30": (-2.3, -3.5),
        "C35": (-2.3, -3.5),
        "C40": (-2.4, -3.5),
        "C45": (-2.5, -3.5),
        "C50": (-2.6, -3.4),
        "C55": (-2.6, -3.4),
        "C60": (-2.7, -3.3),
        "C70": (-2.7, -3.2),
        "C80": (-2.8, -3.1),
        "C90": (-2.9, -3.0),
        "C100": (-3.0, -3.0),
        "C110": (-3.0, -3.0),
        "C120": (-3.0, -3.0),
    }
)

_GF_UNIT_NORMALIZATION = PropertyNormalization(
    kind=NormalizationKind.UNIT_CONVERSION,
    source_convention="G_F expressed in N/m",
    repository_convention="fracture_energy expressed in N/mm; divide source value by 1000",
)
_STRAIN_SIGN_NORMALIZATION = PropertyNormalization(
    kind=NormalizationKind.SIGN_CONVENTION,
    source_convention="compression strain is negative in MC2010 Table 5.1-8",
    repository_convention="compression strain stored as a positive magnitude",
)
_STRAIN_UNIT_NORMALIZATION = PropertyNormalization(
    kind=NormalizationKind.UNIT_CONVERSION,
    source_convention="compression strain expressed in per-mille",
    repository_convention="dimensionless strain; divide magnitude by 1000",
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


def _resolve_configuration(
    profile_parameters: ProfileConfiguration | None,
) -> ProfileConfiguration:
    """Resolve the narrow MC2010 B2 configuration to an effective policy."""

    parameters = {} if profile_parameters is None else dict(profile_parameters.parameters)
    allowed = {"aggregate_type", "reference_age_days"}
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unsupported fib_mc2010 profile parameter(s): {names}")

    aggregate_type = parameters.get("aggregate_type", "quartzite")
    if not isinstance(aggregate_type, str) or aggregate_type not in AGGREGATE_ALPHA_E:
        allowed_names = ", ".join(AGGREGATE_ALPHA_E)
        raise ValueError(
            f"fib_mc2010 aggregate_type must be one of {allowed_names}; got {aggregate_type!r}"
        )

    requested_age = parameters.get("reference_age_days", 28)
    if isinstance(requested_age, bool) or not isinstance(requested_age, int | float):
        raise TypeError("fib_mc2010 reference_age_days must be numeric when supplied")
    if float(requested_age) != REFERENCE_AGE_DAYS:
        raise ValueError(
            "fib_mc2010 B2 implements only the 28-day named-grade reference state; "
            f"got reference_age_days={requested_age!r}"
        )

    return ProfileConfiguration(
        {
            "aggregate_type": aggregate_type,
            "alpha_E": AGGREGATE_ALPHA_E[aggregate_type],
            "reference_age_days": 28,
        }
    )


class FibMc2010Profile:
    """Verified MC2010 named-class physical profile for normal-weight concrete."""

    profile_id = PROFILE_ID

    @staticmethod
    def build(
        class_entry: ConcreteClassEntry,
        profile_parameters: ProfileConfiguration | None = None,
    ) -> tuple[ConcretePhysicalProperties, ProfileConfiguration]:
        """Build one 28-day named-grade MC2010 physical definition."""

        if class_entry.profile != PROFILE_ID:
            raise ValueError(
                f"FibMc2010Profile requires a {PROFILE_ID!r} class entry, "
                f"got {class_entry.profile!r}"
            )

        effective_configuration = _resolve_configuration(profile_parameters)
        aggregate_type = effective_configuration.parameters["aggregate_type"]
        alpha_e = effective_configuration.parameters["alpha_E"]
        if not isinstance(aggregate_type, str) or not isinstance(alpha_e, float | int):
            raise TypeError("resolved fib_mc2010 aggregate configuration is invalid")

        f_ck = class_entry.f_ck_cylinder_mpa
        f_cm = f_ck + 8.0

        if f_ck <= 50.0:
            f_ctm = 0.3 * f_ck ** (2.0 / 3.0)
            f_ctm_equation = "§5.1.5.1, Eq. (5.1-3a)"
        else:
            f_ctm = 2.12 * math.log(1.0 + f_cm / 10.0)
            f_ctm_equation = "§5.1.5.1, Eq. (5.1-3b)"

        f_ctk_lower = 0.7 * f_ctm
        f_ctk_upper = 1.3 * f_ctm

        e_initial = _E_C0_MPA * float(alpha_e) * (f_cm / 10.0) ** (1.0 / 3.0)
        alpha_i = min(0.8 + 0.2 * f_cm / 88.0, 1.0)
        e_secant = alpha_i * e_initial

        poisson_elastic = _POISSON_ELASTIC
        shear_modulus = e_secant / (2.0 * (1.0 + poisson_elastic))

        fracture_energy_n_per_m = 73.0 * f_cm**0.18
        fracture_energy = fracture_energy_n_per_m / 1000.0

        try:
            epsilon_c1_per_mille, epsilon_clim_per_mille = _COMPRESSION_STRAINS_PER_MILLE[
                class_entry.canonical_class_string
            ]
        except KeyError as exc:  # Defensive: registry/table parity is also qualified in tests.
            raise ValueError(
                "No MC2010 Table 5.1-8 compression landmarks for "
                f"{class_entry.canonical_class_string!r}"
            ) from exc
        strain_peak = abs(epsilon_c1_per_mille) / 1000.0
        strain_limit = abs(epsilon_clim_per_mille) / 1000.0

        provenance: dict[str, PropertyProvenance] = {
            "f_ck": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.CHARACTERISTIC,
                equation_or_section="§5.1.2; Table 5.1-3",
                notes=(
                    "Characteristic cylinder compressive strength from the explicit "
                    "MC2010 named-grade registry entry."
                ),
            ),
            "f_cm": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.MEAN,
                equation_or_section="§5.1.4, Eq. (5.1-1)",
                notes="Mean compressive strength with Δf = 8 MPa.",
                derived_from=("f_ck",),
            ),
            "f_ctm": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.MEAN,
                equation_or_section=f_ctm_equation,
                notes=(
                    "Mean uniaxial tensile strength; f_ck <= 50 MPa uses Eq. (5.1-3a), "
                    "f_ck > 50 MPa uses Eq. (5.1-3b)."
                ),
                derived_from=("f_ck", "f_cm"),
            ),
            "f_ctk_lower": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.CHARACTERISTIC,
                equation_or_section="§5.1.5.1, Eq. (5.1-4)",
                notes=(
                    "MC2010 lower-bound characteristic tensile strength f_ctk,min; "
                    "no cross-profile percentile alias is asserted."
                ),
                derived_from=("f_ctm",),
            ),
            "f_ctk_upper": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.CHARACTERISTIC,
                equation_or_section="§5.1.5.1, Eq. (5.1-5)",
                notes=(
                    "MC2010 upper-bound characteristic tensile strength f_ctk,max; "
                    "no cross-profile percentile alias is asserted."
                ),
                derived_from=("f_ctm",),
            ),
            "E_initial": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.MEAN,
                equation_or_section="§5.1.7.2, Eq. (5.1-21); Table 5.1-6",
                notes=(
                    "MC2010 mean initial/tangent modulus E_ci at 28 days. "
                    f"aggregate_type={aggregate_type}; alpha_E={float(alpha_e):g}."
                ),
                derived_from=("f_cm",),
            ),
            "E_secant": _standard_provenance(
                units="MPa",
                statistical_basis=StatisticalBasis.MEAN,
                equation_or_section="§5.1.7.2, Eqs. (5.1-23), (5.1-24); Table 5.1-7",
                notes=(
                    "MC2010 reduced/secant E_c for elastic analysis, not the Sargin "
                    "origin-to-peak secant E_c1."
                ),
                derived_from=("E_initial", "f_cm"),
            ),
            "poisson_elastic": _standard_provenance(
                units="dimensionless",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                equation_or_section="§5.1.7.3",
                notes=(
                    "MC2010 engineering estimate nu_c = 0.20 for the elastic/uncracked "
                    "state; quoted range 0.14-0.26 applies for "
                    "-0.6 f_ck < sigma_c < 0.8 f_ctk."
                ),
            ),
            "shear_modulus_secant_equivalent": PropertyProvenance(
                source_id=PROFILE_ID,
                source_kind=SourceKind.DERIVED,
                edition=SOURCE_EDITION,
                equation_or_section=None,
                units="MPa",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                notes=(
                    "Repository-derived isotropic secant-equivalent shear modulus; "
                    "not claimed as a directly specified MC2010 quantity."
                ),
                overridden=False,
                derived_from=("E_secant", "poisson_elastic"),
            ),
            "fracture_energy": _standard_provenance(
                units="N/mm",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                equation_or_section="§5.1.5.2, Eq. (5.1-9)",
                notes=(
                    "MC2010 G_F for ordinary normal-weight concrete when experimental "
                    "data are unavailable; source equation returns N/m."
                ),
                derived_from=("f_cm",),
                normalizations=(_GF_UNIT_NORMALIZATION,),
            ),
            "strain_peak_compression": _standard_provenance(
                units="dimensionless",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                equation_or_section="§5.1.8.1, Eq. (5.1-26); Table 5.1-8",
                notes=(
                    "Named-grade Sargin/general nonlinear landmark epsilon_c1; source "
                    "table value normalized from negative per-mille."
                ),
                normalizations=(_STRAIN_SIGN_NORMALIZATION, _STRAIN_UNIT_NORMALIZATION),
            ),
            "strain_limit_compression": _standard_provenance(
                units="dimensionless",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                equation_or_section="§5.1.8.1, Eq. (5.1-26); Table 5.1-8",
                notes=(
                    "Named-grade Sargin/general nonlinear epsilon_c,lim; source table "
                    "value normalized from negative per-mille. MC2010 cautions that the "
                    "post-peak limit is size/boundary-condition sensitive."
                ),
                normalizations=(_STRAIN_SIGN_NORMALIZATION, _STRAIN_UNIT_NORMALIZATION),
            ),
            "reference_age_days": _standard_provenance(
                units="days",
                statistical_basis=StatisticalBasis.NOT_APPLICABLE,
                equation_or_section="§5.1.2; §5.1.7.2",
                notes=(
                    "Named-grade B2 reference state fixed to 28 days; arbitrary time "
                    "development is outside G1-B2."
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
                "E_initial",
                "E_secant",
                "poisson_elastic",
                "fracture_energy",
                "strain_peak_compression",
                "strain_limit_compression",
                "reference_age_days",
            )
        }
        resolution["shear_modulus_secant_equivalent"] = PropertyResolutionStatus.DERIVED

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
            fracture_energy=fracture_energy,
            strain_peak_compression=strain_peak,
            strain_limit_compression=strain_limit,
            reference_age_days=REFERENCE_AGE_DAYS,
            provenance=provenance,
            resolution=resolution,
        )
        return physical, effective_configuration
