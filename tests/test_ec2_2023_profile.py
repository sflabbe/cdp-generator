"""G1-B4 qualification for the verified EN 1992-1-1:2023 physical profile."""

import json
import math
from pathlib import Path

import pytest

from cdp_generator.concrete import (
    Concrete,
    CrossProfileConcreteClassError,
    ProfileConfiguration,
    PropertyResolutionStatus,
    SourceKind,
    StatisticalBasis,
    class_entries,
)
from cdp_generator.concrete.provenance import NormalizationKind
from cdp_generator.concrete.standards.ec2_2023 import (
    DEFAULT_K_E,
    DEFAULT_REFERENCE_AGE_DAYS,
    MAX_K_E,
    MAX_REFERENCE_AGE_DAYS,
    MIN_K_E,
    MIN_REFERENCE_AGE_DAYS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FIXTURE = REPO_ROOT / "qualification" / "standards" / "ec2_2023_reference.json"

# Production and the checked-in fixture independently evaluate the frozen equations in binary64.
# These tolerances permit last-bit/libm variation only, never published-table rounding.
EQUATION_REL_TOL = 1e-13
EQUATION_ABS_TOL = 1e-12


def _fixture() -> dict[str, object]:
    return json.loads(REFERENCE_FIXTURE.read_text())


def _class_records() -> dict[str, dict[str, object]]:
    classes = _fixture()["classes"]
    assert isinstance(classes, list)
    return {record["concrete_class"]: record for record in classes}


def _context_records() -> dict[str, dict[str, object]]:
    records = _fixture()["context_cases"]
    assert isinstance(records, list)
    return {record["case"]: record for record in records}


def _ec2(
    concrete_class: str,
    *,
    k_e: float | int | None = None,
    reference_age_days: float | int | None = None,
) -> Concrete:
    parameters: dict[str, float | int] = {}
    if k_e is not None:
        parameters["k_E"] = k_e
    if reference_age_days is not None:
        parameters["reference_age_days"] = reference_age_days
    return Concrete.from_class(
        concrete_class,
        profile="ec2_2023",
        profile_parameters=ProfileConfiguration(parameters) if parameters else None,
    )


def test_independent_reference_fixture_covers_exactly_all_15_ec2_2023_classes():
    fixture = _fixture()
    evidence = fixture["evidence"]
    authority = fixture["authority"]
    assert isinstance(evidence, dict)
    assert isinstance(authority, dict)

    assert fixture["schema_version"] == "ec2_2023_reference.v1"
    assert fixture["profile"] == "ec2_2023"
    assert fixture["reference_configuration"] == {
        "k_E": 9500.0,
        "k_E_basis": "quartzite_assumption",
        "reference_age_days": 28.0,
    }
    assert authority["source_id"] == "en_1992_1_1_2023"
    assert authority["edition"] == "2023"
    assert evidence["structuralcodes_used"] is False
    assert evidence["final_published_en_pdf_directly_inspected"] is False
    assert "without importing cdp_generator production code" in str(evidence["method"])

    records = _class_records()
    registry_entries = tuple(class_entries("ec2_2023"))
    registry_classes = tuple(entry.canonical_class_string for entry in registry_entries)
    assert len(records) == 15
    assert tuple(records) == registry_classes
    assert registry_classes[0] == "C12/15"
    assert registry_classes[-1] == "C100/115"

    for entry in registry_entries:
        record = records[entry.canonical_class_string]
        assert record["f_ck_cylinder_mpa"] == entry.f_ck_cylinder_mpa
        assert record["f_ck_cube_mpa"] == entry.f_ck_cube_mpa


@pytest.mark.parametrize(
    "concrete_class",
    [entry.canonical_class_string for entry in class_entries("ec2_2023")],
)
def test_all_15_named_classes_match_independent_reference_fixture(concrete_class: str):
    expected = _class_records()[concrete_class]
    concrete = _ec2(concrete_class)
    physical = concrete.physical

    assert concrete.physical_profile == "ec2_2023"
    assert physical.f_ck == expected["f_ck_cylinder_mpa"]
    assert physical.f_cm == expected["f_cm_mpa"]
    assert physical.f_ctm == pytest.approx(
        expected["f_ctm_mpa"], rel=EQUATION_REL_TOL, abs=EQUATION_ABS_TOL
    )
    assert physical.f_ctk_lower == pytest.approx(
        expected["f_ctk_lower_mpa"], rel=EQUATION_REL_TOL, abs=EQUATION_ABS_TOL
    )
    assert physical.f_ctk_upper == pytest.approx(
        expected["f_ctk_upper_mpa"], rel=EQUATION_REL_TOL, abs=EQUATION_ABS_TOL
    )
    assert physical.E_secant == pytest.approx(
        expected["E_secant_mpa"], rel=EQUATION_REL_TOL, abs=EQUATION_ABS_TOL
    )
    assert physical.E_initial == pytest.approx(
        expected["E_initial_mpa"], rel=EQUATION_REL_TOL, abs=EQUATION_ABS_TOL
    )
    assert physical.poisson_elastic == expected["poisson_elastic"] == 0.2
    assert physical.shear_modulus_secant_equivalent == pytest.approx(
        expected["shear_modulus_secant_equivalent_mpa"],
        rel=EQUATION_REL_TOL,
        abs=EQUATION_ABS_TOL,
    )
    assert physical.fracture_energy is None
    assert physical.strain_peak_compression == pytest.approx(
        expected["strain_peak_compression"], rel=EQUATION_REL_TOL, abs=1e-15
    )
    assert physical.strain_limit_compression == pytest.approx(
        expected["strain_limit_compression"], rel=EQUATION_REL_TOL, abs=1e-15
    )
    assert physical.reference_age_days == expected["reference_age_days"] == 28.0

    direct_fields = (
        "f_ck",
        "f_cm",
        "f_ctm",
        "f_ctk_lower",
        "f_ctk_upper",
        "E_secant",
        "poisson_elastic",
        "strain_peak_compression",
        "strain_limit_compression",
        "reference_age_days",
    )
    assert all(
        physical.resolution[field] is PropertyResolutionStatus.DIRECT for field in direct_fields
    )
    assert physical.resolution["E_initial"] is PropertyResolutionStatus.DERIVED
    assert (
        physical.resolution["shear_modulus_secant_equivalent"] is PropertyResolutionStatus.DERIVED
    )
    assert physical.resolution["fracture_energy"] is PropertyResolutionStatus.COMPOSED_REQUIRED


def test_default_configuration_makes_quartzite_assumption_explicit():
    concrete = Concrete.from_class("C30/37", profile="ec2_2023")

    assert concrete.profile_parameters.to_dict() == {
        "k_E": DEFAULT_K_E,
        "k_E_basis": "quartzite_assumption",
        "reference_age_days": DEFAULT_REFERENCE_AGE_DAYS,
    }
    assert concrete.physical.E_initial == pytest.approx(1.05 * concrete.physical.E_secant)
    assert concrete.physical.provenance["E_secant"].overridden is False
    assert "basis=quartzite_assumption" in concrete.physical.provenance["E_secant"].notes


def test_explicit_default_k_e_is_distinguishable_from_implicit_quartzite_assumption():
    concrete = _ec2("C30/37", k_e=9500.0)

    assert concrete.profile_parameters.to_dict() == {
        "k_E": 9500.0,
        "k_E_basis": "explicit_standard_parameter",
        "reference_age_days": 28.0,
    }
    provenance = concrete.physical.provenance["E_secant"]
    assert provenance.source_kind is SourceKind.STANDARD
    assert provenance.overridden is True
    assert "basis=explicit_standard_parameter" in provenance.notes


@pytest.mark.parametrize("k_e", [MIN_K_E, DEFAULT_K_E, MAX_K_E])
def test_k_e_closed_interval_endpoints_are_accepted(k_e: float):
    concrete = _ec2("C30/37", k_e=k_e)
    assert concrete.profile_parameters.parameters["k_E"] == k_e
    assert concrete.physical.E_secant == pytest.approx(k_e * 38.0 ** (1.0 / 3.0))


def test_k_e_scales_stiffness_family_linearly_without_moving_strengths():
    low = _ec2("C30/37", k_e=5000.0).physical
    default = _ec2("C30/37").physical
    high = _ec2("C30/37", k_e=13000.0).physical

    for candidate, factor in ((low, 5000.0 / 9500.0), (high, 13000.0 / 9500.0)):
        assert candidate.E_secant == pytest.approx(default.E_secant * factor)
        assert candidate.E_initial == pytest.approx(default.E_initial * factor)
        assert candidate.shear_modulus_secant_equivalent == pytest.approx(
            default.shear_modulus_secant_equivalent * factor
        )
        assert candidate.f_ck == default.f_ck
        assert candidate.f_cm == default.f_cm
        assert candidate.f_ctm == default.f_ctm
        assert candidate.f_ctk_lower == default.f_ctk_lower
        assert candidate.f_ctk_upper == default.f_ctk_upper
        assert candidate.strain_peak_compression == default.strain_peak_compression
        assert candidate.strain_limit_compression == default.strain_limit_compression
        assert candidate.fracture_energy is None


@pytest.mark.parametrize(
    "parameters, error_type, match",
    [
        ({"k_E": 4999.999}, ValueError, "within"),
        ({"k_E": 13000.001}, ValueError, "within"),
        ({"k_E": True}, TypeError, "numeric"),
        ({"k_E": "9500"}, TypeError, "numeric"),
        ({"k_E": math.inf}, ValueError, "finite"),
        ({"k_E": math.nan}, ValueError, "finite"),
        ({"aggregate_type": "quartzite"}, ValueError, "Unsupported"),
        ({"alpha_E": 1.0}, ValueError, "Unsupported"),
        ({"Ecm_factor": 1.0}, ValueError, "Unsupported"),
        ({"kE": 9500}, ValueError, "Unsupported"),
        ({"ke": 9500}, ValueError, "Unsupported"),
    ],
)
def test_configuration_rejects_invalid_k_e_and_cross_profile_keys(
    parameters: dict[str, object], error_type: type[Exception], match: str
):
    with pytest.raises(error_type, match=match):
        Concrete.from_class(
            "C30/37",
            profile="ec2_2023",
            profile_parameters=ProfileConfiguration(parameters),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "age, expected_e_initial",
    [
        (MIN_REFERENCE_AGE_DAYS, True),
        (56.0, False),
        (MAX_REFERENCE_AGE_DAYS, False),
    ],
)
def test_reference_age_closed_interval_and_tangent_resolution(age: float, expected_e_initial: bool):
    concrete = _ec2("C30/37", reference_age_days=age)
    physical = concrete.physical

    assert physical.reference_age_days == age
    assert concrete.profile_parameters.parameters["reference_age_days"] == age
    if expected_e_initial:
        assert physical.E_initial == pytest.approx(1.05 * physical.E_secant)
        assert physical.resolution["E_initial"] is PropertyResolutionStatus.DERIVED
    else:
        assert physical.E_initial is None
        assert physical.resolution["E_initial"] is PropertyResolutionStatus.UNRESOLVED
        provenance = physical.provenance["E_initial"]
        assert provenance.source_kind is SourceKind.DERIVED
        assert provenance.equation_or_section is None
        assert provenance.derived_from == ("E_secant", "reference_age_days")
        assert "E_c,28" in provenance.notes
        assert "age-development context" in provenance.notes


@pytest.mark.parametrize(
    "age, error_type, match",
    [
        (27.999, ValueError, "within"),
        (91.001, ValueError, "within"),
        (True, TypeError, "numeric"),
        ("56", TypeError, "numeric"),
        (math.inf, ValueError, "finite"),
        (math.nan, ValueError, "finite"),
    ],
)
def test_reference_age_outside_standard_context_is_rejected(
    age: object, error_type: type[Exception], match: str
):
    with pytest.raises(error_type, match=match):
        Concrete.from_class(
            "C30/37",
            profile="ec2_2023",
            profile_parameters=ProfileConfiguration(
                {"reference_age_days": age}  # type: ignore[dict-item]
            ),
        )


def test_non_28_reference_state_keeps_reference_properties_but_not_fake_e_c_28():
    c28 = _ec2("C30/37", reference_age_days=28).physical
    c56 = _ec2("C30/37", reference_age_days=56).physical

    assert c56.f_ck == c28.f_ck == 30.0
    assert c56.f_cm == c28.f_cm == 38.0
    assert c56.f_ctm == c28.f_ctm
    assert c56.E_secant == c28.E_secant
    assert c56.E_initial is None
    assert c28.E_initial == pytest.approx(1.05 * c28.E_secant)
    assert "selected t_ref=56 days" in c56.provenance["f_ck"].notes
    assert "selected t_ref=56 days" in c56.provenance["E_secant"].notes


def test_context_fixture_covers_k_e_endpoints_and_non_28_reference_ages():
    expected = _context_records()
    cases = (
        ("C30/37 k_E=5000 at t_ref=28", 5000.0, 28.0),
        ("C30/37 k_E=13000 at t_ref=28", 13000.0, 28.0),
        ("C30/37 k_E=9500 at t_ref=56", 9500.0, 56.0),
        ("C30/37 k_E=9500 at t_ref=91", 9500.0, 91.0),
    )
    for name, k_e, age in cases:
        record = expected[name]
        concrete = _ec2("C30/37", k_e=k_e, reference_age_days=age)
        physical = concrete.physical
        assert physical.E_secant == pytest.approx(record["E_secant_mpa"], rel=EQUATION_REL_TOL)
        assert physical.reference_age_days == record["reference_age_days"]
        assert physical.E_initial == record["E_initial_mpa"] or physical.E_initial == pytest.approx(
            record["E_initial_mpa"], rel=EQUATION_REL_TOL
        )


def test_fctm_c50_uses_lower_branch_and_c55_uses_new_2023_high_branch():
    c50 = _ec2("C50/60").physical
    c55 = _ec2("C55/67").physical

    assert c50.f_ctm == pytest.approx(0.30 * 50.0 ** (2.0 / 3.0))
    assert c55.f_ctm == pytest.approx(1.10 * 55.0 ** (1.0 / 3.0))
    assert c55.f_ctm != pytest.approx(2.12 * math.log(1.0 + c55.f_cm / 10.0), rel=1e-6)
    assert "f_ck <= 50 MPa" in c50.provenance["f_ctm"].notes
    assert "f_ck > 50 MPa" in c55.provenance["f_ctm"].notes


def test_epsilon_c1_cap_transition_is_independent_c55_uncapped_c60_capped():
    c55 = _ec2("C55/67").physical
    c60 = _ec2("C60/75").physical

    assert c55.strain_peak_compression is not None
    assert c55.strain_peak_compression < 0.0028
    assert c55.strain_peak_compression == pytest.approx(0.7 * 63.0 ** (1.0 / 3.0) / 1000.0)
    assert c60.strain_peak_compression == pytest.approx(0.0028)
    assert "uncapped" in c55.provenance["strain_peak_compression"].notes
    assert "capped at 2.8" in c60.provenance["strain_peak_compression"].notes


def test_epsilon_cu1_cap_transition_is_independent_c45_capped_c50_uncapped_c55_uncapped():
    c45 = _ec2("C45/55").physical
    c50 = _ec2("C50/60").physical
    c55 = _ec2("C55/67").physical

    c50_exact_per_mille = 2.8 + 14.0 * (1.0 - 58.0 / 108.0) ** 4
    c55_exact_per_mille = 2.8 + 14.0 * (1.0 - 63.0 / 108.0) ** 4

    assert c45.strain_limit_compression == pytest.approx(0.0035)
    assert c50_exact_per_mille == pytest.approx(3.4431511211968964)
    assert c50.strain_limit_compression == pytest.approx(c50_exact_per_mille / 1000.0)
    assert c55.strain_limit_compression == pytest.approx(c55_exact_per_mille / 1000.0)
    assert "capped at 3.5" in c45.provenance["strain_limit_compression"].notes
    assert "uncapped" in c50.provenance["strain_limit_compression"].notes
    assert "uncapped" in c55.provenance["strain_limit_compression"].notes


def test_upper_endpoint_c100_115_has_expected_new_generation_landmarks():
    physical = _ec2("C100/115").physical
    assert physical.f_ck == 100.0
    assert physical.f_cm == 108.0
    assert physical.f_ctm == pytest.approx(1.1 * 100.0 ** (1.0 / 3.0))
    assert physical.strain_peak_compression == pytest.approx(0.0028)
    assert physical.strain_limit_compression == pytest.approx(0.0028)


def test_compression_landmarks_are_direct_with_unit_conversion_only():
    physical = _ec2("C60/75").physical

    for field in ("strain_peak_compression", "strain_limit_compression"):
        provenance = physical.provenance[field]
        assert physical.resolution[field] is PropertyResolutionStatus.DIRECT
        assert provenance.source_kind is SourceKind.STANDARD
        assert provenance.source_id == "en_1992_1_1_2023"
        assert [normalization.kind for normalization in provenance.normalizations] == [
            NormalizationKind.UNIT_CONVERSION
        ]
        assert all(
            normalization.kind is not NormalizationKind.SIGN_CONVENTION
            for normalization in provenance.normalizations
        )
        assert "per-mille" in provenance.normalizations[0].source_convention
        assert "dimensionless" in provenance.normalizations[0].repository_convention


def test_e_initial_and_shear_have_distinct_repository_derived_semantics():
    physical = _ec2("C30/37").physical
    e_initial_provenance = physical.provenance["E_initial"]
    shear_provenance = physical.provenance["shear_modulus_secant_equivalent"]

    assert physical.E_initial == pytest.approx(1.05 * physical.E_secant)
    assert physical.E_initial != physical.E_secant
    assert physical.resolution["E_initial"] is PropertyResolutionStatus.DERIVED
    assert e_initial_provenance.source_kind is SourceKind.DERIVED
    assert e_initial_provenance.derived_from == ("E_secant", "reference_age_days")

    assert physical.shear_modulus_secant_equivalent == pytest.approx(
        physical.E_secant / (2.0 * (1.0 + physical.poisson_elastic))
    )
    assert (
        physical.resolution["shear_modulus_secant_equivalent"] is PropertyResolutionStatus.DERIVED
    )
    assert shear_provenance.source_kind is SourceKind.DERIVED
    assert shear_provenance.derived_from == ("E_secant", "poisson_elastic")
    assert shear_provenance.equation_or_section is None


def test_fracture_energy_is_truthfully_absent_and_composed_required():
    physical = _ec2("C30/37").physical
    provenance = physical.provenance["fracture_energy"]

    assert physical.fracture_energy is None
    assert physical.resolution["fracture_energy"] is PropertyResolutionStatus.COMPOSED_REQUIRED
    assert provenance.source_id == "en_1992_1_1_2023"
    assert provenance.source_kind is SourceKind.STANDARD
    assert provenance.equation_or_section is None
    assert provenance.derived_from == ()
    assert "composition" in provenance.notes


def test_provenance_source_identity_statistical_bases_and_locations():
    physical = _ec2("C30/37").physical
    expected = {
        "f_ck": (StatisticalBasis.CHARACTERISTIC, "§5.1.3(1)-(3); Table 5.1"),
        "f_cm": (StatisticalBasis.MEAN, "Table 5.1"),
        "f_ctm": (StatisticalBasis.MEAN, "Table 5.1"),
        "f_ctk_lower": (StatisticalBasis.CHARACTERISTIC, "Table 5.1"),
        "f_ctk_upper": (StatisticalBasis.CHARACTERISTIC, "Table 5.1"),
        "E_secant": (StatisticalBasis.MEAN, "§5.1.4(2), Eq. (5.1)"),
        "poisson_elastic": (StatisticalBasis.NOT_APPLICABLE, "§5.1.4(3)"),
        "strain_peak_compression": (StatisticalBasis.NOT_APPLICABLE, "§5.1.6(3), Eq. (5.9)"),
        "strain_limit_compression": (StatisticalBasis.NOT_APPLICABLE, "§5.1.6(3), Eq. (5.10)"),
        "reference_age_days": (StatisticalBasis.NOT_APPLICABLE, "§5.1.3(1)-(3); Table 5.1"),
    }
    for field, (basis, location) in expected.items():
        provenance = physical.provenance[field]
        assert provenance.source_id == "en_1992_1_1_2023"
        assert provenance.edition == "2023"
        assert provenance.source_kind is SourceKind.STANDARD
        assert provenance.statistical_basis is basis
        assert provenance.equation_or_section == location

    assert "5% fractile" in physical.provenance["f_ctk_lower"].notes
    assert "95% fractile" in physical.provenance["f_ctk_upper"].notes
    assert physical.poisson_elastic == 0.20


def test_direct_fields_stay_direct_with_dependencies_and_selected_standard_parameter():
    physical = _ec2("C30/37", k_e=10000.0).physical
    for field in (
        "f_ck",
        "f_cm",
        "f_ctm",
        "f_ctk_lower",
        "f_ctk_upper",
        "E_secant",
        "poisson_elastic",
        "strain_peak_compression",
        "strain_limit_compression",
        "reference_age_days",
    ):
        assert physical.resolution[field] is PropertyResolutionStatus.DIRECT
        assert physical.provenance[field].source_kind is SourceKind.STANDARD

    assert physical.provenance["f_cm"].derived_from == ("f_ck",)
    assert physical.provenance["f_ctm"].derived_from == ("f_ck",)
    assert physical.provenance["E_secant"].derived_from == ("f_cm", "reference_age_days")
    assert physical.provenance["E_secant"].overridden is True


def test_v2_serialization_is_deterministic_for_28_day_complete_state():
    concrete = _ec2("C30/37")
    first = concrete.to_json()
    second = concrete.to_json()
    assert first == second

    payload = json.loads(first)
    assert payload["schema_version"] == "concrete_material_definition.v2"
    assert payload["physical_profile"] == "ec2_2023"
    assert payload["profile_parameters"] == {
        "k_E": 9500.0,
        "k_E_basis": "quartzite_assumption",
        "reference_age_days": 28.0,
    }
    physical = payload["physical"]
    assert physical["schema_version"] == "concrete_physical_properties.v2"
    assert physical["values"]["E_initial"] is not None
    assert physical["resolution"]["E_initial"] == "DERIVED"
    assert physical["values"]["fracture_energy"] is None
    assert physical["resolution"]["fracture_energy"] == "COMPOSED_REQUIRED"


def test_v2_serialization_truthfully_preserves_non_28_unresolved_tangent_state():
    concrete = _ec2("C30/37", reference_age_days=56)
    payload = json.loads(concrete.to_json())

    assert payload["profile_parameters"]["reference_age_days"] == 56.0
    physical = payload["physical"]
    assert physical["values"]["reference_age_days"] == 56.0
    assert physical["values"]["E_initial"] is None
    assert physical["resolution"]["E_initial"] == "UNRESOLVED"
    assert physical["provenance"]["E_initial"]["equation_or_section"] is None
    assert physical["provenance"]["E_initial"]["derived_from"] == [
        "E_secant",
        "reference_age_days",
    ]


def test_default_from_class_now_constructs_verified_ec2_2023_profile():
    concrete = Concrete.from_class("C30/37")
    assert concrete.physical_profile == "ec2_2023"
    assert concrete.physical.reference_age_days == 28.0
    assert concrete.physical.E_initial is not None


def test_all_three_verified_class_profiles_are_operational_together():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    ec2_2004 = Concrete.from_class("C30/37", profile="ec2_2004")
    ec2_2023 = Concrete.from_class("C30/37", profile="ec2_2023")

    assert fib.physical_profile == "fib_mc2010"
    assert ec2_2004.physical_profile == "ec2_2004"
    assert ec2_2023.physical_profile == "ec2_2023"
    assert fib.physical.f_ck == ec2_2004.physical.f_ck == ec2_2023.physical.f_ck == 30.0


def test_verified_profiles_remain_class_based_on_mean_strength_path():
    for profile in ("fib_mc2010", "ec2_2004", "ec2_2023"):
        with pytest.raises(NotImplementedError, match="class-based"):
            Concrete.from_mean_strength(38.0, 0.0022, 0.0035, profile=profile)


def test_cross_profile_parser_protection_remains_strict_after_b4():
    with pytest.raises(CrossProfileConcreteClassError):
        Concrete.from_class("C30", profile="ec2_2023")
    with pytest.raises(CrossProfileConcreteClassError):
        Concrete.from_class("C30/37", profile="fib_mc2010")


def test_structuralcodes_is_not_a_runtime_dependency():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert "structuralcodes" not in pyproject.lower()
