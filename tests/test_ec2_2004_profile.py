"""G1-B3 qualification for the verified EN 1992-1-1:2004 physical profile."""

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
from cdp_generator.concrete.standards.ec2_2004 import AGGREGATE_ECM_FACTOR

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FIXTURE = REPO_ROOT / "qualification" / "standards" / "ec2_2004_reference.json"

# Production and fixture independently evaluate the same published equations in binary64.
# This tight tolerance allows only last-bit/libm variation; it does not absorb table rounding.
EQUATION_REL_TOL = 1e-13
EQUATION_ABS_TOL = 1e-12


def _fixture() -> dict[str, object]:
    return json.loads(REFERENCE_FIXTURE.read_text())


def _class_records() -> dict[str, dict[str, object]]:
    fixture = _fixture()
    classes = fixture["classes"]
    assert isinstance(classes, list)
    return {record["concrete_class"]: record for record in classes}


def _ec2(concrete_class: str, aggregate_type: str = "quartzite") -> Concrete:
    parameters = None
    if aggregate_type != "quartzite":
        parameters = ProfileConfiguration({"aggregate_type": aggregate_type})
    return Concrete.from_class(
        concrete_class,
        profile="ec2_2004",
        profile_parameters=parameters,
    )


def test_independent_reference_fixture_covers_exactly_all_14_ec2_2004_classes():
    fixture = _fixture()
    evidence = fixture["evidence"]
    authority = fixture["authority"]
    assert isinstance(evidence, dict)
    assert isinstance(authority, dict)
    assert fixture["schema_version"] == "ec2_2004_reference.v1"
    assert fixture["profile"] == "ec2_2004"
    assert fixture["reference_configuration"] == {
        "Ecm_factor": 1.0,
        "aggregate_type": "quartzite",
        "reference_age_days": 28,
    }
    assert authority["source_id"] == "en_1992_1_1_2004"
    assert authority["correction"] == "AC:2010 applied to the Table 3.1 epsilon_c1 cap notation"
    assert evidence["structuralcodes_used"] is False
    assert "without importing cdp_generator production code" in str(evidence["method"])

    records = _class_records()
    fixture_classes = tuple(records)
    registry_entries = tuple(class_entries("ec2_2004"))
    registry_classes = tuple(entry.canonical_class_string for entry in registry_entries)
    assert len(fixture_classes) == 14
    assert fixture_classes == registry_classes
    for entry in registry_entries:
        record = records[entry.canonical_class_string]
        assert record["f_ck_cylinder_mpa"] == entry.f_ck_cylinder_mpa
        assert record["f_ck_cube_mpa"] == entry.f_ck_cube_mpa


@pytest.mark.parametrize(
    "concrete_class",
    [entry.canonical_class_string for entry in class_entries("ec2_2004")],
)
def test_all_14_named_classes_match_independent_reference_fixture(concrete_class: str):
    expected = _class_records()[concrete_class]
    concrete = _ec2(concrete_class)
    physical = concrete.physical

    assert concrete.physical_profile == "ec2_2004"
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
        expected["E_secant_quartzite_mpa"], rel=EQUATION_REL_TOL, abs=EQUATION_ABS_TOL
    )
    assert physical.E_initial == pytest.approx(
        expected["E_initial_mpa"], rel=EQUATION_REL_TOL, abs=EQUATION_ABS_TOL
    )
    assert physical.poisson_elastic == expected["poisson_elastic"]
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

    assert all(
        physical.resolution[field] is PropertyResolutionStatus.DIRECT
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
        )
    )
    assert physical.resolution["E_initial"] is PropertyResolutionStatus.DERIVED
    assert (
        physical.resolution["shear_modulus_secant_equivalent"] is PropertyResolutionStatus.DERIVED
    )
    assert physical.resolution["fracture_energy"] is PropertyResolutionStatus.COMPOSED_REQUIRED


def test_default_quartzite_policy_is_explicit_and_v2_serialized():
    concrete = Concrete.from_class("C30/37", profile="ec2_2004")

    assert concrete.profile_parameters.to_dict() == {
        "Ecm_factor": 1.0,
        "aggregate_type": "quartzite",
        "reference_age_days": 28,
    }

    payload = concrete.to_dict()
    assert payload["schema_version"] == "concrete_material_definition.v2"
    assert payload["physical_profile"] == "ec2_2004"
    assert payload["profile_parameters"] == concrete.profile_parameters.to_dict()
    assert payload["physical"]["schema_version"] == "concrete_physical_properties.v2"
    assert payload["physical"]["values"]["f_ck"] == 30.0
    assert payload["physical"]["values"]["fracture_energy"] is None
    assert payload["physical"]["resolution"]["fracture_energy"] == "COMPOSED_REQUIRED"


def test_all_authorized_aggregate_families_scale_only_stiffness_family():
    reference = _ec2("C30/37")
    reference_physical = reference.physical
    reference_e_initial = reference_physical.E_initial
    assert reference_e_initial is not None

    assert dict(AGGREGATE_ECM_FACTOR) == {
        "basalt": 1.2,
        "quartzite": 1.0,
        "limestone": 0.9,
        "sandstone": 0.7,
    }

    for aggregate_type, factor in AGGREGATE_ECM_FACTOR.items():
        concrete = _ec2("C30/37", aggregate_type)
        physical = concrete.physical
        assert concrete.profile_parameters.to_dict() == {
            "Ecm_factor": factor,
            "aggregate_type": aggregate_type,
            "reference_age_days": 28,
        }
        assert physical.E_secant == pytest.approx(reference_physical.E_secant * factor)
        assert physical.E_initial == pytest.approx(reference_e_initial * factor)
        assert physical.shear_modulus_secant_equivalent == pytest.approx(
            reference_physical.shear_modulus_secant_equivalent * factor
        )
        assert physical.f_ck == reference_physical.f_ck
        assert physical.f_cm == reference_physical.f_cm
        assert physical.f_ctm == reference_physical.f_ctm
        assert physical.fracture_energy is None
        assert physical.reference_age_days == 28.0
        assert f"aggregate_type={aggregate_type}" in physical.provenance["E_secant"].notes
        assert f"Ecm_factor={factor:g}" in physical.provenance["E_secant"].notes


@pytest.mark.parametrize(
    "parameters, error_type, match",
    [
        ({"aggregate_type": "Basalt"}, ValueError, "aggregate_type"),
        ({"aggregate_type": "granite"}, ValueError, "aggregate_type"),
        ({"Ecm_factor": 1.1}, ValueError, "Unsupported"),
        ({"alpha_E": 1.0}, ValueError, "Unsupported"),
        ({"kE": 9500}, ValueError, "Unsupported"),
        ({"reference_age_days": 56}, ValueError, "only the 28-day"),
        ({"reference_age_days": True}, TypeError, "must be numeric"),
    ],
)
def test_profile_configuration_rejects_unadjudicated_or_unsupported_context(
    parameters: dict[str, object], error_type: type[Exception], match: str
):
    with pytest.raises(error_type, match=match):
        Concrete.from_class(
            "C30/37",
            profile="ec2_2004",
            profile_parameters=ProfileConfiguration(parameters),  # type: ignore[arg-type]
        )


def test_fctm_c45_uses_lower_strength_branch():
    physical = _ec2("C45/55").physical
    assert physical.f_ctm == pytest.approx(0.30 * 45.0 ** (2.0 / 3.0))
    assert "f_ck <= 50 MPa" in physical.provenance["f_ctm"].notes


def test_fctm_c50_uses_lower_strength_branch():
    physical = _ec2("C50/60").physical
    assert physical.f_ctm == pytest.approx(0.30 * 50.0 ** (2.0 / 3.0))
    assert "f_ck <= 50 MPa" in physical.provenance["f_ctm"].notes


def test_fctm_c55_is_first_named_high_strength_branch():
    physical = _ec2("C55/67").physical
    assert physical.f_ctm == pytest.approx(2.12 * math.log(1.0 + 63.0 / 10.0))
    assert "f_ck > 50 MPa" in physical.provenance["f_ctm"].notes


def test_epsilon_cu1_c45_uses_constant_branch():
    physical = _ec2("C45/55").physical
    assert physical.strain_limit_compression == pytest.approx(3.5 / 1000.0)
    assert "f_ck < 50 MPa" in physical.provenance["strain_limit_compression"].notes


def test_epsilon_cu1_c50_uses_authority_adjudicated_high_strength_expression():
    physical = _ec2("C50/60").physical
    expected_per_mille = 2.8 + 27.0 * ((98.0 - 58.0) / 100.0) ** 4
    assert expected_per_mille == pytest.approx(3.4912)
    assert physical.strain_limit_compression == pytest.approx(expected_per_mille / 1000.0)
    assert "f_ck >= 50 MPa" in physical.provenance["strain_limit_compression"].notes


def test_epsilon_cu1_c55_stays_on_high_strength_expression():
    physical = _ec2("C55/67").physical
    expected_per_mille = 2.8 + 27.0 * ((98.0 - 63.0) / 100.0) ** 4
    assert physical.strain_limit_compression == pytest.approx(expected_per_mille / 1000.0)
    assert "f_ck >= 50 MPa" in physical.provenance["strain_limit_compression"].notes


def test_c50_equation_value_and_published_table_rounding_are_kept_distinct():
    expected = _class_records()["C50/60"]
    physical = _ec2("C50/60").physical

    assert expected["epsilon_cu1_equation_per_mille"] == pytest.approx(3.4912)
    published = expected["published_table"]
    assert isinstance(published, dict)
    assert published["epsilon_cu1_per_mille"] == 3.5
    assert physical.strain_limit_compression == pytest.approx(3.4912 / 1000.0)
    assert physical.strain_limit_compression != 3.5 / 1000.0


def test_epsilon_c1_cap_uses_ac2010_corrected_semantics_and_saturates():
    c70 = _ec2("C70/85").physical
    c80 = _ec2("C80/95").physical
    c90 = _ec2("C90/105").physical

    assert c70.strain_peak_compression < 2.8 / 1000.0
    assert c80.strain_peak_compression == pytest.approx(2.8 / 1000.0)
    assert c90.strain_peak_compression == pytest.approx(2.8 / 1000.0)
    assert "AC:2010" in c80.provenance["strain_peak_compression"].notes
    assert "<= 2.8" in c80.provenance["strain_peak_compression"].notes


def test_ec2_compression_landmarks_use_unit_normalization_only():
    physical = _ec2("C50/60").physical

    for field in ("strain_peak_compression", "strain_limit_compression"):
        provenance = physical.provenance[field]
        assert physical.resolution[field] is PropertyResolutionStatus.DIRECT
        assert provenance.source_kind is SourceKind.STANDARD
        assert provenance.source_id == "en_1992_1_1_2004"
        assert [normalization.kind for normalization in provenance.normalizations] == [
            NormalizationKind.UNIT_CONVERSION
        ]
        assert all(
            normalization.kind is not NormalizationKind.SIGN_CONVENTION
            for normalization in provenance.normalizations
        )


def test_e_initial_is_derived_tangent_approximation_and_not_ecm_alias():
    physical = _ec2("C30/37").physical
    provenance = physical.provenance["E_initial"]

    assert physical.E_initial == pytest.approx(1.05 * physical.E_secant)
    assert physical.E_initial != physical.E_secant
    assert physical.resolution["E_initial"] is PropertyResolutionStatus.DERIVED
    assert provenance.source_kind is SourceKind.DERIVED
    assert provenance.source_id == "en_1992_1_1_2004"
    assert provenance.equation_or_section == "§3.1.4(2)"
    assert provenance.derived_from == ("E_secant",)


def test_shear_is_repository_derived_from_secant_modulus_and_uncracked_poisson():
    physical = _ec2("C30/37").physical
    provenance = physical.provenance["shear_modulus_secant_equivalent"]

    assert physical.poisson_elastic == 0.20
    assert physical.shear_modulus_secant_equivalent == physical.E_secant / (
        2.0 * (1.0 + physical.poisson_elastic)
    )
    assert (
        physical.resolution["shear_modulus_secant_equivalent"] is PropertyResolutionStatus.DERIVED
    )
    assert provenance.source_kind is SourceKind.DERIVED
    assert provenance.derived_from == ("E_secant", "poisson_elastic")
    assert provenance.equation_or_section is None


def test_fracture_energy_is_truthfully_absent_and_requires_explicit_composition():
    physical = _ec2("C30/37").physical
    provenance = physical.provenance["fracture_energy"]

    assert physical.fracture_energy is None
    assert physical.resolution["fracture_energy"] is PropertyResolutionStatus.COMPOSED_REQUIRED
    assert provenance.source_id == "en_1992_1_1_2004"
    assert provenance.source_kind is SourceKind.STANDARD
    assert provenance.equation_or_section is None
    assert provenance.derived_from == ()
    assert "second authority" in provenance.notes


def test_direct_standard_fields_stay_direct_even_when_equations_have_dependencies():
    physical = _ec2("C30/37").physical
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

    for field in direct_fields:
        assert physical.resolution[field] is PropertyResolutionStatus.DIRECT
        assert physical.provenance[field].source_kind is SourceKind.STANDARD
        assert physical.provenance[field].source_id == "en_1992_1_1_2004"

    assert physical.provenance["f_cm"].derived_from == ("f_ck",)
    assert physical.provenance["f_ctm"].derived_from == ("f_ck", "f_cm")
    assert physical.provenance["f_ctk_lower"].derived_from == ("f_ctm",)
    assert physical.provenance["E_secant"].derived_from == ("f_cm",)


def test_provenance_locations_statistical_bases_and_edition_match_authority_contract():
    physical = _ec2("C30/37").physical

    expected = {
        "f_ck": (StatisticalBasis.CHARACTERISTIC, "§3.1.2(1)-(3); Table 3.1"),
        "f_cm": (StatisticalBasis.MEAN, "Table 3.1"),
        "f_ctm": (StatisticalBasis.MEAN, "Table 3.1"),
        "f_ctk_lower": (StatisticalBasis.CHARACTERISTIC, "Table 3.1"),
        "f_ctk_upper": (StatisticalBasis.CHARACTERISTIC, "Table 3.1"),
        "E_secant": (StatisticalBasis.MEAN, "§3.1.3(2); Table 3.1"),
        "poisson_elastic": (StatisticalBasis.NOT_APPLICABLE, "§3.1.3(4)"),
        "strain_peak_compression": (
            StatisticalBasis.NOT_APPLICABLE,
            "§3.1.5; Table 3.1; Eq. (3.14) context",
        ),
        "strain_limit_compression": (
            StatisticalBasis.NOT_APPLICABLE,
            "§3.1.5; Table 3.1; Eq. (3.14) context",
        ),
        "reference_age_days": (StatisticalBasis.NOT_APPLICABLE, "§3.1.2(2); Table 3.1"),
    }

    for field, (basis, location) in expected.items():
        provenance = physical.provenance[field]
        assert provenance.statistical_basis is basis
        assert provenance.equation_or_section == location
        assert provenance.edition == "2004 + AC:2010"

    assert "5% fractile" in physical.provenance["f_ctk_lower"].notes
    assert "95% fractile" in physical.provenance["f_ctk_upper"].notes


def test_serialization_is_deterministic_and_preserves_composed_required_absence():
    concrete = _ec2("C30/37")
    first = concrete.to_json()
    second = concrete.to_json()
    assert first == second

    payload = json.loads(first)
    assert payload["physical_profile"] == "ec2_2004"
    assert payload["physical"]["values"]["fracture_energy"] is None
    assert payload["physical"]["resolution"]["fracture_energy"] == "COMPOSED_REQUIRED"
    assert payload["physical"]["provenance"]["fracture_energy"]["equation_or_section"] is None
    assert payload["physical"]["values"]["f_ck"] == 30.0


def test_ec2_2004_activation_does_not_activate_ec2_2023():
    ec2 = Concrete.from_class("C30/37", profile="ec2_2004")
    assert ec2.physical_profile == "ec2_2004"

    with pytest.raises(NotImplementedError, match="later G1 slice"):
        Concrete.from_class("C30/37", profile="ec2_2023")


def test_fib_mc2010_remains_operational_after_ec2_2004_activation():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    assert fib.physical_profile == "fib_mc2010"
    assert fib.physical.f_ck == 30.0


def test_cross_profile_parser_protection_remains_strict():
    with pytest.raises(CrossProfileConcreteClassError):
        Concrete.from_class("C30", profile="ec2_2004")
    with pytest.raises(CrossProfileConcreteClassError):
        Concrete.from_class("C30/37", profile="fib_mc2010")
