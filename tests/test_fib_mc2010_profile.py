"""G1-B2 qualification for the verified fib Model Code 2010 physical profile."""

import json
from pathlib import Path

import pytest

from cdp_generator.concrete import (
    Concrete,
    ProfileConfiguration,
    PropertyResolutionStatus,
    SourceKind,
    StatisticalBasis,
    class_entries,
)
from cdp_generator.concrete.provenance import NormalizationKind
from cdp_generator.concrete.standards.fib_mc2010 import AGGREGATE_ALPHA_E

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FIXTURE = REPO_ROOT / "qualification" / "standards" / "fib_mc2010_reference.json"

# Production and fixture independently evaluate the same published equations in binary64.
# This tight tolerance allows only last-bit/libm variation; it does not absorb source rounding.
EQUATION_REL_TOL = 1e-13
EQUATION_ABS_TOL = 1e-12


def _fixture() -> dict[str, object]:
    return json.loads(REFERENCE_FIXTURE.read_text())


def _class_records() -> dict[str, dict[str, object]]:
    fixture = _fixture()
    classes = fixture["classes"]
    assert isinstance(classes, list)
    return {record["concrete_class"]: record for record in classes}


def _fib(concrete_class: str, aggregate_type: str = "quartzite") -> Concrete:
    parameters = None
    if aggregate_type != "quartzite":
        parameters = ProfileConfiguration({"aggregate_type": aggregate_type})
    return Concrete.from_class(
        concrete_class,
        profile="fib_mc2010",
        profile_parameters=parameters,
    )


def test_independent_reference_fixture_covers_exactly_all_17_fib_classes():
    fixture = _fixture()
    evidence = fixture["evidence"]
    assert isinstance(evidence, dict)
    assert fixture["schema_version"] == "fib_mc2010_reference.v1"
    assert fixture["profile"] == "fib_mc2010"
    assert fixture["reference_configuration"] == {
        "aggregate_type": "quartzite",
        "alpha_E": 1.0,
        "reference_age_days": 28,
    }
    assert evidence["structuralcodes_used"] is False
    assert "without importing cdp_generator production code" in str(evidence["method"])

    fixture_classes = tuple(_class_records())
    registry_classes = tuple(entry.canonical_class_string for entry in class_entries("fib_mc2010"))
    assert len(fixture_classes) == 17
    assert fixture_classes == registry_classes


@pytest.mark.parametrize(
    "concrete_class",
    [entry.canonical_class_string for entry in class_entries("fib_mc2010")],
)
def test_all_17_named_classes_match_independent_reference_fixture(concrete_class: str):
    expected = _class_records()[concrete_class]
    concrete = _fib(concrete_class)
    physical = concrete.physical

    assert concrete.physical_profile == "fib_mc2010"
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
    assert physical.E_initial == pytest.approx(
        expected["E_initial_mpa"], rel=EQUATION_REL_TOL, abs=EQUATION_ABS_TOL
    )
    assert physical.E_secant == pytest.approx(
        expected["E_secant_mpa"], rel=EQUATION_REL_TOL, abs=EQUATION_ABS_TOL
    )
    assert physical.poisson_elastic == expected["poisson_elastic"]
    assert physical.shear_modulus_secant_equivalent == pytest.approx(
        expected["shear_modulus_secant_equivalent_mpa"], rel=EQUATION_REL_TOL, abs=EQUATION_ABS_TOL
    )
    assert physical.fracture_energy == pytest.approx(
        expected["fracture_energy_n_per_mm"], rel=EQUATION_REL_TOL, abs=EQUATION_ABS_TOL
    )
    assert physical.strain_peak_compression == pytest.approx(
        expected["strain_peak_compression"], abs=1e-15
    )
    assert physical.strain_limit_compression == pytest.approx(
        expected["strain_limit_compression"], abs=1e-15
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
            "E_initial",
            "E_secant",
            "poisson_elastic",
            "fracture_energy",
            "strain_peak_compression",
            "strain_limit_compression",
            "reference_age_days",
        )
    )
    assert (
        physical.resolution["shear_modulus_secant_equivalent"] is PropertyResolutionStatus.DERIVED
    )


def test_default_quartzite_policy_is_explicit_and_serialized():
    concrete = Concrete.from_class("C30", profile="fib_mc2010")

    assert concrete.profile_parameters.to_dict() == {
        "aggregate_type": "quartzite",
        "alpha_E": 1.0,
        "reference_age_days": 28,
    }

    payload = concrete.to_dict()
    assert payload["schema_version"] == "concrete_material_definition.v2"
    assert payload["physical_profile"] == "fib_mc2010"
    assert payload["profile_parameters"] == {
        "aggregate_type": "quartzite",
        "alpha_E": 1.0,
        "reference_age_days": 28,
    }
    assert payload["physical"]["schema_version"] == "concrete_physical_properties.v2"
    assert payload["physical"]["values"]["reference_age_days"] == 28.0
    assert payload["physical"]["resolution"]["reference_age_days"] == "DIRECT"


def test_all_authorized_aggregate_families_scale_modulus_and_are_visible():
    reference = _fib("C30")
    reference_e = reference.physical.E_initial
    reference_ec = reference.physical.E_secant
    assert reference_e is not None

    assert dict(AGGREGATE_ALPHA_E) == {
        "basalt": 1.2,
        "quartzite": 1.0,
        "limestone": 0.9,
        "sandstone": 0.7,
    }

    for aggregate_type, alpha_e in AGGREGATE_ALPHA_E.items():
        concrete = _fib("C30", aggregate_type)
        assert concrete.profile_parameters.to_dict() == {
            "aggregate_type": aggregate_type,
            "alpha_E": alpha_e,
            "reference_age_days": 28,
        }
        assert concrete.physical.E_initial == pytest.approx(reference_e * alpha_e)
        assert concrete.physical.E_secant == pytest.approx(reference_ec * alpha_e)
        assert f"aggregate_type={aggregate_type}" in concrete.physical.provenance["E_initial"].notes
        assert f"alpha_E={alpha_e:g}" in concrete.physical.provenance["E_initial"].notes


@pytest.mark.parametrize(
    "parameters, error_type, match",
    [
        ({"aggregate_type": "Basalt"}, ValueError, "aggregate_type"),
        ({"aggregate_type": "granite"}, ValueError, "aggregate_type"),
        ({"alpha_E": 1.1}, ValueError, "Unsupported"),
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
            "C30",
            profile="fib_mc2010",
            profile_parameters=ProfileConfiguration(parameters),  # type: ignore[arg-type]
        )


def test_fctm_branch_boundary_is_exactly_c50_c55():
    c50 = _fib("C50").physical
    c55 = _fib("C55").physical
    expected = _class_records()

    assert c50.f_ctm == pytest.approx(expected["C50"]["f_ctm_mpa"])
    assert c55.f_ctm == pytest.approx(expected["C55"]["f_ctm_mpa"])
    assert c50.provenance["f_ctm"].equation_or_section == "§5.1.5.1, Eq. (5.1-3a)"
    assert c55.provenance["f_ctm"].equation_or_section == "§5.1.5.1, Eq. (5.1-3b)"


def test_secant_reduction_saturates_at_c80_and_not_before():
    c70 = _fib("C70").physical
    c80 = _fib("C80").physical
    c120 = _fib("C120").physical

    assert c70.E_initial is not None
    assert c70.E_secant < c70.E_initial
    assert c80.E_secant == pytest.approx(c80.E_initial, rel=0.0, abs=1e-12)
    assert c120.E_secant == pytest.approx(c120.E_initial, rel=0.0, abs=1e-12)
    assert "reduced/secant E_c" in c80.provenance["E_secant"].notes
    assert "E_c1" in c80.provenance["E_secant"].notes


def test_fracture_energy_remains_direct_after_explicit_n_per_m_to_n_per_mm_normalization():
    expected = _class_records()["C30"]
    physical = _fib("C30").physical
    provenance = physical.provenance["fracture_energy"]

    assert physical.fracture_energy is not None
    assert physical.fracture_energy * 1000.0 == pytest.approx(
        expected["fracture_energy_source_n_per_m"], rel=EQUATION_REL_TOL
    )
    assert physical.resolution["fracture_energy"] is PropertyResolutionStatus.DIRECT
    assert provenance.source_kind is SourceKind.STANDARD
    assert provenance.source_id == "fib_mc2010_2013"
    assert provenance.equation_or_section == "§5.1.5.2, Eq. (5.1-9)"
    assert provenance.normalizations == (provenance.normalizations[0],)
    assert provenance.normalizations[0].kind is NormalizationKind.UNIT_CONVERSION
    assert "N/m" in provenance.normalizations[0].source_convention
    assert "N/mm" in provenance.normalizations[0].repository_convention


def test_compression_landmarks_are_direct_but_trace_both_sign_and_unit_normalization():
    expected = _class_records()["C90"]
    physical = _fib("C90").physical

    for field, source_key in (
        ("strain_peak_compression", "strain_peak_compression_source_per_mille"),
        ("strain_limit_compression", "strain_limit_compression_source_per_mille"),
    ):
        provenance = physical.provenance[field]
        assert physical.resolution[field] is PropertyResolutionStatus.DIRECT
        assert provenance.source_kind is SourceKind.STANDARD
        assert provenance.source_id == "fib_mc2010_2013"
        assert [normalization.kind for normalization in provenance.normalizations] == [
            NormalizationKind.SIGN_CONVENTION,
            NormalizationKind.UNIT_CONVERSION,
        ]
        assert expected[source_key] < 0
        value = getattr(physical, field)
        assert value == pytest.approx(abs(expected[source_key]) / 1000.0, abs=1e-15)


def test_direct_standard_fields_stay_direct_even_when_their_equations_have_dependencies():
    physical = _fib("C30").physical
    direct_fields = (
        "f_ck",
        "f_cm",
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

    for field in direct_fields:
        assert physical.resolution[field] is PropertyResolutionStatus.DIRECT
        assert physical.provenance[field].source_kind is SourceKind.STANDARD
        assert physical.provenance[field].source_id == "fib_mc2010_2013"

    assert physical.provenance["f_cm"].derived_from == ("f_ck",)
    assert physical.provenance["E_secant"].derived_from == ("E_initial", "f_cm")
    assert physical.provenance["fracture_energy"].derived_from == ("f_cm",)


def test_shear_is_the_only_repository_derived_mc2010_physical_field():
    physical = _fib("C30").physical
    provenance = physical.provenance["shear_modulus_secant_equivalent"]

    assert physical.shear_modulus_secant_equivalent == physical.E_secant / (
        2.0 * (1.0 + physical.poisson_elastic)
    )
    assert physical.shear_modulus == physical.shear_modulus_secant_equivalent
    assert (
        physical.resolution["shear_modulus_secant_equivalent"] is PropertyResolutionStatus.DERIVED
    )
    assert provenance.source_kind is SourceKind.DERIVED
    assert provenance.derived_from == ("E_secant", "poisson_elastic")
    assert provenance.equation_or_section is None
    assert "not claimed as a directly specified MC2010 quantity" in provenance.notes


def test_provenance_locations_and_statistical_bases_match_frozen_authority_contract():
    physical = _fib("C30").physical

    expected = {
        "f_ck": (StatisticalBasis.CHARACTERISTIC, "§5.1.2; Table 5.1-3"),
        "f_cm": (StatisticalBasis.MEAN, "§5.1.4, Eq. (5.1-1)"),
        "f_ctm": (StatisticalBasis.MEAN, "§5.1.5.1, Eq. (5.1-3a)"),
        "f_ctk_lower": (StatisticalBasis.CHARACTERISTIC, "§5.1.5.1, Eq. (5.1-4)"),
        "f_ctk_upper": (StatisticalBasis.CHARACTERISTIC, "§5.1.5.1, Eq. (5.1-5)"),
        "E_initial": (StatisticalBasis.MEAN, "§5.1.7.2, Eq. (5.1-21); Table 5.1-6"),
        "E_secant": (
            StatisticalBasis.MEAN,
            "§5.1.7.2, Eqs. (5.1-23), (5.1-24); Table 5.1-7",
        ),
        "poisson_elastic": (StatisticalBasis.NOT_APPLICABLE, "§5.1.7.3"),
        "fracture_energy": (StatisticalBasis.NOT_APPLICABLE, "§5.1.5.2, Eq. (5.1-9)"),
        "strain_peak_compression": (
            StatisticalBasis.NOT_APPLICABLE,
            "§5.1.8.1, Eq. (5.1-26); Table 5.1-8",
        ),
        "strain_limit_compression": (
            StatisticalBasis.NOT_APPLICABLE,
            "§5.1.8.1, Eq. (5.1-26); Table 5.1-8",
        ),
        "reference_age_days": (StatisticalBasis.NOT_APPLICABLE, "§5.1.2; §5.1.7.2"),
    }
    for field, (basis, location) in expected.items():
        provenance = physical.provenance[field]
        assert provenance.statistical_basis is basis
        assert provenance.equation_or_section == location
        assert provenance.edition == "2013"
        assert provenance.units

    assert "0.14-0.26" in physical.provenance["poisson_elastic"].notes
    assert "-0.6 f_ck < sigma_c < 0.8 f_ctk" in physical.provenance["poisson_elastic"].notes


def test_mc2010_tensile_names_keep_min_max_semantics_not_generic_percentile_aliases():
    physical = _fib("C30").physical
    lower = physical.provenance["f_ctk_lower"]
    upper = physical.provenance["f_ctk_upper"]

    assert "f_ctk,min" in lower.notes
    assert "f_ctk,max" in upper.notes
    assert "percentile alias" in lower.notes
    assert "percentile alias" in upper.notes


def test_mc2010_json_is_deterministic_v2_and_contains_machine_readable_provenance():
    concrete = Concrete.from_class(
        "C30",
        profile="fib_mc2010",
        profile_parameters=ProfileConfiguration({"aggregate_type": "basalt"}),
    )
    first = concrete.to_json()
    second = concrete.to_json()
    assert first == second

    payload = json.loads(first)
    assert payload["schema_version"] == "concrete_material_definition.v2"
    assert payload["profile_parameters"] == {
        "aggregate_type": "basalt",
        "alpha_E": 1.2,
        "reference_age_days": 28,
    }
    physical = payload["physical"]
    assert physical["schema_version"] == "concrete_physical_properties.v2"
    assert physical["resolution"]["fracture_energy"] == "DIRECT"
    assert physical["resolution"]["shear_modulus_secant_equivalent"] == "DERIVED"
    assert physical["provenance"]["shear_modulus_secant_equivalent"]["derived_from"] == [
        "E_secant",
        "poisson_elastic",
    ]
    assert physical["provenance"]["fracture_energy"]["normalizations"][0]["kind"] == (
        "UNIT_CONVERSION"
    )


def test_ec2_2023_physical_construction_remains_non_operational_and_fib_mean_strength_path_is_rejected():
    with pytest.raises(NotImplementedError, match="later G1 slice"):
        Concrete.from_class("C30/37", profile="ec2_2023")
    with pytest.raises(NotImplementedError, match="class-based"):
        Concrete.from_mean_strength(38.0, 0.0022, 0.0035, profile="fib_mc2010")


def test_structuralcodes_is_not_a_runtime_dependency():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert "structuralcodes" not in pyproject.lower()


def test_runtime_resolution_and_source_kind_match_frozen_fib_authority_matrix():
    matrix = json.loads(
        (REPO_ROOT / "qualification" / "standards" / "authority_matrix.json").read_text()
    )
    entries = [
        entry for entry in matrix["field_authority_matrix"] if entry["profile"] == "fib_mc2010"
    ]
    assert len(entries) == 12

    physical = _fib("C30").physical
    field_alias = {"shear_modulus": "shear_modulus_secant_equivalent"}
    for entry in entries:
        field = field_alias.get(entry["physical_field"], entry["physical_field"])
        expected_status = PropertyResolutionStatus(entry["authority_status"].upper())
        assert physical.resolution[field] is expected_status
        if expected_status is PropertyResolutionStatus.DERIVED:
            assert physical.provenance[field].source_kind is SourceKind.DERIVED
        else:
            assert physical.provenance[field].source_kind is SourceKind.STANDARD
            assert physical.provenance[field].source_id == "fib_mc2010_2013"
