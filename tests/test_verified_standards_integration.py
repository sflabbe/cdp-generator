"""G1-Q final integration qualification for verified concrete standards."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from cdp_generator.concrete import (
    Concrete,
    CrossProfileConcreteClassError,
    MalformedConcreteClassError,
    NormalizationKind,
    ProfileConfiguration,
    PropertyResolutionStatus,
    SourceKind,
    UnsupportedConcreteClassError,
    class_entries,
    parse_concrete_class,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "qualification" / "standards" / "g1_integration_manifest.json"

LEGACY_ARGS = {
    "f_cm": 38.0,
    "e_c1": 0.0022,
    "e_clim": 0.0035,
    "profile": "legacy_v1",
}
CLASS_EXAMPLES = {
    "fib_mc2010": "C30",
    "ec2_2004": "C30/37",
    "ec2_2023": "C30/37",
}
SOURCE_IDENTITIES = {
    "fib_mc2010": ("fib_mc2010_2013", "2013"),
    "ec2_2004": ("en_1992_1_1_2004", "2004 + AC:2010"),
    "ec2_2023": ("en_1992_1_1_2023", "2023"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_concrete(
    profile: str,
    concrete_class: str | None = None,
    parameters: dict[str, str | int | float | bool | None] | None = None,
) -> Concrete:
    return Concrete.from_class(
        CLASS_EXAMPLES[profile] if concrete_class is None else concrete_class,
        profile=profile,
        profile_parameters=ProfileConfiguration(parameters) if parameters is not None else None,
    )


def test_integration_manifest_freezes_closed_artifacts_and_program_contract():
    manifest = json.loads(MANIFEST_PATH.read_text())

    assert manifest["schema_version"] == "g1_integration_manifest.v1"
    assert manifest["gate"] == "G1-Q"
    assert manifest["starting_authority"] == {
        "head": "844aac57735e8a82c39c04f13ca2897a7d7afe1f",
        "tree": "4738bd5461cee72a69337707af0dbd6ef29df239",
    }

    for relative_path, expected_sha in manifest["frozen_artifacts"].items():
        assert _sha256(REPO_ROOT / relative_path) == expected_sha

    assert manifest["class_counts"] == {
        "fib_mc2010": 17,
        "ec2_2004": 14,
        "ec2_2023": 15,
        "total": 46,
    }
    assert manifest["reserved_constitutive_backend"] == "cdpm2_grassl_2013"
    assert manifest["master_invariant"] == [
        "LEGACY_COMPATIBILITY",
        "NORMATIVE_AUTHORITY",
        "CONSTITUTIVE_MODEL_CALIBRATION",
    ]


def test_public_resolution_and_construction_domains_remain_separate():
    legacy = Concrete.from_mean_strength(**LEGACY_ARGS)
    assert legacy.physical_profile == "legacy_v1"

    for profile, concrete_class in CLASS_EXAMPLES.items():
        concrete = Concrete.from_class(concrete_class, profile=profile)
        assert concrete.physical_profile == profile
        with pytest.raises(NotImplementedError, match="class-based"):
            Concrete.from_mean_strength(38.0, 0.0022, 0.0035, profile=profile)

    with pytest.raises(NotImplementedError, match="G2"):
        legacy.to_cdpm2(calibration="cdpm2_grassl_2013")


def test_all_46_registry_entries_construct_serialize_and_preserve_resolution_consistency():
    entries = class_entries()
    counts = Counter(entry.profile for entry in entries)
    assert counts == {"fib_mc2010": 17, "ec2_2004": 14, "ec2_2023": 15}
    assert len(entries) == 46

    for entry in entries:
        assert parse_concrete_class(entry.profile, entry.canonical_class_string) == entry
        concrete = Concrete.from_class(entry.canonical_class_string, profile=entry.profile)
        physical = concrete.physical

        assert concrete.physical_profile == entry.profile
        assert physical.f_ck == entry.f_ck_cylinder_mpa
        assert concrete.to_dict()["schema_version"] == "concrete_material_definition.v2"
        assert concrete.to_json() == concrete.to_json()

        values = physical.values_dict()
        assert set(physical.resolution) == set(values)
        assert set(physical.provenance) == set(values)
        for field, status in physical.resolution.items():
            if status in {
                PropertyResolutionStatus.DIRECT,
                PropertyResolutionStatus.DERIVED,
            }:
                assert values[field] is not None
            else:
                assert status in {
                    PropertyResolutionStatus.UNRESOLVED,
                    PropertyResolutionStatus.COMPOSED_REQUIRED,
                }
                assert values[field] is None


def test_parser_isolation_and_ec2_upper_range_boundary():
    cross_profile = (
        ("ec2_2004", "C30"),
        ("ec2_2023", "C30"),
        ("fib_mc2010", "C30/37"),
        ("ec2_2004", "C100/115"),
    )
    for profile, concrete_class in cross_profile:
        with pytest.raises(CrossProfileConcreteClassError):
            parse_concrete_class(profile, concrete_class)

    for malformed in ("30", "30/37", "C30-37", "C30 / 37"):
        with pytest.raises(MalformedConcreteClassError):
            parse_concrete_class("ec2_2023", malformed)

    assert parse_concrete_class("ec2_2023", "C100/115").f_ck_cylinder_mpa == 100.0
    with pytest.raises(UnsupportedConcreteClassError):
        parse_concrete_class("ec2_2004", "C95/110")


def test_ec2_class_string_collision_is_resolved_by_profile_identity():
    ec04 = _class_concrete("ec2_2004")
    ec23 = _class_concrete("ec2_2023")

    assert ec04.physical_profile != ec23.physical_profile
    assert ec04.profile_parameters.to_dict() == {
        "Ecm_factor": 1.0,
        "aggregate_type": "quartzite",
        "reference_age_days": 28,
    }
    assert ec23.profile_parameters.to_dict() == {
        "k_E": 9500.0,
        "k_E_basis": "quartzite_assumption",
        "reference_age_days": 28.0,
    }
    assert ec04.physical.provenance["f_ck"].source_id == "en_1992_1_1_2004"
    assert ec23.physical.provenance["f_ck"].source_id == "en_1992_1_1_2023"
    assert ec04.physical.provenance["f_ck"].edition == "2004 + AC:2010"
    assert ec23.physical.provenance["f_ck"].edition == "2023"


def test_source_identity_matrix_keeps_legacy_standard_and_derived_authorities_distinct():
    legacy = Concrete.from_mean_strength(**LEGACY_ARGS)
    assert legacy.physical.provenance["f_ck"].source_id == "legacy_v1"
    assert legacy.physical.provenance["f_ck"].source_kind is SourceKind.LEGACY_IMPLEMENTATION

    for profile, concrete_class in CLASS_EXAMPLES.items():
        concrete = Concrete.from_class(concrete_class, profile=profile)
        source_id, edition = SOURCE_IDENTITIES[profile]
        provenance = concrete.physical.provenance["f_ck"]
        assert provenance.source_id == source_id
        assert provenance.edition == edition
        assert provenance.source_kind is SourceKind.STANDARD

        shear = concrete.physical.provenance["shear_modulus_secant_equivalent"]
        assert shear.source_kind is SourceKind.DERIVED


def test_serialization_generation_policy_and_v1_boundary():
    legacy = Concrete.from_mean_strength(**LEGACY_ARGS)
    assert legacy.to_dict()["schema_version"] == "concrete_material_definition.v1"
    assert (
        legacy.to_dict(schema_version="v2")["schema_version"] == "concrete_material_definition.v2"
    )

    for profile, concrete_class in CLASS_EXAMPLES.items():
        concrete = Concrete.from_class(concrete_class, profile=profile)
        assert concrete.to_dict()["schema_version"] == "concrete_material_definition.v2"
        with pytest.raises(ValueError, match="reserved for legacy_v1"):
            concrete.to_dict(schema_version="v1")


def test_representative_serializations_are_deterministic_and_preserve_v2_metadata():
    representatives = [
        Concrete.from_mean_strength(**LEGACY_ARGS),
        _class_concrete("fib_mc2010"),
        _class_concrete("ec2_2004"),
        _class_concrete("ec2_2023"),
        _class_concrete("ec2_2023", parameters={"reference_age_days": 56}),
        _class_concrete("ec2_2023", parameters={"k_E": 13000}),
    ]

    for concrete in representatives:
        first = concrete.to_json()
        assert first == concrete.to_json() == concrete.to_json()
        payload = json.loads(first)
        assert payload["physical_profile"] == concrete.physical_profile
        if concrete.physical_profile != "legacy_v1":
            assert payload["profile_parameters"] == concrete.profile_parameters.to_dict()
            assert payload["physical"]["resolution"] == concrete.physical.resolution_dict()
            assert payload["physical"]["provenance"] == concrete.physical.provenance_dict()


def test_configuration_vocabulary_does_not_leak_across_profiles():
    rejected = {
        "fib_mc2010": ({"Ecm_factor": 1.0}, {"k_E": 9500}),
        "ec2_2004": ({"alpha_E": 1.0}, {"k_E": 9500}),
        "ec2_2023": (
            {"aggregate_type": "quartzite"},
            {"alpha_E": 1.0},
            {"Ecm_factor": 1.0},
        ),
    }

    for profile, parameter_sets in rejected.items():
        for parameters in parameter_sets:
            with pytest.raises(ValueError, match="Unsupported"):
                _class_concrete(profile, parameters=parameters)

    assert set(_class_concrete("fib_mc2010").profile_parameters.to_dict()) == {
        "aggregate_type",
        "alpha_E",
        "reference_age_days",
    }
    assert set(_class_concrete("ec2_2004").profile_parameters.to_dict()) == {
        "aggregate_type",
        "Ecm_factor",
        "reference_age_days",
    }
    assert set(_class_concrete("ec2_2023").profile_parameters.to_dict()) == {
        "k_E",
        "k_E_basis",
        "reference_age_days",
    }


def test_reference_age_and_e_initial_policy_matrix():
    for profile in ("fib_mc2010", "ec2_2004"):
        concrete = _class_concrete(profile)
        assert concrete.physical.reference_age_days == 28
        with pytest.raises(ValueError, match="28-day"):
            _class_concrete(profile, parameters={"reference_age_days": 56})

    ec23_28 = _class_concrete("ec2_2023", parameters={"reference_age_days": 28})
    assert ec23_28.physical.E_initial == pytest.approx(1.05 * ec23_28.physical.E_secant)
    assert ec23_28.physical.resolution["E_initial"] is PropertyResolutionStatus.DERIVED

    for age in (56, 91):
        concrete = _class_concrete("ec2_2023", parameters={"reference_age_days": age})
        assert concrete.physical.reference_age_days == age
        assert concrete.physical.E_secant is not None
        assert concrete.physical.E_initial is None
        assert concrete.physical.resolution["E_initial"] is PropertyResolutionStatus.UNRESOLVED
        assert concrete.physical.provenance["E_initial"].equation_or_section is None


def test_e_initial_authority_and_derived_from_matrix():
    fib = _class_concrete("fib_mc2010")
    ec04 = _class_concrete("ec2_2004")
    ec23 = _class_concrete("ec2_2023")

    assert fib.physical.resolution["E_initial"] is PropertyResolutionStatus.DIRECT
    assert fib.physical.provenance["E_initial"].source_kind is SourceKind.STANDARD

    assert ec04.physical.resolution["E_initial"] is PropertyResolutionStatus.DERIVED
    assert ec04.physical.provenance["E_initial"].source_kind is SourceKind.DERIVED
    assert ec04.physical.provenance["E_initial"].derived_from == ("E_secant",)

    assert ec23.physical.resolution["E_initial"] is PropertyResolutionStatus.DERIVED
    assert ec23.physical.provenance["E_initial"].source_kind is SourceKind.DERIVED
    assert ec23.physical.provenance["E_initial"].derived_from == (
        "E_secant",
        "reference_age_days",
    )

    for concrete in (fib, ec04, ec23):
        assert concrete.physical.provenance["shear_modulus_secant_equivalent"].derived_from == (
            "E_secant",
            "poisson_elastic",
        )


def test_fracture_energy_policy_matrix_prevents_implicit_cross_standard_composition():
    fib = _class_concrete("fib_mc2010")
    assert fib.physical.fracture_energy is not None
    assert fib.physical.resolution["fracture_energy"] is PropertyResolutionStatus.DIRECT
    assert fib.physical.provenance["fracture_energy"].source_kind is SourceKind.STANDARD

    for profile in ("ec2_2004", "ec2_2023"):
        concrete = _class_concrete(profile)
        assert concrete.physical.fracture_energy is None
        assert (
            concrete.physical.resolution["fracture_energy"]
            is PropertyResolutionStatus.COMPOSED_REQUIRED
        )
        payload = concrete.to_dict()
        assert payload["physical"]["values"]["fracture_energy"] is None
        assert payload["physical"]["resolution"]["fracture_energy"] == "COMPOSED_REQUIRED"


def test_compression_normalization_policy_is_profile_specific():
    expected = {
        "fib_mc2010": {
            NormalizationKind.SIGN_CONVENTION,
            NormalizationKind.UNIT_CONVERSION,
        },
        "ec2_2004": {NormalizationKind.UNIT_CONVERSION},
        "ec2_2023": {NormalizationKind.UNIT_CONVERSION},
    }

    for profile, kinds in expected.items():
        concrete = _class_concrete(profile)
        for field in ("strain_peak_compression", "strain_limit_compression"):
            actual = {
                normalization.kind
                for normalization in concrete.physical.provenance[field].normalizations
            }
            assert actual == kinds


def test_ec2_edition_difference_sentinels_preserve_independent_normative_laws():
    for concrete_class in ("C30/37", "C50/60", "C90/105"):
        assert _class_concrete("ec2_2004", concrete_class).physical_profile == "ec2_2004"
        assert _class_concrete("ec2_2023", concrete_class).physical_profile == "ec2_2023"

    ec04_55 = _class_concrete("ec2_2004", "C55/67")
    ec23_55 = _class_concrete("ec2_2023", "C55/67")
    expected_2004 = 2.12 * math.log(1.0 + ec04_55.physical.f_cm / 10.0)
    expected_2023 = 1.10 * 55.0 ** (1.0 / 3.0)
    assert ec04_55.physical.f_ctm == pytest.approx(expected_2004)
    assert ec23_55.physical.f_ctm == pytest.approx(expected_2023)
    assert ec04_55.physical.f_ctm != pytest.approx(ec23_55.physical.f_ctm)

    assert _class_concrete("ec2_2023", "C100/115").physical.f_cm == 108.0
    with pytest.raises(CrossProfileConcreteClassError):
        Concrete.from_class("C100/115", profile="ec2_2004")


def test_ec2_2023_default_and_explicit_k_e_keep_assumption_provenance_distinct():
    default = _class_concrete("ec2_2023")
    explicit = _class_concrete("ec2_2023", parameters={"k_E": 9500})

    assert default.physical.E_secant == explicit.physical.E_secant
    assert default.profile_parameters.parameters["k_E_basis"] == "quartzite_assumption"
    assert explicit.profile_parameters.parameters["k_E_basis"] == "explicit_standard_parameter"

    default_provenance = default.physical.provenance["E_secant"]
    explicit_provenance = explicit.physical.provenance["E_secant"]
    assert default_provenance.source_kind is SourceKind.STANDARD
    assert explicit_provenance.source_kind is SourceKind.STANDARD
    assert default_provenance.overridden is False
    assert explicit_provenance.overridden is True


def test_abaqus_boundary_keeps_constitutive_calibration_authority_separate():
    for profile in CLASS_EXAMPLES:
        concrete = _class_concrete(profile)
        abaqus = concrete.to_abaqus_cdp(calibration="abaqus_cdp_legacy")

        assert abaqus.to_dict()["calibration"] == "abaqus_cdp_legacy"
        for provenance in abaqus.provenance.values():
            assert provenance.source_id == "legacy_v1"
            assert provenance.source_kind is SourceKind.LEGACY_IMPLEMENTATION

        payload = concrete.to_dict(abaqus_cdp=abaqus)
        physical_source = concrete.physical.provenance["f_ck"].source_id
        assert payload["physical"]["provenance"]["f_ck"]["source_id"] == physical_source
        constitutive = payload["constitutive_models"]["abaqus_cdp"]
        assert constitutive["calibration"] == "abaqus_cdp_legacy"
        assert {
            field_provenance["source_id"]
            for field_provenance in constitutive["provenance"].values()
        } == {"legacy_v1"}

        assert concrete.to_json(abaqus_cdp=abaqus) == concrete.to_json(abaqus_cdp=abaqus)


def test_cdpm2_reserved_seam_remains_closed_for_all_verified_profiles():
    for profile in CLASS_EXAMPLES:
        concrete = _class_concrete(profile)
        with pytest.raises(NotImplementedError, match="reserved for G2"):
            concrete.to_cdpm2(calibration="cdpm2_grassl_2013")
