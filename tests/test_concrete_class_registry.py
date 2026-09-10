"""G1-B1 qualification for the explicit production class registry/parser."""

import json
from pathlib import Path

import pytest

from cdp_generator.concrete import (
    CrossProfileConcreteClassError,
    MalformedConcreteClassError,
    UnknownPhysicalProfileError,
    UnsupportedConcreteClassError,
    class_entries,
    parse_concrete_class,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_MATRIX = REPO_ROOT / "qualification" / "standards" / "authority_matrix.json"


def test_production_registry_exactly_mirrors_all_46_authority_matrix_entries():
    matrix = json.loads(AUTHORITY_MATRIX.read_text())
    expected = [
        {
            "profile": entry["profile"],
            "canonical_class_string": entry["canonical_class_string"],
            "f_ck_cylinder_mpa": float(entry["f_ck_cylinder_mpa"]),
            "f_ck_cube_mpa": float(entry["f_ck_cube_mpa"]),
            "reference_age_semantics": entry["reference_age_semantics"],
            "profile_range_endpoint": entry["profile_range_endpoint"],
        }
        for entry in matrix["class_authority_matrix"]
    ]
    actual = [
        {
            "profile": entry.profile,
            "canonical_class_string": entry.canonical_class_string,
            "f_ck_cylinder_mpa": entry.f_ck_cylinder_mpa,
            "f_ck_cube_mpa": entry.f_ck_cube_mpa,
            "reference_age_semantics": entry.reference_age_semantics,
            "profile_range_endpoint": entry.profile_range_endpoint,
        }
        for entry in class_entries()
    ]

    assert len(actual) == 46
    assert actual == expected


def test_parser_accepts_every_registry_entry_exactly():
    for entry in class_entries():
        assert parse_concrete_class(entry.profile, entry.canonical_class_string) == entry


def test_parser_accepts_only_canonical_profile_classes():
    assert parse_concrete_class("fib_mc2010", "C30").f_ck_cube_mpa == 37.0
    assert parse_concrete_class("ec2_2004", "C30/37").f_ck_cylinder_mpa == 30.0
    assert parse_concrete_class("ec2_2023", "C100/115").f_ck_cube_mpa == 115.0


@pytest.mark.parametrize("bad", ["30", "30/37", "C30-37", "C30 / 37", " C30/37"])
def test_parser_rejects_malformed_inputs(bad):
    with pytest.raises(MalformedConcreteClassError):
        parse_concrete_class("ec2_2023", bad)


def test_parser_distinguishes_well_formed_but_unsupported_class():
    with pytest.raises(UnsupportedConcreteClassError):
        parse_concrete_class("ec2_2023", "C13/16")
    with pytest.raises(UnsupportedConcreteClassError):
        parse_concrete_class("fib_mc2010", "C13")


def test_parser_distinguishes_cross_profile_class_identity():
    with pytest.raises(CrossProfileConcreteClassError):
        parse_concrete_class("ec2_2023", "C30")
    with pytest.raises(CrossProfileConcreteClassError):
        parse_concrete_class("fib_mc2010", "C30/37")


def test_parser_distinguishes_unknown_profile():
    with pytest.raises(UnknownPhysicalProfileError):
        parse_concrete_class("not_a_profile", "C30/37")
