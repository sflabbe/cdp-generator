"""G2-A qualification for the CDPM2 authority and parameter contracts."""

import hashlib
import json
from pathlib import Path

import pytest

from cdp_generator.concrete import Concrete

REPO_ROOT = Path(__file__).resolve().parents[1]
QROOT = REPO_ROOT / "qualification" / "cdpm2"

FROZEN_G1 = {
    "qualification/legacy_v1_baseline.json": "e3f4529072aab2e57d2b482135297a19c0333a2a0a48b94250bf59bf6c230c32",
    "qualification/standards/authority_matrix.json": "f334f77eae280252fd604b5a6b8ab90383092e08d94cc094a8fe1c76aa0b890f",
    "qualification/standards/fib_mc2010_reference.json": "fefddf141960fee8d0a79eb2089241260b50172217ab417279340c4a0dca8a28",
    "qualification/standards/ec2_2004_reference.json": "ce98440a428858f4d192afe6a036fb1db7b53662acbd74746038a0596c9e2127",
    "qualification/standards/ec2_2023_reference.json": "b9053278e15ad39372e90cfdb14bc8ac4a4fc600a96a7c6bdf53c83413d02ee0",
    "qualification/standards/g1_integration_manifest.json": "428fbdbb382e20e165c40138ba563c4116d70cb59de85fa7775ace75d9379726",
    "docs/standards_sources.md": "a45fd3ae2e169b716fbb574a9d742192a054c38bbd111fb2cdf278fbb3e1bf4a",
}


def _load(name: str) -> dict:
    return json.loads((QROOT / name).read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_g2a_artifacts_are_valid_and_g1_is_frozen():
    authority = _load("authority_matrix.json")
    semantic = _load("semantic_parameter_contract.json")
    backend = _load("backend_compatibility_contract.json")

    assert authority["schema_version"] == "cdpm2_authority_matrix.v1"
    assert semantic["schema_version"] == "cdpm2_semantic_parameter_contract.v1"
    assert backend["schema_version"] == "cdpm2_backend_compatibility_contract.v1"
    assert authority["starting_authority"] == {
        "head": "ea6af160cb3765214c948f7ea1371d20983f524c",
        "tree": "9ff1597cd99eb2e09e52f94deb36a39f825359ea",
    }
    for relative, expected in FROZEN_G1.items():
        assert _sha256(REPO_ROOT / relative) == expected


def test_source_ids_are_unique_and_all_parameter_authorities_exist():
    authority = _load("authority_matrix.json")
    source_ids = set(authority["sources"])
    names = [row["semantic_name"] for row in authority["parameters"]]
    assert len(names) == len(set(names))
    for row in authority["parameters"]:
        assert row["authority_source_id"] in source_ids
        assert row["classification"]
        assert row["units"]
        if row["classification"] == "CONSTITUTIVE_DERIVED":
            assert row["derived_from"]


def test_semantic_contract_is_not_a_raw_24_slot_copy():
    semantic = _load("semantic_parameter_contract.json")
    field_names = [field["name"] for field in semantic["fields"]]
    assert len(field_names) == len(set(field_names))
    assert "LCHAR" not in field_names
    assert "B_s" not in field_names
    assert "ERATETYPE" not in field_names
    assert "SRATETYPE" not in field_names
    assert "FAILFLG" not in field_names
    assert "PRINTFLAG" not in field_names
    assert "runtime_state" not in field_names
    assert semantic["model_id"] == "cdpm2_grassl_2013"
    assert semantic["calibration_id"] == "grassl_2013_static_default_v1"


def test_exact_bilinear_energy_contract_uses_area_relation_not_rounded_reciprocal():
    semantic = _load("semantic_parameter_contract.json")
    derived = semantic["derived_relations"]
    assert derived["baseline_ratios"] == {"f_t1_over_f_t": 0.3, "w_f1_over_w_f": 0.15}
    assert derived["exact_area_coefficient"] == 0.225
    assert derived["w_f"] == "G_Ft/(0.225*f_t)"
    assert derived["w_f1"] == "0.15*w_f"
    assert derived["f_t1"] == "0.3*f_t"


def test_g1_mapping_uses_initial_modulus_mean_strengths_and_truthful_absence_states():
    authority = _load("authority_matrix.json")
    mappings = authority["g1_to_cdpm2_mapping"]

    assert mappings["fib_mc2010"]["mappings"]["E"]["g1_field"] == "E_initial"
    assert mappings["fib_mc2010"]["mappings"]["f_t"]["g1_field"] == "f_ctm"
    assert mappings["fib_mc2010"]["mappings"]["f_c"]["g1_field"] == "f_cm"
    assert mappings["fib_mc2010"]["mappings"]["G_Ft"]["g1_field"] == "fracture_energy"
    assert mappings["fib_mc2010"]["automatic_static_cdpm2_readiness"] == "READY"

    assert mappings["ec2_2004"]["mappings"]["G_Ft"]["state"] == "COMPOSED_REQUIRED"
    assert mappings["ec2_2023_at_28_days"]["mappings"]["G_Ft"]["state"] == "COMPOSED_REQUIRED"
    assert mappings["ec2_2023_above_28_days"]["mappings"]["E"] == {
        "g1_field": "E_initial",
        "reason": "G1 E_initial is intentionally unresolved above 28 days; E_secant fallback is forbidden",
        "state": "UNRESOLVED",
    }
    assert mappings["ec2_2023_above_28_days"]["mappings"]["E"]["g1_field"] != "E_secant"
    assert (
        "E_secant fallback is forbidden"
        in mappings["ec2_2023_above_28_days"]["mappings"]["E"]["reason"]
    )


def test_legacy_v1_is_not_silently_promoted_to_verified_cdpm2_physical_authority():
    authority = _load("authority_matrix.json")
    legacy = authority["g1_to_cdpm2_mapping"]["legacy_v1"]
    assert legacy["automatic_static_cdpm2_readiness"] == "NOT_AUTHORIZED_PHYSICAL_SOURCE"
    assert {entry["state"] for entry in legacy["mappings"].values()} == {"NOT_AUTHORIZED"}


def test_fracture_energy_composition_must_be_explicit():
    authority = _load("authority_matrix.json")
    policy = authority["fracture_energy_composition_policy"]
    assert policy["implicit_cross_standard_composition"] == "FORBIDDEN"
    assert set(policy["future_authorized_mechanisms"]) == {
        "explicit_secondary_physical_authority_with_provenance",
        "explicit_constitutive_G_Ft_override_with_user_constitutive_override_provenance",
    }


def test_corrigendum_as_7_is_dataset_specific_not_global_default():
    authority = _load("authority_matrix.json")
    semantic = _load("semantic_parameter_contract.json")
    correction = authority["corrigendum"]
    assert correction["governing_equations"] == "UNCHANGED_CORRECT"
    assert correction["corrected_A_s"] == 7.0
    assert correction["universal_default"] is False
    assert semantic["named_calibration_policy"]["A_s"] == 15.0


def test_oofem_interface_drift_is_recorded_without_overriding_static_scientific_policy():
    authority = _load("authority_matrix.json")
    discrepancies = {
        row["topic"]: row for row in authority["implementation_interface_discrepancies"]
    }

    hp = discrepancies["OOFEM_H_p_default_drift"]
    assert hp["manual_2022_value"] == 0.01
    assert hp["current_oofem_source_value"] == 0.5
    assert "STATIC_V1_USES_GRASSL_2013" in hp["disposition"]

    damage = discrepancies["OOFEM_damage_flag_numbering_drift"]
    assert "two_damage_variables" in damage["disposition"]
    assert "downstream candidate" in damage["disposition"]


def test_backend_contract_accounts_for_exactly_24_unique_slots():
    backend = _load("backend_compatibility_contract.json")
    slots = backend["slots"]
    indices = [row["slot_index"] for row in slots]
    assert backend["slot_count"] == 24
    assert indices == list(range(1, 25))
    assert len({row["candidate_name"] for row in slots}) == 24


def test_static_adapter_fixed_slots_and_damageflag_quirk_are_frozen():
    backend = _load("backend_compatibility_contract.json")
    by_slot = {row["slot_index"]: row for row in backend["slots"]}
    expected = {14: 0, 15: 1, 16: 1.0, 20: 0, 21: 0, 23: 0, 24: 0}
    assert {slot: by_slot[slot]["fixed_adapter_value"] for slot in expected} == expected
    for slot in expected:
        assert by_slot[slot]["is_fixed_adapter_value"] is True
    assert "two-variable" in by_slot[23]["notes"]
    assert "Eq. (56)" in by_slot[16]["notes"]


def test_lchar_history_and_solver_controls_stay_outside_material_parameters():
    backend = _load("backend_compatibility_contract.json")
    semantic = _load("semantic_parameter_contract.json")
    assert backend["runtime_mesh_context"]["name"] == "LCHAR"
    assert backend["runtime_mesh_context"]["slot"] is None
    assert backend["history_state"]["slot_count"] == 27
    assert backend["history_state"]["included_in_material_parameter_object"] is False
    assert backend["solver_controls"]["included_in_material_parameter_object"] is False
    field_names = {field["name"] for field in semantic["fields"]}
    assert not field_names.intersection(
        {"LCHAR", "yieldtol", "newtoniter", "FD_REL_STEP", "FD_ABS_STEP", "MAX_SUBSTEP_DEPTH"}
    )


def test_bs_is_backend_fixed_one_and_not_a_2013_public_semantic_field():
    authority = _load("authority_matrix.json")
    semantic = _load("semantic_parameter_contract.json")
    bs = next(row for row in authority["parameters"] if row["semantic_name"] == "B_s")
    assert bs["classification"] == "BACKEND_ADAPTER_FIXED"
    assert bs["adapter_fixed_value"] == 1.0
    assert bs["public_field"] is False
    assert "B_s" not in {field["name"] for field in semantic["fields"]}


def test_semantic_units_and_positive_strength_sign_convention_are_explicit():
    semantic = _load("semantic_parameter_contract.json")
    assert semantic["canonical_units"] == {
        "calibration_parameters": "dimensionless",
        "crack_opening": "mm",
        "fracture_energy": "N/mm",
        "strain": "dimensionless",
        "stress_modulus": "MPa",
    }
    assert semantic["sign_convention"] == {
        "epsilon_fc": "positive inelastic-strain magnitude",
        "f_c": "positive magnitude",
        "f_t": "positive magnitude",
    }


def test_cdpm2_production_seam_remains_reserved_during_g2a():
    concrete = Concrete.from_class("C30", profile="fib_mc2010")
    with pytest.raises(NotImplementedError, match="G2"):
        concrete.to_cdpm2(calibration="cdpm2_grassl_2013")


def test_source_dossier_records_three_layer_boundary_and_no_mechanics():
    text = (REPO_ROOT / "docs" / "cdpm2_sources.md").read_text()
    assert "NORMATIVE PHYSICAL AUTHORITY" in text
    assert "CONSTITUTIVE MODEL AUTHORITY" in text
    assert "BACKEND / SOLVER INTERFACE AUTHORITY" in text
    assert "E_initial" in text
    assert "No fallback to `E_secant`" in text or "No fallback to E_secant" in text
    assert "does not implement a constitutive backend" in text
