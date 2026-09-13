"""G2-Q final end-to-end CDPM2 material-factory integration qualification."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from cdp_generator.concrete import (
    Cdpm2ConversionNotReadyError,
    Cdpm2ConversionReadiness,
    Cdpm2FractureEnergyComposition,
    Cdpm2Grassl2013Configuration,
    Concrete,
    ProfileConfiguration,
    class_entries,
)
from cdp_generator.concrete.models.cdpm2 import (
    CDPM2_LEGACY_FIXED_SLOTS,
    CDPM2_LEGACY_SEMANTIC_SLOT_MAP,
    CDPM2_LEGACY_SLOT_COUNT,
    CDPM2_LEGACY_SLOT_NAMES,
    CDPM2_PARAMETER_FIELDS,
    CDPM2_STATIC_CALIBRATION_ID,
    Cdpm2SourceKind,
    adapt_cdpm2_legacy_backend,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (REPO_ROOT / "qualification" / "cdpm2" / "g2q_integration_manifest.json").read_text()
)


def _ec23(age: float) -> Concrete:
    return Concrete.from_class(
        "C30/37",
        profile="ec2_2023",
        profile_parameters=ProfileConfiguration({"reference_age_days": age}),
    )


def _fib_composition() -> Cdpm2FractureEnergyComposition:
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    value = fib.physical.fracture_energy
    assert value is not None
    return Cdpm2FractureEnergyComposition(
        value=value,
        provenance=fib.physical.provenance["fracture_energy"],
    )


def _legacy_full_configuration() -> Cdpm2Grassl2013Configuration:
    return Cdpm2Grassl2013Configuration(
        {
            "E": 30000.0,
            "nu": 0.2,
            "f_t": 3.0,
            "f_c": 38.0,
            "G_Ft": 0.15,
            "eccentricity": 0.52,
            "q_h0": 0.3,
            "H_p": 0.01,
            "D_f": 0.85,
            "A_h": 0.08,
            "B_h": 0.003,
            "C_h": 2.0,
            "D_h": 1e-6,
            "A_s": 15.0,
            "epsilon_fc": 1e-4,
        }
    )


def _assert_backend_matches_semantics(params, payload) -> None:
    assert len(payload.cm) == CDPM2_LEGACY_SLOT_COUNT == 24
    for one_based, field in CDPM2_LEGACY_SEMANTIC_SLOT_MAP.items():
        assert payload.cm[one_based - 1] == getattr(params, field)
    for one_based, expected in CDPM2_LEGACY_FIXED_SLOTS.items():
        assert payload.cm[one_based - 1] == expected


def test_manifest_identity_and_independent_expected_matrix():
    assert MANIFEST["schema_version"] == "cdpm2_g2q_integration_manifest.v1"
    assert MANIFEST["gate"] == "G2-Q"
    assert MANIFEST["starting_head"] == "2c4fff85c45d0f6c61325367a63690d7d6e6ffbb"
    assert MANIFEST["starting_tree"] == "22f2a0c965f271abd4e80414e99fe83295d46e73"
    expected = {
        "fib_mc2010_C30_default": "READY",
        "ec2_2004_C30_37_default": "COMPOSITION_REQUIRED",
        "ec2_2004_C30_37_GFt_override": "READY",
        "ec2_2004_C30_37_physical_composition": "READY",
        "ec2_2023_C30_37_28_default": "COMPOSITION_REQUIRED",
        "ec2_2023_C30_37_56_default": "UNRESOLVED_PHYSICAL_INPUT",
        "ec2_2023_C30_37_56_GFt_only": "UNRESOLVED_PHYSICAL_INPUT",
        "ec2_2023_C30_37_56_E_only": "COMPOSITION_REQUIRED",
        "ec2_2023_C30_37_56_E_GFt": "READY",
        "legacy_v1": "NOT_AUTHORIZED_PHYSICAL_SOURCE",
    }
    assert {
        case["id"]: case["expected_readiness"] for case in MANIFEST["representative_case_matrix"]
    } == expected
    assert MANIFEST["expected_case_method"].startswith("Authored independently")


def test_downstream_moved_additively_without_material_contract_drift():
    downstream = MANIFEST["downstream_compatibility"]
    assert downstream["previous_qualified_head"] == "ce6daff41bb8683db94a7c9818886627e1154f4f"
    assert downstream["current_head"] == "bf7f78b97229e10ecadb0c04dc88f343fb825b79"
    assert downstream["current_tree"] == "6318b2f5bc6483f1e8616ac9d592af0cf73d4fdc"
    assert downstream["current_parent"] == "ce6daff41bb8683db94a7c9818886627e1154f4f"
    assert downstream["c1_parameter_map_blob_sha"] == "f23a109880fd17191a47654b4429ba4f5bf5ce14"
    assert downstream["cdpm2_3d_law_blob_sha"] == "d803a4ab5b3e517e5c7da7f6349d0e298ff8ef5b"
    assert downstream["result"] == "EXACT_MATERIAL_BACKEND_CONTRACT_PRESERVED"


def test_fib_c30_golden_path_is_complete_deterministic_and_non_mutating():
    concrete = Concrete.from_class("C30", profile="fib_mc2010")
    concrete_before = concrete.to_json()
    assessment = concrete.cdpm2_readiness()
    assert assessment.state is Cdpm2ConversionReadiness.READY
    params = concrete.to_cdpm2()
    params_before = params.to_json()
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)

    assert params.model_id == "cdpm2_grassl_2013"
    assert params.calibration_id == CDPM2_STATIC_CALIBRATION_ID
    assert len(params.values_dict()) == 20
    assert set(params.provenance) == set(CDPM2_PARAMETER_FIELDS)
    assert params.provenance["G_Ft"].source_kind is Cdpm2SourceKind.PHYSICAL_SOURCE
    _assert_backend_matches_semantics(params, payload)
    assert payload.characteristic_length == 37.5
    assert params.to_json() == params_before
    assert payload.to_json() == payload.to_json()
    assert concrete.to_json() == concrete_before


def test_all_17_fib_classes_complete_full_factory_chain():
    entries = [entry for entry in class_entries() if entry.profile == "fib_mc2010"]
    assert len(entries) == 17
    for entry in entries:
        concrete = Concrete.from_class(entry.canonical_class_string, profile="fib_mc2010")
        assert concrete.cdpm2_readiness().state is Cdpm2ConversionReadiness.READY
        params = concrete.to_cdpm2()
        payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
        _assert_backend_matches_semantics(params, payload)
        assert params.to_json() == params.to_json()
        assert payload.to_json() == payload.to_json()


def test_all_14_ec2_2004_classes_remain_truthfully_blocked_by_default():
    entries = [entry for entry in class_entries() if entry.profile == "ec2_2004"]
    assert len(entries) == 14
    for entry in entries:
        concrete = Concrete.from_class(entry.canonical_class_string, profile="ec2_2004")
        assessment = concrete.cdpm2_readiness()
        assert assessment.state is Cdpm2ConversionReadiness.COMPOSITION_REQUIRED
        with pytest.raises(Cdpm2ConversionNotReadyError):
            concrete.to_cdpm2()


def test_ec2_2004_explicit_gft_override_completes_end_to_end():
    concrete = Concrete.from_class("C30/37", profile="ec2_2004")
    config = Cdpm2Grassl2013Configuration({"G_Ft": 0.15})
    assert concrete.cdpm2_readiness(configuration=config).state is Cdpm2ConversionReadiness.READY
    params = concrete.to_cdpm2(configuration=config)
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    assert params.provenance["G_Ft"].source_kind is Cdpm2SourceKind.USER_CONSTITUTIVE_OVERRIDE
    assert params.provenance["w_f"].source_kind is Cdpm2SourceKind.GRASSL_2013_DERIVED
    assert "G_Ft" not in CDPM2_LEGACY_SLOT_NAMES
    assert payload.cm[16] == params.w_f
    assert payload.cm[17] == params.w_f1
    assert payload.cm[18] == params.f_t1


def test_ec2_2004_explicit_physical_composition_completes_end_to_end():
    composition = _fib_composition()
    concrete = Concrete.from_class("C30/37", profile="ec2_2004")
    before = concrete.to_json()
    assessment = concrete.cdpm2_readiness(fracture_energy_composition=composition)
    assert assessment.state is Cdpm2ConversionReadiness.READY
    params = concrete.to_cdpm2(fracture_energy_composition=composition)
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    assert concrete.physical_profile == "ec2_2004"
    assert concrete.to_json() == before
    assert params.provenance["G_Ft"].source_kind is Cdpm2SourceKind.PHYSICAL_SOURCE
    assert params.provenance["G_Ft"].source_id == "fib_mc2010_2013"
    _assert_backend_matches_semantics(params, payload)


def test_all_15_ec2_2023_28_day_classes_require_composition_by_default():
    entries = [entry for entry in class_entries() if entry.profile == "ec2_2023"]
    assert len(entries) == 15
    for entry in entries:
        concrete = Concrete.from_class(entry.canonical_class_string, profile="ec2_2023")
        assert concrete.cdpm2_readiness().state is Cdpm2ConversionReadiness.COMPOSITION_REQUIRED


def test_ec2_2023_28_day_explicit_gft_completes_end_to_end_without_e_secant_fallback():
    concrete = _ec23(28.0)
    config = Cdpm2Grassl2013Configuration({"G_Ft": 0.15})
    params = concrete.to_cdpm2(configuration=config)
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    assert concrete.physical.E_initial is not None
    assert concrete.physical.E_initial == params.E
    assert payload.cm[0] == params.E


@pytest.mark.parametrize("age", [56.0, 91.0])
def test_ec2_2023_high_age_default_and_partial_authority_matrix(age: float):
    concrete = _ec23(age)
    assert concrete.physical.E_initial is None
    default = concrete.cdpm2_readiness()
    assert default.state is Cdpm2ConversionReadiness.UNRESOLVED_PHYSICAL_INPUT
    assert default.blockers == (
        "E_initial unresolved; E_secant fallback forbidden",
        "G_Ft composition required",
    )
    gft_only = concrete.cdpm2_readiness(configuration=Cdpm2Grassl2013Configuration({"G_Ft": 0.15}))
    assert gft_only.state is Cdpm2ConversionReadiness.UNRESOLVED_PHYSICAL_INPUT
    e_only = concrete.cdpm2_readiness(configuration=Cdpm2Grassl2013Configuration({"E": 41000.0}))
    assert e_only.state is Cdpm2ConversionReadiness.COMPOSITION_REQUIRED


@pytest.mark.parametrize("age", [56.0, 91.0])
def test_ec2_2023_high_age_explicit_e_and_gft_complete_end_to_end(age: float):
    concrete = _ec23(age)
    explicit_e = 41000.0 if age == 56.0 else 42000.0
    assert concrete.physical.E_secant != explicit_e
    config = Cdpm2Grassl2013Configuration({"E": explicit_e, "G_Ft": 0.15})
    assert concrete.cdpm2_readiness(configuration=config).state is Cdpm2ConversionReadiness.READY
    params = concrete.to_cdpm2(configuration=config)
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    assert explicit_e == params.E
    assert params.provenance["E"].source_kind is Cdpm2SourceKind.USER_CONSTITUTIVE_OVERRIDE
    assert concrete.physical.E_secant != params.E
    assert payload.cm[0] == explicit_e


def test_legacy_is_not_authorized_and_never_reaches_verified_backend_chain():
    legacy = Concrete.from_mean_strength(38.0, 0.0022, 0.0035)
    assert legacy.cdpm2_readiness().state is Cdpm2ConversionReadiness.NOT_AUTHORIZED_PHYSICAL_SOURCE
    assessment = legacy.cdpm2_readiness(configuration=_legacy_full_configuration())
    assert assessment.state is Cdpm2ConversionReadiness.NOT_AUTHORIZED_PHYSICAL_SOURCE
    with pytest.raises(Cdpm2ConversionNotReadyError):
        legacy.to_cdpm2(configuration=_legacy_full_configuration())


def test_default_fib_provenance_category_matrix_survives_full_chain():
    params = Concrete.from_class("C30", profile="fib_mc2010").to_cdpm2()
    for field in ("E", "nu", "f_t", "f_c", "G_Ft"):
        assert params.provenance[field].source_kind is Cdpm2SourceKind.PHYSICAL_SOURCE
    for field in ("w_f", "w_f1", "f_t1", "eccentricity"):
        assert params.provenance[field].source_kind is Cdpm2SourceKind.GRASSL_2013_DERIVED
    for field in ("q_h0", "H_p", "D_f", "A_h", "B_h", "C_h", "D_h"):
        assert (
            params.provenance[field].source_kind is Cdpm2SourceKind.GRASSL_2013_DEFAULT_CALIBRATION
        )
    for field in ("A_s", "epsilon_fc"):
        assert params.provenance[field].source_kind is Cdpm2SourceKind.OOFEM_IMPLEMENTATION_DEFAULT
    for field in ("tensile_softening_type", "damage_formulation"):
        assert params.provenance[field].source_kind is Cdpm2SourceKind.GRASSL_2013_DIRECT
    before = params.to_json()
    adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    assert params.to_json() == before


def test_representative_override_intent_reaches_backend_without_provenance_flattening():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    config = Cdpm2Grassl2013Configuration(
        {
            "E": 39000.0,
            "A_s": 16.0,
            "eccentricity": 0.54,
            "G_Ft": 0.19,
        }
    )
    params = fib.to_cdpm2(configuration=config)
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    for field in ("E", "A_s", "eccentricity", "G_Ft"):
        assert params.provenance[field].source_kind is Cdpm2SourceKind.USER_CONSTITUTIVE_OVERRIDE
    assert payload.cm[0] == 39000.0
    assert payload.cm[2] == 0.54
    assert payload.cm[11] == 16.0
    assert payload.cm[16] == params.w_f
    assert payload.cm[16] == pytest.approx(0.19 / (0.225 * params.f_t))
    assert "provenance" not in payload.to_dict()


def test_backend_is_numerically_independent_of_legitimate_provenance_difference():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    default = fib.to_cdpm2()
    overridden = fib.to_cdpm2(configuration=Cdpm2Grassl2013Configuration({"E": default.E}))
    assert default.values_dict() == overridden.values_dict()
    assert default.provenance["E"].source_kind is Cdpm2SourceKind.PHYSICAL_SOURCE
    assert overridden.provenance["E"].source_kind is Cdpm2SourceKind.USER_CONSTITUTIVE_OVERRIDE
    default_payload = adapt_cdpm2_legacy_backend(default, characteristic_length=37.5)
    override_payload = adapt_cdpm2_legacy_backend(overridden, characteristic_length=37.5)
    assert default_payload.cm == override_payload.cm


def test_real_fib_path_preserves_full_24_slot_contract_and_fixed_flags():
    params = Concrete.from_class("C30", profile="fib_mc2010").to_cdpm2()
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    assert CDPM2_LEGACY_SLOT_NAMES == (
        "E",
        "NU",
        "ECC",
        "QH0",
        "FT",
        "FC",
        "HP",
        "AH",
        "BH",
        "CH",
        "DH",
        "AS",
        "DF",
        "ERATETYPE",
        "TYPE",
        "BS",
        "WF",
        "WF1",
        "FT1",
        "SRATETYPE",
        "FAILFLG",
        "EFC",
        "DAMAGEFLAG",
        "PRINTFLAG",
    )
    _assert_backend_matches_semantics(params, payload)
    assert payload.cm[13] == 0.0
    assert payload.cm[14] == 1.0
    assert payload.cm[15] == 1.0
    assert payload.cm[19] == 0.0
    assert payload.cm[20] == 0.0
    assert payload.cm[22] == 0.0
    assert payload.cm[23] == 0.0
    assert "G_Ft" not in CDPM2_LEGACY_SLOT_NAMES
    assert "LCHAR" not in CDPM2_LEGACY_SLOT_NAMES


def test_lchar_is_independent_of_semantics_in_both_directions():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    params = fib.to_cdpm2()
    params_before = params.to_json()
    first = adapt_cdpm2_legacy_backend(params, characteristic_length=20.0)
    second = adapt_cdpm2_legacy_backend(params, characteristic_length=40.0)
    assert first.cm == second.cm
    assert first.characteristic_length == 20.0
    assert second.characteristic_length == 40.0
    assert params.to_json() == params_before

    modified = fib.to_cdpm2(configuration=Cdpm2Grassl2013Configuration({"E": params.E + 1000.0}))
    third = adapt_cdpm2_legacy_backend(modified, characteristic_length=20.0)
    assert third.characteristic_length == first.characteristic_length
    assert third.cm != first.cm
    assert third.cm[0] == params.E + 1000.0


def test_invalid_lchar_is_rejected_without_default_floor_or_geometry_inference():
    params = Concrete.from_class("C30", profile="fib_mc2010").to_cdpm2()
    for value in (0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, False, "1", None):
        with pytest.raises((TypeError, ValueError)):
            adapt_cdpm2_legacy_backend(
                params,
                characteristic_length=value,  # type: ignore[arg-type]
            )
    for value in (1e-12, 1.0, 37.5, 1e9):
        result = adapt_cdpm2_legacy_backend(params, characteristic_length=value)
        assert result.characteristic_length == value


def test_three_serialization_boundaries_remain_separate_and_deterministic():
    concrete = Concrete.from_class("C30", profile="fib_mc2010")
    params = concrete.to_cdpm2()
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)

    physical_doc = concrete.to_dict()
    semantic_doc = params.to_dict()
    backend_doc = payload.to_dict()

    assert concrete.to_json() == concrete.to_json()
    assert params.to_json() == params.to_json()
    assert payload.to_json() == payload.to_json()

    assert physical_doc["constitutive_models"] == {}
    assert "cdpm2" not in json.dumps(physical_doc).lower()
    assert "cm" not in semantic_doc
    assert "characteristic_length" not in semantic_doc
    assert "provenance" in semantic_doc
    assert set(backend_doc) == {"schema_version", "backend_id", "cm", "characteristic_length"}
    assert "provenance" not in backend_doc
    assert MANIFEST["serialization_policy_adjudication"] == "NO_NEW_PRODUCTION_ENVELOPE"


def test_calibration_selector_preserves_operational_identity_and_old_placeholder_behavior():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    assert fib.to_cdpm2(calibration=CDPM2_STATIC_CALIBRATION_ID).calibration_id == (
        CDPM2_STATIC_CALIBRATION_ID
    )
    with pytest.raises(NotImplementedError, match="model-id placeholder"):
        fib.to_cdpm2(calibration="cdpm2_grassl_2013")


def test_production_cdpm2_imports_have_no_kratos_or_qualification_runtime_dependency():
    module_dir = REPO_ROOT / "cdp_generator" / "concrete" / "models" / "cdpm2"
    for path in module_dir.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            name == "KratosMultiphysics" or name.startswith("KratosMultiphysics.")
            for name in imported
        )
        assert not any(
            name == "qualification" or name.startswith("qualification.") for name in imported
        )


def test_production_cdpm2_surface_contains_no_mechanics_or_runtime_state_module():
    module_dir = REPO_ROOT / "cdp_generator" / "concrete" / "models" / "cdpm2"
    assert {path.name for path in module_dir.glob("*.py")} == {
        "__init__.py",
        "backend.py",
        "configuration.py",
        "mapping.py",
        "provenance.py",
        "schema.py",
    }
    public = set()
    for path in module_dir.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                public.add(node.name)
    forbidden = {
        "return_mapping",
        "stress_integration",
        "commit_state",
        "revert_state",
        "substep",
        "yield_surface",
        "plastic_potential",
    }
    assert public.isdisjoint(forbidden)
