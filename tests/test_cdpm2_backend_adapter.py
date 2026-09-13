"""G2-D qualification for exact legacy24 backend adaptation and explicit LCHAR."""

import inspect
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import cdp_generator.concrete as concrete_api
from cdp_generator.concrete import (
    Cdpm2FractureEnergyComposition,
    Cdpm2Grassl2013Configuration,
    Concrete,
)
from cdp_generator.concrete.models import (
    Cdpm2Legacy24BackendInput,
    adapt_cdpm2_legacy_backend,
)
from cdp_generator.concrete.models.cdpm2 import (
    CDPM2_BACKEND_ID,
    CDPM2_BACKEND_INPUT_SCHEMA_VERSION,
    CDPM2_CHARACTERISTIC_LENGTH_UNITS,
    CDPM2_LEGACY_FIXED_SLOTS,
    CDPM2_LEGACY_SEMANTIC_SLOT_MAP,
    CDPM2_LEGACY_SLOT_COUNT,
    CDPM2_LEGACY_SLOT_NAMES,
    CDPM2_PARAMETER_FIELDS,
    CDPM2_PARAMETER_UNITS,
    Cdpm2DamageFormulation,
    Cdpm2Grassl2013Parameters,
    Cdpm2ParameterProvenance,
    Cdpm2SourceKind,
    Cdpm2TensileSofteningType,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads(
    (REPO_ROOT / "qualification" / "cdpm2" / "g2d_backend_reference.json").read_text()
)
FROZEN_BACKEND_CONTRACT = json.loads(
    (REPO_ROOT / "qualification" / "cdpm2" / "backend_compatibility_contract.json").read_text()
)

EXPECTED_CM = tuple(REFERENCE["sentinel"]["expected_cm"])


def _prov(
    field: str,
    kind: Cdpm2SourceKind,
    *,
    source_id: str,
    derived_from: tuple[str, ...] = (),
    overridden: bool = False,
) -> Cdpm2ParameterProvenance:
    return Cdpm2ParameterProvenance(
        source_id=source_id,
        source_kind=kind,
        edition="G2-D-test",
        equation_or_section="independent sentinel",
        units=CDPM2_PARAMETER_UNITS[field],
        overridden=overridden,
        derived_from=derived_from,
    )


def _sentinel_provenance(*, variant: bool = False) -> dict[str, Cdpm2ParameterProvenance]:
    physical_id = "sentinel_physical_variant" if variant else "sentinel_physical"
    default_id = "sentinel_default_variant" if variant else "sentinel_default"
    provenance: dict[str, Cdpm2ParameterProvenance] = {}

    for field in ("E", "nu", "f_t", "f_c", "G_Ft"):
        provenance[field] = _prov(
            field,
            Cdpm2SourceKind.PHYSICAL_SOURCE,
            source_id=physical_id,
            derived_from=(f"physical.{field}",),
        )

    provenance["w_f"] = _prov(
        "w_f",
        Cdpm2SourceKind.GRASSL_2013_DERIVED,
        source_id="grassl_et_al_2013_cdpm2",
        derived_from=("G_Ft", "f_t"),
    )
    provenance["w_f1"] = _prov(
        "w_f1",
        Cdpm2SourceKind.GRASSL_2013_DERIVED,
        source_id="grassl_et_al_2013_cdpm2",
        derived_from=("w_f",),
    )
    provenance["f_t1"] = _prov(
        "f_t1",
        Cdpm2SourceKind.GRASSL_2013_DERIVED,
        source_id="grassl_et_al_2013_cdpm2",
        derived_from=("f_t",),
    )
    provenance["eccentricity"] = _prov(
        "eccentricity",
        Cdpm2SourceKind.GRASSL_2013_DERIVED,
        source_id="grassl_et_al_2013_cdpm2",
        derived_from=("f_t", "f_c"),
    )

    for field in ("q_h0", "H_p", "D_f", "A_h", "B_h", "C_h", "D_h"):
        provenance[field] = _prov(
            field,
            Cdpm2SourceKind.GRASSL_2013_DEFAULT_CALIBRATION,
            source_id=default_id,
        )

    for field in ("A_s", "epsilon_fc"):
        provenance[field] = _prov(
            field,
            Cdpm2SourceKind.OOFEM_IMPLEMENTATION_DEFAULT,
            source_id="sentinel_oofem_variant" if variant else "sentinel_oofem",
        )

    for field in ("tensile_softening_type", "damage_formulation"):
        provenance[field] = _prov(
            field,
            Cdpm2SourceKind.GRASSL_2013_DIRECT,
            source_id="grassl_et_al_2013_cdpm2",
        )

    assert set(provenance) == set(CDPM2_PARAMETER_FIELDS)
    return provenance


def _sentinel_params(*, variant_provenance: bool = False) -> Cdpm2Grassl2013Parameters:
    values = REFERENCE["sentinel"]["semantic_values"]
    return Cdpm2Grassl2013Parameters(
        E=values["E"],
        nu=values["nu"],
        f_t=values["f_t"],
        f_c=values["f_c"],
        G_Ft=values["G_Ft"],
        w_f=values["w_f"],
        w_f1=values["w_f1"],
        f_t1=values["f_t1"],
        eccentricity=values["eccentricity"],
        q_h0=values["q_h0"],
        H_p=values["H_p"],
        D_f=values["D_f"],
        A_h=values["A_h"],
        B_h=values["B_h"],
        C_h=values["C_h"],
        D_h=values["D_h"],
        A_s=values["A_s"],
        epsilon_fc=values["epsilon_fc"],
        tensile_softening_type=Cdpm2TensileSofteningType.BILINEAR,
        damage_formulation=Cdpm2DamageFormulation.TWO_DAMAGE_VARIABLES,
        provenance=_sentinel_provenance(variant=variant_provenance),
    )


def test_independent_fixture_records_current_downstream_adjudication():
    assert REFERENCE["schema_version"] == "cdpm2_g2d_backend_reference.v1"
    assert REFERENCE["gate"] == "G2-D"
    assert "without importing the production backend adapter" in REFERENCE["method"]
    downstream = REFERENCE["downstream_compatibility"]
    assert downstream["frozen_contract_origin_head"] == ("b98bef2c753bb57e4890ff385d8eea15c185e0ef")
    assert downstream["verified_current_downstream_head"] == (
        "ce6daff41bb8683db94a7c9818886627e1154f4f"
    )
    assert downstream["verified_current_downstream_tree"] == (
        "faf6f04df4d0a1f169fce494460fe96368c44589"
    )
    assert downstream["current_parent"] == "b98bef2c753bb57e4890ff385d8eea15c185e0ef"
    assert downstream["compatibility_result"] == "EXACT_CONTRACT_PRESERVED"


def test_production_contract_identity_matches_frozen_g2a_contract():
    assert FROZEN_BACKEND_CONTRACT["backend_id"] == CDPM2_BACKEND_ID
    assert FROZEN_BACKEND_CONTRACT["slot_count"] == CDPM2_LEGACY_SLOT_COUNT == 24
    assert FROZEN_BACKEND_CONTRACT["slot_indices"] == list(range(1, 25))
    assert (
        tuple(slot["candidate_name"] for slot in FROZEN_BACKEND_CONTRACT["slots"])
        == CDPM2_LEGACY_SLOT_NAMES
    )
    assert CDPM2_BACKEND_INPUT_SCHEMA_VERSION == "cdpm2_legacy24_backend_input.v1"


def test_production_semantic_and_fixed_maps_match_frozen_contract():
    semantic_expected = {
        slot["slot_index"]: slot["semantic_source"]
        for slot in FROZEN_BACKEND_CONTRACT["slots"]
        if not slot["is_fixed_adapter_value"]
    }
    fixed_expected = {
        slot["slot_index"]: float(slot["fixed_adapter_value"])
        for slot in FROZEN_BACKEND_CONTRACT["slots"]
        if slot["is_fixed_adapter_value"]
    }
    assert dict(CDPM2_LEGACY_SEMANTIC_SLOT_MAP) == semantic_expected
    assert dict(CDPM2_LEGACY_FIXED_SLOTS) == fixed_expected


def test_lchar_contract_matches_frozen_runtime_mesh_context():
    context = FROZEN_BACKEND_CONTRACT["runtime_mesh_context"]
    assert context["name"] == "LCHAR"
    assert context["required"] is True
    assert context["slot"] is None
    assert "finite > 0" in context["rule"]
    assert context["units"].startswith("mm")
    assert CDPM2_CHARACTERISTIC_LENGTH_UNITS == "mm"


def test_hand_constructed_sentinel_matches_exact_independent_vector():
    params = _sentinel_params()
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    assert payload.cm == EXPECTED_CM
    assert len(payload.cm) == 24
    assert payload.characteristic_length == 37.5


@pytest.mark.parametrize(
    ("slot", "field"),
    [
        (1, "E"),
        (2, "nu"),
        (3, "eccentricity"),
        (4, "q_h0"),
        (5, "f_t"),
        (6, "f_c"),
        (7, "H_p"),
        (8, "A_h"),
        (9, "B_h"),
        (10, "C_h"),
        (11, "D_h"),
        (12, "A_s"),
        (13, "D_f"),
        (17, "w_f"),
        (18, "w_f1"),
        (19, "f_t1"),
        (22, "epsilon_fc"),
    ],
)
def test_every_semantic_source_lands_in_exact_slot(slot: int, field: str):
    params = _sentinel_params()
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    assert payload.cm[slot - 1] == getattr(params, field)


@pytest.mark.parametrize(
    ("slot", "expected"),
    [(14, 0.0), (15, 1.0), (16, 1.0), (20, 0.0), (21, 0.0), (23, 0.0), (24, 0.0)],
)
def test_every_static_v1_fixed_slot_is_exact(slot: int, expected: float):
    payload = adapt_cdpm2_legacy_backend(_sentinel_params(), characteristic_length=37.5)
    assert payload.cm[slot - 1] == expected


def test_type_damage_and_bs_policy_are_explicitly_frozen():
    params = _sentinel_params()
    assert params.tensile_softening_type is Cdpm2TensileSofteningType.BILINEAR
    assert params.damage_formulation is Cdpm2DamageFormulation.TWO_DAMAGE_VARIABLES
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    assert payload.cm[14] == 1.0  # TYPE
    assert payload.cm[15] == 1.0  # BS
    assert payload.cm[22] == 0.0  # DAMAGEFLAG source-behavior quirk


def test_rate_failure_and_print_flags_remain_disabled():
    payload = adapt_cdpm2_legacy_backend(_sentinel_params(), characteristic_length=37.5)
    assert payload.cm[13] == 0.0  # ERATETYPE
    assert payload.cm[19] == 0.0  # SRATETYPE
    assert payload.cm[20] == 0.0  # FAILFLG
    assert payload.cm[23] == 0.0  # PRINTFLAG


@pytest.mark.parametrize("value", [1e-12, 1.0, 37.5, 1e9])
def test_positive_finite_lchar_is_accepted(value: float):
    payload = adapt_cdpm2_legacy_backend(_sentinel_params(), characteristic_length=value)
    assert payload.characteristic_length == value


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_nonpositive_or_nonfinite_lchar_is_rejected(value: float):
    with pytest.raises(ValueError):
        adapt_cdpm2_legacy_backend(_sentinel_params(), characteristic_length=value)


@pytest.mark.parametrize("value", [True, False, "37.5", None])
def test_nonnumeric_or_bool_lchar_is_rejected(value: object):
    with pytest.raises(TypeError):
        adapt_cdpm2_legacy_backend(_sentinel_params(), characteristic_length=value)


def test_lchar_changes_only_runtime_field_not_cm():
    params = _sentinel_params()
    a = adapt_cdpm2_legacy_backend(params, characteristic_length=1.0)
    b = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    assert a.cm == b.cm
    assert a.characteristic_length != b.characteristic_length
    assert 37.5 not in b.cm


def test_backend_payload_is_immutable_and_validates_its_own_contract():
    payload = adapt_cdpm2_legacy_backend(_sentinel_params(), characteristic_length=37.5)
    with pytest.raises(FrozenInstanceError):
        payload.characteristic_length = 1.0
    with pytest.raises(ValueError, match="exactly 24"):
        Cdpm2Legacy24BackendInput(cm=(1.0,) * 23, characteristic_length=1.0)
    with pytest.raises(ValueError, match="finite"):
        Cdpm2Legacy24BackendInput(cm=(1.0,) * 23 + (float("nan"),), characteristic_length=1.0)


def test_adapter_is_deterministic_and_does_not_mutate_semantic_input():
    params = _sentinel_params()
    before = params.to_json()
    first = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    second = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    assert first == second
    assert first.to_json() == second.to_json()
    assert params.to_json() == before


def test_backend_payload_serialization_shape_is_narrow_and_deterministic():
    payload = adapt_cdpm2_legacy_backend(_sentinel_params(), characteristic_length=37.5)
    serialized = payload.to_dict()
    assert serialized == {
        "schema_version": "cdpm2_legacy24_backend_input.v1",
        "backend_id": "cdpm2_kratos_downstream_b98bef2",
        "cm": list(EXPECTED_CM),
        "characteristic_length": 37.5,
    }
    assert payload.to_json() == payload.to_json()
    assert "provenance" not in serialized
    assert "history" not in serialized
    assert "solver" not in serialized


def test_provenance_differences_do_not_change_mechanical_backend_payload():
    a = _sentinel_params(variant_provenance=False)
    b = _sentinel_params(variant_provenance=True)
    assert a.provenance_dict() != b.provenance_dict()
    payload_a = adapt_cdpm2_legacy_backend(a, characteristic_length=37.5)
    payload_b = adapt_cdpm2_legacy_backend(b, characteristic_length=37.5)
    assert payload_a == payload_b


def test_gft_has_no_direct_backend_slot_and_lchar_is_not_slot_25():
    assert "G_Ft" not in CDPM2_LEGACY_SLOT_NAMES
    assert CDPM2_LEGACY_SLOT_COUNT == 24
    assert len(CDPM2_LEGACY_SLOT_NAMES) == 24
    assert max(CDPM2_LEGACY_FIXED_SLOTS) == 24
    assert REFERENCE["sentinel"]["characteristic_length"] not in EXPECTED_CM


def test_adapter_emits_resolved_values_not_legacy_sentinels():
    params = _sentinel_params()
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    resolved_slots = {
        3: params.eccentricity,
        4: params.q_h0,
        7: params.H_p,
        8: params.A_h,
        9: params.B_h,
        10: params.C_h,
        11: params.D_h,
        12: params.A_s,
        13: params.D_f,
        18: params.w_f1,
        19: params.f_t1,
        22: params.epsilon_fc,
    }
    for slot, expected in resolved_slots.items():
        assert payload.cm[slot - 1] == expected


def test_adapter_rejects_raw_mapping_instead_of_bypassing_semantic_schema():
    with pytest.raises(TypeError, match="Cdpm2Grassl2013Parameters"):
        adapt_cdpm2_legacy_backend({}, characteristic_length=37.5)


def test_core_adapter_succeeds_without_concrete_or_readiness_object():
    params = _sentinel_params()
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    assert payload.cm == EXPECTED_CM


def test_real_fib_chain_adapts_resolved_semantics_only():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    params = fib.to_cdpm2()
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    assert len(payload.cm) == 24
    assert payload.cm[0] == params.E
    assert payload.cm[1] == params.nu
    assert payload.cm[2] == params.eccentricity
    assert payload.cm[16] == params.w_f
    assert payload.cm[21] == params.epsilon_fc
    assert payload.cm[23] == 0.0
    assert payload.characteristic_length == 37.5


def test_ec2_override_flow_has_same_pure_adapter_boundary():
    ec04 = Concrete.from_class("C30/37", profile="ec2_2004")
    params = ec04.to_cdpm2(configuration=Cdpm2Grassl2013Configuration({"G_Ft": 0.15}))
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=22.0)
    assert payload.cm[0] == params.E
    assert payload.cm[16] == params.w_f
    assert payload.cm[17] == params.w_f1
    assert payload.cm[18] == params.f_t1
    assert payload.characteristic_length == 22.0


def test_ec2_explicit_physical_composition_flow_has_same_adapter_boundary():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    ec04 = Concrete.from_class("C30/37", profile="ec2_2004")
    assert fib.physical.fracture_energy is not None
    composition = Cdpm2FractureEnergyComposition(
        value=fib.physical.fracture_energy,
        provenance=fib.physical.provenance["fracture_energy"],
    )
    params = ec04.to_cdpm2(fracture_energy_composition=composition)
    payload = adapt_cdpm2_legacy_backend(params, characteristic_length=22.0)
    assert payload.cm[0] == params.E
    assert payload.cm[16] == params.w_f
    assert payload.characteristic_length == 22.0


def test_concrete_to_cdpm2_signature_and_return_boundary_are_unchanged():
    signature = inspect.signature(Concrete.to_cdpm2)
    assert "characteristic_length" not in signature.parameters
    assert "LCHAR" not in signature.parameters
    params = Concrete.from_class("C30", profile="fib_mc2010").to_cdpm2()
    assert isinstance(params, Cdpm2Grassl2013Parameters)
    assert not isinstance(params, Cdpm2Legacy24BackendInput)


def test_concrete_serialization_shape_is_unchanged_by_backend_adaptation():
    concrete = Concrete.from_class("C30", profile="fib_mc2010")
    before = concrete.to_json()
    params = concrete.to_cdpm2()
    adapt_cdpm2_legacy_backend(params, characteristic_length=37.5)
    after = concrete.to_json()
    assert before == after
    assert '"cm"' not in after
    assert "characteristic_length" not in after
    assert "backend_id" not in after


def test_adapter_exports_are_lower_level_not_high_level_concrete_api():
    from cdp_generator.concrete import models
    from cdp_generator.concrete.models import cdpm2

    assert cdpm2.adapt_cdpm2_legacy_backend is adapt_cdpm2_legacy_backend
    assert models.adapt_cdpm2_legacy_backend is adapt_cdpm2_legacy_backend
    assert not hasattr(concrete_api, "adapt_cdpm2_legacy_backend")
    assert not hasattr(concrete_api, "Cdpm2Legacy24BackendInput")


def test_production_backend_has_no_kratos_g1_mapping_or_solver_dependencies():
    backend_source = (
        REPO_ROOT / "cdp_generator" / "concrete" / "models" / "cdpm2" / "backend.py"
    ).read_text()
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()

    assert "KratosMultiphysics" not in backend_source
    assert "KratosMultiphysics" not in pyproject
    assert "ConcretePhysicalProperties" not in backend_source
    assert "Cdpm2Grassl2013Configuration" not in backend_source
    assert "Cdpm2ReadinessAssessment" not in backend_source
    assert "E_initial" not in backend_source
    assert "fracture_energy" not in backend_source


def test_production_backend_contains_no_history_state_solver_or_mechanics_surface():
    source = (
        REPO_ROOT / "cdp_generator" / "concrete" / "models" / "cdpm2" / "backend.py"
    ).read_text()
    forbidden = (
        "kappaP",
        "omega_t",
        "omega_c",
        "FD_REL_STEP",
        "FD_ABS_STEP",
        "MAX_SUBSTEP_DEPTH",
        "yieldtol",
        "newtoniter",
        "OUTER_POLICY_MODE",
        "return_mapping",
        "stress_update",
        "substepping",
        "tangent",
    )
    for token in forbidden:
        assert token not in source
