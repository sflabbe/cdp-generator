"""G2-B qualification for standalone CDPM2 schema/provenance/configuration."""

import hashlib
import json
import math
from dataclasses import fields, replace
from pathlib import Path

import pytest

from cdp_generator.concrete import Concrete
from cdp_generator.concrete.models.cdpm2 import (
    CDPM2_MODEL_ID,
    CDPM2_OVERRIDE_FIELDS,
    CDPM2_PARAMETER_FIELDS,
    CDPM2_PARAMETER_UNITS,
    CDPM2_STATIC_CALIBRATION_ID,
    Cdpm2ConversionReadiness,
    Cdpm2DamageFormulation,
    Cdpm2Grassl2013Configuration,
    Cdpm2Grassl2013Parameters,
    Cdpm2ParameterProvenance,
    Cdpm2ReadinessAssessment,
    Cdpm2SourceKind,
    Cdpm2TensileSofteningType,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_CONTRACT = REPO_ROOT / "qualification" / "cdpm2" / "semantic_parameter_contract.json"

FROZEN_G2A = {
    "docs/cdpm2_sources.md": "876eae013ec05963fc539d8682fef7d501afbf0c1c11b1701292aad9b3c2eabf",
    "qualification/cdpm2/authority_matrix.json": (
        "f03c5fe35d1bc436ea50c1bfdaa51247bd6dd6966e3df0b568bcd380161f8e43"
    ),
    "qualification/cdpm2/semantic_parameter_contract.json": (
        "bf4821a886a3224e9cf1dc3933e6fd001a098f9924b4169cf5aae4c619d2b5b0"
    ),
    "qualification/cdpm2/backend_compatibility_contract.json": (
        "33eb4ac8e3b7a607e17fd890c3b99b38d2fc2feb26701529206d4752735c6244"
    ),
    "tests/test_cdpm2_authority_contract.py": (
        "d6c36c518110d9ff36e40bcc21bddc2b4208d1a02bf7cd9c9de3d95b2bd735c8"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance() -> dict[str, Cdpm2ParameterProvenance]:
    result: dict[str, Cdpm2ParameterProvenance] = {}
    for field in ("E", "nu", "f_t", "f_c", "G_Ft"):
        result[field] = Cdpm2ParameterProvenance(
            source_id="explicit_test_physical_source",
            source_kind=Cdpm2SourceKind.PHYSICAL_SOURCE,
            edition=None,
            equation_or_section=None,
            units=CDPM2_PARAMETER_UNITS[field],
        )

    result["w_f"] = Cdpm2ParameterProvenance(
        source_id="grassl_2013",
        source_kind=Cdpm2SourceKind.GRASSL_2013_DERIVED,
        edition="2013",
        equation_or_section="bilinear fracture-energy area relation",
        units="mm",
        derived_from=("G_Ft", "f_t"),
    )
    result["w_f1"] = Cdpm2ParameterProvenance(
        source_id="grassl_2013",
        source_kind=Cdpm2SourceKind.GRASSL_2013_DERIVED,
        edition="2013",
        equation_or_section="bilinear static-V1 ratio",
        units="mm",
        derived_from=("w_f",),
    )
    result["f_t1"] = Cdpm2ParameterProvenance(
        source_id="grassl_2013",
        source_kind=Cdpm2SourceKind.GRASSL_2013_DERIVED,
        edition="2013",
        equation_or_section="bilinear static-V1 ratio",
        units="MPa",
        derived_from=("f_t",),
    )

    for field in ("eccentricity", "q_h0", "H_p", "D_f", "A_h", "B_h", "C_h", "D_h"):
        result[field] = Cdpm2ParameterProvenance(
            source_id="grassl_2013",
            source_kind=Cdpm2SourceKind.GRASSL_2013_DEFAULT_CALIBRATION,
            edition="2013",
            equation_or_section="static default calibration",
            units=CDPM2_PARAMETER_UNITS[field],
        )

    for field in ("A_s", "epsilon_fc"):
        result[field] = Cdpm2ParameterProvenance(
            source_id="oofem_grassl_static_baseline",
            source_kind=Cdpm2SourceKind.OOFEM_IMPLEMENTATION_DEFAULT,
            edition="2022-static-baseline",
            equation_or_section="implementation default",
            units=CDPM2_PARAMETER_UNITS[field],
        )

    for field in ("tensile_softening_type", "damage_formulation"):
        result[field] = Cdpm2ParameterProvenance(
            source_id="grassl_2013",
            source_kind=Cdpm2SourceKind.GRASSL_2013_DIRECT,
            edition="2013",
            equation_or_section="static V1 model policy",
            units="enum",
        )
    return result


def _sample(**updates: object) -> Cdpm2Grassl2013Parameters:
    w_f = 0.1 / (0.225 * 3.0)
    values: dict[str, object] = {
        "E": 30000.0,
        "nu": 0.2,
        "f_t": 3.0,
        "f_c": 30.0,
        "G_Ft": 0.1,
        "w_f": w_f,
        "w_f1": 0.15 * w_f,
        "f_t1": 0.3 * 3.0,
        "eccentricity": 0.525,
        "q_h0": 0.3,
        "H_p": 0.01,
        "D_f": 0.85,
        "A_h": 0.08,
        "B_h": 0.003,
        "C_h": 2.0,
        "D_h": 1e-6,
        "A_s": 15.0,
        "epsilon_fc": 1e-4,
        "tensile_softening_type": Cdpm2TensileSofteningType.BILINEAR,
        "damage_formulation": Cdpm2DamageFormulation.TWO_DAMAGE_VARIABLES,
        "provenance": _provenance(),
    }
    values.update(updates)
    return Cdpm2Grassl2013Parameters(**values)  # type: ignore[arg-type]


def test_g2a_authority_files_remain_byte_identical():
    for relative, expected in FROZEN_G2A.items():
        assert _sha256(REPO_ROOT / relative) == expected


def test_production_schema_matches_frozen_g2a_contract():
    contract = json.loads(SEMANTIC_CONTRACT.read_text())
    assert contract["model_id"] == CDPM2_MODEL_ID
    assert contract["calibration_id"] == CDPM2_STATIC_CALIBRATION_ID
    assert tuple(field["name"] for field in contract["fields"]) == CDPM2_PARAMETER_FIELDS
    assert dict(CDPM2_PARAMETER_UNITS) == {
        field["name"]: field["units"] for field in contract["fields"]
    }
    assert (
        tuple(field["name"] for field in contract["fields"] if field["override_permitted"])
        == CDPM2_OVERRIDE_FIELDS
    )
    assert [state.value for state in Cdpm2ConversionReadiness] == contract[
        "conversion_readiness_states"
    ]
    assert [member.value for member in Cdpm2TensileSofteningType] == ["bilinear"]
    assert [member.value for member in Cdpm2DamageFormulation] == ["two_damage_variables"]


def test_valid_fully_resolved_sample_has_exact_20_fields_and_complete_provenance():
    sample = _sample()
    assert sample.model_id == "cdpm2_grassl_2013"
    assert sample.calibration_id == "grassl_2013_static_default_v1"
    assert tuple(sample.values_dict()) == CDPM2_PARAMETER_FIELDS
    assert len(sample.values_dict()) == 20
    assert tuple(sample.provenance_dict()) == CDPM2_PARAMETER_FIELDS
    assert set(sample.provenance) == set(sample.values_dict())


def test_parameter_serialization_is_deterministic_and_backend_agnostic():
    sample = _sample()
    assert sample.to_dict()["schema_version"] == "cdpm2_grassl_2013_parameters.v1"
    assert sample.to_json() == sample.to_json()
    payload = sample.to_dict()
    assert set(payload) == {"schema_version", "model_id", "calibration_id", "values", "provenance"}
    serialized = json.dumps(payload)
    for forbidden in (
        "LCHAR",
        "ERATETYPE",
        "SRATETYPE",
        "FAILFLG",
        "DAMAGEFLAG",
        "PRINTFLAG",
        "MAX_SUBSTEP_DEPTH",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("E", 0.0, "E must be > 0"),
        ("nu", -1e-6, "0 <= nu < 0.5"),
        ("nu", 0.5, "0 <= nu < 0.5"),
        ("f_t", 0.0, "f_t must be > 0"),
        ("f_c", 0.0, "f_c must be > 0"),
        ("G_Ft", 0.0, "G_Ft must be > 0"),
        ("w_f", 0.0, "w_f must be > 0"),
        ("w_f1", 0.0, "w_f1 must be > 0"),
        ("f_t1", 0.0, "f_t1 must be > 0"),
        ("eccentricity", 0.0, "eccentricity must be > 0"),
        ("q_h0", 0.0, "0 < q_h0 <= 1"),
        ("q_h0", 1.01, "0 < q_h0 <= 1"),
        ("H_p", -1e-9, "H_p must be >= 0"),
        ("D_f", 0.5, "D_f must be > 0.5"),
        ("A_h", 0.0, "A_h must be > 0"),
        ("B_h", 0.0, "B_h must be > 0"),
        ("C_h", 0.0, "C_h must be > 0"),
        ("D_h", -1e-9, "D_h must be >= 0"),
        ("A_s", 0.0, "A_s must be > 0"),
        ("epsilon_fc", 0.0, "epsilon_fc must be > 0"),
    ],
)
def test_numeric_domain_failures(field: str, value: float, message: str):
    with pytest.raises(ValueError, match=message):
        _sample(**{field: value})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_parameters_are_rejected(bad: float):
    with pytest.raises(ValueError, match="E must be finite"):
        _sample(E=bad)


def test_bool_and_non_numeric_parameters_are_rejected():
    with pytest.raises(TypeError, match="E must be a finite numeric value"):
        _sample(E=True)
    with pytest.raises(TypeError, match="E must be a finite numeric value"):
        _sample(E="30000")


@pytest.mark.parametrize(
    ("field", "factor", "message"),
    [
        ("w_f", 1.01, "w_f must equal"),
        ("w_f1", 1.01, "w_f1 must equal"),
        ("f_t1", 1.01, "f_t1 must equal"),
    ],
)
def test_bilinear_invariant_mismatches_fail_loudly(field: str, factor: float, message: str):
    sample = _sample()
    with pytest.raises(ValueError, match=message):
        replace(sample, **{field: getattr(sample, field) * factor})


def test_fixed_enum_types_are_strict():
    sample = _sample()
    with pytest.raises(TypeError, match="tensile_softening_type"):
        replace(sample, tensile_softening_type="bilinear")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="damage_formulation"):
        replace(sample, damage_formulation="two_damage_variables")  # type: ignore[arg-type]


def test_provenance_record_validation_and_override_semantics():
    with pytest.raises(ValueError, match="source_id"):
        Cdpm2ParameterProvenance(
            source_id=" ",
            source_kind=Cdpm2SourceKind.PHYSICAL_SOURCE,
            edition=None,
            equation_or_section=None,
            units="MPa",
        )
    with pytest.raises(ValueError, match="units"):
        Cdpm2ParameterProvenance(
            source_id="x",
            source_kind=Cdpm2SourceKind.PHYSICAL_SOURCE,
            edition=None,
            equation_or_section=None,
            units=" ",
        )
    with pytest.raises(ValueError, match="unique"):
        Cdpm2ParameterProvenance(
            source_id="x",
            source_kind=Cdpm2SourceKind.GRASSL_2013_DERIVED,
            edition="2013",
            equation_or_section="x",
            units="mm",
            derived_from=("f_t", "f_t"),
        )
    with pytest.raises(ValueError, match="requires derived_from"):
        Cdpm2ParameterProvenance(
            source_id="x",
            source_kind=Cdpm2SourceKind.GRASSL_2013_DERIVED,
            edition="2013",
            equation_or_section="x",
            units="mm",
        )
    with pytest.raises(ValueError, match="requires overridden=True"):
        Cdpm2ParameterProvenance(
            source_id="user",
            source_kind=Cdpm2SourceKind.USER_CONSTITUTIVE_OVERRIDE,
            edition=None,
            equation_or_section=None,
            units="MPa",
        )
    with pytest.raises(ValueError, match="reserved for USER_CONSTITUTIVE_OVERRIDE"):
        Cdpm2ParameterProvenance(
            source_id="physical",
            source_kind=Cdpm2SourceKind.PHYSICAL_SOURCE,
            edition=None,
            equation_or_section=None,
            units="MPa",
            overridden=True,
        )


def test_parameter_provenance_requires_exact_fields_and_units():
    sample = _sample()
    missing = dict(sample.provenance)
    missing.pop("E")
    with pytest.raises(ValueError, match="missing=E"):
        replace(sample, provenance=missing)

    extra = dict(sample.provenance)
    extra["LCHAR"] = extra["E"]
    with pytest.raises(ValueError, match="extra=LCHAR"):
        replace(sample, provenance=extra)

    wrong_units = dict(sample.provenance)
    wrong_units["E"] = replace(wrong_units["E"], units="Pa")
    with pytest.raises(ValueError, match="Provenance units for E"):
        replace(sample, provenance=wrong_units)


def test_static_derived_provenance_dependencies_are_exact_and_not_overridable():
    sample = _sample()
    provenance = dict(sample.provenance)
    provenance["w_f"] = replace(provenance["w_f"], derived_from=("G_Ft",))
    with pytest.raises(ValueError, match="w_f derived_from"):
        replace(sample, provenance=provenance)

    provenance = dict(sample.provenance)
    provenance["f_t1"] = Cdpm2ParameterProvenance(
        source_id="user",
        source_kind=Cdpm2SourceKind.USER_CONSTITUTIVE_OVERRIDE,
        edition=None,
        equation_or_section=None,
        units="MPa",
        overridden=True,
    )
    with pytest.raises(ValueError, match="f_t1 must use GRASSL_2013_DERIVED"):
        replace(sample, provenance=provenance)


def test_fixed_policy_provenance_is_model_level_not_adapter_fixed():
    sample = _sample()
    provenance = dict(sample.provenance)
    provenance["damage_formulation"] = Cdpm2ParameterProvenance(
        source_id="downstream",
        source_kind=Cdpm2SourceKind.DOWNSTREAM_ADAPTER_FIXED,
        edition=None,
        equation_or_section="DAMAGEFLAG=0",
        units="enum",
    )
    with pytest.raises(ValueError, match="GRASSL_2013_DIRECT"):
        replace(sample, provenance=provenance)


def test_default_configuration_is_empty_and_deterministic():
    config = Cdpm2Grassl2013Configuration()
    assert config.overrides_dict() == {}
    assert config.to_dict() == {
        "schema_version": "cdpm2_grassl_2013_configuration.v1",
        "model_id": "cdpm2_grassl_2013",
        "calibration_id": "grassl_2013_static_default_v1",
        "overrides": {},
    }
    assert config.to_json() == config.to_json()


VALID_OVERRIDE_VALUES = {
    "E": 31000.0,
    "nu": 0.21,
    "f_t": 3.2,
    "f_c": 32.0,
    "G_Ft": 0.12,
    "eccentricity": 0.53,
    "q_h0": 0.31,
    "H_p": 0.02,
    "D_f": 0.9,
    "A_h": 0.09,
    "B_h": 0.004,
    "C_h": 2.1,
    "D_h": 2e-6,
    "A_s": 15.0,
    "epsilon_fc": 2e-4,
}


@pytest.mark.parametrize("field", CDPM2_OVERRIDE_FIELDS)
def test_each_frozen_override_key_is_accepted(field: str):
    config = Cdpm2Grassl2013Configuration({field: VALID_OVERRIDE_VALUES[field]})
    assert config.overrides_dict() == {field: VALID_OVERRIDE_VALUES[field]}


@pytest.mark.parametrize(
    "field",
    [
        "w_f",
        "w_f1",
        "f_t1",
        "tensile_softening_type",
        "damage_formulation",
        "B_s",
        "LCHAR",
        "helem",
        "ERATETYPE",
        "SRATETYPE",
        "FAILFLG",
        "DAMAGEFLAG",
        "PRINTFLAG",
        "yieldtol",
        "newtoniter",
        "FD_REL_STEP",
        "FD_ABS_STEP",
        "MAX_SUBSTEP_DEPTH",
        "history",
        "state",
    ],
)
def test_derived_fixed_runtime_backend_and_solver_overrides_are_rejected(field: str):
    with pytest.raises(ValueError, match="Unsupported CDPM2 static-V1 override"):
        Cdpm2Grassl2013Configuration({field: 1.0})


def test_configuration_rejects_bad_values_and_unknown_calibration():
    with pytest.raises(TypeError, match="E must be a finite numeric value"):
        Cdpm2Grassl2013Configuration({"E": True})
    with pytest.raises(ValueError, match="G_Ft must be finite"):
        Cdpm2Grassl2013Configuration({"G_Ft": math.inf})
    with pytest.raises(ValueError, match="nu must satisfy"):
        Cdpm2Grassl2013Configuration({"nu": 0.5})
    with pytest.raises(ValueError, match="Unknown CDPM2 calibration"):
        Cdpm2Grassl2013Configuration(calibration_id="fib_mc2010")


def test_configuration_preserves_explicit_default_equal_override_and_sorts_keys():
    config = Cdpm2Grassl2013Configuration({"q_h0": 0.3, "A_s": 15.0, "H_p": 0.01})
    assert list(config.to_dict()["overrides"]) == ["A_s", "H_p", "q_h0"]
    assert config.to_dict()["overrides"]["A_s"] == 15.0
    assert config.to_json() == config.to_json()


def test_readiness_vocabulary_and_assessment_invariants():
    assert [state.value for state in Cdpm2ConversionReadiness] == [
        "READY",
        "UNRESOLVED_PHYSICAL_INPUT",
        "COMPOSITION_REQUIRED",
        "NOT_AUTHORIZED_PHYSICAL_SOURCE",
        "UNSUPPORTED_CONFIGURATION",
    ]
    ready = Cdpm2ReadinessAssessment(Cdpm2ConversionReadiness.READY)
    assert ready.to_dict()["blockers"] == []
    assert ready.to_json() == ready.to_json()

    blocked = Cdpm2ReadinessAssessment(
        Cdpm2ConversionReadiness.COMPOSITION_REQUIRED,
        ("G_Ft unavailable",),
    )
    assert blocked.to_dict()["blockers"] == ["G_Ft unavailable"]

    with pytest.raises(ValueError, match="READY assessment cannot contain blockers"):
        Cdpm2ReadinessAssessment(Cdpm2ConversionReadiness.READY, ("unexpected",))
    with pytest.raises(ValueError, match="non-READY assessment requires"):
        Cdpm2ReadinessAssessment(Cdpm2ConversionReadiness.COMPOSITION_REQUIRED)


def test_schema_dataclass_contains_no_runtime_backend_or_state_fields():
    semantic_fields = {field.name for field in fields(Cdpm2Grassl2013Parameters)}
    configuration_fields = {field.name for field in fields(Cdpm2Grassl2013Configuration)}
    forbidden = {
        "B_s",
        "LCHAR",
        "helem",
        "characteristic_length",
        "ERATETYPE",
        "SRATETYPE",
        "FAILFLG",
        "DAMAGEFLAG",
        "PRINTFLAG",
        "yieldtol",
        "newtoniter",
        "FD_REL_STEP",
        "FD_ABS_STEP",
        "MAX_SUBSTEP_DEPTH",
        "history",
        "state",
        "kappaP",
        "omega_t",
        "omega_c",
    }
    assert semantic_fields.isdisjoint(forbidden)
    assert configuration_fields.isdisjoint(forbidden)
    assert set(CDPM2_OVERRIDE_FIELDS).isdisjoint(forbidden)


def test_no_backend_slot_serializer_or_mechanics_surface_is_exposed():
    sample = _sample()
    for name in (
        "to_cm24",
        "to_props",
        "backend_slots",
        "stress_update",
        "integrate_increment",
        "return_mapping",
        "tangent",
    ):
        assert not hasattr(sample, name)


def test_high_level_cdpm2_seam_remains_closed_for_all_verified_profiles():
    examples = {
        "fib_mc2010": "C30",
        "ec2_2004": "C30/37",
        "ec2_2023": "C30/37",
    }
    for profile, concrete_class in examples.items():
        concrete = Concrete.from_class(concrete_class, profile=profile)
        with pytest.raises(NotImplementedError, match="G2"):
            concrete.to_cdpm2(calibration="cdpm2_grassl_2013")


def test_parameter_object_rejects_unknown_calibration_identity():
    with pytest.raises(ValueError, match="Unknown CDPM2 calibration"):
        replace(_sample(), calibration_id="ec2_2023")
