import json
from unittest.mock import patch

import pytest

from cdp_generator.application import (
    AuthorityConcreteRequest,
    AuthorityInputError,
    Cdpm2ConversionRequest,
    cdpm2_override_fields,
    cdpm2_parameter_units,
    parse_cdpm2_overrides,
    run_cdpm2_conversion,
)
from cdp_generator.concrete import Concrete, ProfileConfiguration
from cdp_generator.concrete.models.cdpm2 import (
    CDPM2_LEGACY_SLOT_NAMES,
    CDPM2_OVERRIDE_FIELDS,
    CDPM2_PARAMETER_UNITS,
    Cdpm2Grassl2013Configuration,
    adapt_cdpm2_legacy_backend,
)

MATRIX = [
    ("A", "fib_mc2010", "C30", {}, {}, "READY"),
    ("B", "ec2_2004", "C30/37", {}, {}, "COMPOSITION_REQUIRED"),
    ("C", "ec2_2004", "C30/37", {}, {"G_Ft": 0.15}, "READY"),
    ("D", "ec2_2023", "C30/37", {"reference_age_days": 28}, {}, "COMPOSITION_REQUIRED"),
    ("E", "ec2_2023", "C30/37", {"reference_age_days": 56}, {}, "UNRESOLVED_PHYSICAL_INPUT"),
    (
        "F",
        "ec2_2023",
        "C30/37",
        {"reference_age_days": 56},
        {"G_Ft": 0.15},
        "UNRESOLVED_PHYSICAL_INPUT",
    ),
    ("G", "ec2_2023", "C30/37", {"reference_age_days": 56}, {"E": 41000}, "COMPOSITION_REQUIRED"),
    ("H", "ec2_2023", "C30/37", {"reference_age_days": 56}, {"E": 41000, "G_Ft": 0.15}, "READY"),
]


@pytest.mark.parametrize(
    "case,profile,cls,physical,overrides,state", MATRIX, ids=[row[0] for row in MATRIX]
)
def test_readiness_semantic_backend_parity(case, profile, cls, physical, overrides, state):
    material = AuthorityConcreteRequest(profile, cls, physical)
    domain = Concrete.from_class(
        cls, profile=profile, profile_parameters=ProfileConfiguration(physical)
    )
    config = Cdpm2Grassl2013Configuration(overrides=overrides)
    request = Cdpm2ConversionRequest(material, overrides, 12.5)
    with patch.object(Concrete, "to_cdpm2", autospec=True, side_effect=Concrete.to_cdpm2) as spy:
        result = run_cdpm2_conversion(request)
        assert spy.call_count == (1 if state == "READY" else 0)
    assert result.readiness == domain.cdpm2_readiness(configuration=config).to_dict()
    assert result.readiness["state"] == state
    assert result.configuration == config.to_dict()
    if case in {"B", "D", "G"}:
        assert result.readiness["blockers"] == ["G_Ft composition required"]
    if case == "E":
        assert result.readiness["blockers"] == [
            "E_initial unresolved; E_secant fallback forbidden",
            "G_Ft composition required",
        ]
    if case == "F":
        assert result.readiness["blockers"] == ["E_initial unresolved; E_secant fallback forbidden"]
    if state == "READY":
        parameters = domain.to_cdpm2(configuration=config)
        assert result.semantic_parameters == parameters.to_dict()
        assert len(result.semantic_parameters["values"]) == 20
        assert (
            result.backend
            == adapt_cdpm2_legacy_backend(parameters, characteristic_length=12.5).to_dict()
        )
        assert len(result.backend["cm"]) == 24
        assert result.backend_slot_names == list(CDPM2_LEGACY_SLOT_NAMES)
        assert result.backend["characteristic_length"] == 12.5
        assert "characteristic_length" not in result.semantic_parameters["values"]
        assert "LCHAR" not in result.material.material_definition["physical"]["values"]
        for name in overrides:
            provenance = result.semantic_parameters["provenance"][name]
            assert provenance["source_kind"] == "USER_CONSTITUTIVE_OVERRIDE"
            assert provenance["overridden"]
        for name in ("w_f", "w_f1", "f_t1"):
            assert (
                result.semantic_parameters["provenance"][name]["source_kind"]
                == "GRASSL_2013_DERIVED"
            )
        assert "backend_json" in result.export_capabilities
    else:
        assert result.semantic_parameters is None
        assert result.backend is None
        assert result.export_capabilities == ["material_json", "application_json"]
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False)) == result.to_dict()
    assert result.to_json() == run_cdpm2_conversion(request).to_json()
    assert result.schema_version == "cdpm2_conversion_result.v1"


def test_semantics_without_runtime_context():
    result = run_cdpm2_conversion(Cdpm2ConversionRequest(AuthorityConcreteRequest()))
    assert result.semantic_parameters is not None
    assert result.backend is None
    assert "backend_json" not in result.export_capabilities


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "bad"])
def test_invalid_lchar(value):
    with pytest.raises(AuthorityInputError, match="LCHAR"):
        Cdpm2ConversionRequest(AuthorityConcreteRequest(), characteristic_length_mm=value)


@pytest.mark.parametrize(
    "text",
    ["bad", "[]", '{"E":true}', '{"E":"41000"}', '{"unknown":1}', '{"E":NaN}', '{"G_Ft":-1}'],
)
def test_invalid_override_json(text):
    with pytest.raises(AuthorityInputError):
        parse_cdpm2_overrides(text)


def test_override_catalog_parser_and_immutability():
    assert cdpm2_override_fields() == CDPM2_OVERRIDE_FIELDS
    assert cdpm2_parameter_units() == CDPM2_PARAMETER_UNITS
    overrides = parse_cdpm2_overrides('{"q_h0":0.31,"A_s":16}', {"E": 41000})
    request = Cdpm2ConversionRequest(AuthorityConcreteRequest(), overrides)
    overrides["E"] = 30000
    assert request.overrides["E"] == 41000
    with pytest.raises(AuthorityInputError, match="twice"):
        parse_cdpm2_overrides('{"E":41000}', {"E": 30000})


def test_modern_workflow_never_calls_legacy_curve_kernel():
    with (
        patch("cdp_generator.core.calculate_stress_strain", side_effect=AssertionError("legacy")),
        patch(
            "cdp_generator.core.calculate_stress_strain_temp", side_effect=AssertionError("legacy")
        ),
    ):
        result = run_cdpm2_conversion(
            Cdpm2ConversionRequest(AuthorityConcreteRequest(), characteristic_length_mm=1)
        )
    assert result.readiness["state"] == "READY"
