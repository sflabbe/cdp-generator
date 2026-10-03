import json

import pytest

from cdp_generator.application import (
    AuthorityConcreteRequest,
    AuthorityInputError,
    available_concrete_classes,
    available_physical_profiles,
    build_authority_concrete,
    profile_parameter_specs,
)
from cdp_generator.concrete import Concrete, ProfileConfiguration, class_entries
from cdp_generator.concrete.class_registry import SUPPORTED_CLASS_PROFILES
from cdp_generator.concrete.standards import ec2_2004, ec2_2023, fib_mc2010


def test_profile_catalog_is_domain_owned():
    assert available_physical_profiles() == SUPPORTED_CLASS_PROFILES
    assert set(available_physical_profiles()) == {"fib_mc2010", "ec2_2004", "ec2_2023"}
    for profile in available_physical_profiles():
        assert available_concrete_classes(profile) == tuple(
            e.canonical_class_string for e in class_entries(profile)
        )
        specs = {s.id: s for s in profile_parameter_specs(profile)}
        concrete = Concrete.from_class(available_concrete_classes(profile)[0], profile=profile)
        defaults = concrete.profile_parameters.to_dict()
        for name, spec in specs.items():
            assert spec.default == defaults[name]
        if profile == "ec2_2023":
            assert (specs["k_E"].minimum, specs["k_E"].maximum) == (
                ec2_2023.MIN_K_E,
                ec2_2023.MAX_K_E,
            )
            assert (specs["reference_age_days"].minimum, specs["reference_age_days"].maximum) == (
                ec2_2023.MIN_REFERENCE_AGE_DAYS,
                ec2_2023.MAX_REFERENCE_AGE_DAYS,
            )
        else:
            factors = (
                fib_mc2010.AGGREGATE_ALPHA_E
                if profile == "fib_mc2010"
                else ec2_2004.AGGREGATE_ECM_FACTOR
            )
            assert specs["aggregate_type"].options == tuple(factors)
            assert not specs["reference_age_days"].editable


@pytest.mark.parametrize(
    "profile,cls,parameters",
    [
        ("fib_mc2010", "C30", {}),
        ("fib_mc2010", "C60", {"aggregate_type": "sandstone"}),
        ("ec2_2004", "C30/37", {"aggregate_type": "basalt"}),
        ("ec2_2023", "C30/37", {}),
        ("ec2_2023", "C30/37", {"reference_age_days": 56, "k_E": 10500}),
    ],
)
def test_material_values_resolution_provenance_and_configuration(profile, cls, parameters):
    request = AuthorityConcreteRequest(profile, cls, parameters)
    result = build_authority_concrete(request)
    domain = Concrete.from_class(
        cls, profile=profile, profile_parameters=ProfileConfiguration(parameters)
    )
    assert result.material_definition == domain.to_dict(schema_version="v2")
    assert {p.name: p.value for p in result.physical_properties} == domain.physical.values_dict()
    assert {
        p.name: p.resolution for p in result.physical_properties
    } == domain.physical.resolution_dict()
    assert {
        p.name: p.provenance for p in result.physical_properties if p.provenance is not None
    } == domain.physical.provenance_dict()
    for prop in result.physical_properties:
        if prop.value is None:
            assert prop.resolution in {"UNRESOLVED", "COMPOSED_REQUIRED"}
            # The domain may record negative-source evidence explaining why a
            # value is unresolved. Preserve it, without a producing equation.
            if prop.provenance is not None:
                assert prop.provenance["equation_or_section"] is None
            assert prop.provenance == domain.physical.provenance_dict().get(prop.name)
        elif prop.provenance:
            assert prop.unit == prop.provenance["units"]
    assert result.requested_profile_parameters == parameters
    assert result.effective_profile_parameters == domain.profile_parameters.to_dict()
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False)) == result.to_dict()
    assert result.to_json() == build_authority_concrete(request).to_json()
    assert result.schema_version == "authority_concrete_result.v1"


@pytest.mark.parametrize(
    "profile,cls,parameters",
    [
        ("legacy_v1", "C30", {}),
        ("fib_mc2010", "C30/37", {}),
        ("ec2_2023", "C30/37", {"k_E": 4900}),
        ("ec2_2004", "C30/37", {"reference_age_days": 56}),
        ("ec2_2023", "C30/37", {"reference_age_days": "later"}),
    ],
)
def test_invalid_profile_inputs_are_controlled(profile, cls, parameters):
    with pytest.raises(AuthorityInputError):
        build_authority_concrete(AuthorityConcreteRequest(profile, cls, parameters))


def test_material_request_freezes_caller_mapping():
    data = {"aggregate_type": "basalt"}
    request = AuthorityConcreteRequest(profile_parameters=data)
    data["aggregate_type"] = "sandstone"
    assert request.profile_parameters["aggregate_type"] == "basalt"
    with pytest.raises(TypeError):
        request.profile_parameters["aggregate_type"] = "sandstone"
