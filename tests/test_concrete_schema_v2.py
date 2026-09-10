"""G1-B1 tests for truthful physical-schema v2 representation."""

import json
from dataclasses import FrozenInstanceError

import pytest

from cdp_generator.concrete import (
    Concrete,
    ConcretePhysicalProperties,
    NormalizationKind,
    ProfileConfiguration,
    PropertyNormalization,
    PropertyProvenance,
    PropertyResolutionStatus,
    SourceKind,
    StatisticalBasis,
)


def provenance(
    field: str,
    units: str,
    *,
    source_kind: SourceKind = SourceKind.STANDARD,
    derived_from: tuple[str, ...] = (),
    normalizations: tuple[PropertyNormalization, ...] = (),
) -> PropertyProvenance:
    return PropertyProvenance(
        source_id="test_standard",
        source_kind=source_kind,
        edition="test-edition",
        equation_or_section=f"test:{field}",
        units=units,
        statistical_basis=StatisticalBasis.NOT_APPLICABLE,
        derived_from=derived_from,
        normalizations=normalizations,
    )


def synthetic_v2() -> ConcretePhysicalProperties:
    sign_normalization = PropertyNormalization(
        kind=NormalizationKind.SIGN_CONVENTION,
        source_convention="compression negative",
        repository_convention="compression positive magnitude",
    )
    prov = {
        "f_cm": provenance("f_cm", "MPa"),
        "f_ck": provenance("f_ck", "MPa"),
        "f_ctm": provenance("f_ctm", "MPa"),
        "f_ctk_lower": provenance("f_ctk_lower", "MPa"),
        "f_ctk_upper": provenance("f_ctk_upper", "MPa"),
        "E_secant": provenance("E_secant", "MPa"),
        "poisson_elastic": provenance("poisson_elastic", "dimensionless"),
        "shear_modulus_secant_equivalent": provenance(
            "shear_modulus_secant_equivalent",
            "MPa",
            source_kind=SourceKind.DERIVED,
            derived_from=("E_secant", "poisson_elastic"),
        ),
        "strain_peak_compression": provenance(
            "strain_peak_compression",
            "dimensionless",
            source_kind=SourceKind.DERIVED,
            derived_from=("source:epsilon_c1",),
            normalizations=(sign_normalization,),
        ),
        "strain_limit_compression": provenance(
            "strain_limit_compression",
            "dimensionless",
            source_kind=SourceKind.DERIVED,
            derived_from=("source:epsilon_cu1",),
            normalizations=(sign_normalization,),
        ),
        "reference_age_days": provenance("reference_age_days", "days"),
    }
    resolution = {
        "f_cm": PropertyResolutionStatus.DIRECT,
        "f_ck": PropertyResolutionStatus.DIRECT,
        "f_ctm": PropertyResolutionStatus.DIRECT,
        "f_ctk_lower": PropertyResolutionStatus.DIRECT,
        "f_ctk_upper": PropertyResolutionStatus.DIRECT,
        "E_initial": PropertyResolutionStatus.UNRESOLVED,
        "E_secant": PropertyResolutionStatus.DIRECT,
        "poisson_elastic": PropertyResolutionStatus.DIRECT,
        "shear_modulus_secant_equivalent": PropertyResolutionStatus.DERIVED,
        "fracture_energy": PropertyResolutionStatus.COMPOSED_REQUIRED,
        "strain_peak_compression": PropertyResolutionStatus.DERIVED,
        "strain_limit_compression": PropertyResolutionStatus.DERIVED,
        "reference_age_days": PropertyResolutionStatus.DIRECT,
    }
    return ConcretePhysicalProperties(
        f_cm=38.0,
        f_ck=30.0,
        f_ctm=2.9,
        f_ctk_lower=2.0,
        f_ctk_upper=3.8,
        E_initial=None,
        E_secant=33000.0,
        poisson_elastic=0.2,
        shear_modulus_secant_equivalent=13750.0,
        fracture_energy=None,
        strain_peak_compression=0.0022,
        strain_limit_compression=0.0035,
        reference_age_days=28.0,
        provenance=prov,
        resolution=resolution,
    )


def test_v2_schema_has_canonical_fields_and_truthful_absence():
    physical = synthetic_v2()
    values = physical.values_dict()

    assert tuple(values) == (
        "f_cm",
        "f_ck",
        "f_ctm",
        "f_ctk_lower",
        "f_ctk_upper",
        "E_initial",
        "E_secant",
        "poisson_elastic",
        "shear_modulus_secant_equivalent",
        "fracture_energy",
        "strain_peak_compression",
        "strain_limit_compression",
        "reference_age_days",
    )
    assert values["E_initial"] is None
    assert physical.resolution["E_initial"] is PropertyResolutionStatus.UNRESOLVED
    assert values["fracture_energy"] is None
    assert physical.resolution["fracture_energy"] is PropertyResolutionStatus.COMPOSED_REQUIRED
    assert "E_initial" not in physical.provenance
    assert "fracture_energy" not in physical.provenance


def test_resolution_status_requires_consistent_value_and_source_kind():
    physical = synthetic_v2()
    kwargs = physical.values_dict()

    with pytest.raises(ValueError, match="Populated physical field f_cm"):
        ConcretePhysicalProperties(
            **kwargs,
            provenance=physical.provenance,
            resolution={**physical.resolution, "f_cm": PropertyResolutionStatus.UNRESOLVED},
        )

    bad_provenance = dict(physical.provenance)
    bad_provenance["shear_modulus_secant_equivalent"] = provenance(
        "shear_modulus_secant_equivalent",
        "MPa",
        source_kind=SourceKind.STANDARD,
    )
    with pytest.raises(ValueError, match=r"must use SourceKind.DERIVED"):
        ConcretePhysicalProperties(
            **kwargs,
            provenance=bad_provenance,
            resolution=physical.resolution,
        )


def test_provenance_dependencies_and_normalizations_are_immutable_and_serialized():
    physical = synthetic_v2()
    shear = physical.provenance["shear_modulus_secant_equivalent"]
    strain = physical.provenance["strain_peak_compression"]

    assert shear.derived_from == ("E_secant", "poisson_elastic")
    assert strain.normalizations[0].kind is NormalizationKind.SIGN_CONVENTION
    payload = strain.to_dict()
    assert payload["normalizations"] == [
        {
            "kind": "SIGN_CONVENTION",
            "source_convention": "compression negative",
            "repository_convention": "compression positive magnitude",
        }
    ]

    with pytest.raises(FrozenInstanceError):
        shear.notes = "changed"
    with pytest.raises(FrozenInstanceError):
        strain.normalizations[0].source_convention = "changed"


def test_compression_landmarks_enforce_positive_magnitude():
    physical = synthetic_v2()
    kwargs = physical.values_dict()
    kwargs["strain_peak_compression"] = -0.0022
    with pytest.raises(ValueError, match="positive-magnitude"):
        ConcretePhysicalProperties(
            **kwargs,
            provenance=physical.provenance,
            resolution=physical.resolution,
        )


def test_profile_configuration_is_json_scalar_only_immutable_and_deterministic():
    config = ProfileConfiguration({"reference_age_days": 28, "aggregate_type": "quartzite"})
    assert config.to_dict() == {"aggregate_type": "quartzite", "reference_age_days": 28}

    with pytest.raises(TypeError):
        config.parameters["aggregate_type"] = "basalt"
    with pytest.raises(TypeError, match="JSON scalars"):
        ProfileConfiguration({"bad": [1, 2, 3]})  # type: ignore[dict-item]


def test_v2_serialization_identifies_schema_and_profile_configuration():
    concrete = Concrete(
        physical_profile="test_standard",
        physical=synthetic_v2(),
        profile_parameters=ProfileConfiguration({"reference_age_days": 28, "kE": 9500}),
    )
    first = concrete.to_json()
    second = concrete.to_json()
    assert first == second

    payload = json.loads(first)
    assert payload["schema_version"] == "concrete_material_definition.v2"
    assert payload["physical"]["schema_version"] == "concrete_physical_properties.v2"
    assert payload["physical"]["resolution"]["E_initial"] == "UNRESOLVED"
    assert payload["physical"]["resolution"]["fracture_energy"] == "COMPOSED_REQUIRED"
    assert payload["profile_parameters"] == {"kE": 9500, "reference_age_days": 28}


def test_legacy_adapter_keeps_v1_payload_shape_and_exposes_v2_explicitly():
    concrete = Concrete.from_mean_strength(38.0, 0.0022, 0.0035)

    legacy = concrete.to_dict()
    assert legacy["schema_version"] == "concrete_material_definition.v1"
    assert tuple(legacy["physical"]) == (
        "f_cm",
        "f_ck",
        "f_ctm",
        "E_initial",
        "E_secant",
        "poisson_elastic",
        "shear_modulus",
        "fracture_energy",
        "strain_peak_compression",
        "strain_limit_compression",
    )
    assert "resolution" not in legacy
    assert "profile_parameters" not in legacy
    assert set(legacy["provenance"]["physical"]["shear_modulus"]) == {
        "source_id",
        "source_kind",
        "edition",
        "equation_or_section",
        "units",
        "statistical_basis",
        "notes",
        "overridden",
    }

    physical = concrete.physical
    assert physical.shear_modulus == physical.shear_modulus_secant_equivalent
    assert physical.f_ctk_lower is None
    assert physical.resolution["f_ctk_lower"] is PropertyResolutionStatus.UNRESOLVED
    assert physical.f_ctk_upper is None
    assert physical.resolution["f_ctk_upper"] is PropertyResolutionStatus.UNRESOLVED
    assert physical.reference_age_days is None
    assert physical.resolution["reference_age_days"] is PropertyResolutionStatus.UNRESOLVED
    assert physical.provenance["shear_modulus_secant_equivalent"].derived_from == (
        "E_secant",
        "poisson_elastic",
    )

    v2 = concrete.to_dict(schema_version="v2")
    assert v2["schema_version"] == "concrete_material_definition.v2"
    assert "shear_modulus_secant_equivalent" in v2["physical"]["values"]
    assert "shear_modulus" not in v2["physical"]["values"]
    assert v2["profile_parameters"] == {}


def test_abaqus_cdp_v1_serialization_does_not_gain_v2_provenance_keys():
    abaqus = Concrete.from_mean_strength(38.0, 0.0022, 0.0035).to_abaqus_cdp()
    payload = abaqus.to_dict()
    assert payload["schema_version"] == "abaqus_cdp_parameters.v1"
    assert "derived_from" not in payload["provenance"]["Kc"]
    assert "normalizations" not in payload["provenance"]["Kc"]


def test_characteristic_length_names_are_absent_from_v2_schema():
    values = synthetic_v2().values_dict()
    for forbidden in ("l_ch", "l0", "characteristic_length", "mesh_size"):
        assert forbidden not in values
