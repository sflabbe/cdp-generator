"""G0 tests for the authority-split concrete architecture."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from cdp_generator import (
    calculate_cdp_parameters,
    calculate_characteristic_length,
    calculate_concrete_strength_properties,
    calculate_elastic_modulus,
    calculate_fracture_energy,
    calculate_poisson_ratios,
    calculate_stress_strain,
    calculate_stress_strain_temp,
)
from cdp_generator.concrete import (
    LEGACY_ABAQUS_CALIBRATION,
    LEGACY_PHYSICAL_PROFILE,
    Concrete,
    SourceKind,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE_F_CMS = [20.0, 28.0, 38.0, 58.0, 68.0, 98.0]
E_C1 = 0.0022
E_CLIM = 0.0035


def test_legacy_public_imports_remain_available():
    """G0 must not require downstream users to change legacy imports."""

    for function in (
        calculate_stress_strain,
        calculate_stress_strain_temp,
        calculate_concrete_strength_properties,
        calculate_elastic_modulus,
        calculate_poisson_ratios,
        calculate_cdp_parameters,
        calculate_fracture_energy,
        calculate_characteristic_length,
    ):
        assert callable(function)


def test_console_script_contract_remains_declared():
    """The existing concrete and steel console-script entry points remain unchanged."""

    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'cdp-generator = "cdp_generator.cli:main"' in pyproject
    assert 'cdp-steel = "cdp_generator.steel.cli:main"' in pyproject


def test_legacy_profile_matches_existing_physical_functions_exactly():
    """The new legacy profile is an adapter, not a numerical reimplementation."""

    for f_cm in REPRESENTATIVE_F_CMS:
        concrete = Concrete.from_mean_strength(
            f_cm=f_cm,
            e_c1=E_C1,
            e_clim=E_CLIM,
            profile=LEGACY_PHYSICAL_PROFILE,
        )
        physical = concrete.physical

        strength = calculate_concrete_strength_properties(f_cm)
        elastic = calculate_elastic_modulus(f_cm)
        poisson = calculate_poisson_ratios(f_cm, elastic["E_c"], E_C1)
        fracture_energy = calculate_fracture_energy(f_cm)

        assert physical.f_cm == f_cm
        assert physical.f_ck == strength["f_ck"]
        assert physical.f_ctm == strength["f_ctm"]
        assert physical.E_initial == elastic["E_ci"]
        assert physical.E_secant == elastic["E_c"]
        assert physical.poisson_elastic == poisson["v_ce"]
        assert physical.shear_modulus == elastic["E_c"] / (2 * (1 + poisson["v_ce"]))
        assert physical.fracture_energy == fracture_energy
        assert physical.strain_peak_compression == E_C1
        assert physical.strain_limit_compression == E_CLIM


def test_legacy_abaqus_backend_matches_existing_cdp_function_exactly():
    """ABAQUS-CDP parameters are separated while retaining exact legacy values."""

    for f_cm in REPRESENTATIVE_F_CMS:
        concrete = Concrete.from_mean_strength(f_cm, E_C1, E_CLIM)
        physical = concrete.physical
        poisson = calculate_poisson_ratios(f_cm, physical.E_secant, E_C1)
        legacy = calculate_cdp_parameters(
            f_cm,
            physical.E_secant,
            E_C1,
            poisson["v_c0"],
            poisson["v_ce"],
        )
        abaqus = concrete.to_abaqus_cdp(calibration=LEGACY_ABAQUS_CALIBRATION)

        assert abaqus.dilation_angle == legacy["dilation_angle"]
        assert abaqus.fbfc == legacy["fbfc"]
        assert abaqus.Kc == legacy["K_c"]


def test_physical_schema_has_provenance_for_every_legacy_generated_field():
    concrete = Concrete.from_mean_strength(38.0, E_C1, E_CLIM)
    physical = concrete.physical

    legacy_generated = (
        "f_cm",
        "f_ck",
        "f_ctm",
        "E_initial",
        "E_secant",
        "poisson_elastic",
        "shear_modulus_secant_equivalent",
        "fracture_energy",
        "strain_peak_compression",
        "strain_limit_compression",
    )
    for field in legacy_generated:
        assert physical.values_dict()[field] is not None
        assert field in physical.provenance
        provenance = physical.provenance[field]
        assert provenance.source_id == "legacy_v1"
        assert provenance.source_kind is not SourceKind.STANDARD
        assert provenance.units

    assert physical.f_ctk_lower is None
    assert physical.f_ctk_upper is None
    assert physical.reference_age_days is None


def test_abaqus_schema_has_provenance_and_is_not_generic_physical_data():
    concrete = Concrete.from_mean_strength(38.0, E_C1, E_CLIM)
    abaqus = concrete.to_abaqus_cdp()

    for field in ("dilation_angle", "fbfc", "Kc"):
        provenance = abaqus.provenance[field]
        assert provenance.source_id == "legacy_v1"
        assert provenance.source_kind is SourceKind.LEGACY_IMPLEMENTATION

    assert not hasattr(concrete.physical, "dilation_angle")
    assert not hasattr(concrete.physical, "fbfc")
    assert not hasattr(concrete.physical, "Kc")


def test_characteristic_length_is_not_intrinsic_physical_property():
    physical = Concrete.from_mean_strength(38.0, E_C1, E_CLIM).physical
    values = physical.values_dict()

    assert "l_ch" not in values
    assert "l0" not in values
    assert "characteristic_length" not in values


def test_new_schemas_are_immutable():
    concrete = Concrete.from_mean_strength(38.0, E_C1, E_CLIM)
    physical = concrete.physical
    abaqus = concrete.to_abaqus_cdp()

    with pytest.raises(FrozenInstanceError):
        physical.f_cm = 40.0
    with pytest.raises(TypeError):
        physical.provenance["f_cm"] = physical.provenance["f_cm"]
    with pytest.raises(FrozenInstanceError):
        abaqus.Kc = 1.0
    with pytest.raises(TypeError):
        abaqus.provenance["Kc"] = abaqus.provenance["Kc"]


def test_serialization_is_deterministic_and_keeps_domains_separate():
    concrete = Concrete.from_mean_strength(38.0, E_C1, E_CLIM)
    abaqus = concrete.to_abaqus_cdp()

    first = concrete.to_json(abaqus_cdp=abaqus)
    second = concrete.to_json(abaqus_cdp=abaqus)
    assert first == second

    payload = json.loads(first)
    assert payload["schema_version"] == "concrete_material_definition.v1"
    assert payload["physical_profile"] == "legacy_v1"
    assert payload["physical"]["f_cm"] == 38.0
    assert "dilation_angle" not in payload["physical"]
    assert payload["constitutive_models"]["abaqus_cdp"]["dilation_angle"] == abaqus.dilation_angle
    assert payload["provenance"]["physical"]["f_cm"]["source_id"] == "legacy_v1"


def test_future_standard_and_cdpm2_profiles_are_reserved_not_implemented():
    for profile in ("fib_mc2010", "ec2_2023", "ec2_2004"):
        with pytest.raises(NotImplementedError, match="G1"):
            Concrete.from_mean_strength(38.0, E_C1, E_CLIM, profile=profile)

    with pytest.raises(NotImplementedError, match="G1"):
        Concrete.from_class("C30/37")

    concrete = Concrete.from_mean_strength(38.0, E_C1, E_CLIM)
    with pytest.raises(NotImplementedError, match="G2"):
        concrete.to_cdpm2()
