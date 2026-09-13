"""G2-C qualification for physical-to-semantic CDPM2 mapping and calibration resolution."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from cdp_generator.concrete import (
    Cdpm2ConversionNotReadyError,
    Cdpm2ConversionReadiness,
    Cdpm2FractureEnergyComposition,
    Cdpm2Grassl2013Configuration,
    Concrete,
    ProfileConfiguration,
    PropertyProvenance,
    SourceKind,
    StatisticalBasis,
    class_entries,
)
from cdp_generator.concrete.models.cdpm2 import (
    CDPM2_OVERRIDE_FIELDS,
    Cdpm2DamageFormulation,
    Cdpm2SourceKind,
    Cdpm2TensileSofteningType,
    assess_cdpm2_grassl_2013_readiness,
    resolve_cdpm2_grassl_2013,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads(
    (REPO_ROOT / "qualification" / "cdpm2" / "g2c_mapping_reference.json").read_text()
)


def _case(case_id: str) -> dict[str, object]:
    return next(case for case in REFERENCE["cases"] if case["id"] == case_id)


def _ec23(age: float) -> Concrete:
    return Concrete.from_class(
        "C30/37",
        profile="ec2_2023",
        profile_parameters=ProfileConfiguration({"reference_age_days": age}),
    )


def _fib_composition() -> tuple[Concrete, Cdpm2FractureEnergyComposition]:
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    value = fib.physical.fracture_energy
    assert value is not None
    composition = Cdpm2FractureEnergyComposition(
        value=value,
        provenance=fib.physical.provenance["fracture_energy"],
    )
    return fib, composition


def _assert_values_close(actual: dict[str, float | str], expected: dict[str, object]) -> None:
    assert set(actual) == set(expected)
    for field, expected_value in expected.items():
        actual_value = actual[field]
        if isinstance(expected_value, float):
            assert isinstance(actual_value, float)
            assert actual_value == pytest.approx(expected_value, rel=1e-12, abs=1e-12)
        else:
            assert actual_value == expected_value


def test_independent_reference_fixture_declares_independent_method():
    assert REFERENCE["schema_version"] == "cdpm2_g2c_mapping_reference.v1"
    assert REFERENCE["gate"] == "G2-C"
    method = REFERENCE["evidence"]["method"]
    assert "without importing cdp_generator mapping.py" in method
    assert len(REFERENCE["cases"]) == 10


def test_fib_c30_default_matches_independent_reference():
    concrete = Concrete.from_class("C30", profile="fib_mc2010")
    assessment = concrete.cdpm2_readiness()
    assert assessment.state is Cdpm2ConversionReadiness.READY
    assert assessment.blockers == ()
    params = concrete.to_cdpm2()
    _assert_values_close(params.values_dict(), _case("fib_mc2010_C30_default")["expected_values"])


def test_reference_readiness_matrix_matches_independent_fixture():
    _, composition = _fib_composition()
    matrix = {
        "ec2_2004_C30_37_default": (
            Concrete.from_class("C30/37", profile="ec2_2004"),
            None,
            None,
        ),
        "ec2_2004_C30_37_GFt_override": (
            Concrete.from_class("C30/37", profile="ec2_2004"),
            Cdpm2Grassl2013Configuration({"G_Ft": 0.15}),
            None,
        ),
        "ec2_2004_C30_37_mc2010_composition": (
            Concrete.from_class("C30/37", profile="ec2_2004"),
            None,
            composition,
        ),
        "ec2_2023_C30_37_28_default": (_ec23(28.0), None, None),
        "ec2_2023_C30_37_56_default": (_ec23(56.0), None, None),
        "ec2_2023_C30_37_56_GFt_override_only": (
            _ec23(56.0),
            Cdpm2Grassl2013Configuration({"G_Ft": 0.15}),
            None,
        ),
        "ec2_2023_C30_37_56_E_override_only": (
            _ec23(56.0),
            Cdpm2Grassl2013Configuration({"E": 36000.0}),
            None,
        ),
        "ec2_2023_C30_37_56_E_GFt_overrides": (
            _ec23(56.0),
            Cdpm2Grassl2013Configuration({"E": 36000.0, "G_Ft": 0.15}),
            None,
        ),
    }
    for case_id, (concrete, config, physical_composition) in matrix.items():
        expected = _case(case_id)
        assessment = concrete.cdpm2_readiness(
            configuration=config,
            fracture_energy_composition=physical_composition,
        )
        assert assessment.state.value == expected["readiness"]
        if "blockers" in expected:
            assert list(assessment.blockers) == expected["blockers"]
        if assessment.state is Cdpm2ConversionReadiness.READY:
            params = concrete.to_cdpm2(
                configuration=config,
                fracture_energy_composition=physical_composition,
            )
            _assert_values_close(params.values_dict(), expected["expected_values"])


def test_all_17_fib_classes_are_ready_and_resolve():
    entries = [entry for entry in class_entries() if entry.profile == "fib_mc2010"]
    assert len(entries) == 17
    for entry in entries:
        concrete = Concrete.from_class(entry.canonical_class_string, profile="fib_mc2010")
        assert concrete.cdpm2_readiness().state is Cdpm2ConversionReadiness.READY
        params = concrete.to_cdpm2()
        assert concrete.physical.E_initial == params.E
        assert params.nu == concrete.physical.poisson_elastic
        assert params.f_t == concrete.physical.f_ctm
        assert params.f_c == concrete.physical.f_cm
        assert params.G_Ft == concrete.physical.fracture_energy
        assert params.w_f == pytest.approx(params.G_Ft / (0.225 * params.f_t))
        assert params.eccentricity > 0.0


def test_all_ec2_2004_classes_require_composition_by_default():
    entries = [entry for entry in class_entries() if entry.profile == "ec2_2004"]
    assert len(entries) == 14
    for entry in entries:
        concrete = Concrete.from_class(entry.canonical_class_string, profile="ec2_2004")
        assessment = concrete.cdpm2_readiness()
        assert assessment.state is Cdpm2ConversionReadiness.COMPOSITION_REQUIRED
        assert assessment.blockers == ("G_Ft composition required",)


def test_all_ec2_2023_28_day_classes_require_composition_by_default():
    entries = [entry for entry in class_entries() if entry.profile == "ec2_2023"]
    assert len(entries) == 15
    for entry in entries:
        concrete = Concrete.from_class(entry.canonical_class_string, profile="ec2_2023")
        assessment = concrete.cdpm2_readiness()
        assert assessment.state is Cdpm2ConversionReadiness.COMPOSITION_REQUIRED
        assert assessment.blockers == ("G_Ft composition required",)


@pytest.mark.parametrize("age", [28.0, 56.0, 91.0])
def test_ec2_2023_age_sentinels_never_fall_back_to_e_secant(age: float):
    concrete = _ec23(age)
    if age == 28.0:
        assert concrete.physical.E_initial is not None
        assert concrete.cdpm2_readiness().state is Cdpm2ConversionReadiness.COMPOSITION_REQUIRED
        return

    assert concrete.physical.E_initial is None
    assessment = concrete.cdpm2_readiness()
    assert assessment.state is Cdpm2ConversionReadiness.UNRESOLVED_PHYSICAL_INPUT
    assert assessment.blockers == (
        "E_initial unresolved; E_secant fallback forbidden",
        "G_Ft composition required",
    )

    explicit_e = 41000.0
    config = Cdpm2Grassl2013Configuration({"E": explicit_e, "G_Ft": 0.15})
    params = concrete.to_cdpm2(configuration=config)
    assert explicit_e == params.E
    assert concrete.physical.E_secant != params.E
    assert params.provenance["E"].source_kind is Cdpm2SourceKind.USER_CONSTITUTIVE_OVERRIDE


def test_legacy_is_never_authorized_even_with_full_overrides():
    legacy = Concrete.from_mean_strength(38.0, 0.0022, 0.0035)
    full = {
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
    assessment = legacy.cdpm2_readiness(configuration=Cdpm2Grassl2013Configuration(full))
    assert assessment.state is Cdpm2ConversionReadiness.NOT_AUTHORIZED_PHYSICAL_SOURCE
    assert assessment.blockers == (
        "physical profile is not authorized for verified static CDPM2 mapping",
    )
    with pytest.raises(Cdpm2ConversionNotReadyError) as exc:
        legacy.to_cdpm2(configuration=Cdpm2Grassl2013Configuration(full))
    assert exc.value.assessment == assessment


def test_typed_not_ready_error_preserves_state_and_blockers():
    ec04 = Concrete.from_class("C30/37", profile="ec2_2004")
    with pytest.raises(Cdpm2ConversionNotReadyError) as exc:
        ec04.to_cdpm2()
    assert exc.value.assessment.state is Cdpm2ConversionReadiness.COMPOSITION_REQUIRED
    assert exc.value.assessment.blockers == ("G_Ft composition required",)
    assert "COMPOSITION_REQUIRED" in str(exc.value)


def test_unknown_calibration_fails_loudly_and_model_id_placeholder_is_not_an_alias():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    with pytest.raises(ValueError, match="Unknown CDPM2 calibration"):
        fib.to_cdpm2(calibration="some_other_calibration")
    with pytest.raises(NotImplementedError, match="model-id placeholder"):
        fib.to_cdpm2(calibration="cdpm2_grassl_2013")


def test_standalone_readiness_and_resolver_match_high_level_facade():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    direct_assessment = assess_cdpm2_grassl_2013_readiness(
        physical_profile=fib.physical_profile,
        physical=fib.physical,
    )
    direct = resolve_cdpm2_grassl_2013(
        physical_profile=fib.physical_profile,
        physical=fib.physical,
    )
    assert direct_assessment == fib.cdpm2_readiness()
    assert direct.to_json() == fib.to_cdpm2().to_json()


def test_default_provenance_categories_are_authority_separated():
    params = Concrete.from_class("C30", profile="fib_mc2010").to_cdpm2()
    for field in ("E", "nu", "f_t", "f_c", "G_Ft"):
        assert params.provenance[field].source_kind is Cdpm2SourceKind.PHYSICAL_SOURCE
        assert params.provenance[field].source_id == "fib_mc2010_2013"
    for field in ("w_f", "w_f1", "f_t1", "eccentricity"):
        assert params.provenance[field].source_kind is Cdpm2SourceKind.GRASSL_2013_DERIVED
    for field in ("q_h0", "H_p", "D_f", "A_h", "B_h", "C_h", "D_h"):
        assert (
            params.provenance[field].source_kind is Cdpm2SourceKind.GRASSL_2013_DEFAULT_CALIBRATION
        )
        assert params.provenance[field].source_id == "grassl_et_al_2013_cdpm2"
    for field in ("A_s", "epsilon_fc"):
        assert params.provenance[field].source_kind is Cdpm2SourceKind.OOFEM_IMPLEMENTATION_DEFAULT
        assert params.provenance[field].source_id == "grassl_oofem_cdpm2_manual_2022"
        assert params.provenance[field].edition == "2022-05-18"
    for field in ("tensile_softening_type", "damage_formulation"):
        assert params.provenance[field].source_kind is Cdpm2SourceKind.GRASSL_2013_DIRECT


def test_default_calibration_values_and_exact_authority_locators():
    params = Concrete.from_class("C30", profile="fib_mc2010").to_cdpm2()
    assert {
        "q_h0": params.q_h0,
        "H_p": params.H_p,
        "D_f": params.D_f,
        "A_h": params.A_h,
        "B_h": params.B_h,
        "C_h": params.C_h,
        "D_h": params.D_h,
        "A_s": params.A_s,
        "epsilon_fc": params.epsilon_fc,
    } == {
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
    assert params.provenance["q_h0"].equation_or_section == "Eq. (30); Sec. 5"
    assert params.provenance["H_p"].equation_or_section == "Eqs. (30)-(31); Sec. 5"
    assert params.provenance["D_f"].equation_or_section == "Eqs. (27)-(29); Sec. 5"
    assert params.provenance["A_h"].equation_or_section.startswith("Eq. (33); Sec. 5")


def test_default_eccentricity_uses_final_effective_strengths_and_eq60():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    config = Cdpm2Grassl2013Configuration({"f_t": 3.2, "f_c": 42.0})
    params = fib.to_cdpm2(configuration=config)
    f_bc = 1.16 * params.f_c
    epsilon = (params.f_t / f_bc) * (f_bc**2 - params.f_c**2) / (params.f_c**2 - params.f_t**2)
    expected = (1.0 + epsilon) / (2.0 - epsilon)
    assert params.eccentricity == pytest.approx(expected)
    assert params.provenance["eccentricity"].source_kind is Cdpm2SourceKind.GRASSL_2013_DERIVED
    assert params.provenance["eccentricity"].equation_or_section == "Eq. (60)"
    assert params.provenance["eccentricity"].derived_from == ("f_t", "f_c")


def test_invalid_default_eccentricity_is_unsupported_but_explicit_override_resolves_it():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    invalid = Cdpm2Grassl2013Configuration({"f_t": 38.0, "f_c": 38.0})
    assessment = fib.cdpm2_readiness(configuration=invalid)
    assert assessment.state is Cdpm2ConversionReadiness.UNSUPPORTED_CONFIGURATION
    assert assessment.blockers == ("default eccentricity cannot be derived from final f_t/f_c",)
    fixed = Cdpm2Grassl2013Configuration({"f_t": 38.0, "f_c": 38.0, "eccentricity": 0.525})
    assert fib.cdpm2_readiness(configuration=fixed).state is Cdpm2ConversionReadiness.READY
    assert fib.to_cdpm2(configuration=fixed).eccentricity == 0.525


def test_bilinear_values_and_provenance_derive_from_final_effective_values():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    config = Cdpm2Grassl2013Configuration({"f_t": 3.1, "G_Ft": 0.2})
    params = fib.to_cdpm2(configuration=config)
    assert params.w_f == pytest.approx(0.2 / (0.225 * 3.1))
    assert params.w_f1 == pytest.approx(0.15 * params.w_f)
    assert params.f_t1 == pytest.approx(0.3 * 3.1)
    assert params.provenance["w_f"].source_kind is Cdpm2SourceKind.GRASSL_2013_DERIVED
    assert params.provenance["w_f"].derived_from == ("G_Ft", "f_t")
    assert params.provenance["w_f1"].derived_from == ("w_f",)
    assert params.provenance["f_t1"].derived_from == ("f_t",)


_OVERRIDE_CASES = {
    "E": 31000.0,
    "nu": 0.21,
    "f_t": 3.1,
    "f_c": 39.0,
    "G_Ft": 0.16,
    "eccentricity": 0.53,
    "q_h0": 0.31,
    "H_p": 0.02,
    "D_f": 0.90,
    "A_h": 0.09,
    "B_h": 0.004,
    "C_h": 2.1,
    "D_h": 2e-6,
    "A_s": 16.0,
    "epsilon_fc": 2e-4,
}


@pytest.mark.parametrize("field", CDPM2_OVERRIDE_FIELDS)
def test_all_15_user_overrides_preserve_explicit_constitutive_authority(field: str):
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    value = _OVERRIDE_CASES[field]
    config = Cdpm2Grassl2013Configuration({field: value})
    before = config.to_json()
    params = fib.to_cdpm2(configuration=config)
    assert getattr(params, field) == value
    provenance = params.provenance[field]
    assert provenance.source_kind is Cdpm2SourceKind.USER_CONSTITUTIVE_OVERRIDE
    assert provenance.source_id == "user_constitutive_override"
    assert provenance.overridden is True
    assert config.to_json() == before


def test_explicit_A_s_equal_to_default_remains_user_override():
    params = Concrete.from_class("C30", profile="fib_mc2010").to_cdpm2(
        configuration=Cdpm2Grassl2013Configuration({"A_s": 15.0})
    )
    assert params.A_s == 15.0
    assert params.provenance["A_s"].source_kind is Cdpm2SourceKind.USER_CONSTITUTIVE_OVERRIDE
    assert params.provenance["A_s"].overridden is True


def test_source_override_propagation_respects_derived_authority():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    base = fib.to_cdpm2()
    ft = fib.to_cdpm2(configuration=Cdpm2Grassl2013Configuration({"f_t": 3.2}))
    assert ft.f_t == 3.2
    assert ft.w_f != base.w_f
    assert ft.w_f1 != base.w_f1
    assert ft.f_t1 != base.f_t1
    assert ft.eccentricity != base.eccentricity
    assert ft.provenance["w_f"].source_kind is Cdpm2SourceKind.GRASSL_2013_DERIVED

    fc = fib.to_cdpm2(configuration=Cdpm2Grassl2013Configuration({"f_c": 45.0}))
    assert fc.f_c == 45.0
    assert fc.eccentricity != base.eccentricity
    assert fc.f_t == base.f_t

    gf = fib.to_cdpm2(configuration=Cdpm2Grassl2013Configuration({"G_Ft": 0.2}))
    assert gf.G_Ft == 0.2
    assert gf.w_f != base.w_f
    assert gf.w_f1 != base.w_f1
    assert gf.f_t1 == base.f_t1
    assert gf.provenance["w_f"].source_kind is Cdpm2SourceKind.GRASSL_2013_DERIVED

    ecc = fib.to_cdpm2(configuration=Cdpm2Grassl2013Configuration({"eccentricity": 0.55}))
    assert ecc.eccentricity == 0.55
    assert ecc.f_t == base.f_t
    assert ecc.f_c == base.f_c


def test_explicit_mc2010_to_ec2_composition_is_visible_and_does_not_mutate_ec2():
    fib, composition = _fib_composition()
    ec04 = Concrete.from_class("C30/37", profile="ec2_2004")
    before = ec04.physical.to_dict()
    assessment = ec04.cdpm2_readiness(fracture_energy_composition=composition)
    assert assessment.state is Cdpm2ConversionReadiness.READY
    params = ec04.to_cdpm2(fracture_energy_composition=composition)
    assert params.G_Ft == composition.value
    assert params.provenance["G_Ft"].source_kind is Cdpm2SourceKind.PHYSICAL_SOURCE
    assert params.provenance["G_Ft"].source_id == "fib_mc2010_2013"
    assert "explicit secondary physical" in params.provenance["G_Ft"].notes.lower()
    assert params.provenance["E"].source_id == "en_1992_1_1_2004"
    assert ec04.physical.to_dict() == before
    assert fib.physical.fracture_energy == composition.value


def test_physical_source_ids_remain_profile_specific_after_mapping():
    fib = Concrete.from_class("C30", profile="fib_mc2010").to_cdpm2()
    assert fib.provenance["E"].source_id == "fib_mc2010_2013"

    ec04 = Concrete.from_class("C30/37", profile="ec2_2004").to_cdpm2(
        configuration=Cdpm2Grassl2013Configuration({"G_Ft": 0.15})
    )
    for field in ("E", "nu", "f_t", "f_c"):
        assert ec04.provenance[field].source_id == "en_1992_1_1_2004"
        assert ec04.provenance[field].source_kind is Cdpm2SourceKind.PHYSICAL_SOURCE

    ec23 = Concrete.from_class("C30/37", profile="ec2_2023").to_cdpm2(
        configuration=Cdpm2Grassl2013Configuration({"G_Ft": 0.15})
    )
    for field in ("E", "nu", "f_t", "f_c"):
        assert ec23.provenance[field].source_id == "en_1992_1_1_2023"
        assert ec23.provenance[field].source_kind is Cdpm2SourceKind.PHYSICAL_SOURCE


def test_composition_and_GFt_override_conflict_is_unsupported():
    _, composition = _fib_composition()
    ec04 = Concrete.from_class("C30/37", profile="ec2_2004")
    assessment = ec04.cdpm2_readiness(
        configuration=Cdpm2Grassl2013Configuration({"G_Ft": 0.15}),
        fracture_energy_composition=composition,
    )
    assert assessment.state is Cdpm2ConversionReadiness.UNSUPPORTED_CONFIGURATION
    assert assessment.blockers[0] == (
        "G_Ft physical composition conflicts with explicit constitutive G_Ft override"
    )


def test_redundant_physical_composition_on_fib_is_unsupported():
    fib, composition = _fib_composition()
    assessment = fib.cdpm2_readiness(fracture_energy_composition=composition)
    assert assessment.state is Cdpm2ConversionReadiness.UNSUPPORTED_CONFIGURATION
    assert assessment.blockers == (
        "secondary fracture-energy composition cannot replace an already resolved physical value",
    )


def test_fracture_energy_composition_validation():
    fib, valid = _fib_composition()
    assert valid.to_json() == valid.to_json()

    source = fib.physical.provenance["fracture_energy"]
    with pytest.raises(ValueError, match="finite and > 0"):
        Cdpm2FractureEnergyComposition(0.0, source)
    with pytest.raises(ValueError, match="units"):
        Cdpm2FractureEnergyComposition(0.1, replace(source, units="MPa"))
    with pytest.raises(ValueError, match="must not be overridden"):
        Cdpm2FractureEnergyComposition(0.1, replace(source, overridden=True))

    legacy_source = replace(source, source_kind=SourceKind.LEGACY_IMPLEMENTATION)
    with pytest.raises(ValueError, match="STANDARD, LITERATURE, or DERIVED"):
        Cdpm2FractureEnergyComposition(0.1, legacy_source)

    user_source = replace(source, source_kind=SourceKind.USER_OVERRIDE, overridden=False)
    with pytest.raises(ValueError, match="STANDARD, LITERATURE, or DERIVED"):
        Cdpm2FractureEnergyComposition(0.1, user_source)

    derived = PropertyProvenance(
        source_id="secondary_physical_derivation",
        source_kind=SourceKind.DERIVED,
        edition="1",
        equation_or_section="Eq. X",
        units="N/mm",
        statistical_basis=StatisticalBasis.MEAN,
        derived_from=(),
    )
    with pytest.raises(ValueError, match="needs dependencies"):
        Cdpm2FractureEnergyComposition(0.1, derived)


def test_ec2_2023_readiness_override_matrix_is_exact():
    ec56 = _ec23(56.0)
    assert ec56.cdpm2_readiness().state is Cdpm2ConversionReadiness.UNRESOLVED_PHYSICAL_INPUT
    assert (
        ec56.cdpm2_readiness(configuration=Cdpm2Grassl2013Configuration({"G_Ft": 0.15})).state
        is Cdpm2ConversionReadiness.UNRESOLVED_PHYSICAL_INPUT
    )
    assert (
        ec56.cdpm2_readiness(configuration=Cdpm2Grassl2013Configuration({"E": 36000.0})).state
        is Cdpm2ConversionReadiness.COMPOSITION_REQUIRED
    )
    ready = Cdpm2Grassl2013Configuration({"E": 36000.0, "G_Ft": 0.15})
    assert ec56.cdpm2_readiness(configuration=ready).state is Cdpm2ConversionReadiness.READY


def test_fixed_semantic_model_identities_and_provenance():
    params = Concrete.from_class("C30", profile="fib_mc2010").to_cdpm2()
    assert params.tensile_softening_type is Cdpm2TensileSofteningType.BILINEAR
    assert params.damage_formulation is Cdpm2DamageFormulation.TWO_DAMAGE_VARIABLES
    assert params.provenance["tensile_softening_type"].equation_or_section == (
        "Sec. 2.3.3 and Sec. 5 bilinear study"
    )
    assert params.provenance["damage_formulation"].equation_or_section == "Eq. (1); Sec. 2.1"


def test_resolution_is_deterministic_and_does_not_mutate_inputs():
    fib, composition = _fib_composition()
    ec04 = Concrete.from_class("C30/37", profile="ec2_2004")
    config = Cdpm2Grassl2013Configuration({"A_s": 15.0})
    physical_before = ec04.physical.to_dict()
    config_before = config.to_json()
    composition_before = composition.to_json()
    first = ec04.to_cdpm2(configuration=config, fracture_energy_composition=composition)
    second = ec04.to_cdpm2(configuration=config, fracture_energy_composition=composition)
    assert first.to_json() == second.to_json()
    assert ec04.physical.to_dict() == physical_before
    assert config.to_json() == config_before
    assert composition.to_json() == composition_before
    assert fib.physical.fracture_energy == composition.value


def test_concrete_material_serialization_is_unchanged_by_cdpm2_resolution():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    before_dict = fib.to_dict()
    before_json = fib.to_json()
    params = fib.to_cdpm2()
    assert params.to_dict()["model_id"] == "cdpm2_grassl_2013"
    assert fib.to_dict() == before_dict
    assert fib.to_json() == before_json
    assert "cdpm2" not in fib.to_dict()["constitutive_models"]


def test_output_api_contains_no_backend_slot_or_runtime_names():
    params = Concrete.from_class("C30", profile="fib_mc2010").to_cdpm2()
    values = set(params.values_dict())
    forbidden = {
        "ECC",
        "QH0",
        "HP",
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
        "LCHAR",
        "helem",
        "characteristic_length",
        "history",
        "state",
    }
    assert values.isdisjoint(forbidden)
    assert not hasattr(params, "to_cm24")
    assert not hasattr(params, "backend_slots")
    assert not hasattr(params, "stress_update")


def test_high_level_cdpm2_signature_rejects_LCHAR_and_backend_arguments():
    fib = Concrete.from_class("C30", profile="fib_mc2010")
    with pytest.raises(TypeError):
        fib.to_cdpm2(LCHAR=10.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        fib.to_cdpm2(DAMAGEFLAG=0)  # type: ignore[call-arg]


def test_reserved_constitutive_lifecycle_no_longer_advertises_implemented_model():
    from cdp_generator.concrete import RESERVED_CONSTITUTIVE_PROFILES

    assert "cdpm2_grassl_2013" not in RESERVED_CONSTITUTIVE_PROFILES
