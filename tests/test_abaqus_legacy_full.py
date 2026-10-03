import pytest

from cdp_generator import core
from cdp_generator.application import (
    AbaqusLegacyInputError,
    AbaqusLegacyMaterialRequest,
    AbaqusLegacyValidationError,
    abaqus_legacy_material_text,
    run_abaqus_legacy_material,
    validate_abaqus_legacy_material_tables,
)
from cdp_generator.concrete import Concrete
from cdp_generator.visualization.abaqus_plotly import build_abaqus_figures


@pytest.mark.parametrize("f_cm", [20.0, 28.0, 50.0, 80.0])
def test_full_backend_scalar_parity_with_existing_api(f_cm):
    request = AbaqusLegacyMaterialRequest(f_cm=f_cm)
    result = run_abaqus_legacy_material(request)
    scalar = Concrete.from_mean_strength(f_cm, request.e_c1, request.e_clim).to_abaqus_cdp()
    assert result.plasticity["dilation_angle_deg"] == scalar.dilation_angle
    assert result.plasticity["fbfc"] == scalar.fbfc
    assert result.plasticity["Kc"] == scalar.Kc
    assert result.plasticity["eccentricity"] == 0.1
    assert result.plasticity["viscosity"] == 0.0
    assert result.provenance["plasticity"]["eccentricity"]["source_kind"] == (
        "ABAQUS_DOCUMENTATION_DEFAULT"
    )


def test_default_tables_are_exact_legacy_arrays_except_documented_normalization():
    request = AbaqusLegacyMaterialRequest()
    result = run_abaqus_legacy_material(request)
    raw = core.calculate_stress_strain(28.0, 0.0022, 0.0035, 1.0, [0.0])

    assert [row["stress_mpa"] for row in result.compression_hardening] == pytest.approx(
        raw["compression"]["inelastic stress"][0]
    )
    assert [row["inelastic_strain"] for row in result.compression_hardening] == pytest.approx(
        raw["compression"]["inelastic strain"][0]
    )
    assert result.compression_damage is not None
    assert [row["inelastic_strain"] for row in result.compression_damage] == pytest.approx(
        raw["compression"]["inelastic strain"][0]
    )
    expected_compression_damage = [min(float(v), 0.99) for v in raw["compression"]["damage"]]
    expected_compression_damage[0] = 0.0
    assert [row["damage"] for row in result.compression_damage] == pytest.approx(
        expected_compression_damage
    )

    assert [row["stress_mpa"] for row in result.tension_stiffening] == pytest.approx(
        raw["tension"]["stress"][0]
    )
    assert [row["cracking_displacement_mm"] for row in result.tension_stiffening] == pytest.approx(
        raw["tension"]["crack opening"][0]
    )
    assert result.tension_damage is not None
    expected_tension_damage = [min(float(v), 0.99) for v in raw["tension"]["damage"]]
    assert [row["damage"] for row in result.tension_damage] == pytest.approx(
        expected_tension_damage
    )
    assert raw["compression"]["damage"][0] != 0.0
    assert result.compression_damage[0]["damage"] == 0.0
    assert raw["tension"]["damage"][-1] == 1.0
    assert result.tension_damage[-1]["damage"] == 0.99


def test_power_law_selects_exact_legacy_power_arrays():
    result = run_abaqus_legacy_material(AbaqusLegacyMaterialRequest(tension_law="power_law"))
    raw = core.calculate_stress_strain(28.0, 0.0022, 0.0035, 1.0, [0.0])
    assert [row["stress_mpa"] for row in result.tension_stiffening] == pytest.approx(
        raw["tension"]["stress exponential"][0]
    )
    assert result.tension_damage is not None
    assert [row["damage"] for row in result.tension_damage] == pytest.approx(
        [min(float(v), 0.99) for v in raw["tension"]["damage exponential"]]
    )


def test_validation_and_ref_length_contract():
    result = run_abaqus_legacy_material(
        AbaqusLegacyMaterialRequest(damage_conversion_reference_length_mm=10.0)
    )
    assert result.validation["passed"]
    assert result.validation["compression_plastic_strain_nonnegative"]
    assert result.validation["compression_plastic_strain_monotonic"]
    assert result.validation["tension_plastic_displacement_nonnegative"]
    assert result.validation["tension_plastic_displacement_monotonic"]
    assert result.backend_configuration["damage_conversion_reference_length_mm"] == 10.0
    assert result.backend_configuration["legacy_l_ch_mm"] == 1.0
    assert "not the legacy crack-band l_ch" in result.backend_configuration[
        "reference_length_semantics"
    ]


def test_validator_rejects_decreasing_plastic_sequence():
    result = run_abaqus_legacy_material(AbaqusLegacyMaterialRequest(include_tension_damage=False))
    broken = [dict(row) for row in result.compression_hardening]
    broken[2]["inelastic_strain"] = -0.1
    with pytest.raises(AbaqusLegacyValidationError, match="validity checks failed"):
        validate_abaqus_legacy_material_tables(
            E0=result.elastic["E_mpa"],
            compression_hardening=broken,
            compression_damage=None,
            tension_stiffening=result.tension_stiffening,
            tension_damage=None,
            damage_conversion_reference_length_mm=1.0,
        )


def test_excessive_ref_length_is_rejected_by_documented_tension_conversion():
    with pytest.raises(AbaqusLegacyValidationError, match="tension_plastic"):
        run_abaqus_legacy_material(
            AbaqusLegacyMaterialRequest(damage_conversion_reference_length_mm=1000.0)
        )


def test_inp_keyword_order_variants_and_determinism():
    result = run_abaqus_legacy_material(AbaqusLegacyMaterialRequest())
    text = abaqus_legacy_material_text(result)
    assert text == abaqus_legacy_material_text(result)
    expected_order = [
        "*MATERIAL, NAME=Concrete_Legacy_CDP",
        "*ELASTIC",
        "*CONCRETE DAMAGED PLASTICITY, REF LENGTH=1",
        "*CONCRETE COMPRESSION HARDENING",
        "*CONCRETE COMPRESSION DAMAGE",
        "*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT",
        "*CONCRETE TENSION DAMAGE, TYPE=DISPLACEMENT",
    ]
    positions = [text.index(token) for token in expected_order]
    assert positions == sorted(positions)
    assert "*CONCRETE FAILURE" not in text

    no_damage = run_abaqus_legacy_material(
        AbaqusLegacyMaterialRequest(
            tension_law="power_law",
            include_tension_damage=False,
            include_compression_damage=False,
            damage_conversion_reference_length_mm=12.5,
            viscosity=0.002,
        )
    )
    no_damage_text = abaqus_legacy_material_text(no_damage)
    assert "*CONCRETE COMPRESSION DAMAGE" not in no_damage_text
    assert "*CONCRETE TENSION DAMAGE" not in no_damage_text
    assert "REF LENGTH=12.5" in no_damage_text
    assert no_damage.plasticity["viscosity"] == 0.002
    assert no_damage.provenance["plasticity"]["viscosity"]["source_kind"] == (
        "USER_BACKEND_OVERRIDE"
    )


def test_abaqus_canonical_plots_copy_export_tables_exactly():
    result = run_abaqus_legacy_material(AbaqusLegacyMaterialRequest())
    figures = build_abaqus_figures(result)
    assert len(figures) == 4
    for curve in result.curves:
        trace = figures[curve.group].data[0]
        assert list(trace.x) == curve.x
        assert list(trace.y) == curve.y
        assert curve.metadata["representation"] == "Abaqus export representation"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"damage_conversion_reference_length_mm": 0.0}, "reference_length"),
        ({"eccentricity": 0.0}, "eccentricity"),
        ({"viscosity": -1.0}, "viscosity"),
        ({"tension_law": "bad"}, "tension_law"),
    ],
)
def test_controlled_input_errors(kwargs, match):
    with pytest.raises(AbaqusLegacyInputError, match=match):
        AbaqusLegacyMaterialRequest(**kwargs)
