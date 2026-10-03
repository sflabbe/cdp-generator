"""ABAQUS-Q1 / WEB-M5: dependent legacy Abaqus tables — pure tests, no solver required."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cdp_generator import core, strain_rate
from cdp_generator.application import (
    AbaqusLegacyDependentRequest,
    AbaqusLegacyDependentResult,
    AbaqusLegacyDependentValidationError,
    AbaqusLegacyInputError,
    AbaqusLegacyMaterialRequest,
    AbaqusLegacyValidationError,
    abaqus_legacy_dependent_material_text,
    abaqus_legacy_material_text,
    rate_mapping_audit_rows,
    run_abaqus_legacy_dependent_material,
    run_abaqus_legacy_material,
    temperature_cases,
    temperature_elastic_audit_rows,
)
from cdp_generator.application import abaqus_dependent as dependent_module
from cdp_generator.visualization.abaqus_plotly import build_abaqus_dependent_figures

FIXTURES = Path(__file__).parent / "fixtures"
RATES = (0.0, 2.0, 30.0, 100.0)
ARGS = (28.0, 0.0022, 0.0035, 1.0)


def _rate(**kwargs) -> AbaqusLegacyDependentResult:
    kwargs.setdefault("strain_rates", RATES)
    return run_abaqus_legacy_dependent_material(
        AbaqusLegacyDependentRequest(mode="strain_rate", **kwargs)
    )


def _temp(**kwargs) -> AbaqusLegacyDependentResult:
    return run_abaqus_legacy_dependent_material(
        AbaqusLegacyDependentRequest(mode="temperature", **kwargs)
    )


def _families(rows, key, n):
    size = len(rows) // n
    assert size * n == len(rows)
    return [[row[key] for row in rows[i * size : (i + 1) * size]] for i in range(n)]


# --------------------------------------------------------------------------- shared mappings


@pytest.mark.parametrize("rate,l_ch", [(0.0, 1.0), (2.0, 1.0), (30.0, 12.5), (1e-5, 100.0)])
def test_shared_crack_opening_rate_mapping(rate, l_ch):
    assert strain_rate.legacy_cracking_displacement_rate(rate, l_ch) == rate * l_ch


@pytest.mark.parametrize("rate", [0.0, 1e-5, 2.0, 30.0, 100.0, 1e4])
@pytest.mark.parametrize("l_ch", [1.0, 12.5])
def test_fracture_energy_rate_effect_unchanged_by_refactor(rate, l_ch):
    """Exact reproduction of the historical inline formula (frozen for M5)."""

    G_f = 0.1336
    if rate == 0:
        expected = G_f
    else:
        w_rate = rate * l_ch
        if w_rate > 200:
            b_g = (200 / 0.01) ** (0.08 - 0.62)
            expected = G_f * b_g * (w_rate / 0.01) ** 0.62
        else:
            expected = G_f * (w_rate / 0.01) ** 0.08
    assert strain_rate.apply_fracture_energy_rate_effects(G_f, rate, l_ch) == expected


def test_fracture_energy_rate_effect_uses_the_shared_mapping(monkeypatch):
    calls = []

    def spy(rate, l_ch):
        calls.append((rate, l_ch))
        return rate * l_ch

    monkeypatch.setattr(strain_rate, "legacy_cracking_displacement_rate", spy)
    strain_rate.apply_fracture_energy_rate_effects(0.1, 2.0, 3.0)
    assert calls == [(2.0, 3.0)]


def test_dependent_adapter_has_no_independent_rate_formula():
    source = Path(dependent_module.__file__).read_text(encoding="utf-8")
    assert "legacy_cracking_displacement_rate(" in source
    assert "* request.l_ch" not in source
    assert "np.interp" not in source and "interp(" not in source


# --------------------------------------------------------------------------- temperature E(T)


def test_temperature_elastic_state_matches_kernel_inelastic_modulus(monkeypatch):
    """E(T) is exactly the modulus the kernel passes to calculate_inelastic_compression."""

    seen: list[float] = []
    original = core.calculate_inelastic_compression

    def spy(strain, stress, f_cm, E_c1):
        seen.append(float(E_c1))
        return original(strain, stress, f_cm, E_c1)

    monkeypatch.setattr(core, "calculate_inelastic_compression", spy)
    core.calculate_stress_strain_temp(*ARGS, verbose=False)
    states = core.calculate_temperature_elastic_states(28.0, 0.0022)
    assert [s["E_mpa"] for s in states] == seen
    assert [s["temperature_c"] for s in states] == temperature_cases()


def test_temperature_elastic_state_reconstructs_kernel_families_exactly():
    raw = core.calculate_stress_strain_temp(*ARGS, verbose=False)
    states = core.calculate_temperature_elastic_states(28.0, 0.0022)
    base = core.legacy_temperature_base_properties(28.0, 0.0022)
    table = core.get_eurocode_temperature_table()
    grid = raw["compression"]["inelastic strain"][0]
    for i, state in enumerate(states):
        props = core.apply_temperature_effects(base, state["temperature_c"], table)
        comp = core.calculate_compression_behavior(
            props["f_cm_temp"],
            props["e_c1_temp"],
            props["E_ci_temp"],
            state["E_mpa"],
            40,
            props["eps_cu"] * 2,
        )
        inel = core.calculate_inelastic_compression(
            comp["strain"], comp["stress"], props["f_cm_temp"], state["E_mpa"]
        )
        expected = (
            inel["inelastic_stress"]
            if i == 0
            else np.interp(grid, inel["inelastic_strain"], inel["inelastic_stress"])
        )
        assert np.array_equal(expected, raw["compression"]["inelastic stress"][i])


def test_temperature_nu_is_constant_legacy_room_value():
    states = core.calculate_temperature_elastic_states(28.0, 0.0022)
    raw = core.calculate_stress_strain_temp(*ARGS, verbose=False)
    assert {s["nu"] for s in states} == {raw["properties"]["poisson"]}


# --------------------------------------------------------------------------- rate tables


@pytest.mark.parametrize("law,key", [("bilinear", "stress"), ("power_law", "stress exponential")])
def test_rate_tables_are_exact_kernel_families(law, key):
    result = _rate(tension_law=law)
    raw = core.calculate_stress_strain(*ARGS, list(RATES))
    n = len(RATES)
    comp_s = _families(result.compression_hardening, "stress_mpa", n)
    comp_e = _families(result.compression_hardening, "inelastic_strain", n)
    comp_r = _families(result.compression_hardening, "strain_rate_1_per_s", n)
    ten_s = _families(result.tension_stiffening, "stress_mpa", n)
    ten_u = _families(result.tension_stiffening, "cracking_displacement_mm", n)
    ten_r = _families(result.tension_stiffening, "cracking_displacement_rate_mm_per_s", n)
    for i, rate in enumerate(RATES):
        assert comp_s[i] == [float(v) for v in raw["compression"]["inelastic stress"][i]]
        assert comp_e[i] == [float(v) for v in raw["compression"]["inelastic strain"][i]]
        assert set(comp_r[i]) == {rate}
        assert ten_s[i] == [float(v) for v in raw["tension"][key][i]]
        assert ten_u[i] == [float(v) for v in raw["tension"]["crack opening"][i]]
        assert set(ten_r[i]) == {strain_rate.legacy_cracking_displacement_rate(rate, 1.0)}
        assert comp_e[i][0] == 0.0 and ten_u[i][0] == 0.0
    assert all("temperature_c" not in row for row in result.compression_hardening)


def test_rate_tension_rate_uses_l_ch():
    result = _rate(strain_rates=(0.0, 2.0), l_ch=12.5)
    rates = {row["cracking_displacement_rate_mm_per_s"] for row in result.tension_stiffening}
    assert rates == {0.0, 25.0}
    audit = rate_mapping_audit_rows(result)
    assert audit[1] == {
        "legacy strain rate [1/s]": 2.0,
        "Abaqus compression rate coordinate [1/s]": 2.0,
        "Abaqus tension crack-opening rate [mm/s]": 25.0,
    }


def test_rate_semantics_and_provenance_are_explicit():
    data = _rate().to_dict()
    sem = data["dependency_semantics"]
    assert sem["compression"]["mapping_kind"] == "LEGACY_RATE_AXIS_MAPPING"
    assert "not reconstructed from a loading history" in sem["compression"]["notes"]
    assert sem["tension"]["mapping_kind"] == "LEGACY_CRACK_OPENING_RATE_MAPPING"
    assert sem["damage"]["abaqus_supports_rate_column"] is False
    assert data["provenance"]["legacy_kernel"]["source_kind"] == "LEGACY_IMPLEMENTATION"
    for family, rate in zip(sem["families"], RATES, strict=True):
        assert family["dependency_mode"] == "strain_rate"
        assert family["dependency_value"] == rate
        assert family["dependency_unit"] == "1/s"
        assert family["curve_producer"] == "core.calculate_stress_strain"
        assert family["compression_keyword"] == "*CONCRETE COMPRESSION HARDENING"
    assert data["damage_policy"]["policy"] == "omit"
    assert data["compression_damage"] is None and data["tension_damage"] is None
    assert data["elastic"]["rows"] == [
        {"E_mpa": data["elastic"]["rows"][0]["E_mpa"], "nu": data["elastic"]["rows"][0]["nu"]}
    ]
    for key in (
        "schema_version",
        "workflow",
        "mode",
        "inputs",
        "elastic",
        "plasticity",
        "compression_hardening",
        "tension_stiffening",
        "damage_policy",
        "compression_damage",
        "tension_damage",
        "dependency_semantics",
        "provenance",
        "validation",
        "warnings",
        "documentation_references",
        "export_capabilities",
    ):
        assert key in data


def test_rate_elastic_and_scalars_equal_static_m4():
    result = _rate()
    static = run_abaqus_legacy_material(AbaqusLegacyMaterialRequest())
    assert result.elastic["rows"] == [static.elastic]
    assert result.plasticity == static.plasticity


def test_json_is_strict_and_deterministic():
    a, b = _rate(), _rate()
    assert a.to_json() == b.to_json()
    json.loads(a.to_json(), parse_constant=lambda c: pytest.fail(f"non-finite {c}"))
    assert _temp().to_json() == _temp().to_json()


# --------------------------------------------------------------------------- temperature tables


@pytest.mark.parametrize("law,key", [("bilinear", "stress"), ("power_law", "stress exponential")])
def test_temperature_tables_are_exact_kernel_families(law, key):
    result = _temp(tension_law=law)
    raw = core.calculate_stress_strain_temp(*ARGS, verbose=False)
    temps = temperature_cases()
    n = len(temps)
    comp_s = _families(result.compression_hardening, "stress_mpa", n)
    comp_e = _families(result.compression_hardening, "inelastic_strain", n)
    comp_t = _families(result.compression_hardening, "temperature_c", n)
    comp_r = _families(result.compression_hardening, "strain_rate_1_per_s", n)
    ten_s = _families(result.tension_stiffening, "stress_mpa", n)
    ten_u = _families(result.tension_stiffening, "cracking_displacement_mm", n)
    ten_t = _families(result.tension_stiffening, "temperature_c", n)
    ten_r = _families(result.tension_stiffening, "cracking_displacement_rate_mm_per_s", n)
    for i, temp in enumerate(temps):
        assert comp_s[i] == [float(v) for v in raw["compression"]["inelastic stress"][i]]
        assert comp_e[i] == [float(v) for v in raw["compression"]["inelastic strain"][i]]
        assert set(comp_t[i]) == {temp} and set(comp_r[i]) == {0.0}
        assert ten_s[i] == [float(v) for v in raw["tension"][key][i]]
        assert ten_u[i] == [float(v) for v in raw["tension"]["crack opening"][i]]
        assert set(ten_t[i]) == {temp} and set(ten_r[i]) == {0.0}
        assert comp_e[i][0] == 0.0 and ten_u[i][0] == 0.0


def test_temperature_elasticity_rows_and_audit():
    result = _temp()
    states = core.calculate_temperature_elastic_states(28.0, 0.0022)
    assert result.elastic["temperature_dependent"] is True
    assert result.elastic["nu_assumption"] == "LEGACY_CONSTANT_NU_ASSUMPTION"
    assert result.elastic["rows"] == [
        {"E_mpa": s["E_mpa"], "nu": s["nu"], "temperature_c": s["temperature_c"]} for s in states
    ]
    audit = temperature_elastic_audit_rows(result)
    assert [row["temperature [°C]"] for row in audit] == temperature_cases()
    sem = result.dependency_semantics
    assert sem["elastic"]["nu_mapping"] == "LEGACY_CONSTANT_NU_ASSUMPTION"
    assert "constant" in sem["elastic"]["nu_notes"]
    assert sem["combined_rate_temperature"] is False
    with pytest.raises(AbaqusLegacyInputError):
        rate_mapping_audit_rows(result)


# --------------------------------------------------------------------------- damage policy


def test_reference_damage_equals_m4_static_table_when_first_rate_is_zero():
    result = _rate(strain_rates=(0.0, 2.0), damage_policy="reference_damage")
    static = run_abaqus_legacy_material(AbaqusLegacyMaterialRequest())
    assert result.compression_damage == static.compression_damage
    assert result.tension_damage == static.tension_damage
    assert result.damage_policy["notes"] == (
        "reference damage function reused across all rate-dependent hardening/stiffening families"
    )
    assert result.damage_policy["has_dependency_column"] is False
    assert all(set(row) == {"damage", "inelastic_strain"} for row in result.compression_damage)
    assert result.validation["families_checked"] == 2
    assert all(f["passed"] for f in result.validation["families"])


def test_reference_damage_default_rates_truthful_rejection():
    """Frozen truthful outcome: legacy 30 1/s family breaks compression monotonicity."""

    with pytest.raises(AbaqusLegacyDependentValidationError) as info:
        _rate(damage_policy="reference_damage")
    failures = info.value.failures
    assert failures == (
        {
            "mode": "strain_rate",
            "dependency_value": 30.0,
            "dependency_unit": "1/s",
            "branch": "compression",
            "failed_condition": "compression_plastic_strain_monotonic",
        },
    )
    assert "damage_policy='omit'" in str(info.value)
    assert isinstance(info.value, AbaqusLegacyValidationError)


def test_reference_damage_temperature_truthful_rejection_with_family_modulus():
    with pytest.raises(AbaqusLegacyDependentValidationError) as info:
        _temp(damage_policy="reference_damage")
    failing = {f["dependency_value"] for f in info.value.failures}
    assert 20.0 not in failing
    assert failing == {float(t) for t in temperature_cases()[1:]}
    assert {f["branch"] for f in info.value.failures} == {"compression"}
    assert {f["dependency_unit"] for f in info.value.failures} == {"°C"}


def test_damage_validation_uses_temperature_modulus(monkeypatch):
    """Changing only the family moduli changes the verdict: E(T) is really used."""

    seen = []
    original = dependent_module.abaqus_table_checks

    def spy(**kwargs):
        seen.append(kwargs["E0"])
        return original(**kwargs)

    monkeypatch.setattr(dependent_module, "abaqus_table_checks", spy)
    _temp()
    assert seen == [s["E_mpa"] for s in core.calculate_temperature_elastic_states(28.0, 0.0022)]


def test_omit_policy_never_emits_damage():
    for result in (_rate(), _temp()):
        text = abaqus_legacy_dependent_material_text(result)
        keywords = [line for line in text.splitlines() if line.startswith("*CONCRETE")]
        assert "*CONCRETE COMPRESSION DAMAGE" not in keywords
        assert not any(k.startswith("*CONCRETE TENSION DAMAGE") for k in keywords)
        assert result.compression_damage is None and result.tension_damage is None


def test_synthetic_failure_reports_family_and_branch():
    report = dependent_module._family_failures(
        mode="temperature",
        value=300.0,
        unit="°C",
        report={"hard_failures": ["tension_plastic_displacement_nonnegative"]},
    )
    assert report == [
        {
            "mode": "temperature",
            "dependency_value": 300.0,
            "dependency_unit": "°C",
            "branch": "tension",
            "failed_condition": "tension_plastic_displacement_nonnegative",
        }
    ]


def test_excessive_ref_length_rejects_reference_damage_in_tension():
    with pytest.raises(AbaqusLegacyValidationError, match="tension_plastic"):
        _rate(
            strain_rates=(0.0, 2.0),
            damage_policy="reference_damage",
            damage_conversion_reference_length_mm=1000.0,
        )


# --------------------------------------------------------------------------- requests


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"mode": "both"}, "mode"),
        ({"mode": "strain_rate"}, "requires one or more"),
        ({"mode": "strain_rate", "strain_rates": (2.0, 0.0)}, "strictly increasing"),
        ({"mode": "strain_rate", "strain_rates": (0.0, 0.0)}, "strictly increasing"),
        ({"mode": "strain_rate", "strain_rates": (-1.0,)}, ">= 0"),
        ({"mode": "temperature", "strain_rates": (0.0,)}, "out of scope"),
        ({"mode": "temperature", "damage_policy": "rate_damage"}, "damage_policy"),
        ({"mode": "temperature", "tension_law": "gfi"}, "tension_law"),
        ({"mode": "temperature", "damage_conversion_reference_length_mm": 0.0}, "reference"),
        ({"mode": "temperature", "viscosity": -1.0}, "viscosity"),
        ({"mode": "temperature", "eccentricity": 0.0}, "eccentricity"),
    ],
)
def test_controlled_dependent_input_errors(kwargs, match):
    with pytest.raises(AbaqusLegacyInputError, match=match):
        AbaqusLegacyDependentRequest(**kwargs)


def test_default_dependent_policy_is_omit():
    assert AbaqusLegacyDependentRequest(mode="temperature").damage_policy == "omit"


# --------------------------------------------------------------------------- keyword text


def _normalized_fixture(name: str) -> str:
    # Explicit Windows newline normalization; fixtures are stored with LF bytes.
    return (FIXTURES / name).read_bytes().decode("utf-8").replace("\r\n", "\n")


def _assert_text_matches(actual: str, expected: str) -> None:
    a_lines = actual.split("\n")
    e_lines = expected.split("\n")
    assert len(a_lines) == len(e_lines)
    for a, e in zip(a_lines, e_lines, strict=True):
        if a.startswith("*") or not a:
            assert a == e
        else:
            a_vals = [float(v) for v in a.split(",")]
            e_vals = [float(v) for v in e.split(",")]
            assert a_vals == pytest.approx(e_vals, rel=1e-11, abs=1e-15)


def _sections(text: str) -> list[tuple[str, list[list[float]]]]:
    sections: list[tuple[str, list[list[float]]]] = []
    for line in text.splitlines():
        if line.startswith("**"):
            continue
        if line.startswith("*"):
            sections.append((line, []))
        else:
            sections[-1][1].append([float(v) for v in line.split(",")])
    return sections


def test_static_m4_card_unchanged():
    text = abaqus_legacy_material_text(run_abaqus_legacy_material(AbaqusLegacyMaterialRequest()))
    _assert_text_matches(text, _normalized_fixture("abaqus_m4_static_default.inp"))
    assert "\r" not in text


def test_rate_card_fixture_and_exact_structure():
    result = _rate(strain_rates=(0.0, 2.0), damage_policy="reference_damage")
    text = abaqus_legacy_dependent_material_text(result)
    assert "\r" not in text
    _assert_text_matches(text, _normalized_fixture("abaqus_q1_rate_0_2_reference_damage.inp"))
    sections = _sections(text)
    assert [k for k, _ in sections] == [
        "*MATERIAL, NAME=Concrete_Legacy_CDP",
        "*ELASTIC",
        "*CONCRETE DAMAGED PLASTICITY, REF LENGTH=1",
        "*CONCRETE COMPRESSION HARDENING",
        "*CONCRETE COMPRESSION DAMAGE",
        "*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT",
        "*CONCRETE TENSION DAMAGE, TYPE=DISPLACEMENT",
    ]
    body = dict(sections)
    assert [len(r) for r in body["*ELASTIC"]] == [2]
    assert {len(r) for r in body["*CONCRETE COMPRESSION HARDENING"]} == {3}
    assert {len(r) for r in body["*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT"]} == {3}
    assert {len(r) for r in body["*CONCRETE COMPRESSION DAMAGE"]} == {2}
    assert {len(r) for r in body["*CONCRETE TENSION DAMAGE, TYPE=DISPLACEMENT"]} == {2}
    assert [r[2] for r in body["*CONCRETE COMPRESSION HARDENING"]] == [
        row["strain_rate_1_per_s"] for row in result.compression_hardening
    ]
    assert "GFI" not in text


def test_temperature_card_fixture_custom_scalars_power_law():
    result = _temp(tension_law="power_law", eccentricity=0.12, viscosity=0.001)
    text = abaqus_legacy_dependent_material_text(result)
    _assert_text_matches(text, _normalized_fixture("abaqus_q1_temperature_power_omit_custom.inp"))
    sections = _sections(text)
    assert [k for k, _ in sections] == [
        "*MATERIAL, NAME=Concrete_Legacy_CDP",
        "*ELASTIC",
        "*CONCRETE DAMAGED PLASTICITY, REF LENGTH=1",
        "*CONCRETE COMPRESSION HARDENING",
        "*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT",
    ]
    body = dict(sections)
    assert [r[2] for r in body["*ELASTIC"]] == temperature_cases()
    assert body["*CONCRETE DAMAGED PLASTICITY, REF LENGTH=1"][0][1] == 0.12
    assert body["*CONCRETE DAMAGED PLASTICITY, REF LENGTH=1"][0][4] == 0.001
    hardening = body["*CONCRETE COMPRESSION HARDENING"]
    assert {len(r) for r in hardening} == {4} and {r[2] for r in hardening} == {0.0}
    temps = [r[3] for r in hardening]
    assert temps == sorted(temps)
    stiffening = body["*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT"]
    assert {len(r) for r in stiffening} == {4} and {r[2] for r in stiffening} == {0.0}


def test_temperature_card_bilinear_with_tension_branch_text():
    text = abaqus_legacy_dependent_material_text(_temp())
    stiff = dict(_sections(text))["*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT"]
    assert len(stiff) == 40 * len(temperature_cases())  # legacy temperature kernel: 40 points


def test_dependent_text_refuses_bad_names_and_types():
    result = _rate(strain_rates=(0.0,))
    with pytest.raises(AbaqusLegacyInputError):
        abaqus_legacy_dependent_material_text(result, "bad,name")
    with pytest.raises(AbaqusLegacyInputError):
        abaqus_legacy_dependent_material_text(  # type: ignore[arg-type]
            run_abaqus_legacy_material(AbaqusLegacyMaterialRequest())
        )


# --------------------------------------------------------------------------- plots


def test_dependent_figures_copy_exported_families_only():
    result = _rate(strain_rates=(0.0, 2.0), damage_policy="reference_damage")
    figures = build_abaqus_dependent_figures(result)
    assert len(figures["abaqus_dependent_compression_hardening"].data) == 2
    assert len(figures["abaqus_dependent_tension_stiffening"].data) == 2
    assert len(figures["abaqus_dependent_reference_compression_damage"].data) == 1
    assert len(figures["abaqus_dependent_reference_tension_damage"].data) == 1
    for curve in result.curves:
        traces = [t for t in figures[curve.group].data if t.name == curve.label]
        assert list(traces[0].x) == curve.x and list(traces[0].y) == curve.y
    temp_figs = build_abaqus_dependent_figures(_temp())
    assert len(temp_figs["abaqus_dependent_compression_hardening"].data) == 12
    assert "abaqus_dependent_elastic_modulus" in temp_figs
    assert not any("damage" in group for group in temp_figs)
