import math

import pytest

from cdp_generator.application import (
    AuthorityConcreteRequest,
    AuthorityInputError,
    Cdpm2ConversionRequest,
    cdpm2_tensile_softening_curves,
    run_cdpm2_conversion,
)
from cdp_generator.concrete import Concrete, ProfileConfiguration
from cdp_generator.concrete.standards.fib_mc2010 import (
    estimate_fib_mc2010_fracture_energy,
)
from cdp_generator.visualization.cdpm2_plotly import build_cdpm2_figures


def test_fib_fracture_energy_estimator_is_profile_source_of_truth():
    for cls in ("C20", "C30", "C50", "C90"):
        concrete = Concrete.from_class(cls, profile="fib_mc2010")
        value, provenance = estimate_fib_mc2010_fracture_energy(concrete.physical.f_cm)
        assert value == concrete.physical.fracture_energy
        assert provenance.to_dict() == concrete.physical.provenance["fracture_energy"].to_dict()
        assert provenance.source_id == "fib_mc2010_2013"
        assert provenance.source_kind.value == "STANDARD"
        assert provenance.equation_or_section == "§5.1.5.2, Eq. (5.1-9)"
        assert provenance.derived_from == ("f_cm",)
        assert provenance.normalizations[0].source_convention == "G_F expressed in N/m"


@pytest.mark.parametrize(
    "profile,cls,params,overrides,policy,state,blockers,composition",
    [
        (
            "ec2_2004",
            "C30/37",
            {},
            {},
            "profile_only",
            "COMPOSITION_REQUIRED",
            ["G_Ft composition required"],
            False,
        ),
        (
            "ec2_2004",
            "C30/37",
            {},
            {},
            "fib_mc2010_if_missing",
            "READY",
            [],
            True,
        ),
        (
            "ec2_2023",
            "C30/37",
            {"reference_age_days": 28},
            {},
            "fib_mc2010_if_missing",
            "READY",
            [],
            True,
        ),
        (
            "ec2_2023",
            "C30/37",
            {"reference_age_days": 56},
            {},
            "fib_mc2010_if_missing",
            "UNRESOLVED_PHYSICAL_INPUT",
            ["E_initial unresolved; E_secant fallback forbidden"],
            True,
        ),
        (
            "ec2_2023",
            "C30/37",
            {"reference_age_days": 56},
            {"E": 41000.0},
            "fib_mc2010_if_missing",
            "READY",
            [],
            True,
        ),
        (
            "fib_mc2010",
            "C30",
            {},
            {},
            "fib_mc2010_if_missing",
            "READY",
            [],
            False,
        ),
    ],
)
def test_fib_composition_acceptance_matrix(
    profile, cls, params, overrides, policy, state, blockers, composition
):
    request = Cdpm2ConversionRequest(
        AuthorityConcreteRequest(profile, cls, params),
        overrides,
        fracture_energy_policy=policy,
    )
    result = run_cdpm2_conversion(request)
    assert result.readiness["state"] == state
    assert result.readiness["blockers"] == blockers
    assert (result.fracture_energy_composition is not None) is composition

    primary = Concrete.from_class(
        cls, profile=profile, profile_parameters=ProfileConfiguration(params)
    )
    if composition:
        assert primary.physical.fracture_energy is None
        assert primary.physical.resolution["fracture_energy"].value == "COMPOSED_REQUIRED"
        expected, _ = estimate_fib_mc2010_fracture_energy(primary.physical.f_cm)
        assert result.fracture_energy_composition["strategy"] == "fib_mc2010_if_missing"
        assert result.fracture_energy_composition["value"] == expected
        assert result.fracture_energy_composition["unit"] == "N/mm"
    if state == "READY":
        assert result.semantic_parameters is not None
        if composition:
            expected, _ = estimate_fib_mc2010_fracture_energy(primary.physical.f_cm)
            assert result.semantic_parameters["values"]["G_Ft"] == expected
            provenance = result.semantic_parameters["provenance"]["G_Ft"]
            assert provenance["source_kind"] == "PHYSICAL_SOURCE"
            assert provenance["source_kind"] != "USER_CONSTITUTIVE_OVERRIDE"


def test_fib_composition_conflicts_with_constitutive_override_at_request_boundary():
    with pytest.raises(AuthorityInputError, match="either secondary fib physical composition"):
        Cdpm2ConversionRequest(
            AuthorityConcreteRequest("ec2_2004", "C30/37"),
            {"G_Ft": 0.15},
            fracture_energy_policy="fib_mc2010_if_missing",
        )


def test_cdpm2_softening_curves_are_exact_and_energy_closed():
    result = run_cdpm2_conversion(Cdpm2ConversionRequest(AuthorityConcreteRequest()))
    assert result.semantic_parameters is not None
    values = result.semantic_parameters["values"]
    curve = result.curves[0]
    assert curve.group == "cdpm2_tensile_softening"
    assert curve.x == [0.0, values["w_f1"], values["w_f"]]
    assert curve.y == [values["f_t"], values["f_t1"], 0.0]
    area = sum(
        0.5 * (curve.y[i] + curve.y[i + 1]) * (curve.x[i + 1] - curve.x[i])
        for i in range(2)
    )
    assert math.isclose(area, values["G_Ft"], rel_tol=1e-12, abs_tol=1e-14)
    assert curve.metadata["integrated_area"] == area


def test_cdpm2_regularized_curve_uses_lchar_exactly_and_plotly_copies_data():
    result = run_cdpm2_conversion(
        Cdpm2ConversionRequest(AuthorityConcreteRequest(), characteristic_length_mm=12.5)
    )
    assert len(result.curves) == 2
    opening, regularized = result.curves
    assert regularized.group == "cdpm2_regularized_tensile_softening"
    assert regularized.x == pytest.approx([value / 12.5 for value in opening.x])
    assert regularized.y == opening.y
    figures = build_cdpm2_figures(result.curves)
    for curve in result.curves:
        trace = figures[curve.group].data[0]
        assert list(trace.x) == curve.x
        assert list(trace.y) == curve.y


def test_no_regularized_curve_without_lchar():
    result = run_cdpm2_conversion(Cdpm2ConversionRequest(AuthorityConcreteRequest()))
    assert [curve.group for curve in result.curves] == ["cdpm2_tensile_softening"]
    parameters = Concrete.from_class("C30", profile="fib_mc2010").to_cdpm2()
    assert len(cdpm2_tensile_softening_curves(parameters)) == 1
