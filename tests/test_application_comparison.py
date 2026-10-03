from dataclasses import replace

import pytest

from cdp_generator.application import (
    AuthorityConcreteRequest,
    Cdpm2ConversionRequest,
    ComparisonInputError,
    LegacyConcreteAnalysisRequest,
    SteelAnalysisRequest,
    add_comparison_case,
    authority_comparison_tables,
    compatible_curve_groups,
    make_comparison_case,
    rename_comparison_case,
    run_cdpm2_conversion,
    run_legacy_concrete_analysis,
    run_steel_analysis,
    steel_comparison_table,
)
from cdp_generator.visualization.comparison_plotly import comparison_figure


def steel(label="steel", kind="true"):
    return make_comparison_case(
        run_steel_analysis(
            SteelAnalysisRequest(strain_rates=(0.001,), temperatures=(20,), output_kind=kind)
        ),
        label,
    )


def legacy(label="legacy", fcm=28):
    return make_comparison_case(
        run_legacy_concrete_analysis(LegacyConcreteAnalysisRequest(f_cm=fcm)), label
    )


def authority(label, profile, overrides=None):
    cls = "C30" if profile == "fib_mc2010" else "C30/37"
    return make_comparison_case(
        run_cdpm2_conversion(
            Cdpm2ConversionRequest(
                AuthorityConcreteRequest(physical_profile=profile, concrete_class=cls),
                overrides or {},
            )
        ),
        label,
    )


@pytest.mark.parametrize("factory", [steel, legacy])
def test_curve_overlay_preserves_arrays_and_labels(factory):
    cases = [factory("A"), factory("B")]
    groups = compatible_curve_groups(cases)
    assert groups
    for group in groups:
        figure = comparison_figure(cases, group)
        curves = [
            (case.label, c) for case in cases for c in case.payload["curves"] if c["group"] == group
        ]
        assert len(figure.data) == len(curves)
        for trace, (label, c) in zip(figure.data, curves, strict=True):
            assert list(trace.x) == c["x"]
            assert list(trace.y) == c["y"]
            assert trace.name == f"{label} — {c['label']}"


@pytest.mark.parametrize("field", ["x_quantity", "y_quantity", "x_unit", "y_unit"])
def test_semantic_mismatch_rejected(field):
    first = legacy("first")
    result = run_legacy_concrete_analysis(LegacyConcreteAnalysisRequest())
    curves = [replace(c, **{field: "incompatible"}) for c in result.curves]
    second = make_comparison_case(replace(result, curves=curves), "second")
    assert compatible_curve_groups([first, second]) == ()
    with pytest.raises(ComparisonInputError, match="semantics"):
        comparison_figure([first, second], first.payload["curves"][0]["group"])


def test_cross_family_guard_even_with_matching_axes():
    concrete = run_legacy_concrete_analysis(LegacyConcreteAnalysisRequest())
    metal = run_steel_analysis(SteelAnalysisRequest())
    # Deliberately identical canonical arrays/quantities/units/group; family still rejects.
    metal = replace(metal, curves=concrete.curves)
    cases = [make_comparison_case(concrete, "concrete"), make_comparison_case(metal, "metal")]
    with pytest.raises(ComparisonInputError, match="family"):
        comparison_figure(cases, concrete.curves[0].group)


def test_steel_true_vs_engineering_guard_and_table():
    cases = [steel("true"), steel("engineering", "engineering")]
    assert "Total stress-strain" not in compatible_curve_groups(cases)
    # Plastic x is true in both, but y stress kind differs: reject rather than convert.
    assert "Stress vs true plastic strain" not in compatible_curve_groups(cases)
    with pytest.raises(ComparisonInputError):
        comparison_figure(cases, "Total stress-strain")
    rows = steel_comparison_table(cases)
    assert len(rows) == 26
    assert next(r for r in rows if r["Parameter"] == "fy")["Unit"] == "MPa"
    assert all(r["Data status"] == "approximate_preset" for r in rows)


def test_authority_none_readiness_ready_only_and_units():
    cases = [
        authority("fib", "fib_mc2010"),
        authority("blocked", "ec2_2004"),
        authority("override", "ec2_2004", {"G_Ft": 0.15}),
    ]
    tables = authority_comparison_tables(cases)
    assert {r["State"] for r in tables["CDPM2 readiness"]} == {"READY", "COMPOSITION_REQUIRED"}
    for case in cases:
        for record in case.payload["material"]["physical_properties"]:
            row = next(
                r
                for r in tables["Physical properties"]
                if r["Case"] == case.label and r["Property"] == record["name"]
            )
            assert (row["Value"], row["Unit"], row["Resolution"]) == (
                record["value"],
                record["unit"],
                record["resolution"],
            )
    assert any(r["Value"] is None for r in tables["Physical properties"])
    assert len(tables["Semantic CDPM2"]) == 40
    assert {r["Case"] for r in tables["Semantic CDPM2"]} == {"fib", "override"}
    assert next(r for r in tables["Semantic CDPM2"] if r["Parameter"] == "E")["Unit"] == "MPa"
    assert authority_comparison_tables(cases[:2])["Semantic CDPM2"] == []
    assert compatible_curve_groups(cases) == ()


def test_management_limit_unique_labels_and_snapshot():
    cases = [steel(str(i)) for i in range(4)]
    with pytest.raises(ComparisonInputError, match="Maximum"):
        add_comparison_case(cases, steel("fifth"))
    assert len(add_comparison_case(cases, legacy("extra family"))) == 5
    with pytest.raises(ComparisonInputError, match="ID"):
        add_comparison_case(cases, cases[0])
    with pytest.raises(ComparisonInputError, match="label"):
        add_comparison_case(cases[:1], steel("0"))
    renamed = rename_comparison_case(cases, cases[0].case_id, "new")
    assert renamed[0].label == "new"
    assert renamed[0].case_id == cases[0].case_id
    assert cases[0].label == "0"
    with pytest.raises(ComparisonInputError):
        rename_comparison_case(cases, cases[0].case_id, "1")
    result = run_steel_analysis(SteelAnalysisRequest())
    snapshot = make_comparison_case(result, "snapshot")
    result.curves[0].x[0] = 999
    assert snapshot.payload["curves"][0]["x"][0] != 999
    payload = snapshot.payload
    payload["material"]["fy"] = 0
    assert snapshot.payload["material"]["fy"] != 0
