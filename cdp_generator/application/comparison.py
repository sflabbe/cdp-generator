"""Session comparison snapshots and descriptive, family-guarded projections."""

import json
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from .authority_catalog import cdpm2_parameter_units
from .authority_results import Cdpm2ConversionResult
from .results import CurveSeries, ResultBundle
from .steel_results import SteelAnalysisResult

FAMILIES = ("legacy_concrete", "authority_concrete", "steel_johnson_cook")


class ComparisonInputError(ValueError):
    """Controlled incompatible comparison or case management input."""


@dataclass(frozen=True)
class ComparisonCase:
    case_id: str
    label: str
    family: str
    source_workflow: str
    result_schema: str
    payload_json: str

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.payload_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "family": self.family,
            "source_workflow": self.source_workflow,
            "result_schema": self.result_schema,
            "payload": self.payload,
        }


def make_comparison_case(
    result: ResultBundle | Cdpm2ConversionResult | SteelAnalysisResult,
    label: str,
) -> ComparisonCase:
    if not label.strip():
        raise ComparisonInputError("Enter a case label.")
    family = (
        "legacy_concrete"
        if isinstance(result, ResultBundle)
        else "authority_concrete"
        if isinstance(result, Cdpm2ConversionResult)
        else "steel_johnson_cook"
    )
    return ComparisonCase(
        uuid4().hex, label.strip(), family, family, result.schema_version, result.to_json()
    )


def add_comparison_case(cases: list[ComparisonCase], case: ComparisonCase) -> list[ComparisonCase]:
    if case.family not in FAMILIES:
        raise ComparisonInputError("Unknown comparison family.")
    if any(c.case_id == case.case_id for c in cases):
        raise ComparisonInputError("Case ID already exists.")
    if any(c.family == case.family and c.label == case.label for c in cases):
        raise ComparisonInputError("That label already exists in this family. Choose another.")
    if sum(c.family == case.family for c in cases) >= 4:
        raise ComparisonInputError("Maximum four cases per family. Remove a case first.")
    return [*cases, case]


def rename_comparison_case(
    cases: list[ComparisonCase], case_id: str, label: str
) -> list[ComparisonCase]:
    selected = next((c for c in cases if c.case_id == case_id), None)
    if selected is None:
        raise ComparisonInputError("Unknown case ID.")
    remaining = [c for c in cases if c.case_id != case_id]
    if not label.strip():
        raise ComparisonInputError("Enter a case label.")
    updated = replace(selected, label=label.strip())
    add_comparison_case(remaining, updated)
    return [updated if c.case_id == case_id else c for c in cases]


def _guard(cases: list[ComparisonCase], family: str | None = None) -> str:
    if not cases:
        raise ComparisonInputError("Add cases from the corresponding workflow.")
    actual = cases[0].family
    if (
        actual not in FAMILIES
        or any(c.family != actual for c in cases)
        or (family and actual != family)
    ):
        raise ComparisonInputError("Comparison requires cases from one workflow family.")
    return actual


def curve_semantic_signature(curve: CurveSeries) -> tuple[str, str, str, str, str]:
    # Generic "Stress" in the plastic group still has a true/engineering representation.
    return (
        curve.x_quantity,
        curve.y_quantity,
        curve.x_unit,
        curve.y_unit,
        str(curve.metadata.get("stress_kind", "")),
    )


def case_curves(case: ComparisonCase) -> list[CurveSeries]:
    return [CurveSeries(**curve) for curve in case.payload.get("curves", [])]


def compatible_curve_groups(cases: list[ComparisonCase]) -> tuple[str, ...]:
    family = _guard(cases)
    if family == "authority_concrete":
        return ()
    groups = [set(c.group for c in case_curves(case)) for case in cases]
    common = set.intersection(*groups)
    return tuple(
        sorted(
            group
            for group in common
            if len(
                {
                    curve_semantic_signature(c)
                    for case in cases
                    for c in case_curves(case)
                    if c.group == group
                }
            )
            == 1
        )
    )


def comparison_curves(cases: list[ComparisonCase], group: str) -> list[tuple[str, CurveSeries]]:
    if group not in compatible_curve_groups(cases):
        raise ComparisonInputError("No common compatible quantity/unit semantics for this group.")
    return [(case.label, c) for case in cases for c in case_curves(case) if c.group == group]


def authority_comparison_tables(cases: list[ComparisonCase]) -> dict[str, list[dict[str, Any]]]:
    _guard(cases, "authority_concrete")
    physical: list[dict[str, Any]] = []
    readiness = []
    semantic = []
    ready = [case for case in cases if case.payload["readiness"]["state"] == "READY"]
    units = cdpm2_parameter_units()
    for case in cases:
        payload = case.payload
        material = payload["material"]
        for record in material["physical_properties"]:
            physical.append(
                {
                    "Case": case.label,
                    "Property": record["name"],
                    "Unit": record["unit"],
                    "Value": record["value"],
                    "Resolution": record["resolution"],
                }
            )
        readiness.append(
            {
                "Case": case.label,
                "Profile": material["physical_profile"],
                "Class": material["concrete_class"],
                "State": payload["readiness"]["state"],
                "Blockers": payload["readiness"]["blockers"],
            }
        )
    if len(ready) >= 2:
        common = set.intersection(*(set(c.payload["semantic_parameters"]["values"]) for c in ready))
        for case in ready:
            for name in sorted(common):
                semantic.append(
                    {
                        "Case": case.label,
                        "Parameter": name,
                        "Unit": units[name],
                        "Value": case.payload["semantic_parameters"]["values"][name],
                    }
                )
    return {
        "Physical properties": physical,
        "CDPM2 readiness": readiness,
        "Semantic CDPM2": semantic,
    }


def steel_comparison_table(cases: list[ComparisonCase]) -> list[dict[str, Any]]:
    _guard(cases, "steel_johnson_cook")
    units = {
        "fy": "MPa",
        "fu": "MPa",
        "Agt": "%",
        "E": "MPa",
        "nu": "-",
        "A": "MPa",
        "B": "MPa",
        "n": "-",
        "C": "-",
        "m": "-",
        "epsdot0": "1/s",
        "T_room": "°C",
        "T_melt": "°C",
    }
    rows = []
    for case in cases:
        payload = case.payload
        values = {**payload["material"], **payload["johnson_cook_parameters"]}
        for name, unit in units.items():
            rows.append(
                {
                    "Case": case.label,
                    "Parameter": name,
                    "Value": values[name],
                    "Unit": unit,
                    "Data status": payload["data_status"],
                    "Material": f"{payload['material']['standard']} {payload['material']['grade']}",
                }
            )
    return rows
