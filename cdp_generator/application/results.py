"""Versioned JSON contract; no framework or NumPy objects."""

import json
from dataclasses import asdict, dataclass
from typing import Any

from .requests import AnalysisMode


@dataclass(frozen=True)
class PropertyValue:
    name: str
    value: float
    unit: str


@dataclass(frozen=True)
class CurveSeries:
    id: str
    label: str
    group: str
    x: list[float]
    y: list[float]
    x_quantity: str
    y_quantity: str
    x_unit: str
    y_unit: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ResultBundle:
    schema_version: str
    workflow: str
    mode: AnalysisMode
    inputs: dict[str, Any]
    variables: list[float]
    properties: list[PropertyValue]
    curves: list[CurveSeries]
    warnings: list[str]
    metadata: dict[str, Any]
    export_capabilities: tuple[str, ...] = ("json", "xlsx")

    def to_dict(self) -> dict[str, Any]:
        # Round-trip makes tuples JSON arrays and enforces strict finite JSON.
        return json.loads(self.to_json())

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False)
