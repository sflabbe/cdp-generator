"""Deterministic steel records with explicit data status."""

import json
from dataclasses import asdict, dataclass
from typing import Any

from .results import CurveSeries


@dataclass(frozen=True)
class SteelAnalysisResult:
    material_source: str
    material: dict[str, Any]
    data_status: str
    disclaimer: str
    calibration: dict[str, Any]
    johnson_cook_parameters: dict[str, float]
    analysis: dict[str, Any]
    curves: list[CurveSeries]
    warnings: list[str]
    metadata: dict[str, Any]
    schema_version: str = "steel_analysis_result.v1"
    workflow: str = "steel_johnson_cook"
    export_capabilities: tuple[str, ...] = ("json", "xlsx", "abaqus_experimental")

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False, sort_keys=True, indent=2)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.to_json())
