"""Separate JSON contracts for physical authority and constitutive conversion."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from ..concrete.configuration import JSONScalar
from .results import CurveSeries


class JsonResult:
    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.to_json())

    def to_json(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class PhysicalPropertyRecord:
    name: str
    value: float | None
    unit: str
    resolution: str
    provenance: dict[str, Any] | None


@dataclass(frozen=True)
class AuthorityConcreteResult(JsonResult):
    physical_profile: str
    concrete_class: str
    requested_profile_parameters: dict[str, JSONScalar]
    effective_profile_parameters: dict[str, JSONScalar]
    physical_properties: list[PhysicalPropertyRecord]
    material_definition: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    schema_version: str = "authority_concrete_result.v1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False)


@dataclass(frozen=True)
class Cdpm2ConversionResult(JsonResult):
    material: AuthorityConcreteResult
    calibration: str
    configuration: dict[str, Any]
    readiness: dict[str, Any]
    semantic_parameters: dict[str, Any] | None
    backend: dict[str, Any] | None
    fracture_energy_composition: dict[str, Any] | None
    curves: list[CurveSeries]
    backend_slot_names: list[str]
    warnings: list[str]
    export_capabilities: list[str]
    schema_version: str = "cdpm2_conversion_result.v1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False)
