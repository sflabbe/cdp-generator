"""Immutable frontend inputs; scientific validation is delegated to the domain."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from ..concrete.configuration import JSONScalar, ProfileConfiguration
from ..concrete.models.cdpm2 import Cdpm2Grassl2013Configuration


class AuthorityInputError(ValueError):
    """Expected request or domain-input error suitable for presentation."""


@dataclass(frozen=True)
class AuthorityConcreteRequest:
    physical_profile: str = "fib_mc2010"
    concrete_class: str = "C30"
    profile_parameters: Mapping[str, JSONScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("physical_profile", "concrete_class"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise AuthorityInputError(f"{name} must be a non-empty string.")
        try:
            configuration = ProfileConfiguration(self.profile_parameters)
        except (TypeError, ValueError) as exc:
            raise AuthorityInputError(str(exc)) from exc
        object.__setattr__(self, "profile_parameters", configuration.parameters)

    def to_dict(self) -> dict[str, object]:
        return {
            "physical_profile": self.physical_profile,
            "concrete_class": self.concrete_class,
            "profile_parameters": dict(self.profile_parameters),
        }


@dataclass(frozen=True)
class Cdpm2ConversionRequest:
    material: AuthorityConcreteRequest
    overrides: Mapping[str, float] = field(default_factory=dict)
    characteristic_length_mm: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.material, AuthorityConcreteRequest):
            raise AuthorityInputError("material must be an AuthorityConcreteRequest.")
        try:
            configuration = Cdpm2Grassl2013Configuration(overrides=self.overrides)
        except (TypeError, ValueError) as exc:
            raise AuthorityInputError(str(exc)) from exc
        object.__setattr__(self, "overrides", MappingProxyType(configuration.overrides_dict()))
        value = self.characteristic_length_mm
        if value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise AuthorityInputError("LCHAR must be finite and strictly positive [mm].")
            object.__setattr__(self, "characteristic_length_mm", float(value))

    def to_dict(self) -> dict[str, object]:
        return {
            "material": self.material.to_dict(),
            "overrides": dict(self.overrides),
            "characteristic_length_mm": self.characteristic_length_mm,
        }


def parse_cdpm2_overrides(text: str, common: Mapping[str, float] | None = None) -> dict[str, float]:
    """Parse explicit overrides, then use the domain's fields and validation."""
    try:
        values = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AuthorityInputError(f"Invalid overrides JSON: {exc.msg}") from exc
    if not isinstance(values, dict):
        raise AuthorityInputError("Advanced overrides must be a JSON object.")
    for key, value in values.items():
        if (
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int | float)
        ):
            raise AuthorityInputError("Overrides require string keys and numeric values.")
    common = {} if common is None else dict(common)
    duplicate = sorted(set(values) & set(common))
    if duplicate:
        raise AuthorityInputError("Override entered twice: " + ", ".join(duplicate))
    try:
        return Cdpm2Grassl2013Configuration(overrides={**values, **common}).overrides_dict()
    except (TypeError, ValueError) as exc:
        raise AuthorityInputError(str(exc)) from exc
