"""Immutable, JSON-scalar physical-profile configuration."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

type JSONScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ProfileConfiguration:
    """Structured profile assumptions with deterministic serialization."""

    parameters: Mapping[str, JSONScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen: dict[str, JSONScalar] = {}
        for key, value in self.parameters.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("profile-configuration keys must be non-empty strings")
            if value is not None and not isinstance(value, str | int | float | bool):
                raise TypeError(
                    "profile-configuration values must be JSON scalars "
                    f"(str/int/float/bool/None), got {type(value).__name__} for {key!r}"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"profile-configuration float must be finite: {key!r}")
            frozen[key] = value
        object.__setattr__(self, "parameters", MappingProxyType(frozen))

    def to_dict(self) -> dict[str, JSONScalar]:
        """Return parameters sorted by key for deterministic serialization."""

        return {key: self.parameters[key] for key in sorted(self.parameters)}
