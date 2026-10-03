"""Immutable input contract and frontend-independent numeric parsing."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType


class SteelInputError(ValueError):
    """Expected steel input or domain validation failure."""


def parse_float_series(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise SteelInputError("Enter a comma-separated numeric series.") from exc
    if not values or not all(isfinite(v) for v in values):
        raise SteelInputError("Series must contain finite numbers.")
    return values


@dataclass(frozen=True)
class SteelAnalysisRequest:
    source: str = "preset"
    standard: str = "EC2"
    grade: str = "B500C"
    material_overrides: Mapping[str, float] = field(default_factory=dict)
    assume_rate_temp_neutral: bool = True
    n_default: float | None = None
    C_default: float = 0.0
    m_default: float = 0.0
    epsdot0: float = 1e-3
    strain_rates: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1.0, 10.0, 100.0)
    temperatures: tuple[float, ...] = (20.0, 200.0, 400.0, 600.0, 800.0)
    eps_max: float = 0.20
    n_points: int = 100
    output_kind: str = "true"

    def __post_init__(self) -> None:
        overrides = dict(self.material_overrides)
        allowed = {"fy", "fu", "fu_fy_ratio", "Agt", "E", "nu", "T_room", "T_melt"}
        if overrides.keys() - allowed:
            raise SteelInputError("Unsupported material override.")
        if "fu" in overrides and "fu_fy_ratio" in overrides:
            raise SteelInputError("Choose fu OR fu/fy ratio.")
        values = [*overrides.values(), self.C_default, self.m_default, self.epsdot0, self.eps_max]
        if self.n_default is not None:
            values.append(self.n_default)
        if not all(
            isinstance(v, (float, int)) and not isinstance(v, bool) and isfinite(v) for v in values
        ):
            raise SteelInputError("Inputs must be finite numbers.")
        for name in ("strain_rates", "temperatures"):
            series = tuple(getattr(self, name))
            if not series or not all(
                isinstance(v, (float, int)) and not isinstance(v, bool) and isfinite(v)
                for v in series
            ):
                raise SteelInputError(f"{name} must contain finite numbers.")
            object.__setattr__(self, name, series)
        if self.source not in {"preset", "custom"} or not self.grade.strip():
            raise SteelInputError("Choose preset/custom and a material name.")
        if self.source == "preset" and self.standard == "Custom":
            raise SteelInputError("Custom material requires custom source.")
        if self.epsdot0 <= 0 or self.eps_max <= 0 or overrides.get("E", 1) <= 0:
            raise SteelInputError("E, epsdot0 and maximum strain must be positive.")
        if (
            isinstance(self.n_points, bool)
            or not isinstance(self.n_points, int)
            or self.n_points < 2
        ):
            raise SteelInputError("Number of points must be an integer of at least 2.")
        if self.output_kind not in {"true", "engineering"}:
            raise SteelInputError("Output must be true or engineering.")
        object.__setattr__(self, "material_overrides", MappingProxyType(overrides))
