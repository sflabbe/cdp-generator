"""Validated, frontend-independent legacy analysis inputs."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

AnalysisMode = Literal["strain_rate", "temperature"]


@dataclass(frozen=True)
class LegacyConcreteAnalysisRequest:
    mode: AnalysisMode = "strain_rate"
    f_cm: float = 28.0
    e_c1: float = 0.0022
    e_clim: float = 0.0035
    l_ch: float = 1.0
    strain_rates: tuple[float, ...] = (0.0, 2.0, 30.0, 100.0)

    def __post_init__(self) -> None:
        if self.mode not in ("strain_rate", "temperature"):
            raise ValueError("Choose strain_rate or temperature mode.")
        for name in ("f_cm", "e_c1", "e_clim", "l_ch"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        object.__setattr__(self, "strain_rates", tuple(self.strain_rates))
        if self.mode == "strain_rate" and (
            not self.strain_rates
            or any(not isfinite(rate) or rate < 0 for rate in self.strain_rates)
        ):
            raise ValueError("Enter one or more finite, non-negative strain rates.")


def parse_strain_rates(text: str) -> tuple[float, ...]:
    try:
        return tuple(float(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise ValueError("Use comma-separated numeric strain rates, e.g. 0, 2, 30, 100.") from exc
