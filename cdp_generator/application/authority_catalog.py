"""Small frontend catalog derived from domain registries and constants."""

from dataclasses import dataclass
from typing import Literal

from ..concrete import Concrete, class_entries
from ..concrete.class_registry import SUPPORTED_CLASS_PROFILES
from ..concrete.models.cdpm2 import CDPM2_OVERRIDE_FIELDS, CDPM2_PARAMETER_UNITS
from ..concrete.standards import ec2_2004, ec2_2023, fib_mc2010

PROFILE_LABELS = {
    "fib_mc2010": "fib Model Code 2010",
    "ec2_2004": "EC2 2004 — EN 1992-1-1:2004",
    "ec2_2023": "EC2 2023 — EN 1992-1-1:2023",
}


@dataclass(frozen=True)
class ProfileParameterSpec:
    id: str
    label: str
    kind: Literal["choice", "number"]
    default: str | float
    unit: str = ""
    options: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    editable: bool = True


def available_physical_profiles() -> tuple[str, ...]:
    return SUPPORTED_CLASS_PROFILES


def available_concrete_classes(profile: str) -> tuple[str, ...]:
    return tuple(entry.canonical_class_string for entry in class_entries(profile))


def physical_profile_label(profile: str) -> str:
    return PROFILE_LABELS[profile]


def profile_parameter_specs(profile: str) -> tuple[ProfileParameterSpec, ...]:
    classes = available_concrete_classes(profile)
    # The public builder supplies defaults not exposed as module constants.
    effective = Concrete.from_class(classes[0], profile=profile).profile_parameters.to_dict()
    if profile == fib_mc2010.PROFILE_ID or profile == ec2_2004.PROFILE_ID:
        options = (
            tuple(fib_mc2010.AGGREGATE_ALPHA_E)
            if profile == fib_mc2010.PROFILE_ID
            else tuple(ec2_2004.AGGREGATE_ECM_FACTOR)
        )
        age = (
            fib_mc2010.REFERENCE_AGE_DAYS
            if profile == fib_mc2010.PROFILE_ID
            else ec2_2004.REFERENCE_AGE_DAYS
        )
        return (
            ProfileParameterSpec(
                "aggregate_type",
                "Aggregate type",
                "choice",
                str(effective["aggregate_type"]),
                options=options,
            ),
            ProfileParameterSpec(
                "reference_age_days", "Reference age [days]", "number", age, "days", editable=False
            ),
        )
    return (
        ProfileParameterSpec(
            "k_E",
            "k_E",
            "number",
            ec2_2023.DEFAULT_K_E,
            minimum=ec2_2023.MIN_K_E,
            maximum=ec2_2023.MAX_K_E,
        ),
        ProfileParameterSpec(
            "reference_age_days",
            "Reference age [days]",
            "number",
            ec2_2023.DEFAULT_REFERENCE_AGE_DAYS,
            "days",
            minimum=ec2_2023.MIN_REFERENCE_AGE_DAYS,
            maximum=ec2_2023.MAX_REFERENCE_AGE_DAYS,
        ),
    )


def cdpm2_override_fields() -> tuple[str, ...]:
    return CDPM2_OVERRIDE_FIELDS


def cdpm2_parameter_units() -> dict[str, str]:
    return dict(CDPM2_PARAMETER_UNITS)
