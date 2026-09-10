"""Explicit verified-standard concrete class registry.

Entries mirror ``qualification/standards/authority_matrix.json``. Runtime class
parsing never opens repository qualification artifacts; parity is proven by
qualification tests.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from types import MappingProxyType

SUPPORTED_CLASS_PROFILES: tuple[str, ...] = ("fib_mc2010", "ec2_2004", "ec2_2023")

_MC2010_REFERENCE_AGE = (
    "28 days unless explicitly specified otherwise by the concrete definition/"
    "time-development context"
)
_EC2_2004_REFERENCE_AGE = (
    "28 days for the named strength classes and Table 3.1 reference properties"
)
_EC2_2023_REFERENCE_AGE = "t_ref = 28 days in general; project may specify t_ref from 28 to 91 days"


@dataclass(frozen=True, slots=True)
class ConcreteClassEntry:
    """One authority-adjudicated concrete-class identity."""

    profile: str
    canonical_class_string: str
    f_ck_cylinder_mpa: float
    f_ck_cube_mpa: float
    reference_age_semantics: str
    profile_range_endpoint: bool


class ConcreteClassError(ValueError):
    """Base error for strict class-registry parsing."""


class UnknownPhysicalProfileError(ConcreteClassError):
    """The requested physical profile has no class registry."""


class MalformedConcreteClassError(ConcreteClassError):
    """The supplied class string does not use canonical profile syntax."""


class UnsupportedConcreteClassError(ConcreteClassError):
    """The class syntax is valid but absent from the requested registry."""


class CrossProfileConcreteClassError(ConcreteClassError):
    """The class is canonical for another profile but not the requested one."""


_CLASS_ENTRIES: tuple[ConcreteClassEntry, ...] = (
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C12",
        f_ck_cylinder_mpa=12.0,
        f_ck_cube_mpa=15.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=True,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C16",
        f_ck_cylinder_mpa=16.0,
        f_ck_cube_mpa=20.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C20",
        f_ck_cylinder_mpa=20.0,
        f_ck_cube_mpa=25.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C25",
        f_ck_cylinder_mpa=25.0,
        f_ck_cube_mpa=30.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C30",
        f_ck_cylinder_mpa=30.0,
        f_ck_cube_mpa=37.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C35",
        f_ck_cylinder_mpa=35.0,
        f_ck_cube_mpa=45.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C40",
        f_ck_cylinder_mpa=40.0,
        f_ck_cube_mpa=50.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C45",
        f_ck_cylinder_mpa=45.0,
        f_ck_cube_mpa=55.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C50",
        f_ck_cylinder_mpa=50.0,
        f_ck_cube_mpa=60.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C55",
        f_ck_cylinder_mpa=55.0,
        f_ck_cube_mpa=67.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C60",
        f_ck_cylinder_mpa=60.0,
        f_ck_cube_mpa=75.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C70",
        f_ck_cylinder_mpa=70.0,
        f_ck_cube_mpa=85.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C80",
        f_ck_cylinder_mpa=80.0,
        f_ck_cube_mpa=95.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C90",
        f_ck_cylinder_mpa=90.0,
        f_ck_cube_mpa=105.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C100",
        f_ck_cylinder_mpa=100.0,
        f_ck_cube_mpa=115.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C110",
        f_ck_cylinder_mpa=110.0,
        f_ck_cube_mpa=130.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="fib_mc2010",
        canonical_class_string="C120",
        f_ck_cylinder_mpa=120.0,
        f_ck_cube_mpa=140.0,
        reference_age_semantics=_MC2010_REFERENCE_AGE,
        profile_range_endpoint=True,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C12/15",
        f_ck_cylinder_mpa=12.0,
        f_ck_cube_mpa=15.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=True,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C16/20",
        f_ck_cylinder_mpa=16.0,
        f_ck_cube_mpa=20.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C20/25",
        f_ck_cylinder_mpa=20.0,
        f_ck_cube_mpa=25.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C25/30",
        f_ck_cylinder_mpa=25.0,
        f_ck_cube_mpa=30.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C30/37",
        f_ck_cylinder_mpa=30.0,
        f_ck_cube_mpa=37.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C35/45",
        f_ck_cylinder_mpa=35.0,
        f_ck_cube_mpa=45.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C40/50",
        f_ck_cylinder_mpa=40.0,
        f_ck_cube_mpa=50.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C45/55",
        f_ck_cylinder_mpa=45.0,
        f_ck_cube_mpa=55.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C50/60",
        f_ck_cylinder_mpa=50.0,
        f_ck_cube_mpa=60.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C55/67",
        f_ck_cylinder_mpa=55.0,
        f_ck_cube_mpa=67.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C60/75",
        f_ck_cylinder_mpa=60.0,
        f_ck_cube_mpa=75.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C70/85",
        f_ck_cylinder_mpa=70.0,
        f_ck_cube_mpa=85.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C80/95",
        f_ck_cylinder_mpa=80.0,
        f_ck_cube_mpa=95.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2004",
        canonical_class_string="C90/105",
        f_ck_cylinder_mpa=90.0,
        f_ck_cube_mpa=105.0,
        reference_age_semantics=_EC2_2004_REFERENCE_AGE,
        profile_range_endpoint=True,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C12/15",
        f_ck_cylinder_mpa=12.0,
        f_ck_cube_mpa=15.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=True,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C16/20",
        f_ck_cylinder_mpa=16.0,
        f_ck_cube_mpa=20.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C20/25",
        f_ck_cylinder_mpa=20.0,
        f_ck_cube_mpa=25.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C25/30",
        f_ck_cylinder_mpa=25.0,
        f_ck_cube_mpa=30.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C30/37",
        f_ck_cylinder_mpa=30.0,
        f_ck_cube_mpa=37.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C35/45",
        f_ck_cylinder_mpa=35.0,
        f_ck_cube_mpa=45.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C40/50",
        f_ck_cylinder_mpa=40.0,
        f_ck_cube_mpa=50.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C45/55",
        f_ck_cylinder_mpa=45.0,
        f_ck_cube_mpa=55.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C50/60",
        f_ck_cylinder_mpa=50.0,
        f_ck_cube_mpa=60.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C55/67",
        f_ck_cylinder_mpa=55.0,
        f_ck_cube_mpa=67.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C60/75",
        f_ck_cylinder_mpa=60.0,
        f_ck_cube_mpa=75.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C70/85",
        f_ck_cylinder_mpa=70.0,
        f_ck_cube_mpa=85.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C80/95",
        f_ck_cylinder_mpa=80.0,
        f_ck_cube_mpa=95.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C90/105",
        f_ck_cylinder_mpa=90.0,
        f_ck_cube_mpa=105.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=False,
    ),
    ConcreteClassEntry(
        profile="ec2_2023",
        canonical_class_string="C100/115",
        f_ck_cylinder_mpa=100.0,
        f_ck_cube_mpa=115.0,
        reference_age_semantics=_EC2_2023_REFERENCE_AGE,
        profile_range_endpoint=True,
    ),
)

_REGISTRY = MappingProxyType(
    {
        profile: MappingProxyType(
            {
                entry.canonical_class_string: entry
                for entry in _CLASS_ENTRIES
                if entry.profile == profile
            }
        )
        for profile in SUPPORTED_CLASS_PROFILES
    }
)


def class_entries(profile: str | None = None) -> tuple[ConcreteClassEntry, ...]:
    """Return immutable registry entries, optionally for a single profile."""

    if profile is None:
        return _CLASS_ENTRIES
    if profile not in _REGISTRY:
        raise UnknownPhysicalProfileError(f"Unknown physical concrete profile: {profile!r}")
    return tuple(_REGISTRY[profile].values())


def _canonical_elsewhere(concrete_class: str, requested_profile: str) -> Iterator[str]:
    for profile, entries in _REGISTRY.items():
        if profile != requested_profile and concrete_class in entries:
            yield profile


def _has_profile_syntax(profile: str, concrete_class: str) -> bool:
    if profile == "fib_mc2010":
        return re.fullmatch(r"C[0-9]+", concrete_class) is not None
    return re.fullmatch(r"C[0-9]+/[0-9]+", concrete_class) is not None


def parse_concrete_class(profile: str, concrete_class: str) -> ConcreteClassEntry:
    """Strictly parse a canonical class identity for the selected profile."""

    if profile not in _REGISTRY:
        raise UnknownPhysicalProfileError(f"Unknown physical concrete profile: {profile!r}")

    entry = _REGISTRY[profile].get(concrete_class)
    if entry is not None:
        return entry

    other_profiles = tuple(_canonical_elsewhere(concrete_class, profile))
    if other_profiles:
        names = ", ".join(other_profiles)
        raise CrossProfileConcreteClassError(
            f"Concrete class {concrete_class!r} is canonical for {names}, not for {profile!r}"
        )

    if not _has_profile_syntax(profile, concrete_class):
        raise MalformedConcreteClassError(
            f"Malformed concrete class {concrete_class!r} for profile {profile!r}"
        )

    raise UnsupportedConcreteClassError(
        f"Unsupported concrete class {concrete_class!r} for profile {profile!r}"
    )
