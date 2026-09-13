"""Physical-profile identities and construction-path resolvers."""

from typing import Protocol

from .class_registry import ConcreteClassEntry
from .configuration import ProfileConfiguration
from .schema import ConcretePhysicalProperties
from .standards.ec2_2004 import Ec2_2004Profile
from .standards.ec2_2023 import Ec2_2023Profile
from .standards.fib_mc2010 import FibMc2010Profile
from .standards.legacy_v1 import LegacyV1Profile

LEGACY_PHYSICAL_PROFILE = "legacy_v1"
FIB_MC2010_PHYSICAL_PROFILE = "fib_mc2010"
EC2_2004_PHYSICAL_PROFILE = "ec2_2004"
EC2_2023_PHYSICAL_PROFILE = "ec2_2023"
LEGACY_ABAQUS_CALIBRATION = "abaqus_cdp_legacy"

RESERVED_PHYSICAL_PROFILES: tuple[str, ...] = ()
RESERVED_CONSTITUTIVE_PROFILES: tuple[str, ...] = ()


class ClassPhysicalProfile(Protocol):
    """Construction contract shared by verified named-class physical profiles."""

    profile_id: str

    def build(
        self,
        class_entry: ConcreteClassEntry,
        profile_parameters: ProfileConfiguration | None = None,
    ) -> tuple[ConcretePhysicalProperties, ProfileConfiguration]: ...


def get_physical_profile(profile: str) -> LegacyV1Profile:
    """Resolve the legacy mean-strength construction path only."""

    if profile == LEGACY_PHYSICAL_PROFILE:
        return LegacyV1Profile()
    if profile in (
        FIB_MC2010_PHYSICAL_PROFILE,
        EC2_2004_PHYSICAL_PROFILE,
        EC2_2023_PHYSICAL_PROFILE,
    ):
        raise NotImplementedError(
            f"G1 verified profile {profile!r} is class-based; use Concrete.from_class()"
        )
    if profile in RESERVED_PHYSICAL_PROFILES:
        raise NotImplementedError(
            f"Physical profile {profile!r} is reserved for G1 and is not implemented"
        )
    raise ValueError(f"Unknown physical concrete profile: {profile!r}")


def get_class_physical_profile(profile: str) -> ClassPhysicalProfile:
    """Resolve an implemented verified named-class physical profile."""

    if profile == FIB_MC2010_PHYSICAL_PROFILE:
        return FibMc2010Profile()
    if profile == EC2_2004_PHYSICAL_PROFILE:
        return Ec2_2004Profile()
    if profile == EC2_2023_PHYSICAL_PROFILE:
        return Ec2_2023Profile()
    if profile in RESERVED_PHYSICAL_PROFILES:
        raise NotImplementedError(
            f"Physical profile {profile!r} is reserved for a later G1 slice and is not implemented"
        )
    raise ValueError(f"Unknown class-based physical concrete profile: {profile!r}")
