"""Profile identities and G0 resolver functions."""

from .standards.legacy_v1 import LegacyV1Profile

LEGACY_PHYSICAL_PROFILE = "legacy_v1"
LEGACY_ABAQUS_CALIBRATION = "abaqus_cdp_legacy"

RESERVED_PHYSICAL_PROFILES: tuple[str, ...] = (
    "fib_mc2010",
    "ec2_2023",
    "ec2_2004",
)
RESERVED_CONSTITUTIVE_PROFILES: tuple[str, ...] = ("cdpm2_grassl_2013",)


def get_physical_profile(profile: str) -> LegacyV1Profile:
    """Resolve an implemented physical profile without guessing future authority."""

    if profile == LEGACY_PHYSICAL_PROFILE:
        return LegacyV1Profile()
    if profile in RESERVED_PHYSICAL_PROFILES:
        raise NotImplementedError(
            f"Physical profile {profile!r} is reserved for G1 and is not implemented in G0"
        )
    raise ValueError(f"Unknown physical concrete profile: {profile!r}")
