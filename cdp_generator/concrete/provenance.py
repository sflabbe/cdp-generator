"""Typed provenance metadata for concrete material quantities.

The authority-split architecture distinguishes compatibility provenance,
verified normative authority, derived values, and unresolved properties.  G1-B1
adds machine-readable dependency and normalization metadata without changing any
legacy numerical formula.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class SourceKind(StrEnum):
    """Origin category for a generated material quantity."""

    LEGACY_IMPLEMENTATION = "LEGACY_IMPLEMENTATION"
    STANDARD = "STANDARD"
    LITERATURE = "LITERATURE"
    USER_OVERRIDE = "USER_OVERRIDE"
    DERIVED = "DERIVED"


class StatisticalBasis(StrEnum):
    """Statistical meaning attached to a material quantity."""

    MEAN = "MEAN"
    CHARACTERISTIC = "CHARACTERISTIC"
    DESIGN = "DESIGN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNSPECIFIED = "UNSPECIFIED"


class PropertyResolutionStatus(StrEnum):
    """Machine-readable authority/resolution state for one physical field."""

    DIRECT = "DIRECT"
    DERIVED = "DERIVED"
    UNRESOLVED = "UNRESOLVED"
    COMPOSED_REQUIRED = "COMPOSED_REQUIRED"


class NormalizationKind(StrEnum):
    """Small closed vocabulary for source-to-repository normalization."""

    SIGN_CONVENTION = "SIGN_CONVENTION"
    UNIT_CONVERSION = "UNIT_CONVERSION"


@dataclass(frozen=True, slots=True)
class PropertyNormalization:
    """One explicit source-to-repository normalization step."""

    kind: NormalizationKind
    source_convention: str
    repository_convention: str

    def __post_init__(self) -> None:
        if not self.source_convention.strip():
            raise ValueError("source_convention must be non-empty")
        if not self.repository_convention.strip():
            raise ValueError("repository_convention must be non-empty")

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic JSON-serializable representation."""

        return {
            "kind": self.kind.value,
            "source_convention": self.source_convention,
            "repository_convention": self.repository_convention,
        }


@dataclass(frozen=True, slots=True)
class PropertyProvenance:
    """Immutable provenance record for one generated quantity.

    ``derived_from`` records stable physical-field dependencies. ``normalizations``
    records source-to-repository transformations such as compression-sign or unit
    convention changes.  Both are intentionally small and machine-readable; they
    are not a symbolic algebra system.
    """

    source_id: str
    source_kind: SourceKind
    edition: str | None
    equation_or_section: str | None
    units: str
    statistical_basis: StatisticalBasis = StatisticalBasis.UNSPECIFIED
    notes: str = ""
    overridden: bool = False
    derived_from: tuple[str, ...] = ()
    normalizations: tuple[PropertyNormalization, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not self.units.strip():
            raise ValueError("units must be non-empty")

        derived_from = tuple(self.derived_from)
        normalizations = tuple(self.normalizations)
        if any(not dependency.strip() for dependency in derived_from):
            raise ValueError("derived_from dependencies must be non-empty")
        if len(set(derived_from)) != len(derived_from):
            raise ValueError("derived_from dependencies must be unique")

        object.__setattr__(self, "derived_from", derived_from)
        object.__setattr__(self, "normalizations", normalizations)

    def to_legacy_v1_dict(self) -> dict[str, Any]:
        """Return the pre-G1 provenance payload shape for legacy serialization."""

        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind.value,
            "edition": self.edition,
            "equation_or_section": self.equation_or_section,
            "units": self.units,
            "statistical_basis": self.statistical_basis.value,
            "notes": self.notes,
            "overridden": self.overridden,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable v2 representation."""

        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind.value,
            "edition": self.edition,
            "equation_or_section": self.equation_or_section,
            "units": self.units,
            "statistical_basis": self.statistical_basis.value,
            "notes": self.notes,
            "overridden": self.overridden,
            "derived_from": list(self.derived_from),
            "normalizations": [normalization.to_dict() for normalization in self.normalizations],
        }
