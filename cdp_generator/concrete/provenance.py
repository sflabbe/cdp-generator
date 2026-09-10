"""Typed provenance metadata for concrete material quantities.

G0 deliberately distinguishes compatibility provenance from verified normative
standard authority. Existing formulas are therefore identified as
``legacy_v1`` until later gates adjudicate them independently.
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


@dataclass(frozen=True, slots=True)
class PropertyProvenance:
    """Immutable provenance record for one generated quantity.

    Args:
        source_id: Stable source/profile identity. G0 legacy outputs use
            ``legacy_v1`` rather than an unverified standard citation.
        source_kind: High-level authority/origin category.
        edition: Optional edition/version identity for a future verified source.
        equation_or_section: Formula/function/section identity when known.
        units: Units carried by the generated quantity.
        statistical_basis: Mean/characteristic/design meaning where applicable.
        notes: Free-text clarification that must not overstate authority.
        overridden: Whether the value was replaced by an explicit user override.
    """

    source_id: str
    source_kind: SourceKind
    edition: str | None
    equation_or_section: str | None
    units: str
    statistical_basis: StatisticalBasis = StatisticalBasis.UNSPECIFIED
    notes: str = ""
    overridden: bool = False

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not self.units.strip():
            raise ValueError("units must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable representation."""

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
