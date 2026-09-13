"""Constitutive provenance types for the CDPM2 Grassl 2013 parameter schema."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Cdpm2SourceKind(StrEnum):
    """Authority domain for one effective CDPM2 semantic parameter."""

    PHYSICAL_SOURCE = "PHYSICAL_SOURCE"
    GRASSL_2013_DIRECT = "GRASSL_2013_DIRECT"
    GRASSL_2013_DERIVED = "GRASSL_2013_DERIVED"
    GRASSL_2013_DEFAULT_CALIBRATION = "GRASSL_2013_DEFAULT_CALIBRATION"
    OOFEM_IMPLEMENTATION_DEFAULT = "OOFEM_IMPLEMENTATION_DEFAULT"
    DOWNSTREAM_ADAPTER_FIXED = "DOWNSTREAM_ADAPTER_FIXED"
    USER_CONSTITUTIVE_OVERRIDE = "USER_CONSTITUTIVE_OVERRIDE"


@dataclass(frozen=True, slots=True)
class Cdpm2ParameterProvenance:
    """Immutable constitutive authority record for one CDPM2 semantic field.

    This deliberately does not reuse the G1 physical-property provenance schema:
    constitutive literature/default/calibration authority is a separate domain from
    normative physical-property authority.
    """

    source_id: str
    source_kind: Cdpm2SourceKind
    edition: str | None
    equation_or_section: str | None
    units: str
    notes: str = ""
    overridden: bool = False
    derived_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be non-empty")
        if not isinstance(self.source_kind, Cdpm2SourceKind):
            raise TypeError("source_kind must be Cdpm2SourceKind")
        if self.edition is not None and not isinstance(self.edition, str):
            raise TypeError("edition must be a string or None")
        if self.equation_or_section is not None and not isinstance(self.equation_or_section, str):
            raise TypeError("equation_or_section must be a string or None")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")
        if not isinstance(self.overridden, bool):
            raise TypeError("overridden must be bool")

        derived_from = tuple(self.derived_from)
        if any(not isinstance(name, str) or not name.strip() for name in derived_from):
            raise ValueError("derived_from dependencies must be non-empty strings")
        if len(set(derived_from)) != len(derived_from):
            raise ValueError("derived_from dependencies must be unique")

        if self.source_kind is Cdpm2SourceKind.GRASSL_2013_DERIVED and not derived_from:
            raise ValueError("GRASSL_2013_DERIVED provenance requires derived_from dependencies")
        if self.source_kind is Cdpm2SourceKind.USER_CONSTITUTIVE_OVERRIDE and not self.overridden:
            raise ValueError("USER_CONSTITUTIVE_OVERRIDE provenance requires overridden=True")
        if self.source_kind is not Cdpm2SourceKind.USER_CONSTITUTIVE_OVERRIDE and self.overridden:
            raise ValueError(
                "overridden=True is reserved for USER_CONSTITUTIVE_OVERRIDE provenance"
            )

        object.__setattr__(self, "derived_from", derived_from)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-serializable constitutive provenance."""

        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind.value,
            "edition": self.edition,
            "equation_or_section": self.equation_or_section,
            "units": self.units,
            "notes": self.notes,
            "overridden": self.overridden,
            "derived_from": list(self.derived_from),
        }
