"""Catalog derived exclusively from the existing approximate steel database."""

from ..steel.standards import list_available_standards
from .steel_requests import SteelInputError


def available_steel_standards() -> tuple[str, ...]:
    return tuple(list_available_standards())


def available_steel_grades(standard: str) -> tuple[str, ...]:
    catalog = list_available_standards()
    if standard not in catalog:
        raise SteelInputError(f"Unknown steel standard: {standard}")
    return tuple(catalog[standard])
