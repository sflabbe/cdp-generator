"""Frontend-independent analysis API."""

from .authority_catalog import (
    ProfileParameterSpec,
    available_concrete_classes,
    available_physical_profiles,
    cdpm2_override_fields,
    cdpm2_parameter_units,
    physical_profile_label,
    profile_parameter_specs,
)
from .authority_requests import (
    AuthorityConcreteRequest,
    AuthorityInputError,
    Cdpm2ConversionRequest,
    parse_cdpm2_overrides,
)
from .authority_results import (
    AuthorityConcreteResult,
    Cdpm2ConversionResult,
    PhysicalPropertyRecord,
)
from .authority_services import build_authority_concrete, run_cdpm2_conversion
from .requests import LegacyConcreteAnalysisRequest
from .results import CurveSeries, PropertyValue, ResultBundle
from .services import result_excel_bytes, run_legacy_concrete_analysis, temperature_cases

__all__ = [
    "AuthorityConcreteRequest",
    "AuthorityConcreteResult",
    "AuthorityInputError",
    "Cdpm2ConversionRequest",
    "Cdpm2ConversionResult",
    "CurveSeries",
    "LegacyConcreteAnalysisRequest",
    "PhysicalPropertyRecord",
    "ProfileParameterSpec",
    "PropertyValue",
    "ResultBundle",
    "available_concrete_classes",
    "available_physical_profiles",
    "build_authority_concrete",
    "cdpm2_override_fields",
    "cdpm2_parameter_units",
    "parse_cdpm2_overrides",
    "physical_profile_label",
    "profile_parameter_specs",
    "result_excel_bytes",
    "run_cdpm2_conversion",
    "run_legacy_concrete_analysis",
    "temperature_cases",
]
