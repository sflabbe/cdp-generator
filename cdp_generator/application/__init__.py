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
from .comparison import (
    ComparisonCase,
    ComparisonInputError,
    add_comparison_case,
    authority_comparison_tables,
    compatible_curve_groups,
    curve_semantic_signature,
    make_comparison_case,
    rename_comparison_case,
    steel_comparison_table,
)
from .requests import LegacyConcreteAnalysisRequest
from .results import CurveSeries, PropertyValue, ResultBundle
from .services import result_excel_bytes, run_legacy_concrete_analysis, temperature_cases
from .steel_catalog import available_steel_grades, available_steel_standards
from .steel_requests import SteelAnalysisRequest, SteelInputError, parse_float_series
from .steel_results import SteelAnalysisResult
from .steel_services import run_steel_analysis, steel_result_abaqus_text, steel_result_excel_bytes

__all__ = [
    "AuthorityConcreteRequest",
    "AuthorityConcreteResult",
    "AuthorityInputError",
    "Cdpm2ConversionRequest",
    "Cdpm2ConversionResult",
    "ComparisonCase",
    "ComparisonInputError",
    "CurveSeries",
    "LegacyConcreteAnalysisRequest",
    "PhysicalPropertyRecord",
    "ProfileParameterSpec",
    "PropertyValue",
    "ResultBundle",
    "SteelAnalysisRequest",
    "SteelAnalysisResult",
    "SteelInputError",
    "add_comparison_case",
    "authority_comparison_tables",
    "available_concrete_classes",
    "available_physical_profiles",
    "available_steel_grades",
    "available_steel_standards",
    "build_authority_concrete",
    "cdpm2_override_fields",
    "cdpm2_parameter_units",
    "compatible_curve_groups",
    "curve_semantic_signature",
    "make_comparison_case",
    "parse_cdpm2_overrides",
    "parse_float_series",
    "physical_profile_label",
    "profile_parameter_specs",
    "rename_comparison_case",
    "result_excel_bytes",
    "run_cdpm2_conversion",
    "run_legacy_concrete_analysis",
    "run_steel_analysis",
    "steel_comparison_table",
    "steel_result_abaqus_text",
    "steel_result_excel_bytes",
    "temperature_cases",
]
