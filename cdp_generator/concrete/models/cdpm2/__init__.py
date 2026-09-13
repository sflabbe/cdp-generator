"""Backend-agnostic CDPM2 Grassl 2013 semantic schema foundation."""

from .configuration import (
    CDPM2_CONFIGURATION_SCHEMA_VERSION,
    CDPM2_OVERRIDE_FIELDS,
    CDPM2_READINESS_SCHEMA_VERSION,
    Cdpm2ConversionReadiness,
    Cdpm2Grassl2013Configuration,
    Cdpm2ReadinessAssessment,
)
from .provenance import Cdpm2ParameterProvenance, Cdpm2SourceKind
from .schema import (
    CDPM2_MODEL_ID,
    CDPM2_PARAMETER_FIELDS,
    CDPM2_PARAMETER_UNITS,
    CDPM2_PARAMETERS_SCHEMA_VERSION,
    CDPM2_STATIC_CALIBRATION_ID,
    Cdpm2DamageFormulation,
    Cdpm2Grassl2013Parameters,
    Cdpm2TensileSofteningType,
)

__all__ = [
    "CDPM2_CONFIGURATION_SCHEMA_VERSION",
    "CDPM2_MODEL_ID",
    "CDPM2_OVERRIDE_FIELDS",
    "CDPM2_PARAMETERS_SCHEMA_VERSION",
    "CDPM2_PARAMETER_FIELDS",
    "CDPM2_PARAMETER_UNITS",
    "CDPM2_READINESS_SCHEMA_VERSION",
    "CDPM2_STATIC_CALIBRATION_ID",
    "Cdpm2ConversionReadiness",
    "Cdpm2DamageFormulation",
    "Cdpm2Grassl2013Configuration",
    "Cdpm2Grassl2013Parameters",
    "Cdpm2ParameterProvenance",
    "Cdpm2ReadinessAssessment",
    "Cdpm2SourceKind",
    "Cdpm2TensileSofteningType",
]
