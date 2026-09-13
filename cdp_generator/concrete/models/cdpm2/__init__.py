"""Backend-agnostic CDPM2 Grassl 2013 semantic schema foundation."""

from .backend import (
    CDPM2_BACKEND_ID,
    CDPM2_BACKEND_INPUT_SCHEMA_VERSION,
    CDPM2_CHARACTERISTIC_LENGTH_UNITS,
    CDPM2_LEGACY_FIXED_SLOTS,
    CDPM2_LEGACY_SEMANTIC_SLOT_MAP,
    CDPM2_LEGACY_SLOT_COUNT,
    CDPM2_LEGACY_SLOT_NAMES,
    Cdpm2Legacy24BackendInput,
    adapt_cdpm2_legacy_backend,
)
from .configuration import (
    CDPM2_CONFIGURATION_SCHEMA_VERSION,
    CDPM2_OVERRIDE_FIELDS,
    CDPM2_READINESS_SCHEMA_VERSION,
    Cdpm2ConversionReadiness,
    Cdpm2Grassl2013Configuration,
    Cdpm2ReadinessAssessment,
)
from .mapping import (
    CDPM2_FRACTURE_ENERGY_COMPOSITION_SCHEMA_VERSION,
    Cdpm2ConversionNotReadyError,
    Cdpm2FractureEnergyComposition,
    assess_cdpm2_grassl_2013_readiness,
    resolve_cdpm2_grassl_2013,
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
    "CDPM2_BACKEND_ID",
    "CDPM2_BACKEND_INPUT_SCHEMA_VERSION",
    "CDPM2_CHARACTERISTIC_LENGTH_UNITS",
    "CDPM2_CONFIGURATION_SCHEMA_VERSION",
    "CDPM2_FRACTURE_ENERGY_COMPOSITION_SCHEMA_VERSION",
    "CDPM2_LEGACY_FIXED_SLOTS",
    "CDPM2_LEGACY_SEMANTIC_SLOT_MAP",
    "CDPM2_LEGACY_SLOT_COUNT",
    "CDPM2_LEGACY_SLOT_NAMES",
    "CDPM2_MODEL_ID",
    "CDPM2_OVERRIDE_FIELDS",
    "CDPM2_PARAMETERS_SCHEMA_VERSION",
    "CDPM2_PARAMETER_FIELDS",
    "CDPM2_PARAMETER_UNITS",
    "CDPM2_READINESS_SCHEMA_VERSION",
    "CDPM2_STATIC_CALIBRATION_ID",
    "Cdpm2ConversionNotReadyError",
    "Cdpm2ConversionReadiness",
    "Cdpm2DamageFormulation",
    "Cdpm2FractureEnergyComposition",
    "Cdpm2Grassl2013Configuration",
    "Cdpm2Grassl2013Parameters",
    "Cdpm2Legacy24BackendInput",
    "Cdpm2ParameterProvenance",
    "Cdpm2ReadinessAssessment",
    "Cdpm2SourceKind",
    "Cdpm2TensileSofteningType",
    "adapt_cdpm2_legacy_backend",
    "assess_cdpm2_grassl_2013_readiness",
    "resolve_cdpm2_grassl_2013",
]
