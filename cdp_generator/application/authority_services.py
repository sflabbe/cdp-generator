"""Authority-aware application boundary; all science remains in the domain."""

from ..concrete import Concrete, ProfileConfiguration
from ..concrete.models.cdpm2 import (
    CDPM2_LEGACY_SLOT_NAMES,
    CDPM2_STATIC_CALIBRATION_ID,
    Cdpm2ConversionReadiness,
    Cdpm2Grassl2013Configuration,
    adapt_cdpm2_legacy_backend,
)
from ..concrete.schema import PHYSICAL_UNITS
from .authority_catalog import available_physical_profiles
from .authority_requests import (
    AuthorityConcreteRequest,
    AuthorityInputError,
    Cdpm2ConversionRequest,
)
from .authority_results import (
    AuthorityConcreteResult,
    Cdpm2ConversionResult,
    PhysicalPropertyRecord,
)


def _build_material(request: AuthorityConcreteRequest) -> tuple[Concrete, AuthorityConcreteResult]:
    if request.physical_profile not in available_physical_profiles():
        raise AuthorityInputError("Choose an implemented class-based verified physical profile.")
    try:
        concrete = Concrete.from_class(
            request.concrete_class,
            profile=request.physical_profile,
            profile_parameters=ProfileConfiguration(request.profile_parameters),
        )
    except (ValueError, TypeError) as exc:
        raise AuthorityInputError(str(exc)) from exc
    physical = concrete.physical
    resolution = physical.resolution_dict()
    provenance = physical.provenance_dict()
    result = AuthorityConcreteResult(
        request.physical_profile,
        request.concrete_class,
        dict(request.profile_parameters),
        concrete.profile_parameters.to_dict(),
        [
            PhysicalPropertyRecord(
                name, value, PHYSICAL_UNITS[name], resolution[name], provenance.get(name)
            )
            for name, value in physical.values_dict().items()
        ],
        concrete.to_dict(schema_version="v2"),
    )
    result.to_json()
    return concrete, result


def build_authority_concrete(request: AuthorityConcreteRequest) -> AuthorityConcreteResult:
    return _build_material(request)[1]


def run_cdpm2_conversion(request: Cdpm2ConversionRequest) -> Cdpm2ConversionResult:
    concrete, material = _build_material(request.material)
    configuration = Cdpm2Grassl2013Configuration(overrides=request.overrides)
    readiness = concrete.cdpm2_readiness(configuration=configuration)
    semantics = None
    backend = None
    capabilities = ["material_json", "application_json"]
    if readiness.state is Cdpm2ConversionReadiness.READY:
        parameters = concrete.to_cdpm2(configuration=configuration)
        semantics = parameters.to_dict()
        capabilities.append("semantic_json")
        if request.characteristic_length_mm is not None:
            backend = adapt_cdpm2_legacy_backend(
                parameters, characteristic_length=request.characteristic_length_mm
            ).to_dict()
            capabilities.append("backend_json")
    result = Cdpm2ConversionResult(
        material,
        CDPM2_STATIC_CALIBRATION_ID,
        configuration.to_dict(),
        readiness.to_dict(),
        semantics,
        backend,
        list(CDPM2_LEGACY_SLOT_NAMES),
        [
            "Physical authority, CDPM2 calibration and backend representation are separate. No authority-aware curves are generated."
        ],
        capabilities,
    )
    result.to_json()
    return result
