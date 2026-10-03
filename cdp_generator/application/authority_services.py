"""Authority-aware application boundary; all science remains in the domain."""

import math
from dataclasses import replace

from ..concrete import Concrete, ProfileConfiguration
from ..concrete.models.cdpm2 import (
    CDPM2_LEGACY_SLOT_NAMES,
    CDPM2_STATIC_CALIBRATION_ID,
    Cdpm2ConversionReadiness,
    Cdpm2FractureEnergyComposition,
    Cdpm2Grassl2013Configuration,
    Cdpm2Grassl2013Parameters,
    adapt_cdpm2_legacy_backend,
)
from ..concrete.provenance import PropertyResolutionStatus
from ..concrete.schema import PHYSICAL_UNITS
from ..concrete.standards.fib_mc2010 import estimate_fib_mc2010_fracture_energy
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
from .results import CurveSeries


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


class Cdpm2SofteningInvariantError(RuntimeError):
    """Raised when resolved CDPM2 semantic landmarks violate their energy invariant."""


def fib_mc2010_fracture_energy_composition(
    concrete: Concrete,
) -> Cdpm2FractureEnergyComposition:
    """Compose secondary fib MC2010 G_F authority from the primary material f_cm."""

    physical = concrete.physical
    status = physical.resolution["f_cm"]
    if (
        status not in {PropertyResolutionStatus.DIRECT, PropertyResolutionStatus.DERIVED}
        or "f_cm" not in physical.provenance
    ):
        raise AuthorityInputError(
            "fib MC2010 fracture-energy composition requires resolved primary physical f_cm."
        )
    value, provenance = estimate_fib_mc2010_fracture_energy(float(physical.f_cm))
    provenance = replace(
        provenance,
        notes=(
            provenance.notes
            + f" Applied as secondary physical authority to f_cm resolved by primary profile "
            f"{concrete.physical_profile!r}; the primary material definition is not mutated."
        ),
    )
    return Cdpm2FractureEnergyComposition(value, provenance)


def cdpm2_tensile_softening_curves(
    parameters: Cdpm2Grassl2013Parameters,
    characteristic_length_mm: float | None = None,
) -> list[CurveSeries]:
    """Build exact canonical views of the resolved bilinear CDPM2 tensile-softening law."""

    x = [0.0, float(parameters.w_f1), float(parameters.w_f)]
    y = [float(parameters.f_t), float(parameters.f_t1), 0.0]
    area = sum(0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i]) for i in range(2))
    if not math.isclose(area, float(parameters.G_Ft), rel_tol=1e-12, abs_tol=1e-14):
        raise Cdpm2SofteningInvariantError(
            f"CDPM2 tensile-softening area {area!r} does not equal G_Ft {parameters.G_Ft!r}."
        )
    metadata = {
        "G_Ft": float(parameters.G_Ft),
        "integrated_area": area,
        "tensile_softening_type": parameters.tensile_softening_type.value,
        "calibration_id": parameters.calibration_id,
    }
    curves = [
        CurveSeries(
            "cdpm2.tensile_softening.crack_opening",
            "Resolved bilinear tensile softening",
            "cdpm2_tensile_softening",
            x,
            y,
            "Crack opening",
            "Tensile stress",
            "mm",
            "MPa",
            metadata,
        )
    ]
    if characteristic_length_mm is not None:
        if not math.isfinite(characteristic_length_mm) or characteristic_length_mm <= 0.0:
            raise AuthorityInputError("LCHAR must be finite and strictly positive [mm].")
        curves.append(
            CurveSeries(
                "cdpm2.tensile_softening.regularized",
                "Regularized tensile-softening view",
                "cdpm2_regularized_tensile_softening",
                [value / characteristic_length_mm for value in x],
                y.copy(),
                "Regularized cracking strain",
                "Tensile stress",
                "dimensionless",
                "MPa",
                {**metadata, "characteristic_length_mm": characteristic_length_mm},
            )
        )
    return curves


def run_cdpm2_conversion(request: Cdpm2ConversionRequest) -> Cdpm2ConversionResult:
    concrete, material = _build_material(request.material)
    configuration = Cdpm2Grassl2013Configuration(overrides=request.overrides)
    composition = None
    if request.fracture_energy_policy == "fib_mc2010_if_missing":
        physical = concrete.physical
        physical_gf_usable = (
            physical.fracture_energy is not None
            and physical.resolution["fracture_energy"]
            in {PropertyResolutionStatus.DIRECT, PropertyResolutionStatus.DERIVED}
            and "fracture_energy" in physical.provenance
        )
        if not physical_gf_usable:
            composition = fib_mc2010_fracture_energy_composition(concrete)

    readiness = concrete.cdpm2_readiness(
        configuration=configuration, fracture_energy_composition=composition
    )
    semantics = None
    backend = None
    curves: list[CurveSeries] = []
    capabilities = ["material_json", "application_json"]
    if readiness.state is Cdpm2ConversionReadiness.READY:
        parameters = concrete.to_cdpm2(
            configuration=configuration, fracture_energy_composition=composition
        )
        semantics = parameters.to_dict()
        curves = cdpm2_tensile_softening_curves(parameters, request.characteristic_length_mm)
        capabilities.extend(["semantic_json", "softening_plot_data"])
        if request.characteristic_length_mm is not None:
            backend = adapt_cdpm2_legacy_backend(
                parameters, characteristic_length=request.characteristic_length_mm
            ).to_dict()
            capabilities.append("backend_json")
    result = Cdpm2ConversionResult(
        material=material,
        calibration=CDPM2_STATIC_CALIBRATION_ID,
        configuration=configuration.to_dict(),
        readiness=readiness.to_dict(),
        semantic_parameters=semantics,
        backend=backend,
        fracture_energy_composition=(
            {
                "strategy": request.fracture_energy_policy,
                "value": composition.value,
                "unit": composition.provenance.units,
                "provenance": composition.provenance.to_dict(),
            }
            if composition is not None
            else None
        ),
        curves=curves,
        backend_slot_names=list(CDPM2_LEGACY_SLOT_NAMES),
        warnings=[
            "Physical authority, CDPM2 calibration and backend representation are separate.",
            "CDPM2 curves visualize the resolved bilinear tensile-softening input law; "
            "they are not an integrated uniaxial constitutive response.",
        ],
        export_capabilities=capabilities,
    )
    result.to_json()
    return result
