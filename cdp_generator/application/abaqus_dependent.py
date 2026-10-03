"""Rate- and temperature-dependent legacy Abaqus CDP material backend (ABAQUS-Q1 / WEB-M5).

This module adapts the *existing* legacy rate and temperature curve families to the
dependency columns that Abaqus documents for ``*CONCRETE COMPRESSION HARDENING`` and
``*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT``. It never recomputes, resamples or
re-derives the legacy numerical values:

* compression hardening and tension stiffening rows are the exact kernel arrays;
* the tension rate coordinate comes from the shared legacy mapping
  :func:`cdp_generator.strain_rate.legacy_cracking_displacement_rate`;
* temperature-dependent elasticity comes from
  :func:`cdp_generator.core.calculate_temperature_elastic_states`, i.e. the exact modulus
  used by the legacy inelastic-strain construction;
* damage is either omitted (default) or the single legacy reference-case damage function,
  reused without a rate/temperature column and validated against *every* family.

Abaqus documentation is the authority only for table structure, keyword semantics and the
documented inelastic→plastic conversion; it is not the authority for the legacy values.
The static M4 backend (:mod:`.abaqus_legacy`) is unchanged and remains separate.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Any

from .. import core
from ..strain_rate import legacy_cracking_displacement_rate
from .abaqus_legacy import (
    ABAQUS_DOCUMENTATION_URLS,
    ABAQUS_LEGACY_FULL_CALIBRATION_ID,
    AbaqusCdpSourceKind,
    AbaqusLegacyInputError,
    AbaqusLegacyMaterialRequest,
    AbaqusLegacyValidationError,
    _fmt,
    _normalize_damage,
    abaqus_table_checks,
    run_abaqus_legacy_material,
)
from .results import CurveSeries

ABAQUS_DEPENDENT_SCHEMA_VERSION = "abaqus_legacy_dependent_material_result.v1"
ABAQUS_DEPENDENT_WORKFLOW = "legacy_abaqus_cdp_dependent"
ABAQUS_DEPENDENT_DOCUMENTATION_URLS = (
    *ABAQUS_DOCUMENTATION_URLS,
    ("https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKERRefMap/simaker-c-elasticpyc.htm"),
    (
        "https://docs.software.vt.edu/abaqusv2025/English/"
        "SIMACAEMATRefMap/simamat-c-linearelastic.htm"
    ),
    (
        "https://docs.software.vt.edu/abaqusv2025/English/"
        "SIMACAEMATRefMap/simamat-c-materialdata.htm"
    ),
)

#: Structured mapping identifiers (kept as plain strings in JSON on purpose).
LEGACY_RATE_AXIS_MAPPING = "LEGACY_RATE_AXIS_MAPPING"
LEGACY_CRACK_OPENING_RATE_MAPPING = "LEGACY_CRACK_OPENING_RATE_MAPPING"
LEGACY_CONSTANT_NU_ASSUMPTION = "LEGACY_CONSTANT_NU_ASSUMPTION"
LEGACY_TEMPERATURE_SECANT_MODULUS = "LEGACY_TEMPERATURE_SECANT_MODULUS"
ABAQUS_DOCUMENTATION_REQUIREMENT = "ABAQUS_DOCUMENTATION_REQUIREMENT"

RATE_AXIS_NOTES = (
    "Legacy curve-family control rate is used as the Abaqus compression-hardening rate "
    "dependency coordinate. It is not reconstructed from a loading history."
)
CRACK_OPENING_RATE_NOTES = (
    "Existing legacy crack-opening-rate mapping w_dot = strain_rate * l_ch "
    "(strain_rate.legacy_cracking_displacement_rate, also used by "
    "apply_fracture_energy_rate_effects); units 1/s * mm = mm/s."
)
CONSTANT_NU_NOTES = (
    "Legacy constant-nu assumption: the legacy room/reference elastic Poisson ratio v_ce is "
    "held constant at every temperature; no nu(T) law exists in the legacy kernel."
)
TEMPERATURE_MODULUS_NOTES = (
    "E(T) is the legacy secant modulus E_c1_temp that calculate_stress_strain_temp passes to "
    "calculate_inelastic_compression for each temperature family, so the exported inelastic "
    "strains and the Abaqus elastic strain sigma/E(T) decompose the same legacy total strain. "
    "At 20 °C this value differs from the static M4 E0 (legacy E_c); this is a property of "
    "the legacy temperature kernel, not an Abaqus requirement."
)
REFERENCE_DAMAGE_RATE_NOTES = (
    "reference damage function reused across all rate-dependent hardening/stiffening families"
)
REFERENCE_DAMAGE_TEMPERATURE_NOTES = (
    "reference damage function reused across all temperature-dependent hardening/stiffening "
    "families; the same damage law is applied at all temperatures (no damage(T) is exported "
    "or claimed)"
)


class AbaqusDependentMode(StrEnum):
    STRAIN_RATE = "strain_rate"
    TEMPERATURE = "temperature"


class AbaqusDamagePolicy(StrEnum):
    OMIT = "omit"
    REFERENCE_DAMAGE = "reference_damage"


class AbaqusLegacyDependentValidationError(AbaqusLegacyValidationError):
    """A dependent family violates documented Abaqus conversion constraints.

    ``failures`` lists one mapping per failed check with ``mode``, ``dependency_value``,
    ``dependency_unit``, ``branch`` and ``failed_condition``.
    """

    def __init__(self, message: str, failures: tuple[dict[str, Any], ...]) -> None:
        super().__init__(message)
        self.failures = failures


def _positive_float(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise AbaqusLegacyInputError(f"{name} must be finite and strictly positive.")
    return float(value)


@dataclass(frozen=True, slots=True)
class AbaqusLegacyDependentRequest:
    """Typed request for a rate- or temperature-dependent legacy Abaqus CDP material.

    ``strain_rates`` are the legacy curve-family control rates [1/s] and are required for
    ``mode="strain_rate"``; they must be strictly increasing because Abaqus requires
    dependent data in ascending order of the dependency variable. Temperature cases always
    come from the legacy kernel table and cannot be supplied here.
    """

    mode: str
    f_cm: float = 28.0
    e_c1: float = 0.0022
    e_clim: float = 0.0035
    l_ch: float = 1.0
    strain_rates: tuple[float, ...] = ()
    tension_law: str = "bilinear"
    damage_policy: str = AbaqusDamagePolicy.OMIT.value
    damage_conversion_reference_length_mm: float = 1.0
    eccentricity: float | None = None
    viscosity: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {m.value for m in AbaqusDependentMode}:
            raise AbaqusLegacyInputError("mode must be strain_rate or temperature.")
        object.__setattr__(self, "mode", str(AbaqusDependentMode(self.mode).value))
        for name in ("f_cm", "e_c1", "e_clim", "l_ch", "damage_conversion_reference_length_mm"):
            object.__setattr__(self, name, _positive_float(name, getattr(self, name)))
        if self.tension_law not in {"bilinear", "power_law"}:
            raise AbaqusLegacyInputError("tension_law must be bilinear or power_law.")
        if self.damage_policy not in {p.value for p in AbaqusDamagePolicy}:
            raise AbaqusLegacyInputError("damage_policy must be omit or reference_damage.")
        object.__setattr__(self, "damage_policy", str(AbaqusDamagePolicy(self.damage_policy).value))
        rates = tuple(self.strain_rates)
        for rate in rates:
            if (
                isinstance(rate, bool)
                or not isinstance(rate, int | float)
                or not math.isfinite(rate)
                or rate < 0.0
            ):
                raise AbaqusLegacyInputError("strain_rates must be finite and >= 0.")
        rates = tuple(float(r) for r in rates)
        if self.mode == AbaqusDependentMode.STRAIN_RATE:
            if not rates:
                raise AbaqusLegacyInputError("strain_rate mode requires one or more strain_rates.")
            if any(b <= a for a, b in pairwise(rates)):
                raise AbaqusLegacyInputError(
                    "strain_rates must be unique and strictly increasing: Abaqus requires "
                    "dependent data in ascending order of the dependency variable."
                )
        elif rates:
            raise AbaqusLegacyInputError(
                "temperature mode takes its cases from the legacy kernel; strain_rates must be "
                "empty (combined rate x temperature surfaces are out of scope)."
            )
        object.__setattr__(self, "strain_rates", rates)
        if self.eccentricity is not None:
            object.__setattr__(
                self, "eccentricity", _positive_float("eccentricity", self.eccentricity)
            )
        if self.viscosity is not None:
            if (
                isinstance(self.viscosity, bool)
                or not isinstance(self.viscosity, int | float)
                or not math.isfinite(self.viscosity)
                or self.viscosity < 0.0
            ):
                raise AbaqusLegacyInputError("viscosity must be finite and >= 0 when overridden.")
            object.__setattr__(self, "viscosity", float(self.viscosity))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["strain_rates"] = list(self.strain_rates)
        return data


@dataclass(frozen=True)
class AbaqusLegacyDependentResult:
    mode: str
    inputs: dict[str, Any]
    elastic: dict[str, Any]
    plasticity: dict[str, float]
    compression_hardening: list[dict[str, float]]
    tension_stiffening: list[dict[str, float]]
    damage_policy: dict[str, Any]
    compression_damage: list[dict[str, float]] | None
    tension_damage: list[dict[str, float]] | None
    dependency_semantics: dict[str, Any]
    backend_configuration: dict[str, Any]
    provenance: dict[str, Any]
    validation: dict[str, Any]
    warnings: list[str]
    documentation_references: list[str]
    export_capabilities: list[str]
    curves: list[CurveSeries] = field(default_factory=list)
    workflow: str = ABAQUS_DEPENDENT_WORKFLOW
    schema_version: str = ABAQUS_DEPENDENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.to_json())

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False)

    def dependency_values(self) -> list[float]:
        return [float(row["dependency_value"]) for row in self.dependency_semantics["families"]]


def _floats(values: Any) -> list[float]:
    return [float(v) for v in values]


def _family_label(mode: str, value: float) -> str:
    if mode == AbaqusDependentMode.TEMPERATURE:
        return f"T = {value:g} °C"
    return f"ε̇ = {value:g} 1/s"


def _family_failures(
    *,
    mode: str,
    value: float,
    unit: str,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    failures = []
    for name in report["hard_failures"]:
        if name.startswith("compression_"):
            branch = "compression"
        elif name.startswith("tension_"):
            branch = "tension"
        else:
            branch = "common"
        failures.append(
            {
                "mode": mode,
                "dependency_value": value,
                "dependency_unit": unit,
                "branch": branch,
                "failed_condition": name,
            }
        )
    return failures


def _strictly_increasing(values: list[float]) -> bool:
    return all(b > a for a, b in pairwise(values))


def run_abaqus_legacy_dependent_material(
    request: AbaqusLegacyDependentRequest,
) -> AbaqusLegacyDependentResult:
    """Build and validate a rate- or temperature-dependent legacy Abaqus CDP material.

    Raises :class:`AbaqusLegacyDependentValidationError` (a subclass of
    :class:`AbaqusLegacyValidationError`) identifying mode, dependency value, branch and
    failed condition when any family violates the documented Abaqus constraints — in
    particular when ``damage_policy="reference_damage"`` is incompatible with any family.
    Damage is never silently dropped for a failing family.
    """

    if not isinstance(request, AbaqusLegacyDependentRequest):
        raise AbaqusLegacyInputError("request must be AbaqusLegacyDependentRequest.")

    mode = request.mode
    is_rate = mode == AbaqusDependentMode.STRAIN_RATE
    reference_damage = request.damage_policy == AbaqusDamagePolicy.REFERENCE_DAMAGE

    # Static M4 backend supplies the unchanged scalar calibration, E0/nu and their provenance.
    static = run_abaqus_legacy_material(
        AbaqusLegacyMaterialRequest(
            f_cm=request.f_cm,
            e_c1=request.e_c1,
            e_clim=request.e_clim,
            l_ch=request.l_ch,
            tension_law=request.tension_law,
            damage_conversion_reference_length_mm=request.damage_conversion_reference_length_mm,
            eccentricity=request.eccentricity,
            viscosity=request.viscosity,
            include_tension_damage=False,
            include_compression_damage=False,
        )
    )

    if is_rate:
        raw = core.calculate_stress_strain(
            request.f_cm, request.e_c1, request.e_clim, request.l_ch, request.strain_rates
        )
        values = list(request.strain_rates)
        unit = "1/s"
        E0 = float(static.elastic["E_mpa"])
        nu = float(static.elastic["nu"])
        family_moduli = [E0] * len(values)
        elastic_rows: list[dict[str, float]] = [{"E_mpa": E0, "nu": nu}]
        tension_rates = [legacy_cracking_displacement_rate(v, request.l_ch) for v in values]
        compression_rates = list(values)
        temperatures: list[float] | None = None
        kernel_producer = "core.calculate_stress_strain"
    else:
        raw = core.calculate_stress_strain_temp(
            request.f_cm, request.e_c1, request.e_clim, request.l_ch, verbose=False
        )
        states = core.calculate_temperature_elastic_states(request.f_cm, request.e_c1)
        values = [state["temperature_c"] for state in states]
        unit = "°C"
        family_moduli = [state["E_mpa"] for state in states]
        elastic_rows = [
            {"E_mpa": state["E_mpa"], "nu": state["nu"], "temperature_c": state["temperature_c"]}
            for state in states
        ]
        nu = float(states[0]["nu"])
        tension_rates = [0.0] * len(values)
        compression_rates = [0.0] * len(values)
        temperatures = list(values)
        kernel_producer = "core.calculate_stress_strain_temp"

    n_families = len(values)
    comp_strain = [_floats(a) for a in raw["compression"]["inelastic strain"]]
    comp_stress = [_floats(a) for a in raw["compression"]["inelastic stress"]]
    opening = [_floats(a) for a in raw["tension"]["crack opening"]]
    stress_key = "stress" if request.tension_law == "bilinear" else "stress exponential"
    damage_key = "damage" if request.tension_law == "bilinear" else "damage exponential"
    tension_stress = [_floats(a) for a in raw["tension"][stress_key]]
    if not (
        len(comp_strain) == len(comp_stress) == len(opening) == len(tension_stress) == n_families
    ):
        raise AbaqusLegacyValidationError("Legacy kernel family count does not match the request.")

    def _comp_row(i: int, stress: float, strain: float) -> dict[str, float]:
        row = {
            "stress_mpa": stress,
            "inelastic_strain": strain,
            "strain_rate_1_per_s": compression_rates[i],
        }
        if temperatures is not None:
            row["temperature_c"] = temperatures[i]
        return row

    def _tension_row(i: int, stress: float, displacement: float) -> dict[str, float]:
        row = {
            "stress_mpa": stress,
            "cracking_displacement_mm": displacement,
            "cracking_displacement_rate_mm_per_s": tension_rates[i],
        }
        if temperatures is not None:
            row["temperature_c"] = temperatures[i]
        return row

    compression_families = [
        [_comp_row(i, s, e) for s, e in zip(comp_stress[i], comp_strain[i], strict=True)]
        for i in range(n_families)
    ]
    tension_families = [
        [_tension_row(i, s, u) for s, u in zip(tension_stress[i], opening[i], strict=True)]
        for i in range(n_families)
    ]

    # Reference damage: the legacy kernel's own first-family damage, normalized with the
    # unchanged M4 backend rules. It carries no rate/temperature column.
    comp_damage_values, comp_changes = _normalize_damage(
        _floats(raw["compression"]["damage"]),
        table="compression_damage",
        force_first_zero=True,
    )
    tension_damage_values, tension_changes = _normalize_damage(
        _floats(raw["tension"][damage_key]),
        table="tension_damage",
        force_first_zero=False,
    )
    reference_compression_damage = [
        {"damage": d, "inelastic_strain": e}
        for d, e in zip(comp_damage_values, comp_strain[0], strict=True)
    ]
    reference_tension_damage = [
        {"damage": d, "cracking_displacement_mm": u}
        for d, u in zip(tension_damage_values, opening[0], strict=True)
    ]

    # Per-family validation with the modulus that belongs to each family.
    family_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for i, value in enumerate(values):
        comp_rows = [
            {"stress_mpa": r["stress_mpa"], "inelastic_strain": r["inelastic_strain"]}
            for r in compression_families[i]
        ]
        tens_rows = [
            {
                "stress_mpa": r["stress_mpa"],
                "cracking_displacement_mm": r["cracking_displacement_mm"],
            }
            for r in tension_families[i]
        ]
        report = abaqus_table_checks(
            E0=family_moduli[i],
            compression_hardening=comp_rows,
            compression_damage=reference_compression_damage if reference_damage else None,
            tension_stiffening=tens_rows,
            tension_damage=reference_tension_damage if reference_damage else None,
            damage_conversion_reference_length_mm=request.damage_conversion_reference_length_mm,
        )
        comp_coords = [r["inelastic_strain"] for r in comp_rows]
        tens_coords = [r["cracking_displacement_mm"] for r in tens_rows]
        extra = {
            "compression_inelastic_strain_increasing": _strictly_increasing(comp_coords),
            "tension_cracking_displacement_increasing": _strictly_increasing(tens_coords),
            "elastic_modulus_positive": math.isfinite(family_moduli[i]) and family_moduli[i] > 0,
        }
        for name, passed in extra.items():
            report[name] = passed
            if not passed:
                report["hard_failures"].append(name)
        report["passed"] = not report["hard_failures"]
        family_report = {
            "dependency_value": value,
            "dependency_unit": unit,
            "E_mpa_used_for_conversion": family_moduli[i],
            **report,
        }
        family_reports.append(family_report)
        failures.extend(_family_failures(mode=mode, value=value, unit=unit, report=report))

    ascending = _strictly_increasing(values)
    if not ascending:
        failures.append(
            {
                "mode": mode,
                "dependency_value": None,
                "dependency_unit": unit,
                "branch": "common",
                "failed_condition": "dependency_values_strictly_increasing",
            }
        )
    if failures:
        summary = "; ".join(
            f"mode={f['mode']} value={f['dependency_value']} {f['dependency_unit']} "
            f"branch={f['branch']} condition={f['failed_condition']}"
            for f in failures
        )
        hint = (
            " Select damage_policy='omit' to export hardening/stiffening without damage."
            if reference_damage
            else ""
        )
        raise AbaqusLegacyDependentValidationError(
            "Refusing dependent Abaqus export; validity checks failed: " + summary + "." + hint,
            tuple(failures),
        )

    validation = {
        "passed": True,
        "damage_policy": request.damage_policy,
        "dependency_values_strictly_increasing": ascending,
        "families_checked": n_families,
        "families": family_reports,
        "first_coordinate_tolerance": 0.0,
        "notes": (
            "Every family is checked against the documented Abaqus conversion "
            "plastic = inelastic - d/(1-d) * sigma/E0 (compression) and "
            "u_pl = u_ck - d/(1-d) * sigma * REF_LENGTH / E0 (tension), using the modulus of "
            "that family. With damage omitted the plastic coordinates equal the inelastic ones."
        ),
    }

    compression_hardening = [row for family in compression_families for row in family]
    tension_stiffening = [row for family in tension_families for row in family]

    if is_rate:
        family_meta = [
            {
                "dependency_mode": mode,
                "dependency_value": value,
                "dependency_unit": unit,
                "legacy_strain_rate_1_per_s": value,
                "abaqus_compression_rate_coordinate_1_per_s": compression_rates[i],
                "abaqus_tension_cracking_displacement_rate_mm_per_s": tension_rates[i],
                "compression_mapping": LEGACY_RATE_AXIS_MAPPING,
                "tension_mapping": LEGACY_CRACK_OPENING_RATE_MAPPING,
                "curve_producer": kernel_producer,
                "compression_keyword": "*CONCRETE COMPRESSION HARDENING",
                "tension_keyword": "*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT",
                "source_kind": AbaqusCdpSourceKind.LEGACY_IMPLEMENTATION.value,
                "kernel_family_index": i,
            }
            for i, value in enumerate(values)
        ]
        dependency_semantics: dict[str, Any] = {
            "mode": mode,
            "dependency_unit": unit,
            "compression": {
                "abaqus_coordinate": "inelastic crushing strain rate",
                "unit": "1/s",
                "mapping_kind": LEGACY_RATE_AXIS_MAPPING,
                "notes": RATE_AXIS_NOTES,
            },
            "tension": {
                "abaqus_coordinate": "direct cracking displacement rate",
                "unit": "mm/s",
                "mapping_kind": LEGACY_CRACK_OPENING_RATE_MAPPING,
                "producer": "strain_rate.legacy_cracking_displacement_rate",
                "legacy_l_ch_mm": request.l_ch,
                "notes": CRACK_OPENING_RATE_NOTES,
            },
            "elastic": {
                "temperature_dependent": False,
                "rate_dependent": False,
                "notes": "Abaqus linear elasticity is rate independent; static legacy E0 is used.",
            },
            "damage": {
                "abaqus_supports_rate_column": False,
                "notes": (
                    "Abaqus damage tables have no strain-rate column and the legacy kernel "
                    "provides only reference damage; no rate-dependent damage is exported."
                ),
            },
            "combined_rate_temperature": False,
            "families": family_meta,
        }
    else:
        family_meta = [
            {
                "dependency_mode": mode,
                "dependency_value": value,
                "dependency_unit": unit,
                "temperature_c": value,
                "E_mpa": family_moduli[i],
                "nu": nu,
                "abaqus_compression_rate_coordinate_1_per_s": 0.0,
                "abaqus_tension_cracking_displacement_rate_mm_per_s": 0.0,
                "elastic_mapping": LEGACY_TEMPERATURE_SECANT_MODULUS,
                "nu_mapping": LEGACY_CONSTANT_NU_ASSUMPTION,
                "curve_producer": kernel_producer,
                "elastic_keyword": "*ELASTIC",
                "compression_keyword": "*CONCRETE COMPRESSION HARDENING",
                "tension_keyword": "*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT",
                "source_kind": AbaqusCdpSourceKind.LEGACY_IMPLEMENTATION.value,
                "kernel_family_index": i,
            }
            for i, value in enumerate(values)
        ]
        dependency_semantics = {
            "mode": mode,
            "dependency_unit": unit,
            "temperature_source": "temperature.get_eurocode_temperature_table (legacy kernel)",
            "compression": {
                "abaqus_coordinate": "temperature",
                "rate_coordinate": 0.0,
                "notes": "Legacy temperature branch; rate column fixed to 0 (static).",
            },
            "tension": {
                "abaqus_coordinate": "temperature",
                "rate_coordinate": 0.0,
                "notes": "Legacy temperature branch; rate column fixed to 0 (static).",
            },
            "elastic": {
                "temperature_dependent": True,
                "modulus_mapping": LEGACY_TEMPERATURE_SECANT_MODULUS,
                "modulus_producer": "core.calculate_temperature_elastic_states -> E_c1_temp",
                "modulus_notes": TEMPERATURE_MODULUS_NOTES,
                "nu_mapping": LEGACY_CONSTANT_NU_ASSUMPTION,
                "nu_notes": CONSTANT_NU_NOTES,
            },
            "damage": {
                "temperature_dependent": False,
                "notes": (
                    "The legacy kernel provides only reference-temperature damage; no damage(T) "
                    "law is exported or claimed."
                ),
            },
            "combined_rate_temperature": False,
            "families": family_meta,
        }

    damage_reference_value = values[0]
    if reference_damage:
        damage_notes = (
            REFERENCE_DAMAGE_RATE_NOTES if is_rate else REFERENCE_DAMAGE_TEMPERATURE_NOTES
        )
        damage_policy: dict[str, Any] = {
            "policy": request.damage_policy,
            "damage_exported": True,
            "reference_dependency_value": damage_reference_value,
            "reference_dependency_unit": unit,
            "has_dependency_column": False,
            "notes": damage_notes,
        }
    else:
        damage_policy = {
            "policy": request.damage_policy,
            "damage_exported": False,
            "reference_dependency_value": None,
            "reference_dependency_unit": unit,
            "has_dependency_column": False,
            "notes": (
                "Damage omitted (default for dependent exports): no CONCRETE COMPRESSION DAMAGE "
                "or CONCRETE TENSION DAMAGE keyword is emitted."
            ),
        }

    legacy_kernel_prov = {
        "source_id": "legacy_v1",
        "source_kind": AbaqusCdpSourceKind.LEGACY_IMPLEMENTATION.value,
        "producer": kernel_producer,
        "notes": (
            "Legacy curve families are exported exactly; no adapter interpolation or "
            "resampling. Legacy rate/temperature laws are not asserted as normative "
            "Abaqus, EC2, CEB or fib laws."
        ),
    }
    elastic_prov: dict[str, Any] = {
        "E": static.provenance["elastic"]["E"],
        "nu": static.provenance["elastic"]["nu"],
    }
    if not is_rate:
        elastic_prov = {
            "E": {
                "source_id": "legacy_v1",
                "source_kind": AbaqusCdpSourceKind.LEGACY_IMPLEMENTATION.value,
                "units": "MPa",
                "producer": "core.calculate_temperature_elastic_states -> E_c1_temp",
                "mapping_kind": LEGACY_TEMPERATURE_SECANT_MODULUS,
                "notes": TEMPERATURE_MODULUS_NOTES,
            },
            "nu": {
                "source_id": "legacy_v1",
                "source_kind": AbaqusCdpSourceKind.LEGACY_IMPLEMENTATION.value,
                "units": "dimensionless",
                "producer": "calculate_poisson_ratios -> v_ce",
                "mapping_kind": LEGACY_CONSTANT_NU_ASSUMPTION,
                "notes": CONSTANT_NU_NOTES,
            },
        }
    damage_prov = None
    if reference_damage:
        damage_prov = {
            "source_id": "legacy_v1",
            "source_kind": AbaqusCdpSourceKind.LEGACY_IMPLEMENTATION.value,
            "producer": (
                "calculate_compression_damage; calculate_tension_damage "
                f"(legacy reference case {damage_reference_value:g} {unit})"
            ),
            "notes": damage_policy["notes"],
            "normalizations": comp_changes + tension_changes,
        }
    provenance = {
        "legacy_kernel": legacy_kernel_prov,
        "elastic": elastic_prov,
        "plasticity": static.provenance["plasticity"],
        "compression_hardening": {
            **legacy_kernel_prov,
            "units": "MPa, dimensionless, 1/s" + (", °C" if not is_rate else ""),
            "dependency_mapping": LEGACY_RATE_AXIS_MAPPING if is_rate else "temperature",
        },
        "tension_stiffening": {
            **legacy_kernel_prov,
            "units": "MPa, mm, mm/s" + (", °C" if not is_rate else ""),
            "dependency_mapping": LEGACY_CRACK_OPENING_RATE_MAPPING if is_rate else "temperature",
        },
        "damage": damage_prov,
        "abaqus_requirements": {
            "source_kind": ABAQUS_DOCUMENTATION_REQUIREMENT,
            "items": [
                "first inelastic strain / cracking displacement of each family is 0.0",
                "dependent data are given in ascending order of the dependency variable",
                "damage tables carry no strain-rate column",
                "temperature-dependent isotropic elasticity rows are E, nu, temperature",
            ],
        },
        "families": family_meta,
        "backend_normalizations": (comp_changes + tension_changes) if reference_damage else [],
    }

    warnings = [
        "Legacy compatibility calibration. Verify against project calibration, experiments and "
        "the Abaqus version used.",
        "No Abaqus solver execution has been performed by this backend; run "
        "scripts/qualify_abaqus.py for the optional external solver gate.",
    ]
    if is_rate:
        warnings.append(
            "Compression rate coordinate is a legacy rate-axis mapping, not a kinematically "
            "derived inelastic strain rate."
        )
        if values[0] != 0.0 and reference_damage:
            warnings.append(
                f"Reference damage comes from the first legacy rate family ({values[0]:g} 1/s), "
                "not from the static M4 reference."
            )
    else:
        warnings.append(
            "Temperature-dependent elasticity E(T) uses the legacy secant modulus E_c1_temp; "
            "nu is held constant (legacy constant-nu assumption)."
        )
    if reference_damage:
        warnings.append(
            "Damage is a single reference function reused for every family; it is not "
            + ("rate-dependent damage." if is_rate else "temperature-dependent damage.")
        )

    curves: list[CurveSeries] = []
    for i, value in enumerate(values):
        label = _family_label(mode, value)
        meta = {"representation": "Abaqus export representation", "dependency_value": value}
        curves.append(
            CurveSeries(
                f"abaqus_dependent.compression_hardening.{i}",
                label,
                "abaqus_dependent_compression_hardening",
                comp_strain[i],
                comp_stress[i],
                "Inelastic strain",
                "Compressive stress",
                "dimensionless",
                "MPa",
                dict(meta),
            )
        )
    for i, value in enumerate(values):
        label = _family_label(mode, value)
        meta = {
            "representation": "Abaqus export representation",
            "dependency_value": value,
            "type": "DISPLACEMENT",
        }
        curves.append(
            CurveSeries(
                f"abaqus_dependent.tension_stiffening.{i}",
                label,
                "abaqus_dependent_tension_stiffening",
                opening[i],
                tension_stress[i],
                "Crack opening",
                "Tensile stress",
                "mm",
                "MPa",
                meta,
            )
        )
    if not is_rate:
        curves.append(
            CurveSeries(
                "abaqus_dependent.elastic_modulus",
                "E(T) — legacy E_c1_temp",
                "abaqus_dependent_elastic_modulus",
                list(values),
                list(family_moduli),
                "Temperature",
                "Elastic modulus",
                "°C",
                "MPa",
                {"representation": "Abaqus export representation"},
            )
        )
    if reference_damage:
        ref_meta = {
            "representation": "Abaqus export representation",
            "reused_reference_damage": True,
            "reference_dependency_value": damage_reference_value,
        }
        curves.append(
            CurveSeries(
                "abaqus_dependent.reference_compression_damage",
                "Reference compression damage (reused for all families)",
                "abaqus_dependent_reference_compression_damage",
                comp_strain[0],
                comp_damage_values,
                "Inelastic strain",
                "Damage",
                "dimensionless",
                "dimensionless",
                dict(ref_meta),
            )
        )
        curves.append(
            CurveSeries(
                "abaqus_dependent.reference_tension_damage",
                "Reference tension damage (reused for all families)",
                "abaqus_dependent_reference_tension_damage",
                opening[0],
                tension_damage_values,
                "Crack opening",
                "Damage",
                "mm",
                "dimensionless",
                {**ref_meta, "type": "DISPLACEMENT"},
            )
        )

    result = AbaqusLegacyDependentResult(
        mode=mode,
        inputs=request.to_dict(),
        elastic={
            "temperature_dependent": not is_rate,
            "nu_assumption": None if is_rate else LEGACY_CONSTANT_NU_ASSUMPTION,
            "rows": elastic_rows,
        },
        plasticity=dict(static.plasticity),
        compression_hardening=compression_hardening,
        tension_stiffening=tension_stiffening,
        damage_policy=damage_policy,
        compression_damage=reference_compression_damage if reference_damage else None,
        tension_damage=reference_tension_damage if reference_damage else None,
        dependency_semantics=dependency_semantics,
        backend_configuration={
            "damage_conversion_reference_length_mm": request.damage_conversion_reference_length_mm,
            "reference_length_semantics": static.backend_configuration[
                "reference_length_semantics"
            ],
            "tension_stiffening_type": "DISPLACEMENT",
            "legacy_l_ch_mm": request.l_ch,
            "calibration_id": ABAQUS_LEGACY_FULL_CALIBRATION_ID,
            "combined_rate_temperature": False,
            "rows_per_compression_family": len(comp_strain[0]),
            "rows_per_tension_family": len(opening[0]),
        },
        provenance=provenance,
        validation=validation,
        warnings=warnings,
        documentation_references=list(ABAQUS_DEPENDENT_DOCUMENTATION_URLS),
        export_capabilities=["json", "inp"],
        curves=curves,
    )
    result.to_json()  # Strict finite JSON at the public boundary.
    return result


def build_abaqus_legacy_dependent_material_text(
    result: AbaqusLegacyDependentResult,
    material_name: str = "Concrete_Legacy_CDP",
) -> str:
    """Deterministic Abaqus keyword text for a validated dependent material."""

    if not isinstance(result, AbaqusLegacyDependentResult):
        raise AbaqusLegacyInputError("result must be AbaqusLegacyDependentResult.")
    if not result.validation.get("passed"):
        raise AbaqusLegacyValidationError("Refusing .inp export because validation did not pass.")
    name = material_name.strip()
    if not name or "\n" in name or "," in name:
        raise AbaqusLegacyInputError(
            "material_name must be non-empty and contain no comma/newline."
        )
    is_rate = result.mode == AbaqusDependentMode.STRAIN_RATE
    ref_length = result.backend_configuration["damage_conversion_reference_length_mm"]
    p = result.plasticity
    mapping = (
        "legacy rate-axis mapping (compression); w_dot = strain_rate * l_ch (tension)"
        if is_rate
        else "temperature dependent; E(T) = legacy E_c1_temp; constant nu"
    )
    lines = [
        "** Generated by cdp-generator",
        f"** Calibration: {ABAQUS_LEGACY_FULL_CALIBRATION_ID}",
        f"** Dependent export: {result.mode}; {mapping}",
        f"** Damage policy: {result.damage_policy['policy']}",
        "** Legacy compatibility curves; see JSON export for full provenance.",
        f"*MATERIAL, NAME={name}",
        "*ELASTIC",
    ]
    for row in result.elastic["rows"]:
        values = [row["E_mpa"], row["nu"]]
        if not is_rate:
            values.append(row["temperature_c"])
        lines.append(", ".join(_fmt(v) for v in values))
    lines.append(f"*CONCRETE DAMAGED PLASTICITY, REF LENGTH={_fmt(ref_length)}")
    lines.append(
        ", ".join(
            _fmt(v)
            for v in (
                p["dilation_angle_deg"],
                p["eccentricity"],
                p["fbfc"],
                p["Kc"],
                p["viscosity"],
            )
        )
    )
    lines.append("*CONCRETE COMPRESSION HARDENING")
    for row in result.compression_hardening:
        values = [row["stress_mpa"], row["inelastic_strain"], row["strain_rate_1_per_s"]]
        if not is_rate:
            values.append(row["temperature_c"])
        lines.append(", ".join(_fmt(v) for v in values))
    if result.compression_damage is not None:
        lines.append("*CONCRETE COMPRESSION DAMAGE")
        lines.extend(
            f"{_fmt(row['damage'])}, {_fmt(row['inelastic_strain'])}"
            for row in result.compression_damage
        )
    lines.append("*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT")
    for row in result.tension_stiffening:
        values = [
            row["stress_mpa"],
            row["cracking_displacement_mm"],
            row["cracking_displacement_rate_mm_per_s"],
        ]
        if not is_rate:
            values.append(row["temperature_c"])
        lines.append(", ".join(_fmt(v) for v in values))
    if result.tension_damage is not None:
        lines.append("*CONCRETE TENSION DAMAGE, TYPE=DISPLACEMENT")
        lines.extend(
            f"{_fmt(row['damage'])}, {_fmt(row['cracking_displacement_mm'])}"
            for row in result.tension_damage
        )
    return "\n".join(lines) + "\n"


def abaqus_legacy_dependent_material_text(
    result: AbaqusLegacyDependentResult,
    material_name: str = "Concrete_Legacy_CDP",
) -> str:
    """Public application alias for the dependent deterministic keyword exporter."""

    return build_abaqus_legacy_dependent_material_text(result, material_name)


def rate_mapping_audit_rows(result: AbaqusLegacyDependentResult) -> list[dict[str, float]]:
    """Read-only audit table of the legacy rate mapping (strain-rate mode only)."""

    if result.mode != AbaqusDependentMode.STRAIN_RATE:
        raise AbaqusLegacyInputError("rate mapping audit requires a strain_rate result.")
    return [
        {
            "legacy strain rate [1/s]": row["legacy_strain_rate_1_per_s"],
            "Abaqus compression rate coordinate [1/s]": row[
                "abaqus_compression_rate_coordinate_1_per_s"
            ],
            "Abaqus tension crack-opening rate [mm/s]": row[
                "abaqus_tension_cracking_displacement_rate_mm_per_s"
            ],
        }
        for row in result.dependency_semantics["families"]
    ]


def temperature_elastic_audit_rows(result: AbaqusLegacyDependentResult) -> list[dict[str, float]]:
    """Read-only audit table of E(T) and constant nu (temperature mode only)."""

    if result.mode != AbaqusDependentMode.TEMPERATURE:
        raise AbaqusLegacyInputError("temperature audit requires a temperature result.")
    return [
        {
            "temperature [°C]": row["temperature_c"],
            "E(T) [MPa]": row["E_mpa"],
            "nu [-]": row["nu"],
        }
        for row in result.elastic["rows"]
    ]
