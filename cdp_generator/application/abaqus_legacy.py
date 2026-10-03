"""Full static-reference legacy Abaqus CDP material application service.

The numerical curves remain the repository's historical calibration.  This module
only adapts that calibration to an Abaqus-compatible, auditable representation;
it does not claim normative EC2/fib calibration or execute Abaqus.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any

from .. import core
from ..concrete import Concrete
from .results import CurveSeries

ABAQUS_LEGACY_FULL_CALIBRATION_ID = "abaqus_cdp_legacy_full_v1"
ABAQUS_DOCUMENTATION_URLS = (
    (
        "https://docs.software.vt.edu/abaqusv2025/English/"
        "SIMACAEMATRefMap/simamat-c-concretedamaged.htm"
    ),
    (
        "https://docs.software.vt.edu/abaqusv2025/English/"
        "SIMACAEKEYRefMap/simakey-r-concretedamagedplasticity.htm"
    ),
    (
        "https://docs.software.vt.edu/abaqusv2025/English/"
        "SIMACAEKEYRefMap/simakey-r-concretecompressionhardening.htm"
    ),
    (
        "https://docs.software.vt.edu/abaqusv2025/English/"
        "SIMACAEKEYRefMap/simakey-r-concretecompressiondamage.htm"
    ),
    (
        "https://docs.software.vt.edu/abaqusv2025/English/"
        "SIMACAEKEYRefMap/simakey-r-concretetensiondamage.htm"
    ),
    (
        "https://docs.software.vt.edu/abaqusv2025/English/"
        "SIMACAEKERRefMap/simaker-c-concretetensionstiffeningpyc.htm"
    ),
)


class AbaqusLegacyInputError(ValueError):
    """Controlled invalid request for the legacy Abaqus backend."""


class AbaqusLegacyValidationError(ValueError):
    """Backend representation violates documented Abaqus conversion constraints."""


class AbaqusCdpSourceKind(StrEnum):
    PHYSICAL_SOURCE = "PHYSICAL_SOURCE"
    LEGACY_IMPLEMENTATION = "LEGACY_IMPLEMENTATION"
    ABAQUS_DOCUMENTATION_DEFAULT = "ABAQUS_DOCUMENTATION_DEFAULT"
    ABAQUS_BACKEND_NORMALIZATION = "ABAQUS_BACKEND_NORMALIZATION"
    USER_BACKEND_OVERRIDE = "USER_BACKEND_OVERRIDE"


@dataclass(frozen=True, slots=True)
class AbaqusCdpProvenance:
    source_id: str
    source_kind: AbaqusCdpSourceKind
    units: str
    producer: str | None = None
    source_locator: str | None = None
    notes: str = ""
    normalizations: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.units.strip():
            raise ValueError("Abaqus provenance source_id and units must be non-empty")
        object.__setattr__(self, "normalizations", tuple(dict(v) for v in self.normalizations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind.value,
            "units": self.units,
            "producer": self.producer,
            "source_locator": self.source_locator,
            "notes": self.notes,
            "normalizations": [dict(v) for v in self.normalizations],
        }


@dataclass(frozen=True, slots=True)
class AbaqusLegacyMaterialRequest:
    f_cm: float = 28.0
    e_c1: float = 0.0022
    e_clim: float = 0.0035
    l_ch: float = 1.0
    tension_law: str = "bilinear"
    damage_conversion_reference_length_mm: float = 1.0
    eccentricity: float | None = None
    viscosity: float | None = None
    include_tension_damage: bool = True
    include_compression_damage: bool = True

    def __post_init__(self) -> None:
        for name in ("f_cm", "e_c1", "e_clim", "l_ch", "damage_conversion_reference_length_mm"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise AbaqusLegacyInputError(f"{name} must be finite and strictly positive.")
            object.__setattr__(self, name, float(value))
        if self.tension_law not in {"bilinear", "power_law"}:
            raise AbaqusLegacyInputError("tension_law must be bilinear or power_law.")
        if self.eccentricity is not None:
            if (
                isinstance(self.eccentricity, bool)
                or not isinstance(self.eccentricity, int | float)
                or not math.isfinite(self.eccentricity)
                or self.eccentricity <= 0.0
            ):
                raise AbaqusLegacyInputError("eccentricity must be finite and > 0 when overridden.")
            object.__setattr__(self, "eccentricity", float(self.eccentricity))
        if self.viscosity is not None:
            if (
                isinstance(self.viscosity, bool)
                or not isinstance(self.viscosity, int | float)
                or not math.isfinite(self.viscosity)
                or self.viscosity < 0.0
            ):
                raise AbaqusLegacyInputError("viscosity must be finite and >= 0 when overridden.")
            object.__setattr__(self, "viscosity", float(self.viscosity))
        if not isinstance(self.include_tension_damage, bool) or not isinstance(
            self.include_compression_damage, bool
        ):
            raise AbaqusLegacyInputError("damage include flags must be boolean.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AbaqusLegacyMaterialResult:
    inputs: dict[str, Any]
    elastic: dict[str, float]
    plasticity: dict[str, float]
    compression_hardening: list[dict[str, float]]
    compression_damage: list[dict[str, float]] | None
    tension_stiffening: list[dict[str, float]]
    tension_damage: list[dict[str, float]] | None
    backend_configuration: dict[str, Any]
    provenance: dict[str, Any]
    validation: dict[str, Any]
    warnings: list[str]
    export_capabilities: list[str]
    curves: list[CurveSeries]
    documentation_references: list[str]
    workflow: str = "legacy_abaqus_cdp"
    schema_version: str = "abaqus_legacy_material_result.v1"

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.to_json())

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False)


def _legacy_provenance(*, units: str, producer: str, notes: str) -> AbaqusCdpProvenance:
    return AbaqusCdpProvenance(
        source_id="legacy_v1",
        source_kind=AbaqusCdpSourceKind.LEGACY_IMPLEMENTATION,
        units=units,
        producer=producer,
        notes=notes,
    )


def _backend_setting_provenance(
    *, field: str, units: str, user_override: bool
) -> AbaqusCdpProvenance:
    if user_override:
        return AbaqusCdpProvenance(
            source_id="user_backend_override",
            source_kind=AbaqusCdpSourceKind.USER_BACKEND_OVERRIDE,
            units=units,
            producer="AbaqusLegacyMaterialRequest",
            notes=f"Explicit user backend override for Abaqus {field}.",
        )
    return AbaqusCdpProvenance(
        source_id="abaqus_2025_documentation",
        source_kind=AbaqusCdpSourceKind.ABAQUS_DOCUMENTATION_DEFAULT,
        units=units,
        source_locator=f"*CONCRETE DAMAGED PLASTICITY — {field}",
        notes="Abaqus documented backend default; not a legacy concrete calibration value.",
    )


def _normalize_damage(
    raw: list[float], *, table: str, force_first_zero: bool
) -> tuple[list[float], list[dict[str, Any]]]:
    normalized: list[float] = []
    changes: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        exported = min(float(value), 0.99)
        reason = None
        if force_first_zero and index == 0 and exported != 0.0:
            exported = 0.0
            reason = "Abaqus requires first compression damage point to be exactly zero."
        elif exported != float(value):
            reason = "Abaqus documentation recommends avoiding damage values above 0.99."
        normalized.append(exported)
        if reason is not None:
            changes.append(
                {
                    "table": table,
                    "row": index,
                    "field": "damage",
                    "raw": float(value),
                    "exported": exported,
                    "reason": reason,
                    "source_kind": AbaqusCdpSourceKind.ABAQUS_BACKEND_NORMALIZATION.value,
                }
            )
    return normalized, changes


def _finite_rows(rows: list[dict[str, float]] | None) -> bool:
    return rows is None or all(math.isfinite(float(v)) for row in rows for v in row.values())


def _nondecreasing(values: list[float], tolerance: float) -> bool:
    return all(b + tolerance >= a for a, b in pairwise(values))


def validate_abaqus_legacy_material_tables(
    *,
    E0: float,
    compression_hardening: list[dict[str, float]],
    compression_damage: list[dict[str, float]] | None,
    tension_stiffening: list[dict[str, float]],
    tension_damage: list[dict[str, float]] | None,
    damage_conversion_reference_length_mm: float,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Validate documented Abaqus inelastic→plastic conversion constraints."""

    if not math.isfinite(E0) or E0 <= 0.0:
        raise AbaqusLegacyValidationError("E0 must be finite and positive for validation.")
    if (
        not math.isfinite(damage_conversion_reference_length_mm)
        or damage_conversion_reference_length_mm <= 0
    ):
        raise AbaqusLegacyValidationError("REF LENGTH must be finite and positive for validation.")
    if not compression_hardening or not tension_stiffening:
        raise AbaqusLegacyValidationError(
            "Abaqus hardening and tension-stiffening tables cannot be empty."
        )

    all_finite = all(
        (
            _finite_rows(compression_hardening),
            _finite_rows(compression_damage),
            _finite_rows(tension_stiffening),
            _finite_rows(tension_damage),
        )
    )
    if not all_finite:
        raise AbaqusLegacyValidationError("Abaqus backend tables contain non-finite values.")

    comp_coords = [row["inelastic_strain"] for row in compression_hardening]
    comp_stress = [row["stress_mpa"] for row in compression_hardening]
    compression_first_inelastic_strain_zero = comp_coords[0] == 0.0
    compression_first_damage_zero = True
    compression_damage_below_one = True
    compression_damage_nonnegative = True
    compression_plastic = comp_coords.copy()
    comp_aligned = True
    if compression_damage is not None:
        damage_coords = [row["inelastic_strain"] for row in compression_damage]
        comp_aligned = damage_coords == comp_coords
        damage = [row["damage"] for row in compression_damage]
        compression_first_damage_zero = (
            bool(damage) and damage_coords[0] == 0.0 and damage[0] == 0.0
        )
        compression_damage_below_one = all(d < 1.0 for d in damage)
        compression_damage_nonnegative = all(d >= 0.0 for d in damage)
        compression_plastic = [
            eps - d / (1.0 - d) * sigma / E0
            for eps, d, sigma in zip(comp_coords, damage, comp_stress, strict=True)
        ]

    tension_coords = [row["cracking_displacement_mm"] for row in tension_stiffening]
    tension_stress = [row["stress_mpa"] for row in tension_stiffening]
    tension_first_cracking_displacement_zero = tension_coords[0] == 0.0
    tension_plastic = tension_coords.copy()
    tension_aligned = True
    tension_first_damage_zero = True
    tension_damage_below_one = True
    tension_damage_nonnegative = True
    if tension_damage is not None:
        damage_coords = [row["cracking_displacement_mm"] for row in tension_damage]
        tension_aligned = damage_coords == tension_coords
        damage = [row["damage"] for row in tension_damage]
        tension_first_damage_zero = (
            bool(damage) and damage_coords[0] == 0.0 and damage[0] == 0.0
        )
        tension_damage_below_one = all(d < 1.0 for d in damage)
        tension_damage_nonnegative = all(d >= 0.0 for d in damage)
        tension_plastic = [
            u
            - d
            / (1.0 - d)
            * sigma
            * damage_conversion_reference_length_mm
            / E0
            for u, d, sigma in zip(tension_coords, damage, tension_stress, strict=True)
        ]

    checks = {
        "compression_first_inelastic_strain_zero": compression_first_inelastic_strain_zero,
        "compression_first_damage_zero": compression_first_damage_zero,
        "compression_damage_below_one": compression_damage_below_one,
        "compression_damage_nonnegative": compression_damage_nonnegative,
        "compression_plastic_strain_nonnegative": min(compression_plastic) >= -tolerance,
        "compression_plastic_strain_monotonic": _nondecreasing(compression_plastic, tolerance),
        "tension_first_cracking_displacement_zero": tension_first_cracking_displacement_zero,
        "tension_first_damage_zero": tension_first_damage_zero,
        "tension_damage_below_one": tension_damage_below_one,
        "tension_damage_nonnegative": tension_damage_nonnegative,
        "tension_plastic_displacement_nonnegative": min(tension_plastic) >= -tolerance,
        "tension_plastic_displacement_monotonic": _nondecreasing(tension_plastic, tolerance),
        "coordinate_alignment": comp_aligned and tension_aligned,
        "all_finite": all_finite
        and all(math.isfinite(v) for v in compression_plastic + tension_plastic),
    }
    hard_failures = [name for name, passed in checks.items() if not passed]
    report: dict[str, Any] = {
        **checks,
        "passed": not hard_failures,
        "hard_failures": hard_failures,
        "tolerance": tolerance,
        "compression_plastic_strain_min": min(compression_plastic),
        "compression_plastic_strain_min_increment": min(
            (b - a for a, b in pairwise(compression_plastic)),
            default=0.0,
        ),
        "tension_plastic_displacement_min": min(tension_plastic),
        "tension_plastic_displacement_min_increment": min(
            (b - a for a, b in pairwise(tension_plastic)),
            default=0.0,
        ),
    }
    if hard_failures:
        raise AbaqusLegacyValidationError(
            "Abaqus backend validity checks failed: " + ", ".join(hard_failures)
        )
    return report


def run_abaqus_legacy_material(
    request: AbaqusLegacyMaterialRequest,
) -> AbaqusLegacyMaterialResult:
    """Build a full static-reference legacy CDP material and validate it for export."""

    if not isinstance(request, AbaqusLegacyMaterialRequest):
        raise AbaqusLegacyInputError("request must be AbaqusLegacyMaterialRequest.")

    raw = core.calculate_stress_strain(
        request.f_cm, request.e_c1, request.e_clim, request.l_ch, [0.0]
    )
    legacy_scalar = Concrete.from_mean_strength(
        request.f_cm, request.e_c1, request.e_clim
    ).to_abaqus_cdp()

    E0 = float(raw["properties"]["elasticity"])
    nu = float(raw["properties"]["poisson"])
    eccentricity = 0.1 if request.eccentricity is None else request.eccentricity
    viscosity = 0.0 if request.viscosity is None else request.viscosity

    comp_strain = [float(v) for v in raw["compression"]["inelastic strain"][0]]
    comp_stress = [float(v) for v in raw["compression"]["inelastic stress"][0]]
    compression_hardening = [
        {"stress_mpa": stress, "inelastic_strain": strain}
        for stress, strain in zip(comp_stress, comp_strain, strict=True)
    ]

    raw_comp_damage = [float(v) for v in raw["compression"]["damage"]]
    comp_damage_values, comp_changes = _normalize_damage(
        raw_comp_damage, table="compression_damage", force_first_zero=True
    )
    compression_damage = (
        [
            {"damage": damage, "inelastic_strain": strain}
            for damage, strain in zip(comp_damage_values, comp_strain, strict=True)
        ]
        if request.include_compression_damage
        else None
    )

    opening = [float(v) for v in raw["tension"]["crack opening"][0]]
    stress_key = "stress" if request.tension_law == "bilinear" else "stress exponential"
    damage_key = "damage" if request.tension_law == "bilinear" else "damage exponential"
    tension_stress = [float(v) for v in raw["tension"][stress_key][0]]
    tension_stiffening = [
        {"stress_mpa": stress, "cracking_displacement_mm": displacement}
        for stress, displacement in zip(tension_stress, opening, strict=True)
    ]
    raw_tension_damage = [float(v) for v in raw["tension"][damage_key]]
    tension_damage_values, tension_changes = _normalize_damage(
        raw_tension_damage, table="tension_damage", force_first_zero=False
    )
    tension_damage = (
        [
            {"damage": damage, "cracking_displacement_mm": displacement}
            for damage, displacement in zip(tension_damage_values, opening, strict=True)
        ]
        if request.include_tension_damage
        else None
    )

    validation = validate_abaqus_legacy_material_tables(
        E0=E0,
        compression_hardening=compression_hardening,
        compression_damage=compression_damage,
        tension_stiffening=tension_stiffening,
        tension_damage=tension_damage,
        damage_conversion_reference_length_mm=request.damage_conversion_reference_length_mm,
    )

    normalizations = (
        (comp_changes if request.include_compression_damage else [])
        + (tension_changes if request.include_tension_damage else [])
    )
    elastic_prov = {
        "E": _legacy_provenance(
            units="MPa",
            producer="calculate_stress_strain -> properties.elasticity",
            notes="Legacy physical implementation value; Abaqus is not asserted as its authority.",
        ).to_dict(),
        "nu": _legacy_provenance(
            units="dimensionless",
            producer="calculate_poisson_ratios -> calculate_stress_strain",
            notes="Legacy physical implementation value; Abaqus is not asserted as its authority.",
        ).to_dict(),
    }
    scalar_notes = (
        "Repository legacy ABAQUS-CDP calibration; not an Abaqus default and no normative "
        "EC2/fib authority is asserted."
    )
    plasticity_prov = {
        "dilation_angle": _legacy_provenance(
            units="degrees", producer="calculate_cdp_parameters", notes=scalar_notes
        ).to_dict(),
        "fbfc": _legacy_provenance(
            units="dimensionless", producer="calculate_cdp_parameters", notes=scalar_notes
        ).to_dict(),
        "Kc": _legacy_provenance(
            units="dimensionless", producer="calculate_cdp_parameters", notes=scalar_notes
        ).to_dict(),
        "eccentricity": _backend_setting_provenance(
            field="flow potential eccentricity",
            units="dimensionless",
            user_override=request.eccentricity is not None,
        ).to_dict(),
        "viscosity": _backend_setting_provenance(
            field="viscosity parameter",
            units="time",
            user_override=request.viscosity is not None,
        ).to_dict(),
    }
    compression_hardening_prov = _legacy_provenance(
        units="MPa, dimensionless",
        producer="calculate_compression_behavior; calculate_inelastic_compression",
        notes=(
            "Legacy implementation curve; historical comments are not promoted to verified "
            "normative provenance."
        ),
    ).to_dict()
    compression_damage_prov = _legacy_provenance(
        units="dimensionless, dimensionless",
        producer="calculate_compression_damage",
        notes="Legacy implementation damage adapted only for Abaqus backend compatibility.",
    ).to_dict()
    compression_damage_prov["normalizations"] = (
        comp_changes if request.include_compression_damage else []
    )
    tension_stiffening_prov = _legacy_provenance(
        units="MPa, mm",
        producer="calculate_tension_bilinear"
        if request.tension_law == "bilinear"
        else "calculate_tension_bilinear; calculate_tension_power_law",
        notes=(
            "Legacy implementation stress-crack-opening law exported as TYPE=DISPLACEMENT; "
            "not a verified fib/CEB implementation solely from historical comments."
        ),
    ).to_dict()
    tension_damage_prov = _legacy_provenance(
        units="dimensionless, mm",
        producer="calculate_tension_damage",
        notes="Legacy implementation damage adapted only for Abaqus backend compatibility.",
    ).to_dict()
    tension_damage_prov["normalizations"] = (
        tension_changes if request.include_tension_damage else []
    )

    curves = [
        CurveSeries(
            "abaqus.compression_hardening",
            "Compression hardening — Abaqus export representation",
            "abaqus_compression_hardening",
            comp_strain,
            comp_stress,
            "Inelastic strain",
            "Compressive stress",
            "dimensionless",
            "MPa",
            {"representation": "Abaqus export representation"},
        ),
        CurveSeries(
            "abaqus.tension_stiffening",
            f"Tension stiffening ({request.tension_law}) — Abaqus export representation",
            "abaqus_tension_stiffening",
            opening,
            tension_stress,
            "Crack opening",
            "Tensile stress",
            "mm",
            "MPa",
            {"representation": "Abaqus export representation", "type": "DISPLACEMENT"},
        ),
    ]
    if compression_damage is not None:
        curves.insert(
            1,
            CurveSeries(
                "abaqus.compression_damage",
                "Compression damage — Abaqus export representation",
                "abaqus_compression_damage",
                comp_strain,
                comp_damage_values,
                "Inelastic strain",
                "Damage",
                "dimensionless",
                "dimensionless",
                {"representation": "Abaqus export representation"},
            ),
        )
    if tension_damage is not None:
        curves.append(
            CurveSeries(
                "abaqus.tension_damage",
                "Tension damage — Abaqus export representation",
                "abaqus_tension_damage",
                opening,
                tension_damage_values,
                "Crack opening",
                "Damage",
                "mm",
                "dimensionless",
                {"representation": "Abaqus export representation", "type": "DISPLACEMENT"},
            )
        )

    result = AbaqusLegacyMaterialResult(
        inputs=request.to_dict(),
        elastic={"E_mpa": E0, "nu": nu},
        plasticity={
            "dilation_angle_deg": float(legacy_scalar.dilation_angle),
            "eccentricity": float(eccentricity),
            "fbfc": float(legacy_scalar.fbfc),
            "Kc": float(legacy_scalar.Kc),
            "viscosity": float(viscosity),
        },
        compression_hardening=compression_hardening,
        compression_damage=compression_damage,
        tension_stiffening=tension_stiffening,
        tension_damage=tension_damage,
        backend_configuration={
            "damage_conversion_reference_length_mm": request.damage_conversion_reference_length_mm,
            "reference_length_semantics": (
                "Abaqus REF LENGTH is a backend damage-conversion reference length; it is not "
                "the legacy crack-band l_ch and is not a physical material property."
            ),
            "tension_stiffening_type": "DISPLACEMENT",
            "static_reference_only": True,
            "legacy_l_ch_mm": request.l_ch,
            "normalizations": normalizations,
        },
        provenance={
            "elastic": elastic_prov,
            "plasticity": plasticity_prov,
            "compression_hardening": compression_hardening_prov,
            "compression_damage": (
                compression_damage_prov if request.include_compression_damage else None
            ),
            "tension_stiffening": tension_stiffening_prov,
            "tension_damage": tension_damage_prov if request.include_tension_damage else None,
            "backend_normalizations": normalizations,
        },
        validation=validation,
        warnings=[
            "Legacy compatibility calibration. Verify against project calibration, "
            "experiments and the Abaqus version used.",
            "Static reference calibration only (strain rate 0, room/reference state); "
            "rate/temperature dependent Abaqus damage tables are not exported.",
            "No Abaqus solver execution has been performed by this backend.",
        ],
        export_capabilities=["json", "inp"],
        curves=curves,
        documentation_references=list(ABAQUS_DOCUMENTATION_URLS),
    )
    result.to_json()
    return result


def _fmt(value: float) -> str:
    return format(float(value), ".12g")


def build_abaqus_legacy_cdp_material_text(
    result: AbaqusLegacyMaterialResult,
    material_name: str = "Concrete_Legacy_CDP",
) -> str:
    """Build deterministic Abaqus keyword material text without an Abaqus dependency."""

    if not isinstance(result, AbaqusLegacyMaterialResult):
        raise AbaqusLegacyInputError("result must be AbaqusLegacyMaterialResult.")
    if not result.validation.get("passed"):
        raise AbaqusLegacyValidationError("Refusing .inp export because validation did not pass.")
    name = material_name.strip()
    if not name or "\n" in name or "," in name:
        raise AbaqusLegacyInputError(
            "material_name must be non-empty and contain no comma/newline."
        )

    ref_length = result.backend_configuration["damage_conversion_reference_length_mm"]
    p = result.plasticity
    lines = [
        "** Generated by cdp-generator",
        f"** Calibration: {ABAQUS_LEGACY_FULL_CALIBRATION_ID}",
        "** Legacy compatibility curves; see JSON export for full provenance.",
        f"*MATERIAL, NAME={name}",
        "*ELASTIC",
        f"{_fmt(result.elastic['E_mpa'])}, {_fmt(result.elastic['nu'])}",
        f"*CONCRETE DAMAGED PLASTICITY, REF LENGTH={_fmt(ref_length)}",
        ", ".join(
            _fmt(v)
            for v in (
                p["dilation_angle_deg"],
                p["eccentricity"],
                p["fbfc"],
                p["Kc"],
                p["viscosity"],
            )
        ),
        "*CONCRETE COMPRESSION HARDENING",
    ]
    lines.extend(
        f"{_fmt(row['stress_mpa'])}, {_fmt(row['inelastic_strain'])}"
        for row in result.compression_hardening
    )
    if result.compression_damage is not None:
        lines.append("*CONCRETE COMPRESSION DAMAGE")
        lines.extend(
            f"{_fmt(row['damage'])}, {_fmt(row['inelastic_strain'])}"
            for row in result.compression_damage
        )
    lines.append("*CONCRETE TENSION STIFFENING, TYPE=DISPLACEMENT")
    lines.extend(
        f"{_fmt(row['stress_mpa'])}, {_fmt(row['cracking_displacement_mm'])}"
        for row in result.tension_stiffening
    )
    if result.tension_damage is not None:
        lines.append("*CONCRETE TENSION DAMAGE, TYPE=DISPLACEMENT")
        lines.extend(
            f"{_fmt(row['damage'])}, {_fmt(row['cracking_displacement_mm'])}"
            for row in result.tension_damage
        )
    return "\n".join(lines) + "\n"


def abaqus_legacy_material_text(
    result: AbaqusLegacyMaterialResult,
    material_name: str = "Concrete_Legacy_CDP",
) -> str:
    """Public application alias for the deterministic keyword exporter."""

    return build_abaqus_legacy_cdp_material_text(result, material_name)
