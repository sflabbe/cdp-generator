"""Minimal complete Abaqus qualification decks (ABAQUS-Q1).

Pure text generators: no Abaqus import, no solver execution. Each deck embeds the material
card produced by the canonical deterministic builders, unchanged, and keeps harness-only
content (geometry, boundary conditions, step controls, output requests, initial temperature)
outside the material definition.

Harness design (documented in ``qualification/abaqus/README.md``):

* one fully integrated ``C3D8`` brick, 1 mm x 1 mm x 1 mm (N-mm-MPa-s units), so the
  Abaqus characteristic element length used by ``TYPE=DISPLACEMENT`` is 1 mm and the stress
  state is homogeneous (no hourglass control is involved);
* symmetry-type boundary conditions only: U1=0 on x=0, U2=0 on y=0, U3=0 on z=0. These remove
  rigid-body modes while the lateral faces y=1 and z=1 remain traction free, so the response is
  uniaxial stress — no lateral confinement is imposed;
* displacement control of the x=1 face through ``U1`` in a single ``*STATIC`` step
  (Abaqus/Standard), small fixed maximum increment, no stabilization or extra viscosity: the
  exported material, including its viscosity parameter, is used exactly as generated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .abaqus_dependent import (
    AbaqusDependentMode,
    AbaqusLegacyDependentResult,
    build_abaqus_legacy_dependent_material_text,
)
from .abaqus_legacy import (
    AbaqusLegacyInputError,
    AbaqusLegacyMaterialResult,
    _fmt,
    build_abaqus_legacy_cdp_material_text,
)

QUALIFICATION_ELEMENT_TYPE = "C3D8"
QUALIFICATION_ELEMENT_SIZE_MM = 1.0
QUALIFICATION_MATERIAL_NAME = "CDP_QUALIFICATION"
FIELD_OUTPUT_VARIABLES = ("S", "E", "PE", "PEEQ", "PEEQT", "DAMAGEC", "DAMAGET")
# Default compression target as a multiple of the legacy peak strain e_c1.
COMPRESSION_TARGET_FACTOR_E_C1 = 3.0
# Default tension target crack opening as a fraction of the last exported crack opening.
TENSION_TARGET_CRACK_FRACTION = 0.25


@dataclass(frozen=True)
class QualificationDeck:
    """A complete Abaqus input deck plus the summary needed to evaluate its response."""

    case: str
    job_name: str
    text: str
    loading: str
    input_summary: dict[str, object]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _material_lines(material_text: str) -> list[str]:
    return [line for line in material_text.splitlines() if line]


def _single_element_deck(
    *,
    title: str,
    material_text: str,
    displacement_mm: float,
    initial_temperature_c: float | None = None,
) -> str:
    size = _fmt(QUALIFICATION_ELEMENT_SIZE_MM)
    lines = [
        "*HEADING",
        title,
        "** cdp-generator ABAQUS-Q1 single-element qualification deck.",
        "** Units: N, mm, MPa, s. Harness content below is separate from the material card.",
        "** BCs: U1=0 on x=0, U2=0 on y=0, U3=0 on z=0; lateral faces free (uniaxial stress).",
        "*NODE, NSET=ALLN",
        "1, 0., 0., 0.",
        f"2, {size}, 0., 0.",
        f"3, {size}, {size}, 0.",
        f"4, 0., {size}, 0.",
        f"5, 0., 0., {size}",
        f"6, {size}, 0., {size}",
        f"7, {size}, {size}, {size}",
        f"8, 0., {size}, {size}",
        f"*ELEMENT, TYPE={QUALIFICATION_ELEMENT_TYPE}, ELSET=EALL",
        "1, 1, 2, 3, 4, 5, 6, 7, 8",
        "*NSET, NSET=X0",
        "1, 4, 5, 8",
        "*NSET, NSET=X1",
        "2, 3, 6, 7",
        "*NSET, NSET=Y0",
        "1, 2, 5, 6",
        "*NSET, NSET=Z0",
        "1, 2, 3, 4",
        "*NSET, NSET=REFNODE",
        "7",
        f"*SOLID SECTION, ELSET=EALL, MATERIAL={QUALIFICATION_MATERIAL_NAME}",
        ",",
        "** ---- material card (canonical builder output, unchanged) ----",
        *_material_lines(material_text),
        "** ---- end of material card ----",
    ]
    if initial_temperature_c is not None:
        lines += [
            "*INITIAL CONDITIONS, TYPE=TEMPERATURE",
            f"ALLN, {_fmt(initial_temperature_c)}",
        ]
    lines += [
        "*BOUNDARY",
        "X0, 1, 1",
        "Y0, 2, 2",
        "Z0, 3, 3",
        "*STEP, NAME=LOAD, NLGEOM=NO, INC=10000",
        "*STATIC",
        "0.005, 1., 1e-08, 0.01",
        "*BOUNDARY",
        f"X1, 1, 1, {_fmt(displacement_mm)}",
        "*OUTPUT, FIELD, FREQUENCY=1",
        "*ELEMENT OUTPUT, ELSET=EALL",
        ", ".join(FIELD_OUTPUT_VARIABLES),
        "*NODE OUTPUT",
        "U, RF",
        "*OUTPUT, HISTORY, FREQUENCY=1",
        "*NODE OUTPUT, NSET=REFNODE",
        "U1",
        "*END STEP",
    ]
    return "\n".join(lines) + "\n"


def _require_static(result: AbaqusLegacyMaterialResult) -> None:
    if not isinstance(result, AbaqusLegacyMaterialResult):
        raise AbaqusLegacyInputError("result must be AbaqusLegacyMaterialResult.")


def build_abaqus_static_compression_qualification_deck(
    result: AbaqusLegacyMaterialResult,
    *,
    target_strain: float | None = None,
) -> QualificationDeck:
    """Uniaxial displacement-controlled compression of one C3D8 element (full analysis)."""

    _require_static(result)
    e_c1 = float(result.inputs["e_c1"])
    strain = COMPRESSION_TARGET_FACTOR_E_C1 * e_c1 if target_strain is None else target_strain
    if not strain > 0.0:
        raise AbaqusLegacyInputError("target_strain must be positive (applied as compression).")
    displacement = -strain * QUALIFICATION_ELEMENT_SIZE_MM
    material = build_abaqus_legacy_cdp_material_text(result, QUALIFICATION_MATERIAL_NAME)
    text = _single_element_deck(
        title="cdp-generator ABAQUS-Q1 static_compression",
        material_text=material,
        displacement_mm=displacement,
    )
    stresses = [row["stress_mpa"] for row in result.compression_hardening]
    summary: dict[str, object] = {
        "loading": "uniaxial_compression",
        "element": QUALIFICATION_ELEMENT_TYPE,
        "element_size_mm": QUALIFICATION_ELEMENT_SIZE_MM,
        "applied_displacement_mm": displacement,
        "applied_nominal_strain": -strain,
        "E_mpa": result.elastic["E_mpa"],
        "table_peak_stress_mpa": max(stresses),
        "table_initial_yield_stress_mpa": stresses[0],
        "damage_enabled": result.compression_damage is not None,
        "material_source": "build_abaqus_legacy_cdp_material_text (M4 static)",
    }
    return QualificationDeck(
        "static_compression", "q1_static_compression", text, "compression", summary
    )


def build_abaqus_static_tension_qualification_deck(
    result: AbaqusLegacyMaterialResult,
    *,
    target_displacement_mm: float | None = None,
) -> QualificationDeck:
    """Uniaxial displacement-controlled tension into the softening branch (full analysis)."""

    _require_static(result)
    peak = float(result.tension_stiffening[0]["stress_mpa"])
    w_max = float(result.tension_stiffening[-1]["cracking_displacement_mm"])
    E0 = float(result.elastic["E_mpa"])
    if target_displacement_mm is None:
        displacement = peak / E0 * QUALIFICATION_ELEMENT_SIZE_MM + (
            TENSION_TARGET_CRACK_FRACTION * w_max
        )
    else:
        displacement = target_displacement_mm
    if not displacement > 0.0:
        raise AbaqusLegacyInputError("target_displacement_mm must be positive.")
    material = build_abaqus_legacy_cdp_material_text(result, QUALIFICATION_MATERIAL_NAME)
    text = _single_element_deck(
        title="cdp-generator ABAQUS-Q1 static_tension",
        material_text=material,
        displacement_mm=displacement,
    )
    summary: dict[str, object] = {
        "loading": "uniaxial_tension",
        "element": QUALIFICATION_ELEMENT_TYPE,
        "element_size_mm": QUALIFICATION_ELEMENT_SIZE_MM,
        "applied_displacement_mm": displacement,
        "E_mpa": E0,
        "table_peak_stress_mpa": peak,
        "table_max_cracking_displacement_mm": w_max,
        "tension_law": result.inputs["tension_law"],
        "damage_enabled": result.tension_damage is not None,
        "material_source": "build_abaqus_legacy_cdp_material_text (M4 static)",
    }
    return QualificationDeck("static_tension", "q1_static_tension", text, "tension", summary)


def build_abaqus_dependent_datacheck_deck(
    result: AbaqusLegacyDependentResult,
) -> QualificationDeck:
    """Complete deck that exercises the dependent keyword structure (datacheck gate).

    Temperature decks define an initial temperature of the first legacy case so the
    temperature-dependent tables are referenced; no thermal loading path is claimed.
    """

    if not isinstance(result, AbaqusLegacyDependentResult):
        raise AbaqusLegacyInputError("result must be AbaqusLegacyDependentResult.")
    is_rate = result.mode == AbaqusDependentMode.STRAIN_RATE
    e_c1 = float(result.inputs["e_c1"])
    displacement = -COMPRESSION_TARGET_FACTOR_E_C1 * e_c1 * QUALIFICATION_ELEMENT_SIZE_MM
    material = build_abaqus_legacy_dependent_material_text(result, QUALIFICATION_MATERIAL_NAME)
    values = result.dependency_values()
    case = "rate_datacheck" if is_rate else "temperature_datacheck"
    text = _single_element_deck(
        title=f"cdp-generator ABAQUS-Q1 {case}",
        material_text=material,
        displacement_mm=displacement,
        initial_temperature_c=None if is_rate else values[0],
    )
    summary: dict[str, object] = {
        "loading": "uniaxial_compression (datacheck only)",
        "element": QUALIFICATION_ELEMENT_TYPE,
        "dependency_mode": result.mode,
        "dependency_values": values,
        "dependency_unit": result.dependency_semantics["dependency_unit"],
        "damage_policy": result.damage_policy["policy"],
        "compression_hardening_rows": len(result.compression_hardening),
        "tension_stiffening_rows": len(result.tension_stiffening),
        "material_source": "build_abaqus_legacy_dependent_material_text (M5 dependent)",
    }
    return QualificationDeck(case, f"q1_{case}", text, "datacheck", summary)
