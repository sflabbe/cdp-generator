"""Orchestrate existing steel APIs without rederiving their scientific model."""

from dataclasses import asdict
from typing import Any

from ..steel.export import build_abaqus_material_card_text, build_steel_excel_bytes
from ..steel.johnson_cook import JohnsonCookParams, generate_jc_curves_multicase
from ..steel.standards import calibrate_jc_from_spec, get_steel_spec
from .results import CurveSeries
from .steel_requests import SteelAnalysisRequest, SteelInputError
from .steel_results import SteelAnalysisResult

PRESET_DISCLAIMER = (
    "Built-in approximate preset — verify against the applicable standard / project data."
)
ABAQUS_DISCLAIMER = "Experimental template — verify against the ABAQUS documentation/version used."


def run_steel_analysis(request: SteelAnalysisRequest) -> SteelAnalysisResult:
    calibration = {
        name: getattr(request, name)
        for name in ("assume_rate_temp_neutral", "n_default", "C_default", "m_default", "epsdot0")
    }
    analysis = {
        name: getattr(request, name)
        for name in ("strain_rates", "temperatures", "eps_max", "n_points", "output_kind")
    }
    try:
        spec = get_steel_spec(
            "Custom" if request.source == "custom" else request.standard,
            request.grade,
            overrides=dict(request.material_overrides),
        )
        params = calibrate_jc_from_spec(spec, **calibration, verbose=False)
        raw = generate_jc_curves_multicase(params, spec.E, **analysis)
    except ValueError as exc:
        raise SteelInputError(str(exc)) from exc
    curves = []
    for i, curve in enumerate(raw["curves"]):
        kind = curve["output_kind"]
        metadata = {
            "epsdot": float(curve["epsdot"]),
            "temperature": float(curve["T"]),
            "output_kind": kind,
            "stress_kind": kind,
            "case_id": curve["case_id"],
        }
        label = f"ε̇ = {curve['epsdot']:g} 1/s, T = {curve['T']:g} °C"
        for group, xkey, quantity in (
            ("Total stress-strain", "strain", f"{kind.title()} strain"),
            ("Stress vs true plastic strain", "plastic_strain", "True plastic strain"),
        ):
            curves.append(
                CurveSeries(
                    f"steel_{i}_{xkey}",
                    label,
                    group,
                    curve[xkey].tolist(),
                    curve["stress"].tolist(),
                    quantity,
                    "Stress" if xkey == "plastic_strain" else f"{kind.title()} stress",
                    "-",
                    "MPa",
                    dict(metadata),
                )
            )
    disclaimer = (
        PRESET_DISCLAIMER
        if request.source == "preset"
        else "User-provided material input; not independently verified."
    )
    result = SteelAnalysisResult(
        request.source,
        asdict(spec),
        "approximate_preset" if request.source == "preset" else "user_provided",
        disclaimer,
        {
            **calibration,
            "description": "Existing repository calibrator; automatic n is a heuristic, not experimental fitting.",
        },
        asdict(params),
        analysis,
        curves,
        [disclaimer, ABAQUS_DISCLAIMER],
        {"material_overrides": dict(request.material_overrides)},
    )
    result.to_json()
    return result


def _export_payload(result: SteelAnalysisResult) -> dict[str, Any]:
    """Reconstruct the historical exporter input from stored arrays; no recomputation."""
    total = [c for c in result.curves if c.group == "Total stress-strain"]
    plastic = [c for c in result.curves if c.group == "Stress vs true plastic strain"]
    return {
        "params": JohnsonCookParams(**result.johnson_cook_parameters),
        "E": result.material["E"],
        "strain_rates": result.analysis["strain_rates"],
        "temperatures": result.analysis["temperatures"],
        "curves": [
            {
                "strain": c.x,
                "stress": c.y,
                "plastic_strain": p.x,
                "epsdot": c.metadata["epsdot"],
                "T": c.metadata["temperature"],
                "case_id": c.metadata["case_id"],
                "output_kind": c.metadata["output_kind"],
            }
            for c, p in zip(total, plastic, strict=True)
        ],
    }


def steel_result_excel_bytes(result: SteelAnalysisResult) -> bytes:
    return build_steel_excel_bytes(_export_payload(result))


def steel_result_abaqus_text(result: SteelAnalysisResult) -> str:
    return build_abaqus_material_card_text(
        JohnsonCookParams(**result.johnson_cook_parameters),
        result.material["E"],
        result.material["nu"],
    )
