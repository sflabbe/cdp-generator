"""Translate historical keys without recomputing or resampling data."""

from dataclasses import asdict
from typing import Any

from ..requests import LegacyConcreteAnalysisRequest
from ..results import CurveSeries, PropertyValue, ResultBundle

# group, source, x key, y key, x quantity, y quantity, x unit, y unit, reference only
CURVE_SPECS = (
    (
        "Compression stress-strain",
        "compression",
        "strain",
        "stress",
        "Compressive strain",
        "Compressive stress",
        "-",
        "MPa",
        False,
    ),
    (
        "Compression inelastic",
        "compression",
        "inelastic strain",
        "inelastic stress",
        "Inelastic strain",
        "Compressive stress",
        "-",
        "MPa",
        False,
    ),
    (
        "Compression damage",
        "compression",
        "inelastic strain",
        "damage",
        "Inelastic strain",
        "Damage",
        "-",
        "-",
        True,
    ),
    (
        "Tension crack opening (bilinear)",
        "tension",
        "crack opening",
        "stress",
        "Crack opening",
        "Tensile stress",
        "mm",
        "MPa",
        False,
    ),
    (
        "Tension crack opening (power law)",
        "tension",
        "crack opening",
        "stress exponential",
        "Crack opening",
        "Tensile stress",
        "mm",
        "MPa",
        False,
    ),
    (
        "Tension cracking strain (bilinear)",
        "tension",
        "cracking strain",
        "stress",
        "Cracking strain",
        "Tensile stress",
        "-",
        "MPa",
        False,
    ),
    (
        "Tension cracking strain (power law)",
        "tension",
        "cracking strain",
        "stress exponential",
        "Cracking strain",
        "Tensile stress",
        "-",
        "MPa",
        False,
    ),
    (
        "Tension damage (bilinear)",
        "tension",
        "cracking strain",
        "damage",
        "Cracking strain",
        "Damage",
        "-",
        "-",
        True,
    ),
    (
        "Tension damage (power law)",
        "tension",
        "cracking strain",
        "damage exponential",
        "Cracking strain",
        "Damage",
        "-",
        "-",
        True,
    ),
)
PROPERTY_UNITS = {
    "elasticity": "MPa",
    "shear": "MPa",
    "fracture energy": "N/mm",
    "tensile strength": "MPa",
    "dilation angle": "°",
    "poisson": "-",
    "Kc": "-",
    "fbfc": "-",
    "l0": "m",
}


def adapt_legacy_result(
    raw: dict[str, Any], request: LegacyConcreteAnalysisRequest, variables: list[float]
) -> ResultBundle:
    curves = []
    for group_index, spec in enumerate(CURVE_SPECS):
        group, source, xkey, ykey, xq, yq, xu, yu, reference = spec
        data = raw[source]
        for i, value in enumerate(variables[:1] if reference else variables):
            if xkey == "strain":
                x = data["strain temp"][i] if request.mode == "temperature" else data[xkey]
            else:
                x = data[xkey][i]
            y = data[ykey] if reference else data[ykey][i]
            label = f"T = {value:g} °C" if request.mode == "temperature" else f"ε̇ = {value:g} 1/s"
            curves.append(
                CurveSeries(
                    f"legacy.{group_index}.{i}",
                    label,
                    group,
                    [float(v) for v in x],
                    [float(v) for v in y],
                    xq,
                    yq,
                    xu,
                    yu,
                    {"mode": request.mode, "variable_value": value, "reference_only": reference},
                )
            )
    inputs = asdict(request)
    inputs["strain_rates"] = list(request.strain_rates) if request.mode == "strain_rate" else []
    return ResultBundle(
        "1.0",
        "legacy_concrete",
        request.mode,
        inputs,
        variables,
        [PropertyValue(k, float(v), PROPERTY_UNITS[k]) for k, v in raw["properties"].items()],
        curves,
        ["Legacy curves; not EC2/fib-qualified curves. Damage uses the first case."],
        {
            "provenance": "legacy_v1",
            "curve_authority": None,
            "damage_reference": variables[0],
            "temperature_source": "kernel temperature table"
            if request.mode == "temperature"
            else None,
        },
    )


def legacy_export_data(bundle: ResultBundle) -> dict[str, Any]:
    """Reconstruct only the legacy export structure from canonical curve data."""
    groups = {spec[0]: [c for c in bundle.curves if c.group == spec[0]] for spec in CURVE_SPECS}
    comp, inel, damage, crack, power, strain, _, td, tdp = [groups[s[0]] for s in CURVE_SPECS]
    return {
        "compression": {
            "strain": comp[0].x,
            "strain temp": [c.x for c in comp],
            "stress": [c.y for c in comp],
            "inelastic strain": [c.x for c in inel],
            "inelastic stress": [c.y for c in inel],
            "damage": damage[0].y,
        },
        "tension": {
            "crack opening": [c.x for c in crack],
            "stress": [c.y for c in crack],
            "stress exponential": [c.y for c in power],
            "cracking strain": [c.x for c in strain],
            "damage": td[0].y,
            "damage exponential": tdp[0].y,
        },
    }
