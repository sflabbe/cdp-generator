"""Single calculation entrypoint for any frontend."""

from .. import core
from ..export import build_excel_bytes
from ..temperature import get_eurocode_temperature_table
from .adapters.legacy_concrete import adapt_legacy_result, legacy_export_data
from .requests import LegacyConcreteAnalysisRequest
from .results import ResultBundle


def temperature_cases() -> list[float]:
    return [float(v) for v in get_eurocode_temperature_table()[:, 0]]


def run_legacy_concrete_analysis(request: LegacyConcreteAnalysisRequest) -> ResultBundle:
    args = (request.f_cm, request.e_c1, request.e_clim, request.l_ch)
    if request.mode == "temperature":
        raw = core.calculate_stress_strain_temp(*args, verbose=False)
        variables = temperature_cases()
    else:
        raw = core.calculate_stress_strain(*args, request.strain_rates)
        variables = [float(v) for v in request.strain_rates]
    result = adapt_legacy_result(raw, request, variables)
    result.to_json()  # Reject non-finite kernel output at the public boundary.
    return result


def result_excel_bytes(bundle: ResultBundle) -> bytes:
    return build_excel_bytes(legacy_export_data(bundle), bundle.variables, bundle.mode)
