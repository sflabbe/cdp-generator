import pytest

from cdp_generator.application import SteelAnalysisRequest, run_steel_analysis
from cdp_generator.visualization.steel_plotly import build_steel_figures


@pytest.mark.parametrize("kind", ["true", "engineering"])
def test_canonical_figures(kind):
    result = run_steel_analysis(
        SteelAnalysisRequest(strain_rates=(0.001, 1), temperatures=(20, 400), output_kind=kind)
    )
    figures = build_steel_figures(result)
    assert set(figures) == {"Total stress-strain", "Stress vs true plastic strain"}
    for group, figure in figures.items():
        curves = [c for c in result.curves if c.group == group]
        assert len(figure.data) == 4
        assert figure.layout.xaxis.title.text == f"{curves[0].x_quantity} [-]"
        assert figure.layout.yaxis.title.text == f"{curves[0].y_quantity} [MPa]"
        for trace, curve in zip(figure.data, curves, strict=True):
            assert list(trace.x) == curve.x
            assert list(trace.y) == curve.y
            assert trace.name == curve.label
            assert trace.meta == curve.metadata
