import numpy as np
import pytest

from cdp_generator.application import LegacyConcreteAnalysisRequest, run_legacy_concrete_analysis
from cdp_generator.visualization.plotly import build_figures, figure_for_group


@pytest.mark.parametrize("mode", ["strain_rate", "temperature"])
def test_figures_preserve_semantic_data(mode):
    bundle = run_legacy_concrete_analysis(LegacyConcreteAnalysisRequest(mode=mode))
    figures = build_figures(bundle)
    assert len(figures) == 9
    for group, figure in figures.items():
        curves = [c for c in bundle.curves if c.group == group]
        assert len(figure.data) == len(curves)
        for trace, curve in zip(figure.data, curves, strict=True):
            np.testing.assert_array_equal(trace.x, curve.x)
            np.testing.assert_array_equal(trace.y, curve.y)
            assert trace.name == curve.label
            assert ("°C" if mode == "temperature" else "1/s") in trace.name
        assert figure.layout.xaxis.title.text == f"{curves[0].x_quantity} [{curves[0].x_unit}]"
        assert figure.layout.yaxis.title.text == f"{curves[0].y_quantity} [{curves[0].y_unit}]"
    with pytest.raises(ValueError):
        figure_for_group(bundle, "absent")
