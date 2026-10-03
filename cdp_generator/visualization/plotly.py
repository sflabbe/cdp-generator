"""Pure interactive figure builders consuming canonical curves."""

import plotly.graph_objects as go

from ..application.results import ResultBundle


def figure_for_group(bundle: ResultBundle, group: str) -> go.Figure:
    curves = [curve for curve in bundle.curves if curve.group == group]
    if not curves:
        raise ValueError(f"No curves for group: {group}")
    figure = go.Figure(
        [go.Scatter(x=c.x, y=c.y, name=c.label, mode="lines+markers") for c in curves]
    )
    first = curves[0]
    figure.update_layout(
        title=group,
        xaxis_title=f"{first.x_quantity} [{first.x_unit}]",
        yaxis_title=f"{first.y_quantity} [{first.y_unit}]",
        hovermode="closest",
    )
    return figure


def build_figures(bundle: ResultBundle) -> dict[str, go.Figure]:
    return {
        group: figure_for_group(bundle, group)
        for group in dict.fromkeys(c.group for c in bundle.curves)
    }
