"""Family and semantic guards are enforced before any overlay is built."""

import plotly.graph_objects as go

from ..application.comparison import ComparisonCase, comparison_curves


def comparison_figure(cases: list[ComparisonCase], group: str) -> go.Figure:
    named = comparison_curves(cases, group)
    figure = go.Figure(
        [
            go.Scatter(x=c.x, y=c.y, name=f"{label} — {c.label}", mode="lines", meta=c.metadata)
            for label, c in named
        ]
    )
    first = named[0][1]
    figure.update_layout(
        title=group,
        xaxis_title=f"{first.x_quantity} [{first.x_unit}]",
        yaxis_title=f"{first.y_quantity} [{first.y_unit}]",
        hovermode="closest",
    )
    return figure
