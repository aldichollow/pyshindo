"""Plotly theme and intensity colors grounded in the JMA color guide."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Final

from ..scale import IntensityScale

# Source: Japan Meteorological Agency, "Color usage guideline for weather
# information on the JMA website", Table 2-2 (seismic intensity).
# The guide does not assign a color to intensity 0; the neutral color below is
# a package presentation choice and is kept visually distinct from intensity 1.
JMA_INTENSITY_COLORS: Final = MappingProxyType(
    {
        IntensityScale.ZERO: "#F7F8FA",
        IntensityScale.ONE: "#F2F2FF",
        IntensityScale.TWO: "#00AAFF",
        IntensityScale.THREE: "#0041FF",
        IntensityScale.FOUR: "#FAE696",
        IntensityScale.FIVE_LOWER: "#FFE600",
        IntensityScale.FIVE_UPPER: "#FF9900",
        IntensityScale.SIX_LOWER: "#FF2800",
        IntensityScale.SIX_UPPER: "#A50021",
        IntensityScale.SEVEN: "#B40068",
    }
)

TEXT_ON_INTENSITY: Final = MappingProxyType(
    {
        IntensityScale.ZERO: "#18202A",
        IntensityScale.ONE: "#18202A",
        IntensityScale.TWO: "#07111A",
        IntensityScale.THREE: "#FFFFFF",
        IntensityScale.FOUR: "#18202A",
        IntensityScale.FIVE_LOWER: "#18202A",
        IntensityScale.FIVE_UPPER: "#18202A",
        IntensityScale.SIX_LOWER: "#FFFFFF",
        IntensityScale.SIX_UPPER: "#FFFFFF",
        IntensityScale.SEVEN: "#FFFFFF",
    }
)

LINE_COLORS: Final = MappingProxyType(
    {
        "ns": "#2E5EAA",
        "ew": "#D06A3A",
        "ud": "#7057A8",
        "resultant": "#1E293B",
        "threshold": "#C46A13",
        "realtime": "#087E8B",
        "record_max": "#9D174D",
        "reference": "#3B4252",
        "grid": "#D9DEE7",
        "muted": "#667085",
    }
)

_TEMPLATE_NAME: Final = "pyshindo"


def require_plotly() -> tuple[Any, Any, Any]:
    """Import Plotly lazily and return graph objects, I/O, and subplots."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly import subplots
    except ImportError as exc:
        raise ImportError(
            "Plotting requires the optional dependency group: pip install 'pyshindo[plot]'."
        ) from exc
    return go, pio, subplots


def register_template() -> str:
    """Register and return the package's light Plotly template name."""
    go, pio, _ = require_plotly()
    if _TEMPLATE_NAME not in pio.templates:
        pio.templates[_TEMPLATE_NAME] = go.layout.Template(
            layout={
                "font": {
                    "family": "Inter, Noto Sans JP, Yu Gothic UI, sans-serif",
                    "size": 13,
                    "color": "#18202A",
                },
                "paper_bgcolor": "#FFFFFF",
                "plot_bgcolor": "#FFFFFF",
                "colorway": [
                    LINE_COLORS["ns"],
                    LINE_COLORS["ew"],
                    LINE_COLORS["ud"],
                    LINE_COLORS["realtime"],
                    LINE_COLORS["record_max"],
                ],
                "hoverlabel": {
                    "bgcolor": "#FFFFFF",
                    "bordercolor": "#C9D0DB",
                    "font": {"family": "Inter, Noto Sans JP, sans-serif", "size": 12},
                },
                "legend": {
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "left",
                    "x": 0.0,
                    "bgcolor": "rgba(255,255,255,0)",
                    "font": {"size": 12},
                },
                "margin": {"l": 72, "r": 28, "t": 76, "b": 60},
                "title": {"x": 0.0, "xanchor": "left", "font": {"size": 21}},
                "xaxis": {
                    "showgrid": True,
                    "gridcolor": LINE_COLORS["grid"],
                    "gridwidth": 0.7,
                    "zeroline": False,
                    "showline": True,
                    "linecolor": "#AAB2BF",
                    "ticks": "outside",
                    "tickcolor": "#AAB2BF",
                    "automargin": True,
                },
                "yaxis": {
                    "showgrid": True,
                    "gridcolor": LINE_COLORS["grid"],
                    "gridwidth": 0.7,
                    "zeroline": False,
                    "showline": True,
                    "linecolor": "#AAB2BF",
                    "ticks": "outside",
                    "tickcolor": "#AAB2BF",
                    "automargin": True,
                },
            }
        )
    return _TEMPLATE_NAME


def apply_theme(figure: Any, *, height: int | None = None) -> Any:
    """Apply the package template and restrained interaction defaults."""
    template = register_template()
    updates: dict[str, Any] = {
        "template": template,
        "hovermode": "x unified",
        "modebar": {"orientation": "v"},
    }
    if height is not None:
        updates["height"] = height
    figure.update_layout(**updates)
    return figure
