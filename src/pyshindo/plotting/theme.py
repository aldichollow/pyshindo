"""Plotly theme and intensity colors grounded in the JMA color guide."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Final

from ..long_period.scale import LongPeriodClass
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

# Source: the stylesheet JMA serves with its own long-period observation
# pages, `ltpgm_explain/style_j.css`, classes `td.lv0`-`td.lv4`. The color
# guideline above predates the long-period class and does not assign it
# colors; these are the same family regardless.
LONG_PERIOD_CLASS_COLORS: Final = MappingProxyType(
    {
        LongPeriodClass.ZERO: "#D3D3D3",
        LongPeriodClass.ONE: "#0040FF",
        LongPeriodClass.TWO: "#FFE600",
        LongPeriodClass.THREE: "#FF2800",
        LongPeriodClass.FOUR: "#A50021",
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
        "ns": "#33578E",
        "ew": "#C97A46",
        "ud": "#6B5B95",
        "resultant": "#181818",
        "threshold": "#D97706",
        "realtime": "#1E828C",
        "record_max": "#A14868",
        "reference": "#181818",
        "grid": "#E7E9EC",
        "muted": "#667085",
    }
)

PRIMARY_LINE_WIDTH: Final = 1.6
GUIDE_LINE_WIDTH: Final = 1.6
ACCENT_LINE_WIDTH: Final = 1.8

WAVEFORM_LINE_WIDTH: Final = 1.2

LABEL_TEXT_COLOR: Final = "#141414"
LABEL_OUTLINE_COLOR: Final = "#FFFFFF"

STAGE_COLORS: Final = (
    LINE_COLORS["ns"],
    LINE_COLORS["ew"],
    LINE_COLORS["ud"],
    LINE_COLORS["threshold"],
    LINE_COLORS["realtime"],
    LINE_COLORS["record_max"],
    "#4F8A5B",
    "#A98A3B",
    "#8C6350",
)
"""Nine colors for filter-stage plots, enough for the longest design (eight
named analog factors plus a gain stage). None is near-black, so the solid
combined trace stays distinct from the dotted per-stage traces."""

PRIMARY_LINE: Final = MappingProxyType(
    {"color": LINE_COLORS["reference"], "width": PRIMARY_LINE_WIDTH}
)
"""Style of the primary trace. The rule throughout this package is: solid
near-black for the combined or reference result, dotted color for a
contributing part or an alternative compared against it. Guide lines share
this width -- the dash pattern already tells the two roles apart."""

WAVEFORM_LINE: Final = MappingProxyType(
    {"color": LINE_COLORS["reference"], "width": WAVEFORM_LINE_WIDTH}
)
"""Primary trace at a slightly thinner weight, for a dense waveform -- an
acceleration channel or a resultant -- rather than a smooth analytic curve."""

BOUNDARY_LINE: Final = MappingProxyType({"color": "#FFFFFF", "width": 0.5})
"""Divider between two class bands. White works against both the saturated
edge strip and the pale background tint, where any gray would have to
compete with one or the other."""

GUIDE_LINE_COLOR: Final = "#364F00"
"""Color of every dotted guide line: a threshold crossing, a running
maximum, an FFT-reference level, a peak marker. Green appears nowhere in the
JMA intensity scale, so a guide is never read as part of it."""

INTENSITY_STRIP_FRACTION: Final = 0.025
"""Width of the saturated class strip along the right edge, as a fraction of
the x-domain so it scales with the figure."""

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


_AXIS_STYLE: Final = {
    "showgrid": True,
    "gridcolor": LINE_COLORS["grid"],
    "gridwidth": 0.6,
    "zeroline": False,
    "showline": True,
    "linecolor": "#333333",
    "linewidth": 1.1,
    "mirror": True,
    "ticks": "outside",
    "ticklen": 5,
    "tickwidth": 1.1,
    "tickcolor": "#333333",
    "automargin": True,
}


def register_template() -> str:
    """Register and return the package's light Plotly template name."""
    go, pio, _ = require_plotly()
    if _TEMPLATE_NAME not in pio.templates:
        pio.templates[_TEMPLATE_NAME] = go.layout.Template(
            layout={
                "font": {
                    "family": "Helvetica Neue, Helvetica, Arial, Noto Sans JP, sans-serif",
                    "size": 12,
                    "color": "#1A1A1A",
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
                    "font": {
                        "family": "Helvetica Neue, Helvetica, Arial, Noto Sans JP, sans-serif",
                        "size": 11,
                    },
                },
                "legend": {
                    "orientation": "v",
                    "yanchor": "top",
                    "y": 1.0,
                    "xanchor": "left",
                    "x": 1.02,
                    "bgcolor": "rgba(255,255,255,0)",
                    "font": {"size": 11},
                },
                "margin": {"l": 68, "r": 100, "t": 64, "b": 56},
                "title": {
                    "x": 0.0,
                    "xanchor": "left",
                    "font": {"size": 16, "weight": 500},
                },
                "xaxis": dict(_AXIS_STYLE),
                "yaxis": dict(_AXIS_STYLE),
            }
        )
    return _TEMPLATE_NAME


_LEGEND_FONT_SIZE: Final = 11
_LEGEND_SWATCH_AND_PADDING_PX: Final = 50
_LEGEND_CHARS_TO_PX: Final = 0.66


def _legend_right_margin(figure: Any) -> int:
    """Estimate the right margin a vertical outside-right legend needs.

    The legend's own width isn't fixed, so a constant margin either clips
    figures with long trace names (filter-stage labels carry a frequency,
    e.g. "A1: period effect (0.45 Hz)") or wastes space on figures with
    short ones ("NS"/"EW"/"UD"). Instead this reads the names Plotly will
    actually place in the legend and sizes the margin to the longest one,
    so every figure gets a margin matched to its own content with no
    figure-specific tuning.
    """
    names = [
        trace.name
        for trace in getattr(figure, "data", ())
        if getattr(trace, "name", None) and getattr(trace, "showlegend", None) is not False
    ]
    if not names:
        return _LEGEND_SWATCH_AND_PADDING_PX
    longest = max(len(name) for name in names)
    return _LEGEND_SWATCH_AND_PADDING_PX + round(longest * _LEGEND_FONT_SIZE * _LEGEND_CHARS_TO_PX)


def apply_theme(figure: Any, *, height: int | None = None) -> Any:
    """Apply the package template and restrained interaction defaults."""
    template = register_template()
    updates: dict[str, Any] = {
        "template": template,
        "hovermode": "x unified",
        "modebar": {"orientation": "v"},
        "margin": {"r": _legend_right_margin(figure)},
    }
    if height is not None:
        updates["height"] = height
    figure.update_layout(**updates)
    return figure
