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

# A small, deliberately restrained accent family: one neutral near-black for
# the combined/reference/resultant role -- always the same trace's role
# across every figure, so it stays a plain grayscale line rather than
# picking up a color cast -- and a handful of muted, similarly-toned hues
# for the recurring signal roles (component, threshold, real-time, running
# max). Kept few and tonally close on purpose, rather than a wide
# categorical set, so a figure with several traces still reads as one
# coherent, quiet palette.
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

# Shared widths so every figure's primary trace, guide line, and per-stage
# line reads as the same weight instead of each call site picking its own
# number. PRIMARY_LINE_WIDTH and GUIDE_LINE_WIDTH are equal on purpose: a
# solid primary trace and a dotted guide already read as different roles
# from their dash pattern alone, so giving them the same thickness is what
# makes the whole set of figures feel like one consistent weight.
PRIMARY_LINE_WIDTH: Final = 1.6
GUIDE_LINE_WIDTH: Final = 1.6
ACCENT_LINE_WIDTH: Final = 1.8

# A dense, many-cycle waveform (a raw or filtered acceleration channel, a
# resultant) reads better a touch thinner than a smooth analytic curve at
# the same PRIMARY_LINE color -- same color as PRIMARY_LINE, so it is still
# recognizably the same "primary trace" role, just weighted for the kind of
# data it is drawing.
WAVEFORM_LINE_WIDTH: Final = 1.2

# One flat text color (plus a white halo where a label sits on a saturated
# JMA color -- see figures._add_outlined_label) instead of switching text
# color per background, so every label in the package reads the same way.
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
"""Nine colors for filter-stage breakdown plots -- the one place a figure
legitimately needs more than a couple of hues (the improved and low-rate
designs each have eight named analog factors plus a gain stage). Tonally
matched to LINE_COLORS rather than a wide rainbow, so the extra variety
still reads as part of the same restrained palette. None is near-black, so
the solid combined/reference trace (PRIMARY_LINE) always stands apart from
the dotted per-stage traces."""

PRIMARY_LINE: Final = MappingProxyType(
    {"color": LINE_COLORS["reference"], "width": PRIMARY_LINE_WIDTH}
)
"""Shared style for the one solid, near-black 'combined' or 'reference' trace
in a figure that also shows dotted contributing components -- the visual
rule used throughout pyshindo.plotting is: solid gray = combined/primary
result, dotted color = a contributing part or an alternative being compared
against it."""

WAVEFORM_LINE: Final = MappingProxyType(
    {"color": LINE_COLORS["reference"], "width": WAVEFORM_LINE_WIDTH}
)
"""Same color and role as :data:`PRIMARY_LINE`, at :data:`WAVEFORM_LINE_WIDTH`
-- for a raw or filtered acceleration channel or a resultant, rather than a
smooth analytic curve."""

BOUNDARY_LINE: Final = MappingProxyType({"color": "#9AA0AC", "width": 0.5})
"""Thin, pale line marking a JMA intensity-class boundary. Deliberately
understated -- a visual tick mark, not a data series -- so it stays legible
against both the pale background tint and the saturated edge strip in
:func:`~pyshindo.plotting.figures.add_intensity_bands` without competing
with either. Drawn with additional shape-level opacity on top of this
already-light color (see the ``opacity`` passed alongside it) for a
deliberately understated mark."""

GUIDE_LINE_COLOR: Final = "#364F00"
"""The one color used for every dotted guide/marker line across the package
(a threshold crossing, a running maximum, an FFT-reference level, a peak
annotation) -- a deep, clear green, rather than letting each figure's guide
line pick its own hue (previously an amber threshold line in one figure and
a rose "record max" line in another, which read as arbitrary, slightly-too-
warm accents next to the neutral primary trace). Green also does not appear
anywhere in the JMA intensity scale, so a guide line is never mistaken for
part of that scale."""

INTENSITY_STRIP_FRACTION: Final = 0.025
"""Width of the saturated JMA-color edge strip in
:func:`~pyshindo.plotting.figures.add_intensity_bands`, as a fraction of the
plot's x-domain (so it scales with the figure instead of a fixed pixel
guess)."""

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
                # The legend sits outside the axes, to the right, rather than
                # stacked above the plot -- it grows downward as entries are
                # added, so it can never collide with the title regardless of
                # how many traces a figure has. apply_theme() replaces "r"
                # with a value sized to the actual legend labels; the value
                # here is only a fallback for a figure built without it.
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
