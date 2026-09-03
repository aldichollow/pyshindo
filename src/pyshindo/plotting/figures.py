"""High-level Plotly figures for algorithm inspection and reporting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ..duration import amplitude_duration_curve, duration_threshold_at
from ..exceptions import UnstableFilterError
from ..filters.jma import jma_filter_components, jma_filter_response
from ..filters.realtime import (
    RealtimeFilter,
    design_realtime_filter,
    filter_stage_response,
    realtime_filter_response,
)
from ..models import (
    IntensityComparisonResult,
    MeasuredIntensityResult,
    RealtimeIntensityResult,
    RecursiveFilterDesign,
)
from ..scale import INTENSITY_INTERVALS
from .theme import (
    ACCENT_LINE_WIDTH,
    BOUNDARY_LINE,
    GUIDE_LINE_COLOR,
    GUIDE_LINE_WIDTH,
    INTENSITY_STRIP_FRACTION,
    JMA_INTENSITY_COLORS,
    LABEL_OUTLINE_COLOR,
    LABEL_TEXT_COLOR,
    LINE_COLORS,
    PRIMARY_LINE,
    STAGE_COLORS,
    WAVEFORM_LINE,
    apply_theme,
    require_plotly,
)

_COMPONENT_NAMES = ("NS", "EW", "UD")
_FILTER_STYLES = {
    RealtimeFilter.KUNUGI_2008: ("2008 approximation", "#8B5E3C", "dot"),
    RealtimeFilter.KUNUGI_2012: ("2012 approximation", LINE_COLORS["realtime"], "dot"),
    RealtimeFilter.JP7681907_LOWRATE: (
        "Generalized low-rate approximation",
        "#6D5A98",
        "dot",
    ),
}
# Sub-decade minor gridlines for the log-log filter response figures -- with
# only major gridlines at powers of ten, a value like "3 Hz" or a gain of
# "0.03" has nothing nearby to read it off against.
_LOG_MINOR_GRID = {
    "xaxis_minor_showgrid": True,
    "xaxis_minor_gridcolor": LINE_COLORS["grid"],
    "xaxis_minor_gridwidth": 0.4,
    "yaxis_minor_showgrid": True,
    "yaxis_minor_gridcolor": LINE_COLORS["grid"],
    "yaxis_minor_gridwidth": 0.4,
}
# The 8 one-pixel offsets around a point -- used to fake a text stroke (see
# _add_outlined_label) since Plotly annotations have no native outline.
_TEXT_OUTLINE_OFFSETS = tuple(
    (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)
)


def _time_axis(
    sample_count: int,
    sampling_rate_hz: float,
    time_s: npt.ArrayLike | None,
) -> np.ndarray:
    if time_s is None:
        return np.arange(sample_count, dtype=np.float64) / sampling_rate_hz
    time = np.asarray(time_s, dtype=np.float64)
    if time.shape != (sample_count,):
        raise ValueError(f"time_s must have shape ({sample_count},).")
    return time


def _component_labels(count: int, names: Sequence[str] | None) -> tuple[str, ...]:
    if names is None:
        return _COMPONENT_NAMES[:count]
    labels = tuple(names)
    if len(labels) != count:
        raise ValueError(f"component_names must contain {count} labels.")
    return labels


def _thin_indices(size: int, max_points: int) -> np.ndarray:
    if max_points < 2:
        raise ValueError("max_points must be at least two.")
    if size <= max_points:
        return np.arange(size)
    return np.unique(np.linspace(0, size - 1, max_points, dtype=np.int64))


def _rowcol_kwargs(row: int | None, col: int | None) -> dict[str, Any]:
    """Build the ``row``/``col`` kwargs shared by shape- and trace-adding calls."""
    kwargs: dict[str, Any] = {}
    if row is not None:
        kwargs["row"] = row
    if col is not None:
        kwargs["col"] = col
    return kwargs


def _add_stacked_channels(
    figure: Any,
    go: Any,
    *,
    time: np.ndarray,
    values: np.ndarray,
    labels: Sequence[str],
    unit_label: str,
) -> None:
    """Add one monochrome row per channel, each labeled by its own y-axis.

    Shared by :func:`acceleration_figure` and :func:`measured_result_figure`
    so a "these are the acceleration channels" panel looks identical --
    same color, same width, one row per channel -- wherever it appears,
    rather than each figure styling its channels differently. All rows
    share one symmetric y-range (the largest sample across every channel)
    instead of each auto-scaling on its own, so a channel's amplitude can
    be compared directly against the others just by eye.
    """
    limit = float(np.max(np.abs(values))) * 1.05 if values.size else 1.0
    if limit <= 0.0:
        limit = 1.0
    for index, label in enumerate(labels):
        row = index + 1
        figure.add_trace(
            go.Scattergl(
                x=time,
                y=values[:, index],
                mode="lines",
                name=label,
                line=dict(WAVEFORM_LINE),
                showlegend=False,
                hovertemplate=(
                    f"{label}<br>%{{x:.3f}} s<br>%{{y:.4g}} {unit_label}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(
            title_text=f"{label} [{unit_label}]", range=[-limit, limit], row=row, col=1
        )


def _add_endpoint_label(
    figure: Any,
    go: Any,
    *,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    row: int | None = None,
    col: int | None = None,
) -> Any:
    """Mark the last finite sample of ``y`` with its value.

    A single small label at the current reading -- the same touch used in
    live monitoring displays -- rather than making the reader trace the
    curve to its end and compare against the axis.
    """
    finite = np.flatnonzero(np.isfinite(y))
    if finite.size == 0:
        return figure
    index = finite[-1]
    figure.add_trace(
        go.Scatter(
            x=[x[index]],
            y=[y[index]],
            mode="markers+text",
            text=[f"{y[index]:.2f}"],
            textposition="top left",
            textfont={"size": 11, "color": color},
            marker={"color": color, "size": 5},
            showlegend=False,
            hoverinfo="skip",
        ),
        **_rowcol_kwargs(row, col),
    )
    return figure


def _add_outlined_label(
    figure: Any,
    *,
    x: float,
    xref: str,
    y: float,
    text: str,
    xanchor: str,
    font_size: int,
    row: int | None,
    col: int | None,
) -> None:
    """Draw ``text`` in :data:`LABEL_TEXT_COLOR` with a white halo.

    Plotly annotations have no native text-stroke, so the halo is built
    from eight copies of the same text nudged one pixel in every direction
    (``xshift``/``yshift``) and drawn first, with the real text on top.
    This keeps every label the same flat color regardless of what it sits
    on, instead of picking a different text color per background.
    """
    kwargs = _rowcol_kwargs(row, col)
    base = {
        "x": x,
        "xref": xref,
        "y": y,
        "text": text,
        "showarrow": False,
        "xanchor": xanchor,
    }
    for dx, dy in _TEXT_OUTLINE_OFFSETS:
        figure.add_annotation(
            **base,
            xshift=dx,
            yshift=dy,
            font={"size": font_size, "color": LABEL_OUTLINE_COLOR},
            **kwargs,
        )
    figure.add_annotation(
        **base,
        font={"size": font_size, "color": LABEL_TEXT_COLOR},
        **kwargs,
    )


def add_intensity_bands(
    figure: Any,
    *,
    row: int | None = None,
    col: int | None = None,
    opacity: float = 0.10,
    y_min: float = -3.0,
    y_max: float = 7.0,
    bands: bool = True,
    strip: bool = False,
    boundaries: bool = False,
    annotate: bool = True,
) -> Any:
    """Add JMA intensity-class reference marks to a y-axis.

    ``bands`` tints the full plot width with each class's color, very
    faintly, for context. ``strip`` instead paints a narrow, fully
    saturated band along the right edge -- like a tab poking out of the
    page -- so the classes stay identifiable by color without washing out
    the data underneath. ``boundaries`` adds a thin line at each class
    edge. ``annotate`` labels each class: on the strip, as black text with
    a white halo (see :func:`_add_outlined_label`) so one flat text color
    stays legible on every JMA color; otherwise beside the plain axis edge
    with a translucent background instead.
    """
    shape_kwargs = _rowcol_kwargs(row, col)
    for scale, (lower, upper) in INTENSITY_INTERVALS.items():
        visible_lower = max(lower, y_min)
        visible_upper = min(upper, y_max)
        if visible_upper <= visible_lower:
            continue
        if bands:
            figure.add_hrect(
                y0=visible_lower,
                y1=visible_upper,
                fillcolor=JMA_INTENSITY_COLORS[scale],
                opacity=opacity,
                line_width=0,
                layer="below",
                **shape_kwargs,
            )
        if strip:
            figure.add_shape(
                type="rect",
                xref="x domain",
                x0=1.0 - INTENSITY_STRIP_FRACTION,
                x1=1.0,
                y0=visible_lower,
                y1=visible_upper,
                fillcolor=JMA_INTENSITY_COLORS[scale],
                opacity=0.95,
                line_width=0,
                layer="below",
                **shape_kwargs,
            )
        if boundaries and np.isfinite(lower) and lower > y_min:
            figure.add_hline(
                y=lower,
                line=dict(BOUNDARY_LINE),
                opacity=0.6,
                layer="below",
                **shape_kwargs,
            )
        if annotate:
            label_y = (visible_lower + visible_upper) / 2.0
            if strip:
                _add_outlined_label(
                    figure,
                    x=1.0 - INTENSITY_STRIP_FRACTION / 2.0,
                    xref="x domain",
                    y=label_y,
                    text=scale.japanese,
                    xanchor="center",
                    font_size=10,
                    row=row,
                    col=col,
                )
            else:
                annotation_kwargs: dict[str, Any] = {
                    "x": 1.0,
                    "xref": "x domain",
                    "y": label_y,
                    "text": scale.japanese,
                    "showarrow": False,
                    "xanchor": "right",
                    "font": {"size": 10, "color": LINE_COLORS["muted"]},
                    "bgcolor": "rgba(255,255,255,0.82)",
                    "borderpad": 2,
                }
                figure.add_annotation(**annotation_kwargs, **shape_kwargs)
    return figure


def filter_response_figure(
    sampling_rate_hz: float = 100.0,
    *,
    filter_names: Sequence[str | RealtimeFilter] = (
        RealtimeFilter.KUNUGI_2008,
        RealtimeFilter.KUNUGI_2012,
    ),
    points: int = 3000,
    skip_unstable: bool = True,
) -> Any:
    """Compare the JMA response with selected digital approximations."""
    go, _, _ = require_plotly()
    if points < 32:
        raise ValueError("points must be at least 32.")
    nyquist = sampling_rate_hz / 2.0
    lower = max(0.01, nyquist / 100_000.0)
    frequency = np.geomspace(lower, nyquist * (1.0 - 1e-8), points)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frequency,
            y=jma_filter_response(frequency),
            mode="lines",
            name="JMA frequency-domain reference",
            line=dict(PRIMARY_LINE),
            hovertemplate="%{x:.4g} Hz<br>gain %{y:.5g}<extra></extra>",
        )
    )
    for item in filter_names:
        selected = RealtimeFilter.parse(item)
        if selected is RealtimeFilter.AUTO:
            design = design_realtime_filter(sampling_rate_hz, filter_name=selected)
            selected = RealtimeFilter.parse(design.name)
        else:
            try:
                design = design_realtime_filter(sampling_rate_hz, filter_name=selected)
            except UnstableFilterError:
                if skip_unstable:
                    continue
                raise
        label, color, dash = _FILTER_STYLES[selected]
        response = realtime_filter_response(design, frequency)
        figure.add_trace(
            go.Scatter(
                x=frequency,
                y=response.amplitude,
                mode="lines",
                name=label,
                line={"color": color, "width": ACCENT_LINE_WIDTH, "dash": dash},
                hovertemplate="%{x:.4g} Hz<br>gain %{y:.5g}<extra></extra>",
            )
        )
    figure.update_layout(
        title="Intensity-filter amplitude response",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Amplitude gain",
        xaxis_type="log",
        yaxis_type="log",
        **_LOG_MINOR_GRID,
        height=520,
    )
    return apply_theme(figure)


def filter_stages_figure(
    design: RecursiveFilterDesign,
    frequency_hz: npt.ArrayLike | None = None,
    *,
    points: int = 3000,
    show_combined: bool = True,
) -> Any:
    """Plot each named stage of a real-time filter design on its own.

    Each analog factor that makes up ``design`` (see :attr:`RecursiveFilterDesign.stages`)
    is drawn as its own labeled curve, so the role of each stage in the overall
    response is visible individually rather than only as a combined result.
    """
    go, _, _ = require_plotly()
    if frequency_hz is None:
        nyquist = design.nyquist_hz
        lower = max(0.01, nyquist / 100_000.0)
        frequency = np.geomspace(lower, nyquist * (1.0 - 1e-8), points)
    else:
        frequency = np.asarray(frequency_hz, dtype=np.float64)
    figure = go.Figure()
    for index, stage in enumerate(design.stages):
        response = filter_stage_response(design, stage, frequency)
        color = STAGE_COLORS[index % len(STAGE_COLORS)]
        label = (
            stage.name
            if stage.characteristic_frequency_hz is None
            else f"{stage.name} ({stage.characteristic_frequency_hz:g} Hz)"
        )
        figure.add_trace(
            go.Scatter(
                x=frequency,
                y=response.amplitude,
                mode="lines",
                name=label,
                line={"color": color, "width": ACCENT_LINE_WIDTH, "dash": "dot"},
                hovertemplate=f"{label}<br>%{{x:.4g}} Hz<br>gain %{{y:.5g}}<extra></extra>",
            )
        )
    if show_combined:
        combined = realtime_filter_response(design, frequency)
        figure.add_trace(
            go.Scatter(
                x=frequency,
                y=combined.amplitude,
                mode="lines",
                name=f"{design.name} (combined)",
                line=dict(PRIMARY_LINE),
                hovertemplate="combined<br>%{x:.4g} Hz<br>gain %{y:.5g}<extra></extra>",
            )
        )
    figure.update_layout(
        title=f"Filter stages [{design.name}]",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Amplitude gain",
        xaxis_type="log",
        yaxis_type="log",
        **_LOG_MINOR_GRID,
        height=560,
    )
    return apply_theme(figure)


def jma_filter_components_figure(
    *,
    minimum_hz: float = 0.01,
    maximum_hz: float = 100.0,
    points: int = 2000,
) -> Any:
    """Show the three published factors and their combined response."""
    go, _, _ = require_plotly()
    frequency = np.geomspace(minimum_hz, maximum_hz, points)
    components = jma_filter_components(frequency)
    component_traces = (
        (components.period_effect, "Period effect", STAGE_COLORS[0]),
        (components.low_cut, "Low cut", STAGE_COLORS[3]),
        (components.high_cut, "High cut", STAGE_COLORS[4]),
    )
    figure = go.Figure()
    for values, name, color in component_traces:
        figure.add_trace(
            go.Scatter(
                x=frequency,
                y=values,
                mode="lines",
                name=name,
                line={"color": color, "width": ACCENT_LINE_WIDTH, "dash": "dot"},
                hovertemplate=f"{name}<br>%{{x:.4g}} Hz<br>gain %{{y:.5g}}<extra></extra>",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=frequency,
            y=components.combined,
            mode="lines",
            name="Combined",
            line=dict(PRIMARY_LINE),
            hovertemplate="Combined<br>%{x:.4g} Hz<br>gain %{y:.5g}<extra></extra>",
        )
    )
    figure.update_layout(
        title="JMA intensity-filter factors",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Amplitude gain",
        xaxis_type="log",
        yaxis_type="log",
        **_LOG_MINOR_GRID,
        height=520,
    )
    return apply_theme(figure)


def measured_result_figure(
    result: MeasuredIntensityResult,
    *,
    time_s: npt.ArrayLike | None = None,
    component_names: Sequence[str] | None = None,
) -> Any:
    """Visualize filtered components and the cumulative-duration threshold.

    The filtered channels are drawn with the same stacked, one-row-per-
    channel layout as :func:`acceleration_figure` -- since both are
    fundamentally "plot these acceleration channels", just at different
    stages of the calculation, they should look like the same kind of
    figure rather than one overlaying colored components and the other
    stacking monochrome ones.
    """
    if result.filtered_acceleration_gal is None or result.resultant_acceleration_gal is None:
        raise ValueError("The result does not retain intermediate waveforms.")
    go, _, subplots = require_plotly()
    filtered = result.filtered_acceleration_gal
    resultant = result.resultant_acceleration_gal
    time = _time_axis(filtered.shape[0], result.sampling_rate_hz, time_s)
    labels = _component_labels(filtered.shape[1], component_names)
    channel_count = filtered.shape[1]
    resultant_row = channel_count + 1

    figure = subplots.make_subplots(
        rows=resultant_row,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.4 / resultant_row,
        row_heights=[*([0.55 / channel_count] * channel_count), 0.45],
        subplot_titles=(
            *([None] * channel_count),
            "Resultant acceleration and threshold",
        ),
    )
    _add_stacked_channels(figure, go, time=time, values=filtered, labels=labels, unit_label="gal")
    figure.add_trace(
        go.Scattergl(
            x=time,
            y=resultant,
            mode="lines",
            name="Resultant",
            line=dict(WAVEFORM_LINE),
            fill="tozeroy",
            fillcolor="rgba(24,24,24,0.06)",
            hovertemplate="%{x:.3f} s<br>%{y:.4g} gal<extra></extra>",
        ),
        row=resultant_row,
        col=1,
    )
    figure.add_hline(
        y=result.threshold_acceleration_gal,
        line={"color": GUIDE_LINE_COLOR, "width": GUIDE_LINE_WIDTH, "dash": "dot"},
        annotation_text=(
            f"a = {result.threshold_acceleration_gal:.4g} [gal]   "
            f"I = {result.intensity:.1f} (震度{result.scale.japanese})"
        ),
        annotation_position="top right",
        row=resultant_row,
        col=1,
    )
    figure.update_yaxes(title_text="Resultant [gal]", row=resultant_row, col=1)
    figure.update_xaxes(title_text="Time [s]", row=resultant_row, col=1)
    figure.update_layout(
        # Parentheses (not the "[...]" used elsewhere for a filter name or a
        # unit) mark a descriptive qualifier that is neither.
        title="Instrumental seismic intensity (frequency-domain reference)",
        height=170 * channel_count + 320,
    )
    return apply_theme(figure)


def amplitude_duration_figure(
    resultant_acceleration_gal: npt.ArrayLike,
    sampling_rate_hz: float,
    *,
    duration_s: float = 0.3,
    max_points: int = 20_000,
) -> Any:
    """Visualize the descending amplitude-duration relation and selected level."""
    go, _, _ = require_plotly()
    curve = amplitude_duration_curve(resultant_acceleration_gal, sampling_rate_hz)
    threshold = duration_threshold_at(
        resultant_acceleration_gal,
        sampling_rate_hz,
        duration_s=duration_s,
    )
    indices = _thin_indices(curve.amplitude.size, max_points)
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=curve.exceedance_duration_s[indices],
            y=curve.amplitude[indices],
            mode="lines",
            name="Ordered resultant",
            line=dict(PRIMARY_LINE),
            hovertemplate="%{x:.4g} s<br>%{y:.5g} gal<extra></extra>",
        )
    )
    figure.add_vline(
        x=duration_s,
        line={"color": GUIDE_LINE_COLOR, "width": GUIDE_LINE_WIDTH, "dash": "dot"},
    )
    figure.add_hline(
        y=threshold,
        line={"color": GUIDE_LINE_COLOR, "width": GUIDE_LINE_WIDTH, "dash": "dot"},
        annotation_text=f"{duration_s:g} [s]   {threshold:.4g} [gal]",
        annotation_position="top right",
    )
    figure.update_layout(
        title="Amplitude-duration selection",
        xaxis_title="Cumulative exceedance duration [s]",
        yaxis_title="Resultant acceleration [gal]",
        height=500,
    )
    return apply_theme(figure)


def realtime_result_figure(
    result: RealtimeIntensityResult,
    *,
    time_s: npt.ArrayLike | None = None,
    y_range: tuple[float, float] = (-3.0, 7.0),
) -> Any:
    """Visualize filtered resultant acceleration and real-time intensity."""
    go, _, subplots = require_plotly()
    time = _time_axis(result.sample_count, result.sampling_rate_hz, time_s)
    figure = subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.44, 0.56],
        subplot_titles=("Filtered resultant acceleration", "Real-time seismic intensity"),
    )
    figure.add_trace(
        go.Scattergl(
            x=time,
            y=result.resultant_acceleration_gal,
            mode="lines",
            name="Resultant",
            line=dict(WAVEFORM_LINE),
            fill="tozeroy",
            fillcolor="rgba(24,24,24,0.06)",
            hovertemplate="%{x:.3f} s<br>%{y:.4g} gal<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=time,
            y=result.intensity_raw,
            mode="lines",
            name="Real-time intensity Ir",
            line=dict(PRIMARY_LINE),
            hovertemplate="%{x:.3f} s<br>Ir %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scattergl(
            x=time,
            y=result.record_max_intensity_raw,
            mode="lines",
            name="Record maximum Ia",
            line={"color": GUIDE_LINE_COLOR, "width": GUIDE_LINE_WIDTH, "dash": "dot"},
            hovertemplate="%{x:.3f} s<br>Ia %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    _add_endpoint_label(
        figure,
        go,
        x=time,
        y=result.intensity_raw,
        color=LINE_COLORS["reference"],
        row=2,
        col=1,
    )
    # Plotly silently drops row/col-targeted shapes added before any trace
    # exists in that subplot cell, so this must come after the traces above.
    add_intensity_bands(
        figure,
        row=2,
        col=1,
        y_min=y_range[0],
        y_max=y_range[1],
        strip=True,
        boundaries=True,
        annotate=True,
    )
    # The class boundary lines already mark the meaningful y levels; the
    # generic every-integer gridline underneath the bands/strip just adds
    # visual noise (most obviously where it crosses the saturated strip).
    figure.update_yaxes(showgrid=False, row=2, col=1)
    figure.update_yaxes(title_text="Acceleration [gal]", row=1, col=1)
    figure.update_yaxes(title_text="Instrumental intensity", range=list(y_range), row=2, col=1)
    figure.update_xaxes(title_text="Time [s]", row=2, col=1)
    figure.update_layout(
        title=f"Real-time seismic intensity [{result.filter_name}]",
        height=780,
    )
    return apply_theme(figure)


def intensity_comparison_figure(
    comparison: IntensityComparisonResult,
    *,
    y_range: tuple[float, float] = (-3.0, 7.0),
) -> Any:
    """Compare the real-time time series with the FFT-reference scalar."""
    go, _, _ = require_plotly()
    result = comparison.realtime
    time = np.arange(result.sample_count, dtype=np.float64) / result.sampling_rate_hz
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=time,
            y=result.intensity_raw,
            mode="lines",
            name="Real-time intensity Ir",
            line=dict(PRIMARY_LINE),
            hovertemplate="%{x:.3f} s<br>Ir %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=time,
            y=result.record_max_intensity_raw,
            mode="lines",
            name="Running maximum Ia",
            line={"color": GUIDE_LINE_COLOR, "width": GUIDE_LINE_WIDTH, "dash": "dot"},
            hovertemplate="%{x:.3f} s<br>Ia %{y:.3f}<extra></extra>",
        )
    )
    _add_endpoint_label(
        figure,
        go,
        x=time,
        y=result.intensity_raw,
        color=LINE_COLORS["reference"],
    )
    figure.add_hline(
        y=comparison.measured.intensity_raw,
        line={"color": GUIDE_LINE_COLOR, "width": GUIDE_LINE_WIDTH, "dash": "dot"},
        annotation_text=(
            f"FFT reference = {comparison.measured.intensity_raw:.3f}   "
            f"ΔI = {comparison.raw_difference:+.3f}"
        ),
        annotation_position="top left",
    )
    # Plotly silently drops row/col-targeted shapes added before any trace
    # exists in that subplot cell; this figure has no subplots, so order
    # relative to the traces above does not matter here.
    add_intensity_bands(figure, y_min=y_range[0], y_max=y_range[1], strip=True, boundaries=True)
    # See the matching call in realtime_result_figure: the boundary lines
    # already mark the meaningful levels, so the generic gridline is dropped.
    figure.update_yaxes(showgrid=False)
    figure.update_layout(
        title="Reference and real-time seismic intensity",
        xaxis_title="Time [s]",
        yaxis_title="Instrumental intensity",
        yaxis_range=list(y_range),
        height=560,
    )
    return apply_theme(figure)


def acceleration_figure(
    acceleration: npt.ArrayLike,
    sampling_rate_hz: float,
    *,
    time_s: npt.ArrayLike | None = None,
    component_names: Sequence[str] | None = None,
    title: str = "Acceleration",
    unit_label: str = "gal",
) -> Any:
    """Plot each acceleration channel in its own stacked row.

    One row per channel sharing a time axis -- the layout used for
    multi-channel seismograms and similar instrument records -- rather than
    overlaying every component on a single axis, so a channel with much
    smaller amplitude (commonly the vertical component) still gets a
    readable scale of its own. The single largest sample across all
    channels is marked.
    """
    go, _, subplots = require_plotly()
    values = np.asarray(acceleration, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError("acceleration must be one- or two-dimensional.")
    time = _time_axis(values.shape[0], sampling_rate_hz, time_s)
    labels = _component_labels(values.shape[1], component_names)
    channel_count = values.shape[1]

    figure = subplots.make_subplots(
        rows=channel_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.4 / channel_count,
    )
    _add_stacked_channels(
        figure, go, time=time, values=values, labels=labels, unit_label=unit_label
    )
    peak_sample, peak_channel = np.unravel_index(np.argmax(np.abs(values)), values.shape)
    peak_value = values[peak_sample, peak_channel]
    figure.add_annotation(
        x=time[peak_sample],
        y=peak_value,
        row=int(peak_channel) + 1,
        col=1,
        text=f"{peak_value:.4g} [{unit_label}]",
        showarrow=True,
        arrowhead=2,
        arrowcolor=GUIDE_LINE_COLOR,
        font={"size": 11, "color": GUIDE_LINE_COLOR},
        ax=0,
        ay=-28,
    )
    figure.update_xaxes(title_text="Time [s]", row=channel_count, col=1)
    figure.update_layout(
        title=title,
        height=180 * channel_count + 120,
    )
    return apply_theme(figure)
