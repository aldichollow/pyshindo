"""High-level Plotly figures for algorithm inspection and reporting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ..duration import amplitude_duration_curve, duration_threshold_at
from ..exceptions import UnstableFilterError
from ..filters.jma import jma_filter_components, jma_filter_response
from ..filters.realtime import RealtimeFilter, design_realtime_filter, realtime_filter_response
from ..models import IntensityComparisonResult, MeasuredIntensityResult, RealtimeIntensityResult
from ..scale import INTENSITY_INTERVALS
from .theme import JMA_INTENSITY_COLORS, LINE_COLORS, apply_theme, require_plotly

_COMPONENT_KEYS = ("ns", "ew", "ud")
_COMPONENT_NAMES = ("NS", "EW", "UD")
_FILTER_STYLES = {
    RealtimeFilter.KUNUGI_2008: ("2008 approximation", "#8B5E3C", "dash"),
    RealtimeFilter.KUNUGI_2012: ("2012 approximation", LINE_COLORS["realtime"], "solid"),
    RealtimeFilter.JP7681907_LOWRATE: (
        "Generalized low-rate approximation",
        "#6D5A98",
        "dot",
    ),
}


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


def add_intensity_bands(
    figure: Any,
    *,
    row: int | None = None,
    col: int | None = None,
    opacity: float = 0.13,
    y_min: float = -3.0,
    y_max: float = 7.0,
    annotate: bool = True,
) -> Any:
    """Add subtle JMA intensity-class bands to a Plotly y-axis."""
    shape_kwargs: dict[str, Any] = {}
    if row is not None:
        shape_kwargs["row"] = row
    if col is not None:
        shape_kwargs["col"] = col
    for scale, (lower, upper) in INTENSITY_INTERVALS.items():
        visible_lower = max(lower, y_min)
        visible_upper = min(upper, y_max)
        if visible_upper <= visible_lower:
            continue
        figure.add_hrect(
            y0=visible_lower,
            y1=visible_upper,
            fillcolor=JMA_INTENSITY_COLORS[scale],
            opacity=opacity,
            line_width=0,
            layer="below",
            **shape_kwargs,
        )
        if annotate:
            annotation_kwargs: dict[str, Any] = {
                "x": 1.0,
                "xref": "x domain",
                "y": (visible_lower + visible_upper) / 2.0,
                "text": scale.japanese,
                "showarrow": False,
                "xanchor": "right",
                "font": {"size": 10, "color": LINE_COLORS["muted"]},
                "bgcolor": "rgba(255,255,255,0.58)",
                "borderpad": 2,
            }
            if row is not None and col is not None:
                figure.add_annotation(row=row, col=col, **annotation_kwargs)
            else:
                figure.add_annotation(**annotation_kwargs)
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
            line={"color": LINE_COLORS["reference"], "width": 2.6},
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
                line={"color": color, "width": 2.0, "dash": dash},
                hovertemplate="%{x:.4g} Hz<br>gain %{y:.5g}<extra></extra>",
            )
        )
    figure.update_layout(
        title="Intensity-filter amplitude response",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Amplitude gain",
        xaxis_type="log",
        yaxis_type="log",
        height=520,
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
    traces = (
        (components.period_effect, "Period effect", "#326273", "dot"),
        (components.low_cut, "Low cut", "#6C8E3F", "dash"),
        (components.high_cut, "High cut", "#C66A2B", "dashdot"),
        (components.combined, "Combined", LINE_COLORS["reference"], "solid"),
    )
    figure = go.Figure()
    for values, name, color, dash in traces:
        figure.add_trace(
            go.Scatter(
                x=frequency,
                y=values,
                mode="lines",
                name=name,
                line={"color": color, "width": 2.4 if name == "Combined" else 1.7, "dash": dash},
                hovertemplate="%{x:.4g} Hz<br>gain %{y:.5g}<extra></extra>",
            )
        )
    figure.update_layout(
        title="JMA intensity-filter factors",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Amplitude gain",
        xaxis_type="log",
        yaxis_type="log",
        height=520,
    )
    return apply_theme(figure)


def measured_result_figure(
    result: MeasuredIntensityResult,
    *,
    time_s: npt.ArrayLike | None = None,
    component_names: Sequence[str] | None = None,
) -> Any:
    """Visualize filtered components and the cumulative-duration threshold."""
    if result.filtered_acceleration_gal is None or result.resultant_acceleration_gal is None:
        raise ValueError("The result does not retain intermediate waveforms.")
    go, _, subplots = require_plotly()
    filtered = result.filtered_acceleration_gal
    resultant = result.resultant_acceleration_gal
    time = _time_axis(filtered.shape[0], result.sampling_rate_hz, time_s)
    labels = _component_labels(filtered.shape[1], component_names)

    figure = subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.58, 0.42],
        subplot_titles=("Filtered acceleration", "Resultant acceleration and threshold"),
    )
    for index, label in enumerate(labels):
        key = _COMPONENT_KEYS[min(index, len(_COMPONENT_KEYS) - 1)]
        figure.add_trace(
            go.Scattergl(
                x=time,
                y=filtered[:, index],
                mode="lines",
                name=label,
                line={"color": LINE_COLORS[key], "width": 1.15},
                hovertemplate=f"{label}<br>%{{x:.3f}} s<br>%{{y:.4g}} gal<extra></extra>",
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scattergl(
            x=time,
            y=resultant,
            mode="lines",
            name="Resultant",
            line={"color": LINE_COLORS["resultant"], "width": 1.25},
            fill="tozeroy",
            fillcolor="rgba(30,41,59,0.08)",
            hovertemplate="%{x:.3f} s<br>%{y:.4g} gal<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_hline(
        y=result.threshold_acceleration_gal,
        line={"color": LINE_COLORS["threshold"], "width": 2.0, "dash": "dash"},
        annotation_text=(
            f"a = {result.threshold_acceleration_gal:.4g} gal · "
            f"I = {result.intensity:.1f} · 震度{result.scale.japanese}"
        ),
        annotation_position="top right",
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="Acceleration [gal]", row=1, col=1)
    figure.update_yaxes(title_text="Resultant [gal]", row=2, col=1)
    figure.update_xaxes(title_text="Time [s]", row=2, col=1)
    figure.update_layout(
        title="Instrumental seismic intensity · frequency-domain reference",
        height=760,
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
            line={"color": LINE_COLORS["resultant"], "width": 1.8},
            hovertemplate="%{x:.4g} s<br>%{y:.5g} gal<extra></extra>",
        )
    )
    figure.add_vline(
        x=duration_s,
        line={"color": LINE_COLORS["threshold"], "width": 1.6, "dash": "dash"},
    )
    figure.add_hline(
        y=threshold,
        line={"color": LINE_COLORS["threshold"], "width": 1.6, "dash": "dash"},
        annotation_text=f"{duration_s:g} s · {threshold:.4g} gal",
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
            line={"color": LINE_COLORS["resultant"], "width": 1.1},
            fill="tozeroy",
            fillcolor="rgba(30,41,59,0.08)",
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
            line={"color": LINE_COLORS["realtime"], "width": 2.1},
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
            line={"color": LINE_COLORS["record_max"], "width": 1.6, "dash": "dot"},
            hovertemplate="%{x:.3f} s<br>Ia %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    add_intensity_bands(
        figure,
        row=2,
        col=1,
        y_min=y_range[0],
        y_max=y_range[1],
        annotate=True,
    )
    figure.update_yaxes(title_text="Acceleration [gal]", row=1, col=1)
    figure.update_yaxes(title_text="Instrumental intensity", range=list(y_range), row=2, col=1)
    figure.update_xaxes(title_text="Time [s]", row=2, col=1)
    title_suffix = (
        "not available"
        if np.isnan(result.approximate_intensity_raw)
        else f"Ia = {result.approximate_intensity:.1f}"
    )
    figure.update_layout(
        title=f"Real-time seismic intensity · {result.filter_name} · {title_suffix}",
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
            line={"color": LINE_COLORS["realtime"], "width": 2.0},
            hovertemplate="%{x:.3f} s<br>Ir %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=time,
            y=result.record_max_intensity_raw,
            mode="lines",
            name="Running maximum Ia",
            line={"color": LINE_COLORS["record_max"], "width": 1.5, "dash": "dot"},
            hovertemplate="%{x:.3f} s<br>Ia %{y:.3f}<extra></extra>",
        )
    )
    figure.add_hline(
        y=comparison.measured.intensity_raw,
        line={"color": LINE_COLORS["reference"], "width": 2.0, "dash": "dash"},
        annotation_text=(
            f"FFT reference {comparison.measured.intensity_raw:.3f} · "
            f"ΔI = {comparison.raw_difference:+.3f}"
        ),
        annotation_position="top right",
    )
    add_intensity_bands(figure, y_min=y_range[0], y_max=y_range[1])
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
    line_colors: Mapping[str, str] | None = None,
) -> Any:
    """Create a reusable component-wise acceleration figure."""
    go, _, _ = require_plotly()
    values = np.asarray(acceleration, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError("acceleration must be one- or two-dimensional.")
    time = _time_axis(values.shape[0], sampling_rate_hz, time_s)
    labels = _component_labels(values.shape[1], component_names)
    colors = dict(LINE_COLORS if line_colors is None else line_colors)
    figure = go.Figure()
    for index, label in enumerate(labels):
        key = _COMPONENT_KEYS[min(index, len(_COMPONENT_KEYS) - 1)]
        figure.add_trace(
            go.Scattergl(
                x=time,
                y=values[:, index],
                mode="lines",
                name=label,
                line={"color": colors[key], "width": 1.15},
                hovertemplate=f"{label}<br>%{{x:.3f}} s<br>%{{y:.4g}} {unit_label}<extra></extra>",
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="Time [s]",
        yaxis_title=f"Acceleration [{unit_label}]",
        height=470,
    )
    return apply_theme(figure)
