from __future__ import annotations

import numpy as np
import pytest

from pyshindo import (
    IntensityComparisonResult,
    RealtimeFilter,
    calculate_measured_intensity,
    calculate_realtime_intensity,
    design_realtime_filter,
    synthetic_three_component_motion,
)
from pyshindo.long_period import (
    LONG_PERIOD_CLASS_INTERVALS,
    LongPeriodClass,
    calculate_long_period_class,
)
from pyshindo.plotting import (
    acceleration_figure,
    amplitude_duration_figure,
    filter_response_figure,
    filter_stages_figure,
    intensity_comparison_figure,
    jma_filter_components_figure,
    long_period_spectrum_figure,
    measured_result_figure,
    realtime_result_figure,
)
from pyshindo.plotting.theme import (
    LONG_PERIOD_CLASS_COLORS,
    PRIMARY_LINE_WIDTH,
    WAVEFORM_LINE_WIDTH,
)


def test_plotting_functions_return_figures() -> None:
    values = synthetic_three_component_motion(duration_s=3.0, noise_std_gal=0.0)
    measured = calculate_measured_intensity(values)
    realtime = calculate_realtime_intensity(values)
    assert measured.resultant_acceleration_gal is not None
    figures = [
        acceleration_figure(values, 100.0),
        filter_response_figure(100.0, points=128),
        jma_filter_components_figure(points=128),
        measured_result_figure(measured),
        amplitude_duration_figure(measured.resultant_acceleration_gal, 100.0),
        realtime_result_figure(realtime),
        intensity_comparison_figure(
            IntensityComparisonResult(measured=measured, realtime=realtime)
        ),
    ]
    assert all(len(figure.data) >= 1 for figure in figures)


def test_the_two_intensity_figures_draw_the_same_pair_of_series() -> None:
    # Both are built from one helper, so a change to either must show up in
    # both: same names, same styling, drawn from the same arrays.
    values = synthetic_three_component_motion(duration_s=3.0, noise_std_gal=0.0)
    realtime = calculate_realtime_intensity(values)
    measured = calculate_measured_intensity(values)
    stacked = realtime_result_figure(realtime)
    compared = intensity_comparison_figure(
        IntensityComparisonResult(measured=measured, realtime=realtime)
    )

    def named(figure: object, prefix: str) -> object:
        return next(t for t in figure.data if t.name and t.name.startswith(prefix))

    a, b = named(stacked, "Real-time intensity Ir"), named(compared, "Real-time intensity Ir")
    assert a.line.width == b.line.width == PRIMARY_LINE_WIDTH
    np.testing.assert_array_equal(a.y, b.y)
    # The running-maximum series is the same data under two names, because one
    # figure covers a finished record and the other a running comparison.
    assert named(stacked, "Record maximum Ia").line.dash == "dot"
    assert named(compared, "Running maximum Ia").line.dash == "dot"
    np.testing.assert_array_equal(
        named(stacked, "Record maximum Ia").y, named(compared, "Running maximum Ia").y
    )


def test_filter_stages_figure_has_one_trace_per_stage_plus_combined() -> None:
    design = design_realtime_filter(100.0, filter_name=RealtimeFilter.KUNUGI_2012)
    figure = filter_stages_figure(design, points=64)
    assert len(figure.data) == len(design.stages) + 1
    stage_traces = figure.data[:-1]
    assert all(trace.line.dash == "dot" for trace in stage_traces)
    for stage, trace in zip(design.stages, stage_traces, strict=True):
        assert trace.name.startswith(stage.name)
    combined_trace = figure.data[-1]
    assert combined_trace.line.dash is None  # solid
    assert combined_trace.line.width == PRIMARY_LINE_WIDTH


def test_acceleration_figure_stacks_one_row_per_channel() -> None:
    values = synthetic_three_component_motion(duration_s=3.0, noise_std_gal=0.0)
    figure = acceleration_figure(values, 100.0)
    assert len(figure.data) == values.shape[1]
    assert [trace.yaxis for trace in figure.data] == ["y", "y2", "y3"]
    assert all(trace.line.width == WAVEFORM_LINE_WIDTH for trace in figure.data)

    # All channels share one symmetric y-range so amplitudes compare by eye.
    ranges = [figure.layout[f"yaxis{n}" if n > 1 else "yaxis"].range for n in (1, 2, 3)]
    assert len({tuple(r) for r in ranges}) == 1
    shared_limit = ranges[0][1]
    assert shared_limit == pytest.approx(float(np.max(np.abs(values))) * 1.05)


def test_acceleration_figure_draws_every_channel_beyond_three() -> None:
    values = np.random.default_rng(0).standard_normal((200, 4))
    figure = acceleration_figure(values, 100.0)
    assert len(figure.data) == 4
    assert [trace.name for trace in figure.data] == [
        "Channel 1",
        "Channel 2",
        "Channel 3",
        "Channel 4",
    ]


def test_acceleration_figure_rejects_an_empty_record() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        acceleration_figure(np.zeros((0, 3)), 100.0)


def test_long_period_spectrum_figure_shades_the_class_bands() -> None:
    values = np.ascontiguousarray(
        synthetic_three_component_motion(duration_s=20.0)[:, :2]
    ) * 12.0
    result = calculate_long_period_class(values, 100.0)
    figure = long_period_spectrum_figure(result)

    # The spectrum itself, drawn with the shared primary-trace style.
    spectrum = next(t for t in figure.data if t.name == "Sva (horizontal composite)")
    assert spectrum.line.width == PRIMARY_LINE_WIDTH
    np.testing.assert_allclose(spectrum.y, result.sva_cm_s)

    # Bands are drawn in JMA's own long-period class colors, the same
    # mechanism the intensity figures use.
    fills = {shape.fillcolor for shape in figure.layout.shapes if shape.fillcolor}
    assert fills <= set(LONG_PERIOD_CLASS_COLORS.values())
    assert LONG_PERIOD_CLASS_COLORS[result.long_period_class] in fills

    # Labels are positioned by the logarithm of the value on a log axis; a
    # regression here would park them at 10^(threshold).
    axis_top = figure.layout.yaxis.range[1]
    assert all(
        annotation.y <= axis_top + 1e-9 for annotation in figure.layout.annotations
    )
    assert figure.layout.yaxis.type == "log"


def test_long_period_spectrum_figure_keeps_a_quiet_record_readable() -> None:
    # A class-0 record must not be squashed against the bottom of a range
    # stretching to the 100 cm/s boundary it never approaches.
    values = np.ascontiguousarray(
        synthetic_three_component_motion(duration_s=20.0)[:, :2]
    ) * 0.05
    result = calculate_long_period_class(values, 100.0)
    assert result.long_period_class is LongPeriodClass.ZERO

    figure = long_period_spectrum_figure(result)
    axis_top = 10.0 ** figure.layout.yaxis.range[1]
    assert axis_top < LONG_PERIOD_CLASS_INTERVALS[LongPeriodClass.ONE][0]

    # The axis must also not run far below the data: a spectrum spanning one
    # decade should not be drawn inside a three-decade window.
    axis_bottom = 10.0 ** figure.layout.yaxis.range[0]
    assert axis_bottom > float(result.sva_cm_s.min()) / 3.0
