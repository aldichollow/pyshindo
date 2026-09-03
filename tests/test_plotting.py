from __future__ import annotations

import numpy as np
import pytest

from pyshindo import (
    RealtimeFilter,
    calculate_measured_intensity,
    calculate_realtime_intensity,
    design_realtime_filter,
    synthetic_three_component_motion,
)
from pyshindo.plotting import (
    acceleration_figure,
    amplitude_duration_figure,
    filter_response_figure,
    filter_stages_figure,
    measured_result_figure,
    realtime_result_figure,
)
from pyshindo.plotting.theme import PRIMARY_LINE_WIDTH, WAVEFORM_LINE_WIDTH


def test_plotting_functions_return_figures() -> None:
    values = synthetic_three_component_motion(duration_s=3.0, noise_std_gal=0.0)
    measured = calculate_measured_intensity(values)
    realtime = calculate_realtime_intensity(values)
    assert measured.resultant_acceleration_gal is not None
    figures = [
        acceleration_figure(values, 100.0),
        filter_response_figure(100.0, points=128),
        measured_result_figure(measured),
        amplitude_duration_figure(measured.resultant_acceleration_gal, 100.0),
        realtime_result_figure(realtime),
    ]
    assert all(len(figure.data) >= 1 for figure in figures)


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
