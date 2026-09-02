from __future__ import annotations

from pyshindo import (
    calculate_measured_intensity,
    calculate_realtime_intensity,
    synthetic_three_component_motion,
)
from pyshindo.plotting import (
    acceleration_figure,
    amplitude_duration_figure,
    filter_response_figure,
    measured_result_figure,
    realtime_result_figure,
)


def test_plotting_functions_return_figures() -> None:
    values = synthetic_three_component_motion(duration_s=3.0, noise_std_gal=0.0)
    measured = calculate_measured_intensity(values)
    realtime = calculate_realtime_intensity(values)
    figures = [
        acceleration_figure(values, 100.0),
        filter_response_figure(100.0, points=128),
        measured_result_figure(measured),
        amplitude_duration_figure(measured.resultant_acceleration_gal, 100.0),
        realtime_result_figure(realtime),
    ]
    assert all(len(figure.data) >= 1 for figure in figures)
