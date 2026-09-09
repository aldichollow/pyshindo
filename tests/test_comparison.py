from __future__ import annotations

import pytest

from pyshindo.comparison import compare_intensity_methods
from pyshindo.synthetic import synthetic_three_component_motion

RATE = 100.0


def _record():
    return synthetic_three_component_motion(sampling_rate_hz=RATE, duration_s=10.0)


def test_comparison_result_properties() -> None:
    comparison = compare_intensity_methods(_record(), RATE)

    assert comparison.raw_difference == pytest.approx(
        comparison.measured.intensity_raw - comparison.realtime.approximate_intensity_raw
    )
    assert comparison.reported_difference == pytest.approx(
        comparison.measured.intensity - comparison.realtime.approximate_intensity
    )
    assert comparison.absolute_raw_difference == abs(comparison.raw_difference)
    assert isinstance(comparison.scale_agreement, bool)
    assert comparison.scale_agreement == (
        comparison.measured.scale is comparison.realtime.approximate_scale
    )


def test_algorithm_specific_options_are_applied() -> None:
    # retain_intermediates isn't a common input option, so it should reach
    # calculate_measured_intensity through measured_options.
    comparison = compare_intensity_methods(
        _record(), RATE, measured_options={"retain_intermediates": True}
    )
    assert comparison.measured.filtered_acceleration_gal is not None


@pytest.mark.parametrize(
    "options_kwarg", ["measured_options", "realtime_options"]
)
def test_common_options_cannot_be_overridden(options_kwarg: str) -> None:
    with pytest.raises(ValueError, match="must not override common input option"):
        compare_intensity_methods(_record(), RATE, **{options_kwarg: {"sampling_rate_hz": 50.0}})
