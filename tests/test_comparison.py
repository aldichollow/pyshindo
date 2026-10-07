from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from pyshindo import calculate_measured_intensity
from pyshindo.comparison import compare_intensity_methods, compare_results
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


# --------------------------------------------------------------------------
# compare_results: generic field-by-field equality for a result dataclass
# with array fields, where the auto-generated __eq__ (eq=False) is disabled
# because `a.array_field == b.array_field` is itself an array, not a bool.
# --------------------------------------------------------------------------


def test_compare_results_on_real_results_from_the_same_input() -> None:
    # synthetic_three_component_motion defaults to a fixed seed, so two
    # independently computed results from the same call are a real,
    # deterministic equality case -- not a synthetic dataclass stand-in.
    acceleration = _record()
    first = calculate_measured_intensity(acceleration, RATE, unit="gal")
    second = calculate_measured_intensity(acceleration, RATE, unit="gal")
    report = compare_results(first, second)
    assert report.equal
    assert bool(report) is True
    assert report.differences == ()


def test_compare_results_ignores_timing_even_though_it_always_differs() -> None:
    acceleration = _record()
    first = calculate_measured_intensity(acceleration, RATE, unit="gal")
    second = calculate_measured_intensity(acceleration, RATE, unit="gal")
    assert first.timing != second.timing  # wall-clock timing is never identical
    assert compare_results(first, second).equal  # but that alone must not fail the comparison


def test_compare_results_detects_a_real_numeric_difference() -> None:
    acceleration = _record()
    first = calculate_measured_intensity(acceleration, RATE, unit="gal")
    second = calculate_measured_intensity(acceleration * 1.5, RATE, unit="gal")
    report = compare_results(first, second)
    assert not report.equal
    assert bool(report) is False
    names = {difference.field_name for difference in report.differences}
    assert "intensity_raw" in names
    assert "ComparisonReport" in str(report)


@dataclass(frozen=True, slots=True, eq=False)
class _Sample:
    label: str
    value: float
    values: np.ndarray
    optional_values: np.ndarray | None = None
    timing: float = field(default=0.0, compare=False)


def test_compare_results_requires_matching_types() -> None:
    with pytest.raises(TypeError, match="same type"):
        compare_results(_Sample("a", 1.0, np.array([1.0])), object())


def test_compare_results_requires_dataclass_instances() -> None:
    with pytest.raises(TypeError, match="dataclass instances"):
        compare_results(_Sample, _Sample)


def test_compare_results_treats_nan_as_agreeing_in_floats_and_arrays() -> None:
    a = _Sample("x", float("nan"), np.array([1.0, float("nan"), 3.0]))
    b = _Sample("x", float("nan"), np.array([1.0, float("nan"), 3.0]))
    assert compare_results(a, b).equal


def test_compare_results_detects_an_array_shape_mismatch() -> None:
    a = _Sample("x", 1.0, np.array([1.0, 2.0]))
    b = _Sample("x", 1.0, np.array([1.0, 2.0, 3.0]))
    report = compare_results(a, b)
    assert not report.equal
    assert "shape mismatch" in report.differences[0].reason


def test_compare_results_detects_an_optional_array_present_on_only_one_side() -> None:
    a = _Sample("x", 1.0, np.array([1.0]), optional_values=np.array([1.0]))
    b = _Sample("x", 1.0, np.array([1.0]), optional_values=None)
    report = compare_results(a, b)
    assert not report.equal
    assert report.differences[0].field_name == "optional_values"


def test_compare_results_both_optional_arrays_none_is_not_a_difference() -> None:
    a = _Sample("x", 1.0, np.array([1.0]))
    b = _Sample("x", 1.0, np.array([1.0]))
    assert compare_results(a, b).equal


def test_compare_results_respects_rtol_atol() -> None:
    a = _Sample("x", 1.0, np.array([100.0]))
    b = _Sample("x", 1.0, np.array([100.0001]))
    assert not compare_results(a, b, rtol=0.0, atol=1e-9).equal
    assert compare_results(a, b, rtol=0.0, atol=1e-3).equal


def test_compare_results_reports_absolute_and_relative_error() -> None:
    a = _Sample("x", 1.0, np.array([100.0, 200.0]))
    b = _Sample("x", 1.0, np.array([101.0, 202.0]))
    difference = compare_results(a, b, rtol=0.0, atol=0.0).differences[0]
    assert difference.max_absolute_error == pytest.approx(2.0)
    assert difference.max_relative_error == pytest.approx(2.0 / 202.0)


def test_compare_results_falls_back_to_equality_for_non_numeric_fields() -> None:
    a = _Sample("a", 1.0, np.array([1.0]))
    b = _Sample("b", 1.0, np.array([1.0]))
    report = compare_results(a, b)
    assert not report.equal
    assert report.differences[0].field_name == "label"


def test_compare_results_recurses_into_nested_dataclasses() -> None:
    @dataclass(frozen=True, slots=True, eq=False)
    class _Wrapper:
        inner: _Sample

    a = _Wrapper(_Sample("x", 1.0, np.array([1.0])))
    b = _Wrapper(_Sample("y", 1.0, np.array([1.0])))
    report = compare_results(a, b)
    assert not report.equal
    assert "nested mismatch" in report.differences[0].reason
