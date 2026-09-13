from __future__ import annotations

import numpy as np
import pytest

from pyshindo.signal import (
    cosine_taper,
    detect_clipping,
    detrend_acceleration,
    peak_ground_acceleration,
    remove_offset,
    resample_acceleration,
    time_axis,
    vector_resultant,
)
from pyshindo.synthetic import synthetic_three_component_motion
from pyshindo.units import convert_acceleration
from pyshindo.validation import sampling_diagnostics


def test_vector_resultant_and_pga() -> None:
    values = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]])
    assert np.array_equal(vector_resultant(values), [5.0, 12.0])
    assert peak_ground_acceleration(values) == 12.0


def test_remove_offset_uses_initial_baseline() -> None:
    values = np.array([[2.0, 3.0], [2.0, 3.0], [4.0, 5.0]])
    corrected = remove_offset(values, baseline_samples=2)
    assert np.array_equal(corrected, [[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]])


def test_detrend_acceleration_linear_removes_a_known_slope() -> None:
    t = np.arange(200, dtype=np.float64)
    values = np.column_stack([2.0 * t + 5.0, -3.0 * t + 1.0])
    detrended = detrend_acceleration(values, mode="linear")
    np.testing.assert_allclose(detrended, 0.0, atol=1e-9)


def test_detrend_acceleration_constant_only_removes_the_mean() -> None:
    t = np.arange(200, dtype=np.float64)
    values = np.column_stack([2.0 * t + 5.0, np.full(200, 3.0)])
    detrended = detrend_acceleration(values, mode="constant")
    # A genuine slope survives constant detrending; a pure offset does not.
    assert not np.allclose(detrended[:, 0], 0.0)
    np.testing.assert_allclose(detrended[:, 1], 0.0, atol=1e-9)


def test_cosine_taper_attenuates_the_edges_and_keeps_the_middle() -> None:
    values = np.ones((200, 2))
    tapered = cosine_taper(values, fraction=0.1)
    assert tapered[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert tapered[-1, 0] == pytest.approx(0.0, abs=1e-6)
    assert tapered[100, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("fraction", [-0.1, 0.6, float("nan"), float("inf")])
def test_cosine_taper_rejects_an_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        cosine_taper(np.ones((10, 1)), fraction=fraction)


def test_time_axis_applies_the_start_offset() -> None:
    axis = time_axis(5, 100.0, start_s=2.0)
    np.testing.assert_allclose(axis, [2.0, 2.01, 2.02, 2.03, 2.04])


def test_time_axis_rejects_a_negative_sample_count() -> None:
    with pytest.raises(ValueError, match="sample_count"):
        time_axis(-1, 100.0)


def test_time_axis_rejects_a_non_finite_start() -> None:
    with pytest.raises(ValueError, match="start_s"):
        time_axis(10, 100.0, start_s=float("nan"))


def test_resample_changes_sample_count() -> None:
    values = np.arange(30, dtype=float).reshape(10, 3)
    resampled = resample_acceleration(values, original_rate_hz=50.0, target_rate_hz=100.0)
    assert resampled.shape == (20, 3)


def test_unit_conversion() -> None:
    assert convert_acceleration([1.0], "m/s^2", "gal")[0] == pytest.approx(100.0)
    assert convert_acceleration([1.0], "g", "m/s^2")[0] == pytest.approx(9.80665)


def test_sampling_diagnostics() -> None:
    timestamps = np.arange(100, dtype=np.float64) / 100.0
    diagnostics = sampling_diagnostics(timestamps)
    assert diagnostics.sampling_rate_hz == pytest.approx(100.0)
    assert diagnostics.is_uniform


def test_sampling_diagnostics_rejects_non_increasing_timestamps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        sampling_diagnostics([0.0, 0.02, 0.01, 0.03])


@pytest.mark.parametrize("timestamps", [[0.0], np.zeros((5, 2))])
def test_sampling_diagnostics_rejects_the_wrong_shape(timestamps: object) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        sampling_diagnostics(timestamps)


def test_sampling_diagnostics_rejects_non_finite_timestamps() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        sampling_diagnostics([0.0, 0.01, float("nan"), 0.03])


# --------------------------------------------------------------------------
# detect_clipping
# --------------------------------------------------------------------------


def test_detect_clipping_finds_nothing_in_a_clean_synthetic_record() -> None:
    acc = synthetic_three_component_motion(sampling_rate_hz=100.0, duration_s=20.0)
    report = detect_clipping(acc)
    assert not report.any_suspected
    assert report.components_affected == ()
    assert report.component_count == 3
    assert report.sample_count == acc.shape[0]


def test_detect_clipping_flags_saturation_at_a_known_range() -> None:
    acc = synthetic_three_component_motion(sampling_rate_hz=100.0, duration_s=5.0)[:, :1]
    acc[100:106, 0] = 2000.0  # pinned at a hypothetical +-2000 gal digitizer range
    report = detect_clipping(acc, max_range_gal=2000.0)
    assert report.any_suspected
    interval = next(i for i in report.intervals if i.method == "known_range")
    assert interval.component == 0
    assert interval.start_sample == 100
    assert interval.end_sample == 106
    assert interval.value == pytest.approx(2000.0)


def test_detect_clipping_known_range_respects_tolerance() -> None:
    acc = np.full((50, 1), 1995.0)  # within 0.1% tolerance of 2000, but not exactly at it
    report = detect_clipping(acc, max_range_gal=2000.0, range_tolerance=0.01)
    assert any(i.method == "known_range" for i in report.intervals)

    report_strict = detect_clipping(acc, max_range_gal=2000.0, range_tolerance=0.0001)
    assert not any(i.method == "known_range" for i in report_strict.intervals)


def test_detect_clipping_flags_a_repeated_value_near_the_extreme() -> None:
    rng = np.random.default_rng(0)
    acc = rng.normal(scale=5.0, size=(2000, 1))
    acc[500:505, 0] = 900.0  # a saturated run, far above the rest of the record
    report = detect_clipping(acc)
    assert report.any_suspected
    interval = next(i for i in report.intervals if i.method == "repeated_extreme")
    assert interval.component == 0
    assert interval.start_sample == 500
    assert interval.end_sample == 505


def test_detect_clipping_does_not_flag_a_repeated_quiet_noise_floor() -> None:
    # A quiet pre-event segment that happens to repeat the same quantized value
    # several times, far from the record's own peak, must not be flagged --
    # only a repeat near the component's own extreme counts.
    rng = np.random.default_rng(1)
    acc = rng.normal(scale=2.0, size=(2000, 1))
    acc[100:110, 0] = 0.05  # quiet, repeated, nowhere near the peak below
    acc[1500, 0] = 300.0  # one large, isolated, legitimate sample -- the real peak
    report = detect_clipping(acc)
    assert not report.any_suspected


def test_detect_clipping_requires_the_configured_run_length() -> None:
    acc = np.zeros((100, 1))
    acc[10:12, 0] = 500.0  # only two repeats
    report = detect_clipping(acc, repeat_threshold=3)
    assert not report.any_suspected
    report_lower = detect_clipping(acc, repeat_threshold=2)
    assert report_lower.any_suspected


def test_detect_clipping_only_flags_the_affected_component() -> None:
    acc = np.zeros((200, 3))
    acc[:, 0] = np.random.default_rng(2).normal(scale=1.0, size=200)
    acc[50:55, 1] = 800.0  # only component 1 saturates
    acc[:, 2] = np.random.default_rng(3).normal(scale=1.0, size=200)
    report = detect_clipping(acc)
    assert report.components_affected == (1,)


def test_detect_clipping_handles_a_record_shorter_than_repeat_threshold() -> None:
    acc = np.array([[1.0], [1.0]])  # only 2 samples, default repeat_threshold=3
    report = detect_clipping(acc)
    assert not report.any_suspected


def test_detect_clipping_handles_an_all_zero_component() -> None:
    acc = np.zeros((50, 1))
    report = detect_clipping(acc)
    assert not report.any_suspected


def test_clipping_report_str_summarizes_instead_of_dumping_every_interval() -> None:
    acc = np.zeros((200, 3))
    acc[50:55, 1] = 800.0
    acc[100:105, 1] = 800.0
    report = detect_clipping(acc)
    text = str(report)
    assert "2 suspected interval(s)" in text
    assert "component 1: 2" in text
    assert "ClippingInterval" not in text  # a summary, not a field dump


def test_clipping_report_str_reports_no_suspected_clipping_when_clean() -> None:
    acc = np.zeros((50, 2))
    report = detect_clipping(acc)
    assert "no suspected clipping" in str(report)


def test_detect_clipping_never_corrects_the_input() -> None:
    acc = np.full((20, 1), 2000.0)
    before = acc.copy()
    detect_clipping(acc, max_range_gal=2000.0)
    np.testing.assert_array_equal(acc, before)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"repeat_threshold": 1}, "repeat_threshold"),
        ({"extreme_fraction": 0.0}, "extreme_fraction"),
        ({"extreme_fraction": 1.5}, "extreme_fraction"),
        ({"range_tolerance": -0.1}, "range_tolerance"),
        ({"range_tolerance": 1.0}, "range_tolerance"),
        ({"max_range_gal": -5.0}, "max_range_gal"),
        ({"max_range_gal": 0.0}, "max_range_gal"),
    ],
)
def test_detect_clipping_validates_its_parameters(kwargs: dict, match: str) -> None:
    acc = np.zeros((10, 1))
    with pytest.raises(ValueError, match=match):
        detect_clipping(acc, **kwargs)
