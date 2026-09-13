from __future__ import annotations

import numpy as np
import pytest

from pyshindo import synthetic_three_component_motion
from pyshindo._spectral_response import design_oscillator_bank, relative_velocity_response
from pyshindo.exceptions import InvalidAccelerationError
from pyshindo.spectrum_intensity import (
    DEFAULT_DAMPING_RATIO,
    INTEGRATION_WIDTH_S,
    LOWER_PERIOD_S,
    UPPER_PERIOD_S,
    SpectrumIntensityEstimator,
    calculate_spectrum_intensity,
    default_periods_s,
)

RATE = 100.0


def record(duration_s: float = 15.0) -> np.ndarray:
    return synthetic_three_component_motion(sampling_rate_hz=RATE, duration_s=duration_s)


# --------------------------------------------------------------------------
# Oscillator response against an independent step-by-step evaluation
# --------------------------------------------------------------------------


def _naive_recurrence_peaks(bank, acceleration: np.ndarray) -> np.ndarray:
    """Step the published recurrence in a plain Python loop, period by period.

    An independent re-implementation of the same closed-form coefficients as
    relative_velocity_response, to check the fast lfilter path is solving the
    same equation, not a different one.
    """
    components = acceleration.shape[1]
    peaks = np.zeros((bank.period_count, components))
    for k in range(bank.period_count):
        displacement = np.zeros(components)
        velocity = -acceleration[0] / RATE
        previous = acceleration[0].copy()
        peak = np.abs(velocity).copy()
        for current in acceleration[1:]:
            next_displacement = (
                bank.a11[k] * displacement
                + bank.a12[k] * velocity
                + bank.b11[k] * previous
                + bank.b12[k] * current
            )
            next_velocity = (
                bank.a21[k] * displacement
                + bank.a22[k] * velocity
                + bank.b21[k] * previous
                + bank.b22[k] * current
            )
            displacement, velocity, previous = next_displacement, next_velocity, current
            np.maximum(peak, np.abs(velocity), out=peak)
        peaks[k] = peak
    return peaks


def test_relative_velocity_response_matches_a_naive_recurrence_loop() -> None:
    acc = record()[:, :2]
    periods = np.array([0.15, 0.5, 1.0, 1.7, 2.4])
    bank = design_oscillator_bank(periods, DEFAULT_DAMPING_RATIO, RATE)

    fast_peaks, _ = relative_velocity_response(bank, acc)
    slow_peaks = _naive_recurrence_peaks(bank, acc)

    np.testing.assert_allclose(fast_peaks, slow_peaks, rtol=1e-10)


def test_relative_velocity_response_handles_empty_and_single_sample_input() -> None:
    bank = design_oscillator_bank(np.array([1.0, 2.0]), DEFAULT_DAMPING_RATIO, RATE)

    empty_peaks, empty_series = relative_velocity_response(
        bank, np.zeros((0, 2)), collect=True
    )
    assert empty_peaks.shape == (2, 2)
    assert np.all(empty_peaks == 0.0)
    assert empty_series.shape == (0, 2, 2)

    one_peaks, _ = relative_velocity_response(bank, np.array([[5.0, -3.0]]))
    # A single sample can only reach the published VEL(1) = -A(1)*dt state.
    expected = np.abs(np.array([5.0, -3.0])) / RATE
    np.testing.assert_allclose(one_peaks, np.tile(expected, (2, 1)))


def test_relative_velocity_response_agrees_with_naive_loop_on_a_two_sample_record() -> None:
    # Exactly two samples exercises the seeded-but-no-lfilter-call branch.
    acc = record()[:2, :2]
    bank = design_oscillator_bank(np.array([0.5, 1.5]), DEFAULT_DAMPING_RATIO, RATE)
    fast_peaks, _ = relative_velocity_response(bank, acc)
    slow_peaks = _naive_recurrence_peaks(bank, acc)
    np.testing.assert_allclose(fast_peaks, slow_peaks, rtol=1e-10)


def test_relative_velocity_response_does_not_add_ground_velocity() -> None:
    # A pure step offset has a large ground velocity (dt * offset accumulated
    # every sample) but a bounded relative velocity response once the
    # oscillator's own transient settles -- if ground velocity leaked in here,
    # the peak would keep growing with record length instead of converging.
    offset = np.full((3000, 1), 50.0)
    bank = design_oscillator_bank(np.array([1.0]), DEFAULT_DAMPING_RATIO, RATE)
    peaks, _ = relative_velocity_response(bank, offset)
    assert np.isfinite(peaks).all()
    assert peaks[0, 0] < 1000.0  # relative response stays bounded, not accumulating


# --------------------------------------------------------------------------
# calculate_spectrum_intensity
# --------------------------------------------------------------------------


def test_default_period_grid_spans_the_published_range() -> None:
    periods = default_periods_s()
    assert periods[0] == LOWER_PERIOD_S
    assert periods[-1] == UPPER_PERIOD_S
    assert periods.shape == (121,)


def test_si_value_matches_the_integral_definition_directly() -> None:
    # SI = (1/2.4) * integral[0.1, 2.5] Sv(T, h) dT, computed here by hand from
    # the same oscillator bank, independent of calculate_spectrum_intensity's
    # own internals.
    acc = record()[:, :1]
    periods = default_periods_s()
    bank = design_oscillator_bank(periods, DEFAULT_DAMPING_RATIO, RATE)
    peaks, _ = relative_velocity_response(bank, acc)
    expected = np.trapezoid(peaks[:, 0], periods) / INTEGRATION_WIDTH_S

    result = calculate_spectrum_intensity(acc, RATE, unit="gal")

    assert result.si_cm_s[0] == pytest.approx(expected)


def test_si_value_is_reported_per_component_not_combined() -> None:
    acc = record()
    result = calculate_spectrum_intensity(acc, RATE, unit="gal")
    assert result.si_cm_s.shape == (3,)
    # Combining would collapse this to a scalar; per-component values differ.
    assert len(set(np.round(result.si_cm_s, 6))) == 3


def test_accepts_one_to_three_components() -> None:
    acc = record()
    for count in (1, 2, 3):
        result = calculate_spectrum_intensity(acc[:, :count], RATE, unit="gal")
        assert result.si_cm_s.shape == (count,)
        assert result.component_count == count


def test_unit_conversion_scales_si_consistently() -> None:
    acc_gal = record()[:, :1]
    result_gal = calculate_spectrum_intensity(acc_gal, RATE, unit="gal")
    result_ms2 = calculate_spectrum_intensity(acc_gal / 100.0, RATE, unit="m/s^2")
    np.testing.assert_allclose(result_gal.si_cm_s, result_ms2.si_cm_s, rtol=1e-10)


def test_retain_spectrum_returns_the_full_response_only_when_asked() -> None:
    acc = record(duration_s=3.0)[:, :2]
    default = calculate_spectrum_intensity(acc, RATE, unit="gal")
    retained = calculate_spectrum_intensity(acc, RATE, unit="gal", retain_spectrum=True)
    assert default.sv_time_series_cm_s is None
    assert retained.sv_time_series_cm_s is not None
    assert retained.sv_time_series_cm_s.shape == (acc.shape[0], len(default.periods_s), 2)
    # The peak-per-period spectrum is cheap and always present, regardless of
    # retain_spectrum -- it is what a response-spectrum plot actually needs.
    assert default.sv_cm_s.shape == (len(default.periods_s), 2)
    np.testing.assert_allclose(
        retained.sv_cm_s, np.abs(retained.sv_time_series_cm_s).max(axis=0)
    )


def test_custom_damping_ratio_changes_the_result() -> None:
    acc = record()[:, :1]
    lightly_damped = calculate_spectrum_intensity(acc, RATE, unit="gal", damping_ratio=0.05)
    default = calculate_spectrum_intensity(acc, RATE, unit="gal")
    assert lightly_damped.si_cm_s[0] != pytest.approx(default.si_cm_s[0])
    assert lightly_damped.damping_ratio == 0.05


def test_custom_period_grid_is_honored() -> None:
    acc = record()[:, :1]
    periods = np.linspace(0.2, 2.0, 21)
    result = calculate_spectrum_intensity(acc, RATE, unit="gal", periods_s=periods)
    np.testing.assert_array_equal(result.periods_s, periods)


def test_invalid_damping_ratio_is_rejected() -> None:
    with pytest.raises(ValueError, match="damping_ratio"):
        calculate_spectrum_intensity(record()[:, :1], RATE, unit="gal", damping_ratio=1.5)


def test_four_components_are_rejected() -> None:
    acc = np.zeros((1000, 4))
    with pytest.raises(InvalidAccelerationError):
        calculate_spectrum_intensity(acc, RATE, unit="gal")


def test_default_periods_s_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least two"):
        default_periods_s(1)


def test_record_duration_s_is_sample_count_over_rate() -> None:
    result = calculate_spectrum_intensity(record(duration_s=3.0)[:, :1], RATE, unit="gal")
    assert result.record_duration_s == pytest.approx(3.0)


# --------------------------------------------------------------------------
# Convergence of the default period grid (documents the choice in
# _reports/si_value_implementation_plan.md as an executable check)
# --------------------------------------------------------------------------


def test_default_grid_is_converged_relative_to_a_much_finer_grid() -> None:
    acc = record(duration_s=10.0)[:, :1]
    coarse = calculate_spectrum_intensity(acc, RATE, unit="gal", periods_s=default_periods_s(121))
    fine = calculate_spectrum_intensity(acc, RATE, unit="gal", periods_s=default_periods_s(769))
    relative_error = abs(coarse.si_cm_s[0] - fine.si_cm_s[0]) / fine.si_cm_s[0]
    assert relative_error < 1e-4


# --------------------------------------------------------------------------
# SpectrumIntensityEstimator
# --------------------------------------------------------------------------

STREAM_PERIODS = np.array([0.2, 0.5, 1.0, 1.7, 2.4])


def test_chunked_processing_is_identical_to_one_shot_processing() -> None:
    acc = record()[:, :2]
    one_shot = SpectrumIntensityEstimator(RATE, periods_s=STREAM_PERIODS)
    one_shot.process(acc)

    chunked = SpectrumIntensityEstimator(RATE, periods_s=STREAM_PERIODS)
    rng = np.random.default_rng(20260910)
    position = 0
    while position < acc.shape[0]:
        size = int(rng.integers(1, 500))
        chunked.process(acc[position : position + size])
        position += size

    # Bit-identical: chunking must not change the arithmetic at all.
    np.testing.assert_array_equal(chunked.sv_cm_s, one_shot.sv_cm_s)
    np.testing.assert_array_equal(chunked.si_cm_s, one_shot.si_cm_s)


def test_sample_by_sample_streaming_is_identical_to_one_shot_processing() -> None:
    acc = record(duration_s=6.0)[:, :2]
    one_shot = SpectrumIntensityEstimator(RATE, periods_s=STREAM_PERIODS)
    one_shot.process(acc)

    sampled = SpectrumIntensityEstimator(RATE, periods_s=STREAM_PERIODS)
    update = None
    for sample in acc:
        update = sampled.process_sample(sample)

    assert update is not None
    np.testing.assert_array_equal(sampled.sv_cm_s, one_shot.sv_cm_s)
    assert update.sample_count == acc.shape[0]
    np.testing.assert_array_equal(update.si_so_far_cm_s, one_shot.si_cm_s)


def test_streaming_agrees_with_the_batch_calculation() -> None:
    # The estimator steps the raw recurrence sample by sample (needed for a
    # true streaming API); calculate_spectrum_intensity runs the fast lfilter
    # path. Same equation, different arithmetic order -- like the "filter" vs
    # "recurrence" solvers in pyshindo.long_period -- so agreement is at
    # floating-point rounding level rather than bit-identical.
    acc = record()[:, :2]
    batch = calculate_spectrum_intensity(acc, RATE, unit="gal", periods_s=STREAM_PERIODS)
    estimator = SpectrumIntensityEstimator(RATE, periods_s=STREAM_PERIODS)
    estimator.process(acc)
    np.testing.assert_allclose(estimator.sv_cm_s, batch.sv_cm_s, rtol=1e-9)
    np.testing.assert_allclose(estimator.si_cm_s, batch.si_cm_s, rtol=1e-9)


def test_streaming_maximum_never_decreases() -> None:
    acc = record()[:, :1]
    estimator = SpectrumIntensityEstimator(RATE, periods_s=STREAM_PERIODS)
    previous = 0.0
    for start in range(0, acc.shape[0], 500):
        update = estimator.process(acc[start : start + 500])
        current = float(update.si_so_far_cm_s[0])
        assert current >= previous
        previous = current


def test_estimator_result_matches_the_batch_result() -> None:
    acc = record()[:, :2]
    batch = calculate_spectrum_intensity(acc, RATE, unit="gal", periods_s=STREAM_PERIODS)
    estimator = SpectrumIntensityEstimator(RATE, periods_s=STREAM_PERIODS)
    estimator.process(acc)
    streamed = estimator.result()
    np.testing.assert_allclose(streamed.sv_cm_s, batch.sv_cm_s, rtol=1e-9)
    np.testing.assert_allclose(streamed.si_cm_s, batch.si_cm_s, rtol=1e-9)
    assert streamed.sample_count == batch.sample_count
    assert streamed.component_count == batch.component_count
    assert streamed.sv_time_series_cm_s is None


def test_estimator_accepts_one_to_three_components() -> None:
    acc = record()
    for count in (1, 2, 3):
        estimator = SpectrumIntensityEstimator(RATE, periods_s=STREAM_PERIODS)
        estimator.process(acc[:, :count])
        assert estimator.component_count == count
        assert estimator.si_cm_s.shape == (count,)


def test_estimator_rejects_a_component_count_change_mid_stream() -> None:
    acc = record()
    estimator = SpectrumIntensityEstimator(RATE, periods_s=STREAM_PERIODS)
    estimator.process(acc[:, :2])
    with pytest.raises(ValueError, match="components"):
        estimator.process(acc[:, :3])


def test_process_sample_rejects_the_wrong_number_of_components() -> None:
    estimator = SpectrumIntensityEstimator(RATE, periods_s=STREAM_PERIODS)
    with pytest.raises(InvalidAccelerationError):
        estimator.process_sample(np.zeros(4))


def test_estimator_before_any_input_reports_empty_state() -> None:
    estimator = SpectrumIntensityEstimator(RATE, periods_s=STREAM_PERIODS)
    assert estimator.sample_count == 0
    assert estimator.component_count is None
    assert estimator.si_cm_s.shape == (0,)
    assert estimator.sv_cm_s.shape == (len(STREAM_PERIODS), 0)


def test_estimator_periods_s_is_read_only() -> None:
    estimator = SpectrumIntensityEstimator(RATE, periods_s=STREAM_PERIODS)
    with pytest.raises(ValueError, match="read-only"):
        estimator.periods_s[0] = 99.0


def test_estimator_default_damping_ratio_matches_the_batch_function() -> None:
    estimator = SpectrumIntensityEstimator(RATE, periods_s=STREAM_PERIODS)
    assert estimator.damping_ratio == DEFAULT_DAMPING_RATIO
