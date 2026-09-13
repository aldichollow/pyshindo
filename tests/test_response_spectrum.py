from __future__ import annotations

import numpy as np
import pytest

from pyshindo import synthetic_three_component_motion
from pyshindo._spectral_response import design_oscillator_bank, relative_displacement_response
from pyshindo.response_spectrum import calculate_response_spectrum

RATE = 100.0
DAMPING = 0.05
PERIODS = np.array([0.2, 0.5, 1.0, 1.7, 2.4])


def record(duration_s: float = 15.0) -> np.ndarray:
    return synthetic_three_component_motion(sampling_rate_hz=RATE, duration_s=duration_s)


# --------------------------------------------------------------------------
# relative_displacement_response against an independent step-by-step loop
# --------------------------------------------------------------------------


def _naive_displacement_peaks(bank, acceleration: np.ndarray) -> np.ndarray:
    """Step the published recurrence in a plain Python loop, tracking displacement.

    Independent of relative_displacement_response's own transfer-function
    path, the same way test_spectrum_intensity.py's naive loop checks velocity.
    """
    components = acceleration.shape[1]
    peaks = np.zeros((bank.period_count, components))
    for k in range(bank.period_count):
        displacement = np.zeros(components)
        velocity = -acceleration[0] / RATE
        previous = acceleration[0].copy()
        peak = np.abs(displacement).copy()
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
            np.maximum(peak, np.abs(displacement), out=peak)
        peaks[k] = peak
    return peaks


def test_relative_displacement_response_matches_a_naive_recurrence_loop() -> None:
    acc = record()[:, :2]
    bank = design_oscillator_bank(PERIODS, DAMPING, RATE)

    fast_peaks, _ = relative_displacement_response(bank, acc)
    slow_peaks = _naive_displacement_peaks(bank, acc)

    np.testing.assert_allclose(fast_peaks, slow_peaks, rtol=1e-10)


def test_relative_displacement_response_agrees_on_a_two_sample_record() -> None:
    # Exactly two samples exercises the seeded-but-no-lfilter-call branch.
    acc = record()[:2, :2]
    bank = design_oscillator_bank(PERIODS[:2], DAMPING, RATE)
    fast_peaks, _ = relative_displacement_response(bank, acc)
    slow_peaks = _naive_displacement_peaks(bank, acc)
    np.testing.assert_allclose(fast_peaks, slow_peaks, rtol=1e-10)


def test_relative_displacement_response_handles_empty_and_single_sample_input() -> None:
    bank = design_oscillator_bank(np.array([1.0, 2.0]), DAMPING, RATE)

    empty_peaks, empty_series = relative_displacement_response(
        bank, np.zeros((0, 2)), collect=True
    )
    assert empty_peaks.shape == (2, 2)
    assert np.all(empty_peaks == 0.0)
    assert empty_series.shape == (0, 2, 2)

    # DIS(1) = 0 by the published initialization, for every period.
    one_peaks, _ = relative_displacement_response(bank, np.array([[5.0, -3.0]]))
    assert np.all(one_peaks == 0.0)


# --------------------------------------------------------------------------
# calculate_response_spectrum
# --------------------------------------------------------------------------


def test_damping_ratio_and_periods_s_are_required() -> None:
    acc = record()[:, :1]
    with pytest.raises(TypeError):
        calculate_response_spectrum(acc, RATE, unit="gal", periods_s=PERIODS)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        calculate_response_spectrum(acc, RATE, unit="gal", damping_ratio=DAMPING)  # type: ignore[call-arg]


def test_sd_and_sv_match_the_shared_solver_directly() -> None:
    acc = record()[:, :2]
    bank = design_oscillator_bank(PERIODS, DAMPING, RATE)
    from pyshindo._spectral_response import relative_velocity_response

    expected_sd, _ = relative_displacement_response(bank, acc)
    expected_sv, _ = relative_velocity_response(bank, acc)

    result = calculate_response_spectrum(
        acc, RATE, unit="gal", damping_ratio=DAMPING, periods_s=PERIODS
    )

    np.testing.assert_allclose(result.sd_cm, expected_sd)
    np.testing.assert_allclose(result.sv_cm_s, expected_sv)


def test_response_is_reported_per_component_not_combined() -> None:
    acc = record()
    result = calculate_response_spectrum(
        acc, RATE, unit="gal", damping_ratio=DAMPING, periods_s=PERIODS
    )
    assert result.sd_cm.shape == (len(PERIODS), 3)
    assert result.sv_cm_s.shape == (len(PERIODS), 3)
    assert result.psv_cm_s.shape == (len(PERIODS), 3)
    assert result.psa_gal.shape == (len(PERIODS), 3)


def test_accepts_one_to_three_components() -> None:
    acc = record()
    for count in (1, 2, 3):
        result = calculate_response_spectrum(
            acc[:, :count], RATE, unit="gal", damping_ratio=DAMPING, periods_s=PERIODS
        )
        assert result.sd_cm.shape == (len(PERIODS), count)
        assert result.component_count == count


def test_unit_conversion_scales_response_consistently() -> None:
    acc_gal = record()[:, :1]
    result_gal = calculate_response_spectrum(
        acc_gal, RATE, unit="gal", damping_ratio=DAMPING, periods_s=PERIODS
    )
    result_ms2 = calculate_response_spectrum(
        acc_gal / 100.0, RATE, unit="m/s^2", damping_ratio=DAMPING, periods_s=PERIODS
    )
    np.testing.assert_allclose(result_gal.sd_cm, result_ms2.sd_cm, rtol=1e-10)
    np.testing.assert_allclose(result_gal.sv_cm_s, result_ms2.sv_cm_s, rtol=1e-10)


def test_retain_time_series_flags_are_independent() -> None:
    acc = record(duration_s=3.0)[:, :2]
    expected_shape = (acc.shape[0], len(PERIODS), 2)

    default = calculate_response_spectrum(
        acc, RATE, unit="gal", damping_ratio=DAMPING, periods_s=PERIODS
    )
    assert default.sd_time_series_cm is None
    assert default.sv_time_series_cm_s is None

    displacement_only = calculate_response_spectrum(
        acc,
        RATE,
        unit="gal",
        damping_ratio=DAMPING,
        periods_s=PERIODS,
        retain_displacement_time_series=True,
    )
    assert displacement_only.sd_time_series_cm is not None
    assert displacement_only.sd_time_series_cm.shape == expected_shape
    assert displacement_only.sv_time_series_cm_s is None

    velocity_only = calculate_response_spectrum(
        acc,
        RATE,
        unit="gal",
        damping_ratio=DAMPING,
        periods_s=PERIODS,
        retain_velocity_time_series=True,
    )
    assert velocity_only.sv_time_series_cm_s is not None
    assert velocity_only.sv_time_series_cm_s.shape == expected_shape
    assert velocity_only.sd_time_series_cm is None

    both = calculate_response_spectrum(
        acc,
        RATE,
        unit="gal",
        damping_ratio=DAMPING,
        periods_s=PERIODS,
        retain_displacement_time_series=True,
        retain_velocity_time_series=True,
    )
    np.testing.assert_allclose(
        both.sd_cm, np.abs(both.sd_time_series_cm).max(axis=0)
    )
    np.testing.assert_allclose(
        both.sv_cm_s, np.abs(both.sv_time_series_cm_s).max(axis=0)
    )


def test_psv_and_psa_are_derived_from_sd() -> None:
    acc = record()[:, :1]
    result = calculate_response_spectrum(
        acc, RATE, unit="gal", damping_ratio=DAMPING, periods_s=PERIODS
    )
    omega = 2.0 * np.pi / PERIODS
    np.testing.assert_allclose(result.psv_cm_s, omega[:, np.newaxis] * result.sd_cm)
    np.testing.assert_allclose(result.psa_gal, (omega**2)[:, np.newaxis] * result.sd_cm)


def test_invalid_damping_ratio_is_rejected() -> None:
    with pytest.raises(ValueError, match="damping_ratio"):
        calculate_response_spectrum(
            record()[:, :1], RATE, unit="gal", damping_ratio=1.5, periods_s=PERIODS
        )


def test_record_duration_s_is_sample_count_over_rate() -> None:
    result = calculate_response_spectrum(
        record(duration_s=3.0)[:, :1],
        RATE,
        unit="gal",
        damping_ratio=DAMPING,
        periods_s=PERIODS,
    )
    assert result.record_duration_s == pytest.approx(3.0)


def test_different_damping_ratios_give_different_results() -> None:
    acc = record()[:, :1]
    light = calculate_response_spectrum(
        acc, RATE, unit="gal", damping_ratio=0.02, periods_s=PERIODS
    )
    heavy = calculate_response_spectrum(
        acc, RATE, unit="gal", damping_ratio=0.20, periods_s=PERIODS
    )
    assert not np.allclose(light.sd_cm, heavy.sd_cm)
