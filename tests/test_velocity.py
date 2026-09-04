from __future__ import annotations

import numpy as np
import pytest

from pyshindo import (
    component_peak_velocity,
    integrate_to_velocity,
    peak_ground_velocity,
    remove_offset,
)


def test_integrated_sine_matches_the_analytic_velocity() -> None:
    # For a(t) = A sin(wt), the zero-initial-velocity integral is
    # v(t) = A/w * (1 - cos(wt)), whose peak is 2A/w.
    rate = 200.0
    frequency_hz = 1.0
    amplitude_gal = 50.0
    time = np.arange(0, 4.0, 1.0 / rate)
    angular = 2.0 * np.pi * frequency_hz
    acceleration = amplitude_gal * np.sin(angular * time)

    velocity = integrate_to_velocity(acceleration, rate)
    expected = amplitude_gal / angular * (1.0 - np.cos(angular * time))

    assert velocity.shape == (time.size, 1)
    # The trapezoidal rule under-integrates a sinusoid by a relative
    # (w * dt)^2 / 12, so the agreement is checked against that analytic
    # bound rather than an arbitrary tolerance: a looser result would mean
    # the integration is wrong, a much tighter one would mean this test is
    # not actually exercising trapezoidal integration.
    discretization_bound = np.max(np.abs(expected)) * (angular / rate) ** 2 / 12.0
    error = np.max(np.abs(velocity[:, 0] - expected))
    assert error == pytest.approx(discretization_bound, rel=0.05)


def test_velocity_starts_at_zero_and_scales_with_the_input_unit() -> None:
    rate = 100.0
    acceleration_gal = np.full((int(rate) * 2, 3), 1.0)

    from_gal = integrate_to_velocity(acceleration_gal, rate, unit="gal")
    from_mps2 = integrate_to_velocity(acceleration_gal / 100.0, rate, unit="m/s^2")

    assert np.all(from_gal[0] == 0.0)
    # 1 m/s^2 is 100 gal, so the same numbers in m/s^2 must integrate to the
    # same cm/s result once converted.
    np.testing.assert_allclose(from_gal, from_mps2, rtol=1e-12)


def test_constant_acceleration_drifts_linearly() -> None:
    # Documented behavior, not a defect: integration cannot distinguish a
    # baseline offset from real long-period motion.
    rate = 100.0
    duration_s = 10.0
    offset_gal = 2.0
    acceleration = np.full((int(rate * duration_s), 1), offset_gal)

    velocity = integrate_to_velocity(acceleration, rate)

    final_time_s = (velocity.shape[0] - 1) / rate
    assert velocity[-1, 0] == pytest.approx(offset_gal * final_time_s)
    # Removing the offset first is what suppresses the drift.
    corrected = integrate_to_velocity(remove_offset(acceleration), rate)
    assert np.max(np.abs(corrected)) == pytest.approx(0.0, abs=1e-12)


def test_component_and_resultant_peaks_agree_with_direct_computation() -> None:
    rng = np.random.default_rng(20260904)
    rate = 100.0
    acceleration = rng.normal(0.0, 10.0, size=(500, 3))

    velocity = integrate_to_velocity(acceleration, rate)
    component_peaks = component_peak_velocity(acceleration, rate)
    resultant_peak = peak_ground_velocity(acceleration, rate)

    np.testing.assert_allclose(component_peaks, np.max(np.abs(velocity), axis=0))
    expected_resultant = np.max(np.linalg.norm(velocity, axis=1))
    assert resultant_peak == pytest.approx(expected_resultant)
    # The resultant peak can never be below the largest single-component peak.
    assert resultant_peak >= np.max(component_peaks) - 1e-12


def test_peak_ground_velocity_is_invariant_to_horizontal_rotation() -> None:
    # The resultant is a Euclidean norm, so rotating the horizontal pair must
    # not change PGV. This is what lets ObsPy "1"/"2" channels be used without
    # knowing their true azimuth.
    rng = np.random.default_rng(7)
    rate = 100.0
    acceleration = rng.normal(0.0, 10.0, size=(400, 3))
    angle = 0.723
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    original = peak_ground_velocity(acceleration, rate)
    rotated = peak_ground_velocity(acceleration @ rotation.T, rate)

    assert rotated == pytest.approx(original)
