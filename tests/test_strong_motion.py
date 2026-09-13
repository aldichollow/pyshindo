from __future__ import annotations

import numpy as np
import pytest

from pyshindo import (
    apply_strong_motion_displacement_filter,
    apply_strong_motion_velocity_filter,
    synthetic_three_component_motion,
)

RATE = 100.0


def test_velocity_filter_rejects_a_non_reference_sampling_rate() -> None:
    acceleration = synthetic_three_component_motion(sampling_rate_hz=200.0, duration_s=5.0)
    with pytest.raises(ValueError, match="100"):
        apply_strong_motion_velocity_filter(acceleration, 200.0)


def test_displacement_filter_rejects_a_non_reference_sampling_rate() -> None:
    acceleration = synthetic_three_component_motion(sampling_rate_hz=200.0, duration_s=5.0)
    with pytest.raises(ValueError, match="100"):
        apply_strong_motion_displacement_filter(acceleration, 200.0)


def test_velocity_filter_preserves_shape_and_scales_with_the_input_unit() -> None:
    acceleration_gal = synthetic_three_component_motion(sampling_rate_hz=RATE, duration_s=10.0)

    from_gal = apply_strong_motion_velocity_filter(acceleration_gal, RATE, unit="gal")
    from_mps2 = apply_strong_motion_velocity_filter(acceleration_gal / 100.0, RATE, unit="m/s^2")

    assert from_gal.shape == acceleration_gal.shape
    # A generous tolerance: the gal and m/s^2 paths take different floating-point
    # routes to the same recursive filter (a divide-then-multiply-back on every
    # sample versus none), and that rounding difference compounds a little
    # differently across platforms as the recursion runs -- observed up to
    # about 2e-8 relative on Linux/Python 3.13 in CI, comfortably under this.
    np.testing.assert_allclose(from_gal, from_mps2, rtol=1e-6)


def test_displacement_filter_preserves_shape_and_scales_with_the_input_unit() -> None:
    acceleration_gal = synthetic_three_component_motion(sampling_rate_hz=RATE, duration_s=10.0)

    from_gal = apply_strong_motion_displacement_filter(acceleration_gal, RATE, unit="gal")
    from_mps2 = apply_strong_motion_displacement_filter(
        acceleration_gal / 100.0, RATE, unit="m/s^2"
    )

    assert from_gal.shape == acceleration_gal.shape
    # A generous tolerance: the gal and m/s^2 paths take different floating-point
    # routes to the same recursive filter (a divide-then-multiply-back on every
    # sample versus none), and that rounding difference compounds a little
    # differently across platforms as the recursion runs -- observed up to
    # about 2e-8 relative on Linux/Python 3.13 in CI, comfortably under this.
    np.testing.assert_allclose(from_gal, from_mps2, rtol=1e-6)


def test_displacement_filter_settles_to_a_bounded_offset_response() -> None:
    # This is a mass-spring-damper response, not an integrator: a sustained
    # acceleration settles the "mass" at a new equilibrium position rather
    # than driving displacement to grow without bound the way a literal
    # double integration of the same offset would. The finite steady-state
    # gain is 1/(2*pi/period_s)**2 in continuous time; check it is bounded
    # and stays put, not that it decays to zero.
    offset_gal = 5.0
    short = np.full(3000, offset_gal)
    long_ = np.full(6000, offset_gal)

    settled_short = apply_strong_motion_displacement_filter(short, RATE)[-1, 0]
    settled_long = apply_strong_motion_displacement_filter(long_, RATE)[-1, 0]

    assert settled_short == pytest.approx(settled_long, rel=1e-6)
    natural_period_s = 6.0
    expected_steady_state = offset_gal / (2.0 * np.pi / natural_period_s) ** 2
    assert settled_long == pytest.approx(expected_steady_state, rel=0.05)


def test_velocity_filter_suppresses_a_constant_offset() -> None:
    offset = np.full((3000, 3), 5.0)
    filtered = apply_strong_motion_velocity_filter(offset, RATE)
    assert np.max(np.abs(filtered[-1])) < 1e-6


def test_displacement_filter_matches_a_direct_recurrence_reference() -> None:
    # Cross-check against the recursion written out exactly as JMA's page
    # states it, rather than through scipy.signal.lfilter, on a record with
    # no special structure.
    rng = np.random.default_rng(20260913)
    acceleration = rng.normal(0.0, 20.0, size=1000)

    h0, c1, c2 = 1.0, -1.988438073558305, 0.9885471048650272
    d0, d1, d2 = 0.00002485615736514583, 0.00004971231473029166, 0.00002485615736514583
    d = np.zeros(acceleration.shape[0])
    for t in range(acceleration.shape[0]):
        a_t = acceleration[t]
        a_t1 = acceleration[t - 1] if t >= 1 else 0.0
        a_t2 = acceleration[t - 2] if t >= 2 else 0.0
        d_t1 = d[t - 1] if t >= 1 else 0.0
        d_t2 = d[t - 2] if t >= 2 else 0.0
        d[t] = h0 * (d0 * a_t + d1 * a_t1 + d2 * a_t2) - (c1 * d_t1 + c2 * d_t2)

    filtered = apply_strong_motion_displacement_filter(acceleration, RATE)
    np.testing.assert_allclose(filtered[:, 0], d, rtol=1e-10)
