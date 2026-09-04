from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg, signal

from pyshindo import remove_offset, synthetic_three_component_motion
from pyshindo.exceptions import InvalidAccelerationError, NonstandardSamplingRateWarning
from pyshindo.long_period import (
    OFFICIAL_DAMPING_RATIO,
    OFFICIAL_PERIODS_S,
    LongPeriodClass,
    LongPeriodEstimator,
    _core,
    band_period_range_s,
    calculate_long_period_class,
    classify_long_period,
    long_period_class_label,
)

RATE = 100.0


def horizontal_record(duration_s: float = 40.0, scale: float = 10.0) -> np.ndarray:
    values = synthetic_three_component_motion(sampling_rate_hz=RATE, duration_s=duration_s)
    return np.ascontiguousarray(values[:, :2]) * scale


# --------------------------------------------------------------------------
# Published constants and period grid
# --------------------------------------------------------------------------


def test_official_period_grid_is_exactly_the_published_32_values() -> None:
    assert OFFICIAL_PERIODS_S.shape == (32,)
    assert OFFICIAL_PERIODS_S[0] == 1.6
    assert OFFICIAL_PERIODS_S[-1] == 7.8
    # Built from integer tenths, so the spacing is exact rather than drifting.
    np.testing.assert_array_equal(
        OFFICIAL_PERIODS_S, np.array(range(16, 80, 2), dtype=np.float64) / 10.0
    )


def test_published_high_pass_constants_are_used_verbatim_at_100_hz() -> None:
    # Regression on the exact decimal literals from JMA 7th committee, 資料2 p.13.
    design = _core.design_high_pass(100.0)
    assert design.is_published_reference
    np.testing.assert_array_equal(design.numerator, [1.0, -2.0, 1.0])
    np.testing.assert_array_equal(design.denominator, [1.0, -1.995438545842, 0.995448925627])
    assert design.gain == 0.997721867867


def test_high_pass_blocks_dc_and_passes_the_class_period_band() -> None:
    design = _core.design_high_pass(100.0)
    numerator = design.numerator * design.gain
    # A double zero at z = 1 means exactly zero response to a constant offset.
    assert abs(np.polyval(numerator, 1.0)) == 0.0
    assert np.max(np.abs(np.roots(design.denominator))) < 1.0

    frequency, response = signal.freqz(numerator, design.denominator, worN=200_000, fs=100.0)
    magnitude = np.abs(response)
    # Near-unity across the 1.6-7.8 s band, with the mild roll-off at the long
    # end that is part of the published definition.
    assert np.interp(1 / 1.6, frequency, magnitude) == pytest.approx(1.0, abs=1e-4)
    assert np.interp(1 / 7.8, frequency, magnitude) == pytest.approx(0.9874, abs=1e-3)
    # Half power close to the nominal 20-second corner.
    corner = 1.0 / frequency[np.argmin(np.abs(magnitude - 1 / np.sqrt(2)))]
    assert corner == pytest.approx(19.5, abs=0.5)


def test_non_reference_rate_reproduces_the_published_design_at_100_hz() -> None:
    # The rate-generalized reconstruction is only trustworthy if it collapses
    # onto the published constants at the rate they were published for.
    published = _core.design_high_pass(100.0)
    other = _core.design_high_pass(200.0)
    assert not other.is_published_reference

    zeta = _core._HPF_PROTOTYPE_DAMPING
    pole = _core._HPF_PROTOTYPE_OMEGA_N * (-zeta + 1j * np.sqrt(1 - zeta**2))
    reconstructed = np.real(np.poly([np.exp(pole / 100.0), np.conj(np.exp(pole / 100.0))]))
    assert np.max(np.abs(reconstructed - published.denominator)) < 1e-8
    assert zeta == pytest.approx(1 / np.sqrt(2), abs=1e-6)


# --------------------------------------------------------------------------
# Oscillator recurrence against an independent discretization
# --------------------------------------------------------------------------


def foh_oracle(period_s: float, damping: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact first-order-hold discretization by matrix exponential."""
    w = 2 * np.pi / period_s
    block = np.zeros((4, 4))
    block[:2, :2] = [[0.0, 1.0], [-w * w, -2 * damping * w]]
    block[:2, 2] = [0.0, -1.0]
    block[2, 3] = 1.0
    expanded = linalg.expm(block * dt)
    constant, slope = expanded[:2, 2], expanded[:2, 3]
    return expanded[:2, :2], np.column_stack([constant - slope / dt, slope / dt])


def test_published_oscillator_coefficients_match_a_matrix_exponential_oracle() -> None:
    bank = _core.design_oscillator_bank(OFFICIAL_PERIODS_S, OFFICIAL_DAMPING_RATIO, RATE)
    for index, period in enumerate(OFFICIAL_PERIODS_S):
        expected_a, expected_b = foh_oracle(float(period), OFFICIAL_DAMPING_RATIO, 1 / RATE)
        actual_a = np.array(
            [[bank.a11[index], bank.a12[index]], [bank.a21[index], bank.a22[index]]]
        )
        actual_b = np.array(
            [[bank.b11[index], bank.b12[index]], [bank.b21[index], bank.b22[index]]]
        )
        np.testing.assert_allclose(actual_a, expected_a, atol=1e-14)
        np.testing.assert_allclose(actual_b, expected_b, atol=1e-12)


def test_initial_conditions_follow_the_published_recurrence() -> None:
    # JMA initializes DIS(1)=0 and VEL(1) = -A(1)*dt; the velocity is not zero.
    bank = _core.design_oscillator_bank(np.array([3.0]), 0.05, RATE)
    state = _core.ResponseState(bank)
    first = np.array([[7.0, -3.0]])
    state.advance(first)
    np.testing.assert_allclose(state.displacement, 0.0)
    np.testing.assert_allclose(state.velocity, -first / RATE)


# --------------------------------------------------------------------------
# Invariances the definition requires
# --------------------------------------------------------------------------


def test_result_is_invariant_to_horizontal_rotation() -> None:
    # Vector combination of the horizontals is a Euclidean norm, so the class
    # cannot depend on how the sensor is oriented in the horizontal plane.
    values = horizontal_record()
    angle = 0.9123
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    original = calculate_long_period_class(values, RATE)
    rotated = calculate_long_period_class(values @ rotation.T, RATE)
    np.testing.assert_allclose(rotated.sva_cm_s, original.sva_cm_s, rtol=1e-9)


def test_result_is_invariant_to_component_swap_and_sign() -> None:
    values = horizontal_record()
    original = calculate_long_period_class(values, RATE)
    swapped = calculate_long_period_class(values[:, ::-1], RATE)
    negated = calculate_long_period_class(-values, RATE)
    np.testing.assert_allclose(swapped.sva_cm_s, original.sva_cm_s, rtol=1e-9)
    np.testing.assert_allclose(negated.sva_cm_s, original.sva_cm_s, rtol=1e-9)


def test_unit_conversion_leaves_the_result_unchanged() -> None:
    values = horizontal_record()
    in_gal = calculate_long_period_class(values, RATE, unit="gal")
    in_mps2 = calculate_long_period_class(values / 100.0, RATE, unit="m/s^2")
    np.testing.assert_allclose(in_mps2.sva_cm_s, in_gal.sva_cm_s, rtol=1e-9)


def test_sva_scales_linearly_with_input_amplitude() -> None:
    values = horizontal_record()
    single = calculate_long_period_class(values, RATE)
    doubled = calculate_long_period_class(values * 2.0, RATE)
    np.testing.assert_allclose(doubled.sva_cm_s, 2.0 * single.sva_cm_s, rtol=1e-9)


def test_a_constant_offset_appears_as_a_start_up_transient() -> None:
    """A standing offset is not simply removed by the high-pass.

    The filter has an exact zero at DC, so it rejects a constant in the
    steady state -- but a record that *begins* offset presents a step at the
    first sample, and the filter's transient response to that step sits
    squarely inside the 1.6-7.8 s band being measured. The effect is large:
    a few gal of offset can multiply the reported Sva several times over.

    This is the filter behaving correctly, and it matches what JMA computes
    from its own records, which begin from a quiet pre-event interval. It is
    a caveat for anyone feeding in a record that does not.
    """
    # A record containing nothing but a constant still produces a response,
    # entirely from that transient.
    constant = np.full((6000, 2), 5.0)
    spurious = calculate_long_period_class(constant, RATE)
    assert spurious.max_sva_cm_s > 1.0

    # Removing the offset first makes it vanish, which is the practical
    # guidance for any record that does not begin from rest.
    corrected = calculate_long_period_class(remove_offset(constant), RATE)
    assert corrected.max_sva_cm_s == pytest.approx(0.0, abs=1e-12)

    # On a real record the same correction is exactly reversible: shifting a
    # record and then removing the offset returns the original answer.
    values = horizontal_record()
    np.testing.assert_allclose(
        calculate_long_period_class(remove_offset(values + 5.0), RATE).sva_cm_s,
        calculate_long_period_class(remove_offset(values), RATE).sva_cm_s,
        rtol=1e-9,
    )


# --------------------------------------------------------------------------
# End-to-end against an ODE solver
# --------------------------------------------------------------------------


def test_pipeline_matches_an_independent_ode_solution() -> None:
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d

    values = horizontal_record(duration_s=8.0)
    periods = np.array([2.0, 5.0])
    result = calculate_long_period_class(values, RATE, periods_s=periods)

    design = _core.design_high_pass(RATE)
    filtered, _ = _core.apply_high_pass(design, values, _core.high_pass_initial_state(design))
    time = np.arange(filtered.shape[0]) / RATE
    dt = 1 / RATE
    damping = OFFICIAL_DAMPING_RATIO

    for index, period in enumerate(periods):
        w = 2 * np.pi / period
        relative = []
        for component in range(2):
            forcing = interp1d(time, filtered[:, component], kind="linear", assume_sorted=True)
            solution = solve_ivp(
                lambda t, y, w=w, f=forcing: [
                    y[1],
                    -2 * damping * w * y[1] - w * w * y[0] - float(f(t)),
                ],
                (time[0], time[-1]),
                [0.0, -filtered[0, component] * dt],
                t_eval=time,
                rtol=1e-11,
                atol=1e-12,
                method="DOP853",
            )
            relative.append(solution.y[1])
        ground = [
            np.concatenate([[0.0], np.cumsum(0.5 * dt * (filtered[1:, c] + filtered[:-1, c]))])
            for c in range(2)
        ]
        expected = np.hypot(relative[0] + ground[0], relative[1] + ground[1]).max()
        assert result.sva_cm_s[index] == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------
# Streaming equivalence
# --------------------------------------------------------------------------


def test_chunked_streaming_is_identical_to_batch() -> None:
    values = horizontal_record()
    batch = calculate_long_period_class(values, RATE, solver="recurrence")
    estimator = LongPeriodEstimator(RATE)
    rng = np.random.default_rng(20260904)
    position = 0
    while position < values.shape[0]:
        size = int(rng.integers(1, 500))
        estimator.process(values[position : position + size])
        position += size
    # Bit-identical: chunking must not change the arithmetic at all.
    np.testing.assert_array_equal(estimator.sva_cm_s, batch.sva_cm_s)


def test_sample_by_sample_streaming_is_identical_to_batch() -> None:
    values = horizontal_record(duration_s=6.0)
    batch = calculate_long_period_class(values, RATE, solver="recurrence")
    estimator = LongPeriodEstimator(RATE)
    for sample in values:
        update = estimator.process_sample(sample)
    np.testing.assert_array_equal(estimator.sva_cm_s, batch.sva_cm_s)
    assert update.sample_count == values.shape[0]
    assert update.class_so_far is batch.long_period_class


def test_both_solvers_agree_to_floating_point_rounding() -> None:
    values = horizontal_record()
    fast = calculate_long_period_class(values, RATE, solver="filter")
    reference = calculate_long_period_class(values, RATE, solver="recurrence")
    # Same equation, different arithmetic order, so agreement is at rounding
    # level rather than exact -- but the class must never be able to differ.
    np.testing.assert_allclose(fast.sva_cm_s, reference.sva_cm_s, rtol=1e-10)
    assert fast.long_period_class is reference.long_period_class
    assert [b.long_period_class for b in fast.bands] == [
        b.long_period_class for b in reference.bands
    ]


def test_retained_response_agrees_between_solvers() -> None:
    values = horizontal_record(duration_s=6.0)
    fast = calculate_long_period_class(values, RATE, solver="filter", retain_response=True)
    reference = calculate_long_period_class(
        values, RATE, solver="recurrence", retain_response=True
    )
    assert fast.absolute_velocity_cm_s is not None
    assert reference.absolute_velocity_cm_s is not None
    np.testing.assert_allclose(
        fast.absolute_velocity_cm_s, reference.absolute_velocity_cm_s, rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("samples", [1, 2, 3])
def test_both_solvers_agree_on_records_shorter_than_the_filter_state(samples: int) -> None:
    values = horizontal_record(duration_s=1.0)[:samples]
    fast = calculate_long_period_class(values, RATE, solver="filter")
    reference = calculate_long_period_class(values, RATE, solver="recurrence")
    np.testing.assert_allclose(fast.sva_cm_s, reference.sva_cm_s, rtol=1e-10, atol=1e-15)


def test_unknown_solver_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown solver"):
        calculate_long_period_class(horizontal_record(duration_s=2.0), RATE, solver="newmark")


def test_streaming_maximum_never_decreases() -> None:
    values = horizontal_record()
    estimator = LongPeriodEstimator(RATE)
    previous = 0.0
    for start in range(0, values.shape[0], 500):
        update = estimator.process(values[start : start + 500])
        assert update.max_sva_so_far_cm_s >= previous
        previous = update.max_sva_so_far_cm_s


def test_estimator_result_matches_the_batch_result() -> None:
    values = horizontal_record()
    batch = calculate_long_period_class(values, RATE, solver="recurrence")
    estimator = LongPeriodEstimator(RATE)
    estimator.process(values)
    streamed = estimator.result()
    np.testing.assert_array_equal(streamed.sva_cm_s, batch.sva_cm_s)
    assert streamed.long_period_class is batch.long_period_class
    assert streamed.critical_period_s == batch.critical_period_s
    assert [b.long_period_class for b in streamed.bands] == [
        b.long_period_class for b in batch.bands
    ]


# --------------------------------------------------------------------------
# Classification and period bands
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, LongPeriodClass.ZERO),
        (4.999999, LongPeriodClass.ZERO),
        (5.0, LongPeriodClass.ONE),
        (14.999999, LongPeriodClass.ONE),
        (15.0, LongPeriodClass.TWO),
        (49.999999, LongPeriodClass.TWO),
        (50.0, LongPeriodClass.THREE),
        (99.999999, LongPeriodClass.THREE),
        (100.0, LongPeriodClass.FOUR),
        (1000.0, LongPeriodClass.FOUR),
    ],
)
def test_class_boundaries_are_lower_bound_inclusive(
    value: float, expected: LongPeriodClass
) -> None:
    assert classify_long_period(value) is expected


def test_jma_worked_example_boundary() -> None:
    # JMA 8th committee 資料3 compares the two combination methods at one
    # station: 14.67 cm/s is class 1 and 15.12 cm/s is class 2.
    assert classify_long_period(14.67) is LongPeriodClass.ONE
    assert classify_long_period(15.12) is LongPeriodClass.TWO


def test_period_bands_group_by_integer_second() -> None:
    result = calculate_long_period_class(horizontal_record(), RATE)
    assert [band.band_second for band in result.bands] == [1, 2, 3, 4, 5, 6, 7]
    assert result.band(1).period_range_s == (1.6, 1.8)
    assert result.band(2).period_range_s == (2.0, 2.8)
    assert result.band(7).period_range_s == (7.0, 7.8)
    # The overall maximum is the largest band maximum, by construction.
    assert result.max_sva_cm_s == pytest.approx(max(b.max_sva_cm_s for b in result.bands))
    assert result.band(3).japanese_label == "3秒台"


def test_bands_are_omitted_for_a_non_official_period_grid() -> None:
    result = calculate_long_period_class(
        horizontal_record(duration_s=10.0), RATE, periods_s=np.array([2.0, 3.0])
    )
    assert result.bands == ()
    assert not result.reference_conditions_met


# --------------------------------------------------------------------------
# Reference conditions and input validation
# --------------------------------------------------------------------------


def test_reference_conditions_require_the_published_configuration() -> None:
    values = horizontal_record(duration_s=10.0)
    assert calculate_long_period_class(values, RATE).reference_conditions_met
    no_filter = calculate_long_period_class(values, RATE, high_pass=False)
    assert not no_filter.reference_conditions_met
    assert not calculate_long_period_class(
        values, RATE, damping_ratio=0.02
    ).reference_conditions_met
    with pytest.warns(NonstandardSamplingRateWarning):
        other = calculate_long_period_class(values[::2], 50.0)
    assert not other.reference_conditions_met


def test_three_components_are_rejected_with_a_useful_message() -> None:
    values = synthetic_three_component_motion(duration_s=3.0)
    with pytest.raises(InvalidAccelerationError, match="horizontal components only"):
        calculate_long_period_class(values, RATE)


def test_invalid_shapes_and_values_are_rejected() -> None:
    with pytest.raises(InvalidAccelerationError, match="two horizontal"):
        calculate_long_period_class(np.zeros((100, 1)), RATE)
    with pytest.raises(InvalidAccelerationError, match="dimension"):
        calculate_long_period_class(np.zeros(100), RATE)
    with pytest.raises(InvalidAccelerationError, match="at least one sample"):
        calculate_long_period_class(np.zeros((0, 2)), RATE)
    broken = np.zeros((10, 2))
    broken[3, 1] = np.nan
    with pytest.raises(InvalidAccelerationError, match="non-finite"):
        calculate_long_period_class(broken, RATE)


def test_quiet_record_is_class_zero_and_finite() -> None:
    result = calculate_long_period_class(np.zeros((3000, 2)), RATE)
    assert result.long_period_class is LongPeriodClass.ZERO
    assert result.max_sva_cm_s == 0.0
    assert np.all(np.isfinite(result.sva_cm_s))


def test_retain_response_returns_the_full_history_only_when_asked() -> None:
    values = horizontal_record(duration_s=5.0)
    assert calculate_long_period_class(values, RATE).absolute_velocity_cm_s is None
    retained = calculate_long_period_class(values, RATE, retain_response=True)
    assert retained.absolute_velocity_cm_s is not None
    assert retained.absolute_velocity_cm_s.shape == (values.shape[0], 32)
    # The reported spectrum is exactly the column-wise maximum of that history.
    np.testing.assert_allclose(
        retained.absolute_velocity_cm_s.max(axis=0), retained.sva_cm_s, rtol=1e-12
    )


def test_class_labels_and_band_ranges() -> None:
    assert long_period_class_label(20.0) == "長周期地震動階級2"
    assert long_period_class_label(20.0, language="en").endswith("class 2")
    with pytest.raises(ValueError, match="language"):
        long_period_class_label(20.0, language="fr")

    assert LongPeriodClass.THREE.japanese == "3"
    assert LongPeriodClass.THREE.english == "3"

    assert band_period_range_s(1) == (1.6, 1.8)
    with pytest.raises(ValueError, match="band must be one of"):
        band_period_range_s(9)


def test_classification_rejects_undefined_input() -> None:
    with pytest.raises(ValueError, match="NaN"):
        classify_long_period(float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        classify_long_period(-1.0)


def test_core_designs_reject_impossible_parameters() -> None:
    with pytest.raises(ValueError, match="sampling_rate_hz"):
        _core.design_high_pass(0.0)
    with pytest.raises(ValueError, match="periods_s"):
        _core.design_oscillator_bank(np.array([]), 0.05, RATE)
    with pytest.raises(ValueError, match="periods_s"):
        _core.design_oscillator_bank(np.array([-1.0]), 0.05, RATE)
    with pytest.raises(ValueError, match="damping_ratio"):
        _core.design_oscillator_bank(OFFICIAL_PERIODS_S, 1.5, RATE)


def test_advancing_an_empty_block_is_a_no_op() -> None:
    bank = _core.design_oscillator_bank(OFFICIAL_PERIODS_S, OFFICIAL_DAMPING_RATIO, RATE)
    state = _core.ResponseState(bank)
    assert state.advance(np.zeros((0, 2))) is None
    assert state.advance(np.zeros((0, 2)), collect=True).shape == (0, 32)
    assert state.sample_count == 0
