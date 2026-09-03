from __future__ import annotations

import numpy as np
import pytest

from pyshindo.exceptions import UnstableFilterError
from pyshindo.filters.jma import jma_filter_components, jma_filter_response
from pyshindo.filters.realtime import (
    RealtimeFilter,
    design_realtime_filter,
    kunugi_2012_analog_amplitude,
    realtime_filter_response,
)


def test_jma_response_is_zero_at_dc_and_finite_elsewhere() -> None:
    frequency = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 50.0])
    components = jma_filter_components(frequency)
    assert np.isinf(components.period_effect[0])
    assert components.combined[0] == 0.0
    assert np.all(np.isfinite(components.combined))
    assert np.all(components.combined >= 0.0)


def test_2012_filter_has_six_biquads_and_is_stable_at_100_hz() -> None:
    design = design_realtime_filter(100.0, filter_name=RealtimeFilter.KUNUGI_2012)
    assert design.sos.shape == (6, 6)
    assert design.stable
    assert design.max_pole_radius < 1.0


def test_2012_analog_response_reproduces_reported_approximation_range() -> None:
    frequency = np.geomspace(0.1, 50.0, 20_000)
    ratio = kunugi_2012_analog_amplitude(frequency) / jma_filter_response(frequency)
    assert np.min(ratio) >= 0.974 - 5e-4
    assert np.max(ratio) <= 1.029 + 5e-4


def test_2012_digital_response_is_close_in_primary_band_at_100_hz() -> None:
    design = design_realtime_filter(100.0)
    frequency = np.geomspace(0.1, 10.0, 10_000)
    response = realtime_filter_response(design, frequency).amplitude
    ratio = response / jma_filter_response(frequency)
    assert np.min(ratio) >= 0.974 - 5e-4
    assert np.max(ratio) <= 1.029 + 5e-4


def test_explicit_2012_design_rejects_unstable_low_rate() -> None:
    with pytest.raises(UnstableFilterError):
        design_realtime_filter(50.0, filter_name=RealtimeFilter.KUNUGI_2012)


def test_auto_uses_generalized_design_below_80_hz() -> None:
    design = design_realtime_filter(50.0)
    assert design.name == RealtimeFilter.JP7681907_LOWRATE.value
    assert design.stable


def test_lowrate_and_2012_are_equivalent_at_100_hz() -> None:
    conventional = design_realtime_filter(100.0, filter_name=RealtimeFilter.KUNUGI_2012)
    generalized = design_realtime_filter(
        100.0,
        filter_name=RealtimeFilter.JP7681907_LOWRATE,
    )
    np.testing.assert_allclose(generalized.sos, conventional.sos, rtol=0.0, atol=1e-14)


def test_2008_filter_can_be_constructed() -> None:
    design = design_realtime_filter(100.0, filter_name=RealtimeFilter.KUNUGI_2008)
    assert design.sos.shape == (3, 6)
    assert design.stable


def test_2012_100_hz_coefficients_regression() -> None:
    expected = np.array(
        [
            [0.683841286630, -1.121132630715, 0.437291344085, 1.0, -1.773993457792, 0.779517257629],
            [0.335099020523, -0.321629559579, 0.019220136557, 1.0, -1.335552634567, 0.368242232068],
            [1.007672569921, -1.953000416904, 0.946292010555, 1.0, -1.953000416904, 0.953964580476],
            [0.027448001625, 0.274480016246, 0.027448001625, 1.0, -0.884296655928, 0.213672675423],
            [0.069790163958, 0.697901639579, 0.069790163958, 1.0, -0.362781663242, 0.200263630737],
            [0.121994491519, 1.219944915195, 0.121994491519, 1.0, 0.395903163288, 0.068030734945],
        ]
    )
    design = design_realtime_filter(100.0, filter_name=RealtimeFilter.KUNUGI_2012)
    np.testing.assert_allclose(design.sos, expected, rtol=0.0, atol=5e-13)


def test_jma_fft_filter_matches_integer_cycle_sine_gain() -> None:
    from pyshindo.filters.jma import apply_jma_filter_fft

    sampling_rate_hz = 100.0
    sample_count = 2_000
    frequency_hz = 2.0
    time_s = np.arange(sample_count, dtype=np.float64) / sampling_rate_hz
    wave = np.sin(2.0 * np.pi * frequency_hz * time_s)
    filtered, _, _ = apply_jma_filter_fft(wave[:, np.newaxis], sampling_rate_hz)
    expected_gain = jma_filter_response(np.array([frequency_hz]))[0]
    np.testing.assert_allclose(filtered[:, 0], expected_gain * wave, rtol=0.0, atol=2e-13)


def test_published_lowrate_table_policies_are_stable_at_50_hz() -> None:
    from pyshindo.filters.realtime import LowRateGammaPolicy

    for policy in (LowRateGammaPolicy.PIECEWISE, LowRateGammaPolicy.CONSTANT_STABLE):
        design = design_realtime_filter(
            50.0,
            filter_name=RealtimeFilter.JP7681907_LOWRATE,
            lowrate_gamma_policy=policy,
        )
        assert design.stable


def test_accuracy_constant_lowrate_policy_exposes_instability() -> None:
    from pyshindo.filters.realtime import LowRateGammaPolicy

    with pytest.raises(UnstableFilterError):
        design_realtime_filter(
            50.0,
            filter_name=RealtimeFilter.JP7681907_LOWRATE,
            lowrate_gamma_policy=LowRateGammaPolicy.CONSTANT_ACCURATE,
        )


@pytest.mark.parametrize(
    ("filter_name", "sampling_rate_hz", "expected_stage_count"),
    [
        (RealtimeFilter.KUNUGI_2008, 100.0, 6),
        (RealtimeFilter.KUNUGI_2012, 100.0, 9),
        (RealtimeFilter.JP7681907_LOWRATE, 50.0, 9),
    ],
)
def test_filter_stages_cascade_reproduces_combined_response(
    filter_name: RealtimeFilter,
    sampling_rate_hz: float,
    expected_stage_count: int,
) -> None:
    from scipy.signal import sosfreqz

    design = design_realtime_filter(sampling_rate_hz, filter_name=filter_name)
    assert len(design.stages) == expected_stage_count
    assert design.stages[-1].name == "gain"

    stage_sos = np.vstack([stage.sos for stage in design.stages])
    frequency = np.linspace(0.01, design.nyquist_hz * 0.99, 2000)
    _, combined_response = sosfreqz(design.sos, worN=frequency, fs=sampling_rate_hz)
    _, staged_response = sosfreqz(stage_sos, worN=frequency, fs=sampling_rate_hz)
    np.testing.assert_allclose(staged_response, combined_response, rtol=0.0, atol=1e-9)


def test_filter_stage_response_matches_its_own_row_in_the_combined_cascade() -> None:
    from pyshindo.filters.realtime import filter_stage_response

    design = design_realtime_filter(100.0, filter_name=RealtimeFilter.KUNUGI_2012)
    gain_stage = design.stages[-1]
    response = filter_stage_response(design, gain_stage, np.array([0.0, 1.0, 10.0]))
    np.testing.assert_allclose(response.amplitude, gain_stage.sos[0], rtol=0.0, atol=1e-12)
