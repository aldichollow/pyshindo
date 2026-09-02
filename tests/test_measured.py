from __future__ import annotations

import numpy as np
import pytest

from pyshindo.measured import calculate_measured_intensity, measured_intensity


def _synthetic_record(scale: float = 1.0) -> np.ndarray:
    sampling_rate = 100.0
    time = np.arange(20 * int(sampling_rate), dtype=np.float64) / sampling_rate
    envelope = np.exp(-0.5 * ((time - 8.0) / 1.8) ** 2)
    return scale * np.column_stack(
        [
            60.0 * envelope * np.sin(2.0 * np.pi * 1.2 * time),
            45.0 * envelope * np.sin(2.0 * np.pi * 2.3 * time + 0.4),
            25.0 * envelope * np.sin(2.0 * np.pi * 4.8 * time + 1.1),
        ]
    )


def test_zero_record_returns_negative_infinity_and_scale_zero() -> None:
    result = calculate_measured_intensity(np.zeros((100, 3)))
    assert result.intensity_raw == -np.inf
    assert result.intensity == -np.inf
    assert result.scale.value == "0"


def test_reference_result_exposes_intermediates() -> None:
    values = _synthetic_record()
    result = calculate_measured_intensity(values)
    assert result.filtered_acceleration_gal is not None
    assert result.filtered_acceleration_gal.shape == values.shape
    assert result.resultant_acceleration_gal is not None
    assert result.resultant_acceleration_gal.shape == (values.shape[0],)
    assert result.frequency_hz is not None
    assert result.filter_response is not None
    assert result.duration_samples == 30
    assert result.reference_conditions_met


def test_linear_amplitude_scaling_has_expected_raw_intensity_shift() -> None:
    base = calculate_measured_intensity(_synthetic_record(), retain_intermediates=False)
    scaled = calculate_measured_intensity(_synthetic_record(scale=10.0), retain_intermediates=False)
    assert scaled.intensity_raw - base.intensity_raw == pytest.approx(2.0, abs=1e-12)


def test_simple_api_matches_detailed_result() -> None:
    values = _synthetic_record()
    detailed = calculate_measured_intensity(values, retain_intermediates=False)
    assert measured_intensity(values) == detailed.intensity
    assert measured_intensity(values, reported=False) == detailed.intensity_raw


def test_timing_is_reported_and_internally_consistent() -> None:
    result = calculate_measured_intensity(_synthetic_record())
    timing = result.timing
    assert timing.fft_filter_s >= 0.0
    assert timing.duration_threshold_s >= 0.0
    assert timing.total_s >= timing.fft_filter_s + timing.duration_threshold_s
