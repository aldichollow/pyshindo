from __future__ import annotations

import numpy as np
import pytest

from pyshindo.signal import (
    cosine_taper,
    detrend_acceleration,
    peak_ground_acceleration,
    remove_offset,
    resample_acceleration,
    time_axis,
    vector_resultant,
)
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
