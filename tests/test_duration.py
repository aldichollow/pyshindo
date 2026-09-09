from __future__ import annotations

import numpy as np
import pytest

from pyshindo.duration import (
    amplitude_duration_curve,
    duration_sample_count,
    duration_threshold,
    exceedance_duration,
)
from pyshindo.exceptions import FractionalDurationWarning, InsufficientDataError


def test_standard_duration_is_30_samples() -> None:
    assert duration_sample_count(0.3, 100.0) == 30


def test_fractional_duration_uses_ceil() -> None:
    with pytest.warns(FractionalDurationWarning):
        assert duration_sample_count(0.3, 128.0) == 39


def test_duration_threshold_is_kth_largest() -> None:
    values = np.array([1.0, 10.0, 4.0, 8.0, 3.0])
    assert duration_threshold(values, 2) == 8.0


def test_duration_threshold_rejects_too_few_samples() -> None:
    with pytest.raises(InsufficientDataError, match="At least 5"):
        duration_threshold(np.array([1.0, 2.0, 3.0]), 5)


def test_duration_sample_count_rejects_an_unknown_policy() -> None:
    with pytest.raises(ValueError, match="policy"):
        duration_sample_count(0.3, 100.0, policy="round")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "amplitude", [np.array([]), np.array([[1.0, 2.0]]), np.array([1.0, -2.0, 3.0])]
)
def test_duration_threshold_rejects_invalid_amplitude(amplitude: np.ndarray) -> None:
    with pytest.raises(ValueError):
        duration_threshold(amplitude, 1)


def test_duration_threshold_rejects_non_finite_amplitude() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        duration_threshold(np.array([1.0, float("nan"), 3.0]), 1)


def test_exceedance_duration_counts_samples_at_or_above_threshold() -> None:
    values = np.array([1.0, 5.0, 5.0, 2.0, 8.0])
    assert exceedance_duration(values, 5.0, 100.0) == pytest.approx(0.03)


def test_exceedance_duration_rejects_a_negative_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        exceedance_duration(np.array([1.0, 2.0]), -1.0, 100.0)


def test_exceedance_duration_rejects_a_non_positive_sampling_rate() -> None:
    with pytest.raises(ValueError, match="sampling_rate_hz"):
        exceedance_duration(np.array([1.0, 2.0]), 1.0, 0.0)


def test_amplitude_duration_curve_is_sorted_descending() -> None:
    curve = amplitude_duration_curve(np.array([3.0, 1.0, 2.0]), 100.0)
    np.testing.assert_array_equal(curve.amplitude, [3.0, 2.0, 1.0])
    np.testing.assert_allclose(curve.exceedance_duration_s, [0.01, 0.02, 0.03])


def test_amplitude_duration_curve_rejects_a_non_positive_sampling_rate() -> None:
    with pytest.raises(ValueError, match="sampling_rate_hz"):
        amplitude_duration_curve(np.array([1.0, 2.0]), -100.0)
