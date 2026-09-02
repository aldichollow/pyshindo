from __future__ import annotations

import numpy as np
import pytest

from pyshindo.signal import (
    peak_ground_acceleration,
    remove_offset,
    resample_acceleration,
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
