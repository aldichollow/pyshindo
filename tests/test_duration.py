from __future__ import annotations

import numpy as np
import pytest

from pyshindo._order import RollingKthLargest
from pyshindo.duration import duration_sample_count, duration_threshold
from pyshindo.exceptions import FractionalDurationWarning


def test_standard_duration_is_30_samples() -> None:
    assert duration_sample_count(0.3, 100.0) == 30


def test_fractional_duration_uses_ceil() -> None:
    with pytest.warns(FractionalDurationWarning):
        assert duration_sample_count(0.3, 128.0) == 39


def test_duration_threshold_is_kth_largest() -> None:
    values = np.array([1.0, 10.0, 4.0, 8.0, 3.0])
    assert duration_threshold(values, 2) == 8.0


def test_rolling_kth_largest_matches_brute_force() -> None:
    generator = np.random.default_rng(42)
    values = generator.normal(size=4_000)
    window_size = 137
    rank = 11
    rolling = RollingKthLargest(window_size, rank)
    history: list[float] = []
    for value in values:
        history.append(float(value))
        actual = rolling.update(float(value))
        window = history[-window_size:]
        expected = None if len(window) < rank else sorted(window, reverse=True)[rank - 1]
        assert actual == expected


def test_rolling_kth_largest_handles_duplicate_values() -> None:
    rolling = RollingKthLargest(window_size=5, k=3)
    values = [4.0, 4.0, 1.0, 4.0, 2.0, 4.0, 0.0]
    history: list[float] = []
    for value in values:
        history.append(value)
        actual = rolling.update(value)
        window = history[-5:]
        expected = None if len(window) < 3 else sorted(window, reverse=True)[2]
        assert actual == expected


def test_rolling_kth_largest_bounds_lazy_heap_storage() -> None:
    rolling = RollingKthLargest(window_size=64, k=7)
    generator = np.random.default_rng(123)
    for value in generator.normal(size=20_000):
        rolling.update(float(value))
        assert len(rolling._top) + len(rolling._rest) <= 2 * rolling.window_size + 64
