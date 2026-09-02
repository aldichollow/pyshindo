from __future__ import annotations

import numpy as np
import pytest

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
