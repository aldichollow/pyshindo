from __future__ import annotations

import numpy as np
import pytest

from pyshindo.exceptions import InvalidAccelerationError, MissingComponentWarning
from pyshindo.measured import measured_intensity
from pyshindo.synthetic import (
    scale_acceleration_to_intensity,
    synthetic_three_component_motion,
)

RATE = 100.0


def _record() -> np.ndarray:
    return synthetic_three_component_motion(sampling_rate_hz=RATE, duration_s=10.0)


def test_scale_acceleration_to_intensity_hits_the_target() -> None:
    scaled, _ = scale_acceleration_to_intensity(_record(), 5.0, RATE)
    achieved = measured_intensity(scaled, RATE, reported=False)
    assert achieved == pytest.approx(5.0, abs=1e-9)


def test_scale_acceleration_to_intensity_factor_matches_the_log_relationship() -> None:
    record = _record()
    current = measured_intensity(record, RATE, reported=False)
    _, factor = scale_acceleration_to_intensity(record, 4.0, RATE)
    assert 2.0 * np.log10(factor) == pytest.approx(4.0 - current, abs=1e-9)


def test_scale_acceleration_to_intensity_keeps_one_dimensional_shape() -> None:
    single_component = _record()[:, 0]
    with pytest.warns(MissingComponentWarning):
        scaled, _ = scale_acceleration_to_intensity(
            single_component, 4.0, RATE, allow_fewer_components=True
        )
    assert scaled.ndim == 1
    assert scaled.shape == single_component.shape


def test_scale_acceleration_to_intensity_rejects_fewer_components_by_default() -> None:
    single_component = _record()[:, 0]
    with pytest.raises(InvalidAccelerationError, match="Three"):
        scale_acceleration_to_intensity(single_component, 4.0, RATE)


def test_scale_acceleration_to_intensity_respects_component_axis() -> None:
    record = _record()
    transposed = record.T  # (components, samples)
    scaled, _ = scale_acceleration_to_intensity(transposed, 5.0, RATE, component_axis=0)
    assert scaled.shape == transposed.shape
    # Same physical result regardless of which axis carried the components.
    scaled_default, _ = scale_acceleration_to_intensity(record, 5.0, RATE)
    np.testing.assert_allclose(scaled.T, scaled_default)


def test_scale_acceleration_to_intensity_rejects_a_non_finite_target() -> None:
    with pytest.raises(ValueError, match="target_intensity_raw"):
        scale_acceleration_to_intensity(_record(), float("nan"), RATE)


def test_scale_acceleration_to_intensity_rejects_an_unscalable_record() -> None:
    silent = np.zeros((1000, 3))
    with pytest.raises(ValueError, match="no positive intensity threshold"):
        scale_acceleration_to_intensity(silent, 5.0, RATE)
