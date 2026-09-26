from __future__ import annotations

import numpy as np
import pytest

from pyshindo.spatial import ValueTransform
from pyshindo.spatial._transform import forward_transform, inverse_transform


def test_identity_transform_is_a_no_op() -> None:
    values = np.array([-5.0, 0.0, 5.0])
    assert forward_transform(values, ValueTransform.IDENTITY) is values
    assert inverse_transform(values, ValueTransform.IDENTITY) is values


def test_log_transform_round_trips() -> None:
    values = np.array([0.1, 1.0, 100.0])
    forward = forward_transform(values, ValueTransform.LOG)
    np.testing.assert_allclose(inverse_transform(forward, ValueTransform.LOG), values)


def test_log_transform_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        forward_transform(np.array([1.0, 0.0, 2.0]), ValueTransform.LOG)
    with pytest.raises(ValueError, match="strictly positive"):
        forward_transform(np.array([1.0, -3.0]), ValueTransform.LOG)


def test_log_inverse_transform_passes_nan_through() -> None:
    result = inverse_transform(np.array([1.0, np.nan]), ValueTransform.LOG)
    assert np.isnan(result[1])
    assert result[0] == pytest.approx(np.e)
