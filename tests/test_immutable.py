from __future__ import annotations

import numpy as np
import pytest

from pyshindo._immutable import frozen_array, frozen_mapping


def test_frozen_array_is_read_only() -> None:
    array = frozen_array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="read-only"):
        array[0] = 99.0


def test_frozen_array_copies_rather_than_aliasing_the_input() -> None:
    source = np.array([1.0, 2.0, 3.0])
    array = frozen_array(source)
    source[0] = 99.0
    assert array[0] == 1.0


def test_frozen_array_preserves_dtype_by_default() -> None:
    array = frozen_array(np.array([1, 2, 3], dtype=np.int64))
    assert array.dtype == np.int64


def test_frozen_array_casts_to_an_explicit_dtype() -> None:
    array = frozen_array([1, 2, 3], dtype=np.float64)
    assert array.dtype == np.float64


def test_frozen_mapping_is_read_only() -> None:
    mapping = frozen_mapping({"a": 1.0})
    with pytest.raises(TypeError):
        mapping["a"] = 2.0  # type: ignore[index]


def test_frozen_mapping_copies_rather_than_aliasing_the_input() -> None:
    source = {"a": 1.0}
    mapping = frozen_mapping(source)
    source["a"] = 99.0
    assert mapping["a"] == 1.0
