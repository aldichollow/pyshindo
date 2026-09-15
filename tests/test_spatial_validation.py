from __future__ import annotations

import numpy as np
import pytest

from pyshindo.spatial.validation import validate_station_coordinates, validate_values


def test_coordinates_must_be_one_dimensional_and_equal_length() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_station_coordinates(np.zeros((2, 2)), np.zeros((2, 2)), minimum_count=1)
    with pytest.raises(ValueError, match="same length"):
        validate_station_coordinates(np.zeros(3), np.zeros(4), minimum_count=1)


def test_coordinates_require_the_minimum_station_count() -> None:
    with pytest.raises(ValueError, match="At least 3"):
        validate_station_coordinates(np.zeros(2), np.zeros(2), minimum_count=3)


def test_coordinates_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_station_coordinates(
            np.array([35.0, np.nan]), np.array([135.0, 136.0]), minimum_count=1
        )


def test_latitude_and_longitude_ranges_are_enforced() -> None:
    with pytest.raises(ValueError, match=r"\[-90, 90\]"):
        validate_station_coordinates(np.array([91.0]), np.array([135.0]), minimum_count=1)
    with pytest.raises(ValueError, match=r"\[-180, 180\]"):
        validate_station_coordinates(np.array([35.0]), np.array([181.0]), minimum_count=1)


def test_duplicate_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        validate_station_coordinates(
            np.array([35.0, 35.0]), np.array([135.0, 135.0]), minimum_count=1
        )


def test_valid_coordinates_return_contiguous_float64_arrays() -> None:
    lat, lon = validate_station_coordinates([35, 36], [135, 136], minimum_count=1)
    assert lat.dtype == np.float64
    assert lon.dtype == np.float64
    assert lat.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(lat, [35.0, 36.0])


def test_values_must_match_station_count() -> None:
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        validate_values([1.0, 2.0], station_count=3)


def test_values_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_values([1.0, np.nan, 3.0], station_count=3)


def test_valid_values_return_a_contiguous_float64_array() -> None:
    array = validate_values([1, 2, 3], station_count=3)
    assert array.dtype == np.float64
    assert array.flags["C_CONTIGUOUS"]
