from __future__ import annotations

import numpy as np
import pytest

from pyshindo.scale import (
    IntensityScale,
    acceleration_from_intensity,
    classify_intensity,
    intensity_from_acceleration,
    intensity_label,
    report_intensity,
)


@pytest.mark.parametrize("intensity", [-2.0, 0.0, 4.5, 6.732])
def test_acceleration_intensity_round_trip(intensity: float) -> None:
    acceleration = acceleration_from_intensity(intensity)
    assert intensity_from_acceleration(acceleration) == pytest.approx(intensity)


def test_zero_acceleration_maps_to_negative_infinity() -> None:
    assert intensity_from_acceleration(0.0) == -np.inf


@pytest.mark.parametrize(
    ("raw", "reported"),
    [
        (4.444, 4.4),
        (4.445, 4.4),
        (4.449, 4.4),
        (4.450, 4.4),
        (4.499, 4.5),
        (-1.999, -2.0),
    ],
)
def test_report_intensity_two_step_decimal_rule(raw: float, reported: float) -> None:
    assert report_intensity(raw) == reported


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-np.inf, IntensityScale.ZERO),
        (0.499, IntensityScale.ZERO),
        (0.5, IntensityScale.ONE),
        (1.5, IntensityScale.TWO),
        (2.5, IntensityScale.THREE),
        (3.5, IntensityScale.FOUR),
        (4.5, IntensityScale.FIVE_LOWER),
        (5.0, IntensityScale.FIVE_UPPER),
        (5.5, IntensityScale.SIX_LOWER),
        (6.0, IntensityScale.SIX_UPPER),
        (6.5, IntensityScale.SEVEN),
        (np.inf, IntensityScale.SEVEN),
    ],
)
def test_classification_boundaries(value: float, expected: IntensityScale) -> None:
    assert classify_intensity(value) is expected


def test_intensity_label() -> None:
    assert intensity_label(5.2, language="ja") == "震度5強"
    assert intensity_label(5.2, language="en") == "Intensity 5 upper"
