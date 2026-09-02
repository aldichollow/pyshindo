"""Instrumental-intensity conversion, reporting, and scale classification."""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from enum import StrEnum
from math import isfinite, log10
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt

from .units import ArrayLike, FloatArray

_HUNDREDTH_PLACE: Final = Decimal("0.01")
_TENTH_PLACE: Final = Decimal("0.1")


class IntensityScale(StrEnum):
    """JMA seismic-intensity classes with machine-friendly values."""

    ZERO = "0"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE_LOWER = "5-"
    FIVE_UPPER = "5+"
    SIX_LOWER = "6-"
    SIX_UPPER = "6+"
    SEVEN = "7"

    @property
    def japanese(self) -> str:
        """Return the conventional Japanese class label."""
        return {
            IntensityScale.ZERO: "0",
            IntensityScale.ONE: "1",
            IntensityScale.TWO: "2",
            IntensityScale.THREE: "3",
            IntensityScale.FOUR: "4",
            IntensityScale.FIVE_LOWER: "5弱",
            IntensityScale.FIVE_UPPER: "5強",
            IntensityScale.SIX_LOWER: "6弱",
            IntensityScale.SIX_UPPER: "6強",
            IntensityScale.SEVEN: "7",
        }[self]

    @property
    def english(self) -> str:
        """Return a readable English class label."""
        return {
            IntensityScale.ZERO: "0",
            IntensityScale.ONE: "1",
            IntensityScale.TWO: "2",
            IntensityScale.THREE: "3",
            IntensityScale.FOUR: "4",
            IntensityScale.FIVE_LOWER: "5 lower",
            IntensityScale.FIVE_UPPER: "5 upper",
            IntensityScale.SIX_LOWER: "6 lower",
            IntensityScale.SIX_UPPER: "6 upper",
            IntensityScale.SEVEN: "7",
        }[self]


INTENSITY_INTERVALS: Final = MappingProxyType(
    {
        IntensityScale.ZERO: (-np.inf, 0.5),
        IntensityScale.ONE: (0.5, 1.5),
        IntensityScale.TWO: (1.5, 2.5),
        IntensityScale.THREE: (2.5, 3.5),
        IntensityScale.FOUR: (3.5, 4.5),
        IntensityScale.FIVE_LOWER: (4.5, 5.0),
        IntensityScale.FIVE_UPPER: (5.0, 5.5),
        IntensityScale.SIX_LOWER: (5.5, 6.0),
        IntensityScale.SIX_UPPER: (6.0, 6.5),
        IntensityScale.SEVEN: (6.5, np.inf),
    }
)
"""Published instrumental-intensity intervals keyed by intensity class."""

_SCALE_LOWER_BOUNDS: Final[tuple[tuple[float, IntensityScale], ...]] = tuple(
    (lower, scale)
    for scale, (lower, _upper) in reversed(tuple(INTENSITY_INTERVALS.items()))
    if np.isfinite(lower)
)


def intensity_from_acceleration(threshold_acceleration_gal: float) -> float:
    """Convert the 0.3-second threshold acceleration to raw intensity.

    Zero acceleration maps to negative infinity. Positive acceleration is
    transformed as ``2 * log10(a) + 0.94``, where ``a`` is in gal.
    """
    if not isfinite(threshold_acceleration_gal) or threshold_acceleration_gal < 0.0:
        raise ValueError("threshold_acceleration_gal must be finite and non-negative.")
    if threshold_acceleration_gal == 0.0:
        return -np.inf
    return 2.0 * log10(threshold_acceleration_gal) + 0.94


def acceleration_from_intensity(intensity: float) -> float:
    """Return the threshold acceleration in gal implied by a raw intensity value."""
    if not isfinite(intensity):
        raise ValueError("intensity must be finite.")
    return 10.0 ** ((intensity - 0.94) / 2.0)


def intensity_series_from_acceleration(values_gal: ArrayLike) -> FloatArray:
    """Vectorize the raw intensity conversion over acceleration values.

    Zero values map to negative infinity. Negative and NaN values map to NaN so
    an unavailable rolling threshold remains distinguishable from quiet motion.
    """
    values = np.asarray(values_gal, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    positive = values > 0.0
    zero = values == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        output[positive] = 2.0 * np.log10(values[positive]) + 0.94
    output[zero] = -np.inf
    return output


def report_intensity(value: float) -> float:
    """Apply the official two-step decimal treatment to an intensity value.

    The calculation first rounds the third decimal place and then discards the
    second decimal place. Decimal arithmetic is used so boundary behavior does
    not depend on binary floating-point tie handling.
    """
    if not isfinite(value):
        return value
    decimal_value = Decimal(str(value))
    rounded_to_hundredth = decimal_value.quantize(_HUNDREDTH_PLACE, rounding=ROUND_HALF_UP)
    reported = rounded_to_hundredth.quantize(_TENTH_PLACE, rounding=ROUND_DOWN)
    return float(reported)


def report_intensity_array(values: npt.ArrayLike) -> FloatArray:
    """Apply :func:`report_intensity` element by element to an array."""
    array = np.asarray(values, dtype=np.float64)
    flat = np.fromiter((report_intensity(float(item)) for item in array.flat), dtype=np.float64)
    return flat.reshape(array.shape)


def classify_intensity(value: float) -> IntensityScale:
    """Map an instrumental-intensity value to a JMA intensity class."""
    if np.isnan(value):
        raise ValueError("NaN cannot be classified as an intensity scale.")
    for lower_bound, scale in _SCALE_LOWER_BOUNDS:
        if value >= lower_bound:
            return scale
    return IntensityScale.ZERO


def classify_intensity_array(values: npt.ArrayLike) -> npt.NDArray[np.str_]:
    """Return machine-friendly class labels for an array of intensity values."""
    array = np.asarray(values, dtype=np.float64)
    if np.any(np.isnan(array)):
        raise ValueError("NaN cannot be classified as an intensity scale.")
    output = np.full(array.shape, IntensityScale.ZERO.value, dtype="<U2")
    for lower_bound, scale in reversed(_SCALE_LOWER_BOUNDS):
        output[array >= lower_bound] = scale.value
    return output


def intensity_interval(scale: str | IntensityScale) -> tuple[float, float]:
    """Return the lower-inclusive, upper-exclusive interval of a class."""
    selected = scale if isinstance(scale, IntensityScale) else IntensityScale(scale)
    return INTENSITY_INTERVALS[selected]


def intensity_label(value: float, *, language: str = "ja", reported: bool = True) -> str:
    """Return a display label such as ``震度5弱`` or ``Intensity 5 lower``."""
    basis = report_intensity(value) if reported else value
    scale = classify_intensity(basis)
    normalized = language.strip().lower()
    if normalized in {"ja", "jp", "japanese"}:
        return f"震度{scale.japanese}"
    if normalized in {"en", "english"}:
        return f"Intensity {scale.english}"
    raise ValueError("language must be 'ja' or 'en'.")
