"""General signal utilities used around seismic-intensity calculations."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import signal as scipy_signal

from .units import ArrayLike, FloatArray
from .validation import as_acceleration_array, validate_sampling_rate

DetrendMode = Literal["constant", "linear"]


def time_axis(sample_count: int, sampling_rate_hz: float, *, start_s: float = 0.0) -> FloatArray:
    """Return an exact sample-index-based time axis."""
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative.")
    if not math.isfinite(start_s):
        raise ValueError("start_s must be finite.")
    rate = validate_sampling_rate(sampling_rate_hz, warn_nonstandard=False)
    return start_s + np.arange(sample_count, dtype=np.float64) / rate


def vector_resultant(acceleration: npt.ArrayLike, *, component_axis: int = -1) -> FloatArray:
    """Return the Euclidean norm across acceleration components."""
    values = np.asarray(acceleration, dtype=np.float64)
    if values.ndim == 1:
        if not np.all(np.isfinite(values)):
            raise ValueError("acceleration contains non-finite values.")
        return np.abs(values)
    if values.ndim != 2:
        raise ValueError("acceleration must be one- or two-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError("acceleration contains non-finite values.")
    try:
        return np.linalg.norm(values, axis=component_axis)
    except np.AxisError as exc:
        raise ValueError(f"component_axis={component_axis} is invalid for {values.shape}.") from exc


def component_peak_acceleration(
    acceleration: ArrayLike,
    *,
    component_axis: int = -1,
) -> FloatArray:
    """Return the maximum absolute acceleration of each component."""
    values = as_acceleration_array(
        acceleration,
        component_axis=component_axis,
        warn_fewer_components=False,
    )
    return np.max(np.abs(values), axis=0)


def peak_ground_acceleration(
    acceleration: npt.ArrayLike,
    *,
    component_axis: int = -1,
) -> float:
    """Return the maximum vector-resultant acceleration in the input unit."""
    resultant = vector_resultant(acceleration, component_axis=component_axis)
    if resultant.size == 0:
        raise ValueError("acceleration must contain at least one sample.")
    return float(np.max(resultant))


def remove_offset(
    acceleration: ArrayLike,
    *,
    baseline_samples: int | None = None,
    component_axis: int = -1,
) -> FloatArray:
    """Subtract a per-component mean from a complete or initial baseline interval.

    This operation is not part of the published intensity definition. It is a
    practical preprocessing helper for records containing offsets or a static
    gravity component. Its use and baseline interval should be reported.
    """
    values = as_acceleration_array(
        acceleration,
        component_axis=component_axis,
        warn_fewer_components=False,
        copy=True,
    )
    if baseline_samples is None:
        reference = values
    else:
        if baseline_samples < 1 or baseline_samples > values.shape[0]:
            raise ValueError("baseline_samples must lie within the record.")
        reference = values[:baseline_samples]
    values -= np.mean(reference, axis=0, keepdims=True)
    return values


def detrend_acceleration(
    acceleration: ArrayLike,
    *,
    mode: DetrendMode = "linear",
    component_axis: int = -1,
) -> FloatArray:
    """Remove a constant or linear trend independently from each component.

    Detrending is intentionally explicit because it changes the input record
    and is not an implicit step of the reference intensity implementation.
    """
    values = as_acceleration_array(
        acceleration,
        component_axis=component_axis,
        warn_fewer_components=False,
    )
    return np.ascontiguousarray(scipy_signal.detrend(values, axis=0, type=mode))


def resample_acceleration(
    acceleration: ArrayLike,
    original_rate_hz: float,
    target_rate_hz: float = 100.0,
    *,
    component_axis: int = -1,
    max_denominator: int = 100_000,
    window: str | tuple[str, float] = ("kaiser", 5.0),
) -> FloatArray:
    """Resample acceleration with a polyphase anti-aliasing filter.

    The rational approximation to ``target_rate_hz / original_rate_hz`` is
    limited by ``max_denominator``. The caller remains responsible for checking
    whether the source bandwidth, calibration, and anti-alias filtering support
    the target analysis.
    """
    original = validate_sampling_rate(original_rate_hz, warn_nonstandard=False)
    target = validate_sampling_rate(target_rate_hz, warn_nonstandard=False)
    if max_denominator < 1:
        raise ValueError("max_denominator must be at least one.")
    values = as_acceleration_array(
        acceleration,
        component_axis=component_axis,
        warn_fewer_components=False,
    )
    ratio = Fraction(target / original).limit_denominator(max_denominator)
    resampled = scipy_signal.resample_poly(
        values,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=0,
        window=window,
    )
    return np.ascontiguousarray(resampled, dtype=np.float64)


def cosine_taper(
    acceleration: ArrayLike,
    *,
    fraction: float = 0.05,
    component_axis: int = -1,
) -> FloatArray:
    """Apply a symmetric cosine taper to a fraction of each record edge.

    ``fraction=0.05`` tapers the first and last five percent. Tapering changes
    the reference intensity result and is never applied automatically.
    """
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 0.5:
        raise ValueError("fraction must be between zero and 0.5.")
    values = as_acceleration_array(
        acceleration,
        component_axis=component_axis,
        warn_fewer_components=False,
        copy=True,
    )
    window = scipy_signal.windows.tukey(values.shape[0], alpha=2.0 * fraction)
    values *= window[:, np.newaxis]
    return values
