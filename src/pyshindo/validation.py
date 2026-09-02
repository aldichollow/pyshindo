"""Input validation shared by batch and streaming calculations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .exceptions import (
    InvalidAccelerationError,
    MissingComponentWarning,
    NonstandardSamplingRateWarning,
)
from .units import ArrayLike, FloatArray

STANDARD_SAMPLING_RATE_HZ = 100.0


@dataclass(frozen=True, slots=True)
class SamplingDiagnostics:
    """Summary statistics for timestamp-derived sampling intervals."""

    sampling_rate_hz: float
    median_interval_s: float
    mean_interval_s: float
    max_absolute_jitter_s: float
    relative_peak_jitter: float
    is_uniform: bool


def validate_sampling_rate(
    sampling_rate_hz: float,
    *,
    warn_nonstandard: bool = True,
    stacklevel: int = 2,
) -> float:
    """Validate a sampling rate and optionally warn when it is not 100 Hz."""
    sampling_rate_hz = float(sampling_rate_hz)
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and greater than zero.")
    if warn_nonstandard and not np.isclose(
        sampling_rate_hz,
        STANDARD_SAMPLING_RATE_HZ,
        rtol=0.0,
        atol=1e-12,
    ):
        warnings.warn(
            "The published reference and real-time algorithms are normally evaluated at "
            "100 Hz. Coefficients and duration samples are being recomputed for "
            f"{sampling_rate_hz:g} Hz; validate the result for the intended application.",
            NonstandardSamplingRateWarning,
            stacklevel=stacklevel,
        )
    return sampling_rate_hz


def as_acceleration_array(
    acceleration: ArrayLike,
    *,
    component_axis: int = -1,
    allow_fewer_components: bool = True,
    warn_fewer_components: bool = True,
    copy: bool = False,
) -> FloatArray:
    """Return finite acceleration as ``(samples, components)`` float64 data.

    One-dimensional input is treated as a single component. Two-dimensional
    input may contain one, two, or three components. Three components are
    required for the standard instrumental-intensity calculation; fewer
    components are accepted for diagnostics and controlled experiments.
    """
    array = (
        np.array(acceleration, dtype=np.float64, copy=True)
        if copy
        else np.asarray(acceleration, dtype=np.float64)
    )
    if array.ndim == 1:
        array = array[:, np.newaxis]
    elif array.ndim == 2:
        try:
            array = np.moveaxis(array, component_axis, -1)
        except np.AxisError as exc:
            raise InvalidAccelerationError(
                f"component_axis={component_axis} is invalid for shape {array.shape}."
            ) from exc
    else:
        raise InvalidAccelerationError(
            f"Acceleration must be one- or two-dimensional; received shape {array.shape}."
        )

    if array.shape[0] == 0:
        raise InvalidAccelerationError("Acceleration data must contain at least one sample.")
    if not 1 <= array.shape[1] <= 3:
        raise InvalidAccelerationError(
            "Acceleration data must contain one, two, or three components; "
            f"received {array.shape[1]}."
        )
    if not np.all(np.isfinite(array)):
        indices = np.argwhere(~np.isfinite(array))
        first = tuple(int(value) for value in indices[0])
        raise InvalidAccelerationError(f"Acceleration contains a non-finite value at {first}.")
    if array.shape[1] < 3:
        if not allow_fewer_components:
            raise InvalidAccelerationError(
                "Three orthogonal acceleration components are required for this calculation."
            )
        if warn_fewer_components:
            warnings.warn(
                f"Only {array.shape[1]} acceleration component(s) were supplied. The resulting "
                "intensity is not the standard three-component value.",
                MissingComponentWarning,
                stacklevel=2,
            )
    return np.ascontiguousarray(array)


def sampling_diagnostics(
    timestamps_s: npt.ArrayLike,
    *,
    relative_tolerance: float = 1e-3,
) -> SamplingDiagnostics:
    """Estimate sampling rate and timing jitter from monotonically increasing timestamps."""
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("timestamps_s must be a one-dimensional array with at least two values.")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps_s contains non-finite values.")
    intervals = np.diff(timestamps)
    if np.any(intervals <= 0.0):
        raise ValueError("timestamps_s must be strictly increasing.")
    median_interval = float(np.median(intervals))
    mean_interval = float(np.mean(intervals))
    max_jitter = float(np.max(np.abs(intervals - median_interval)))
    relative_peak_jitter = max_jitter / median_interval
    return SamplingDiagnostics(
        sampling_rate_hz=1.0 / median_interval,
        median_interval_s=median_interval,
        mean_interval_s=mean_interval,
        max_absolute_jitter_s=max_jitter,
        relative_peak_jitter=relative_peak_jitter,
        is_uniform=relative_peak_jitter <= relative_tolerance,
    )
