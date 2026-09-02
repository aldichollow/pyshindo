"""Duration-to-sample conversion and amplitude order statistics."""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt

from .exceptions import FractionalDurationWarning, InsufficientDataError
from .models import AmplitudeDurationCurve

DurationSamplePolicy = Literal["ceil", "nearest", "floor"]


def duration_sample_count(
    duration_s: float,
    sampling_rate_hz: float,
    *,
    policy: DurationSamplePolicy = "ceil",
    warn_fractional: bool = True,
) -> int:
    """Convert a physical duration to a positive number of samples.

    ``ceil`` is the default because it never represents less than the requested
    duration. At 100 Hz, the standard 0.3-second condition maps exactly to 30
    samples and all policies agree.
    """
    duration_s = float(duration_s)
    sampling_rate_hz = float(sampling_rate_hz)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and greater than zero.")
    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and greater than zero.")

    exact = duration_s * sampling_rate_hz
    nearest = round(exact)
    is_integer = math.isclose(exact, nearest, rel_tol=0.0, abs_tol=1e-12)
    if warn_fractional and not is_integer:
        warnings.warn(
            f"{duration_s:g} s corresponds to {exact:.6g} samples at "
            f"{sampling_rate_hz:g} Hz. The {policy!r} sample-count policy is used.",
            FractionalDurationWarning,
            stacklevel=2,
        )

    if policy == "ceil":
        count = math.ceil(exact - 1e-12)
    elif policy == "nearest":
        count = math.floor(exact + 0.5)
    elif policy == "floor":
        count = math.floor(exact + 1e-12)
    else:
        raise ValueError("policy must be 'ceil', 'nearest', or 'floor'.")
    if count < 1:
        raise ValueError("The selected duration maps to fewer than one sample.")
    return count


def _validated_amplitude(values: npt.ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("amplitude must be one-dimensional.")
    if array.size == 0:
        raise ValueError("amplitude must contain at least one sample.")
    if not np.all(np.isfinite(array)):
        raise ValueError("amplitude contains non-finite values.")
    if np.any(array < 0.0):
        raise ValueError("amplitude must be non-negative.")
    return array


def duration_threshold(
    resultant_acceleration_gal: npt.ArrayLike,
    sample_count: int,
) -> float:
    """Return the ``sample_count``-th largest resultant acceleration.

    This order statistic is the sample-domain implementation of the cumulative
    duration condition. Ties can make the number of samples greater than or
    equal to the returned threshold exceed ``sample_count``.
    """
    values = _validated_amplitude(resultant_acceleration_gal)
    if sample_count < 1:
        raise ValueError("sample_count must be at least one.")
    if values.size < sample_count:
        raise InsufficientDataError(
            f"At least {sample_count} samples are required; received {values.size}."
        )
    index = values.size - sample_count
    return float(np.partition(values, index)[index])


def duration_threshold_at(
    resultant_acceleration_gal: npt.ArrayLike,
    sampling_rate_hz: float,
    *,
    duration_s: float = 0.3,
    policy: DurationSamplePolicy = "ceil",
) -> float:
    """Return the amplitude selected by a physical cumulative duration.

    This convenience wrapper converts ``duration_s`` to a sample count and then
    calls :func:`duration_threshold`.
    """
    sample_count = duration_sample_count(duration_s, sampling_rate_hz, policy=policy)
    return duration_threshold(resultant_acceleration_gal, sample_count)


def exceedance_duration(
    amplitude: npt.ArrayLike,
    threshold: float,
    sampling_rate_hz: float,
) -> float:
    """Return total time with amplitude greater than or equal to ``threshold``."""
    values = _validated_amplitude(amplitude)
    threshold = float(threshold)
    sampling_rate_hz = float(sampling_rate_hz)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative.")
    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and greater than zero.")
    return float(np.count_nonzero(values >= threshold) / sampling_rate_hz)


def amplitude_duration_curve(
    amplitude: npt.ArrayLike,
    sampling_rate_hz: float,
) -> AmplitudeDurationCurve:
    """Return a descending amplitude-duration curve for inspection or plotting.

    Entry ``k`` contains the ``(k + 1)``-th largest amplitude and cumulative
    sample duration ``(k + 1) / sampling_rate_hz``. The curve makes the standard
    0.3-second selection directly visible without changing the calculation.
    """
    values = _validated_amplitude(amplitude)
    sampling_rate_hz = float(sampling_rate_hz)
    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and greater than zero.")
    ordered = np.sort(values)[::-1].copy()
    duration = np.arange(1, ordered.size + 1, dtype=np.float64) / sampling_rate_hz
    return AmplitudeDurationCurve(amplitude=ordered, exceedance_duration_s=duration)
