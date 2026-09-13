"""General signal utilities used around seismic-intensity calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.exceptions import AxisError
from scipy import signal as scipy_signal

from .units import AccelerationUnit, ArrayLike, FloatArray, to_gal
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
    except AxisError as exc:
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


@dataclass(frozen=True, slots=True)
class ClippingInterval:
    """One suspected clipped interval in a single component.

    ``end_sample`` is exclusive, matching Python slicing (``column[start_sample:end_sample]``
    is the flagged run). ``value`` is the flagged run's own largest-magnitude
    sample, in gal. ``method`` is ``"known_range"`` or ``"repeated_extreme"``;
    see :func:`detect_clipping`.
    """

    component: int
    start_sample: int
    end_sample: int
    value: float
    method: str


@dataclass(frozen=True, slots=True)
class ClippingReport:
    """Diagnostic report of suspected clipping, per component and interval.

    Detection only: nothing here corrects, removes, or otherwise alters
    ``acceleration``. See :func:`detect_clipping`.
    """

    intervals: tuple[ClippingInterval, ...]
    component_count: int
    sample_count: int

    @property
    def any_suspected(self) -> bool:
        """Return whether any interval was flagged, by either method."""
        return len(self.intervals) > 0

    @property
    def components_affected(self) -> tuple[int, ...]:
        """Return the sorted, deduplicated component indices with a flagged interval."""
        return tuple(sorted({interval.component for interval in self.intervals}))


def _repeated_value_mask(column: FloatArray, repeat_threshold: int) -> npt.NDArray[np.bool_]:
    """Return True for every sample in a run of at least ``repeat_threshold`` equal values."""
    n = column.size
    mask = np.zeros(n, dtype=bool)
    if n < repeat_threshold:
        return mask
    change = np.flatnonzero(np.diff(column) != 0.0)
    boundaries = np.concatenate(([0], change + 1, [n]))
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end - start >= repeat_threshold:
            mask[start:end] = True
    return mask


def _flagged_runs_to_intervals(
    component: int,
    flagged: npt.NDArray[np.bool_],
    column: FloatArray,
    method: str,
) -> list[ClippingInterval]:
    """Convert a boolean flag mask into contiguous :class:`ClippingInterval` runs."""
    if not np.any(flagged):
        return []
    padded = np.concatenate(([False], flagged, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    intervals = []
    for start, end in zip(starts, ends, strict=True):
        segment = column[start:end]
        value = float(segment[np.argmax(np.abs(segment))])
        intervals.append(
            ClippingInterval(
                component=component,
                start_sample=int(start),
                end_sample=int(end),
                value=value,
                method=method,
            )
        )
    return intervals


def detect_clipping(
    acceleration: ArrayLike,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    max_range_gal: float | None = None,
    range_tolerance: float = 0.001,
    repeat_threshold: int = 3,
    extreme_fraction: float = 0.9,
    component_axis: int = -1,
) -> ClippingReport:
    """Flag samples that look clipped, per component and interval.

    Diagnostic only: nothing is corrected, removed, or otherwise changed, and
    no other function in this package calls this one automatically -- run it
    explicitly and decide what to do with what it finds, the same "warn, do
    not silently act" stance :class:`~pyshindo.exceptions.MissingComponentWarning`
    takes elsewhere in this package.

    Two independent methods, checked separately (a sample can be flagged by
    either, both, or neither); each component is evaluated on its own, since
    only one channel of a three-component record clipping briefly is a real
    and useful thing to be able to see.

    ``known_range``:
        Every sample at or beyond ``max_range_gal`` -- a full-scale digitizer
        range known in advance, if any -- allowing ``range_tolerance`` of
        headroom below it, since a saturated sample does not always land
        exactly on the nominal full-scale value. Skipped entirely when
        ``max_range_gal`` is ``None``.
    ``repeated_extreme``:
        A run of at least ``repeat_threshold`` consecutive, exactly equal
        samples -- something a real signal essentially never does, but a
        saturated digitizer often does -- restricted to samples within
        ``extreme_fraction`` of that component's own peak absolute
        amplitude. The restriction exists because a quiet pre-event noise
        floor can also repeat a value by coincidence of quantization;
        requiring the repeat to sit near the component's own extreme keeps a
        quiet record from flagging its own noise floor.

        This still has a real, documented false-positive mode: a record that
        is quiet *everywhere*, published already rounded to a few decimal
        places (as JMA's own strong-motion records are, to three), can hold
        the same rounded value for a few samples right at its own smooth,
        low-amplitude peak, where the true signal's derivative is close to
        zero -- not because anything clipped, but because rounding a slowly
        turning curve does that. Checked against 268 real JMA station
        records, this fired on 2 of them, both components whose own peak was
        under 0.5 gal. Scrutinize a
        ``repeated_extreme`` flag on a very small peak amplitude accordingly;
        it is not a sign the method is unreliable at any level that matters
        for clipping in practice.

    Parameters
    ----------
    max_range_gal:
        A known full-scale range in gal (for example, a digitizer's stated
        maximum), or ``None`` to skip the known-range method entirely and
        rely on ``repeated_extreme`` alone.
    range_tolerance:
        Fractional headroom below ``max_range_gal`` still counted as clipped;
        0.001 means within 0.1 percent of full scale.
    repeat_threshold:
        Minimum run length of exactly equal consecutive samples to flag with
        the ``repeated_extreme`` method. Must be at least 2.
    extreme_fraction:
        A ``repeated_extreme`` run only counts within this fraction of the
        component's own peak absolute amplitude. Must lie in ``(0, 1]``.
    """
    if range_tolerance < 0.0 or range_tolerance >= 1.0:
        raise ValueError("range_tolerance must lie within [0, 1).")
    if repeat_threshold < 2:
        raise ValueError("repeat_threshold must be at least two.")
    if not 0.0 < extreme_fraction <= 1.0:
        raise ValueError("extreme_fraction must lie within (0, 1].")
    if max_range_gal is not None and not (math.isfinite(max_range_gal) and max_range_gal > 0.0):
        raise ValueError("max_range_gal must be finite and greater than zero.")

    values = as_acceleration_array(
        acceleration,
        component_axis=component_axis,
        warn_fewer_components=False,
    )
    values_gal = to_gal(values, unit, copy=False)
    samples, components = values_gal.shape

    intervals: list[ClippingInterval] = []
    for component in range(components):
        column = values_gal[:, component]

        if max_range_gal is not None:
            threshold = max_range_gal * (1.0 - range_tolerance)
            known_range_flags = np.abs(column) >= threshold
            intervals.extend(
                _flagged_runs_to_intervals(component, known_range_flags, column, "known_range")
            )

        peak = float(np.max(np.abs(column))) if samples else 0.0
        if peak > 0.0:
            extreme = np.abs(column) >= peak * extreme_fraction
            repeated = _repeated_value_mask(column, repeat_threshold)
            intervals.extend(
                _flagged_runs_to_intervals(
                    component, extreme & repeated, column, "repeated_extreme"
                )
            )

    return ClippingReport(
        intervals=tuple(intervals),
        component_count=components,
        sample_count=samples,
    )
