"""Complete-record calculation of the JMA long-period ground motion class."""

from __future__ import annotations

import time
import warnings

import numpy as np
import numpy.typing as npt
from numpy.exceptions import AxisError

from ..exceptions import InvalidAccelerationError, NonstandardSamplingRateWarning
from ..units import AccelerationUnit, ArrayLike, FloatArray, to_gal
from ..validation import validate_sampling_rate
from ._core import (
    REFERENCE_SAMPLING_RATE_HZ,
    ResponseState,
    apply_high_pass,
    design_high_pass,
    design_oscillator_bank,
    high_pass_initial_state,
)
from .models import LongPeriodBandResult, LongPeriodResult, LongPeriodTiming
from .scale import (
    OFFICIAL_DAMPING_RATIO,
    OFFICIAL_PERIODS_S,
    PERIOD_BANDS,
    PERIOD_DECISECONDS,
    band_of_period_deciseconds,
    band_period_range_s,
    classify_long_period,
)


def as_horizontal_acceleration(
    acceleration: ArrayLike,
    *,
    component_axis: int = -1,
) -> FloatArray:
    """Return finite ``(samples, 2)`` horizontal acceleration.

    Deliberately not :func:`pyshindo.validation.as_acceleration_array`. That
    validator is built for the intensity calculation, where three components
    are the standard case and fewer are a shortfall worth warning about. Here
    the vertical component is not missing, it is *irrelevant*: the JMA class
    is defined on the two horizontal components only. Reusing a validator
    whose warning says components are missing would describe the situation
    backwards.
    """
    array = np.asarray(acceleration, dtype=np.float64)
    if array.ndim != 2:
        raise InvalidAccelerationError(
            "Long-period ground motion needs two horizontal acceleration components "
            f"shaped (samples, 2); received an array with {array.ndim} dimension(s)."
        )
    try:
        array = np.moveaxis(array, component_axis, -1)
    except AxisError as exc:
        raise InvalidAccelerationError(
            f"component_axis={component_axis} is invalid for shape {array.shape}."
        ) from exc
    if array.shape[1] == 3:
        raise InvalidAccelerationError(
            "Three components were supplied, but the long-period ground motion class "
            "is defined on the horizontal components only. Pass the two horizontal "
            "columns explicitly, for example acceleration[:, :2] for NS/EW/UD data."
        )
    if array.shape[1] != 2:
        raise InvalidAccelerationError(
            "Long-period ground motion needs exactly two horizontal acceleration "
            f"components; received {array.shape[1]}."
        )
    if array.shape[0] == 0:
        raise InvalidAccelerationError("Acceleration data must contain at least one sample.")
    if not np.all(np.isfinite(array)):
        indices = np.argwhere(~np.isfinite(array))
        first = tuple(int(value) for value in indices[0])
        raise InvalidAccelerationError(f"Acceleration contains a non-finite value at {first}.")
    return np.ascontiguousarray(array)


def resolve_periods(periods_s: npt.ArrayLike | None) -> FloatArray:
    """Return the period grid to evaluate, defaulting to the official 32."""
    if periods_s is None:
        return OFFICIAL_PERIODS_S
    periods = np.asarray(periods_s, dtype=np.float64)
    if periods.ndim != 1 or periods.size == 0:
        raise ValueError("periods_s must be a non-empty one-dimensional array.")
    return periods


def uses_official_period_grid(periods_s: FloatArray) -> bool:
    """Return whether a period array is exactly the published 1.6-7.8 s grid."""
    return periods_s.shape == OFFICIAL_PERIODS_S.shape and bool(
        np.array_equal(periods_s, OFFICIAL_PERIODS_S)
    )


def summarize_bands(
    sva_cm_s: FloatArray,
    periods_s: FloatArray,
) -> tuple[LongPeriodBandResult, ...]:
    """Return the per-band maxima and classes JMA publishes alongside the total.

    Only defined for the official period grid, whose bands group periods by
    their integer second (1.6-1.8 s, 2.0-2.8 s, ... 7.0-7.8 s). Returns an
    empty tuple for any other grid rather than inventing bands for it.
    """
    if not uses_official_period_grid(periods_s):
        return ()
    bands = []
    deciseconds = np.array(PERIOD_DECISECONDS)
    for band in PERIOD_BANDS:
        mask = np.array([band_of_period_deciseconds(d) == band for d in deciseconds])
        peak = float(np.max(sva_cm_s[mask]))
        bands.append(
            LongPeriodBandResult(
                band_second=band,
                period_range_s=band_period_range_s(band),
                max_sva_cm_s=peak,
                long_period_class=classify_long_period(peak),
            )
        )
    return tuple(bands)


def warn_if_non_reference_rate(sampling_rate_hz: float, stacklevel: int) -> None:
    """Warn when the published high-pass constants do not apply to a rate."""
    if sampling_rate_hz == REFERENCE_SAMPLING_RATE_HZ:
        return
    warnings.warn(
        "The published long-period high-pass constants are stated for 100 Hz. "
        f"They are being re-derived for {sampling_rate_hz:g} Hz from the analog "
        "prototype behind them; the result is no longer the published reference path.",
        NonstandardSamplingRateWarning,
        stacklevel=stacklevel,
    )


def meets_reference_conditions(
    *,
    high_pass: bool,
    published_high_pass: bool,
    sampling_rate_hz: float,
    periods_s: FloatArray,
    damping_ratio: float,
    component_count: int,
) -> bool:
    """Return whether every condition of the published definition is in force.

    One predicate for both the batch and streaming paths, so a result can never
    claim to be the reference calculation in one and not the other.
    """
    return (
        high_pass
        and published_high_pass
        and sampling_rate_hz == REFERENCE_SAMPLING_RATE_HZ
        and uses_official_period_grid(periods_s)
        and damping_ratio == OFFICIAL_DAMPING_RATIO
        and component_count == 2
    )


def calculate_long_period_class(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    component_axis: int = -1,
    damping_ratio: float = OFFICIAL_DAMPING_RATIO,
    periods_s: npt.ArrayLike | None = None,
    high_pass: bool = True,
    warn_nonstandard_rate: bool = True,
    retain_response: bool = False,
) -> LongPeriodResult:
    """Calculate the JMA long-period ground motion class for one record.

    The published sequence: apply the 20-second second-order high-pass to each
    horizontal acceleration component, drive a bank of damped single-degree-of-
    freedom oscillators with the filtered acceleration using the linear
    acceleration method, add the ground velocity to each relative velocity
    response to obtain the absolute velocity response, combine the two
    horizontal components as a vector at every instant, take the maximum over
    time for each period, and classify the largest of those.

    Parameters
    ----------
    acceleration:
        Two horizontal acceleration components, shaped ``(samples, 2)``. The
        vertical component is not part of this definition; see
        :func:`as_horizontal_acceleration`.
    sampling_rate_hz:
        100 Hz is the reference condition the published filter constants are
        stated for. Other rates are accepted with a warning and re-derive the
        high-pass from the analog prototype behind those constants.
    unit:
        ``"gal"``, ``"m/s^2"``, or ``"g"``. Sva is always returned in cm/s.
    damping_ratio:
        5 percent for the published class definition. Other values are
        available for study but are not the JMA quantity.
    periods_s:
        Defaults to the official 1.6-7.8 s grid in 0.2 s steps. Per-band
        results are reported only for that grid.
    high_pass:
        Apply the 20-second high-pass. Leave this on for the JMA definition;
        turning it off is for isolating the oscillator response in tests and
        experiments.
    retain_response:
        Keep the full ``(samples, periods)`` horizontal absolute velocity.
        Off by default because, unlike the intensity result, this is one array
        per period: a five-minute record at 100 Hz costs about 7 MB.

    Notes
    -----
    No offset removal, detrending, or tapering is applied. The 20-second
    high-pass is part of the published definition and is applied for that
    reason alone, not as a general baseline correction.
    """
    total_started = time.perf_counter()
    rate = validate_sampling_rate(sampling_rate_hz, warn_nonstandard=False)
    if warn_nonstandard_rate:
        warn_if_non_reference_rate(rate, stacklevel=3)
    parsed_unit = AccelerationUnit.parse(unit)
    values = as_horizontal_acceleration(acceleration, component_axis=component_axis)
    values_gal = to_gal(values, parsed_unit, copy=False)
    periods = resolve_periods(periods_s)

    high_pass_started = time.perf_counter()
    if high_pass:
        design = design_high_pass(rate)
        state = high_pass_initial_state(design)
        filtered, _ = apply_high_pass(design, values_gal, state)
        published_high_pass = design.is_published_reference
    else:
        filtered = values_gal
        published_high_pass = False
    high_pass_elapsed = time.perf_counter() - high_pass_started

    response_started = time.perf_counter()
    bank = design_oscillator_bank(periods, damping_ratio, rate)
    response = ResponseState(bank)
    collected = response.advance(filtered, collect=retain_response)
    response_elapsed = time.perf_counter() - response_started

    sva = response.running_max.copy()
    critical_index = int(np.argmax(sva))
    max_sva = float(sva[critical_index])
    reference_conditions_met = meets_reference_conditions(
        high_pass=high_pass,
        published_high_pass=published_high_pass,
        sampling_rate_hz=rate,
        periods_s=periods,
        damping_ratio=damping_ratio,
        component_count=values_gal.shape[1],
    )
    return LongPeriodResult(
        sva_cm_s=sva,
        periods_s=periods,
        max_sva_cm_s=max_sva,
        critical_period_s=float(periods[critical_index]),
        long_period_class=classify_long_period(max_sva),
        bands=summarize_bands(sva, periods),
        damping_ratio=float(damping_ratio),
        sampling_rate_hz=rate,
        sample_count=values_gal.shape[0],
        component_count=values_gal.shape[1],
        high_pass_applied=high_pass,
        reference_conditions_met=reference_conditions_met,
        absolute_velocity_cm_s=collected,
        timing=LongPeriodTiming(
            high_pass_s=high_pass_elapsed,
            response_s=response_elapsed,
            total_s=time.perf_counter() - total_started,
        ),
    )
