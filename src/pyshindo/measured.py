"""Frequency-domain reference calculation of instrumental seismic intensity."""

from __future__ import annotations

import math
import time

import numpy as np

from .duration import DurationSamplePolicy, duration_sample_count, duration_threshold
from .filters.jma import apply_jma_filter_fft
from .models import MeasuredIntensityResult, MeasuredIntensityTiming
from .scale import classify_intensity, intensity_from_acceleration, report_intensity
from .signal import component_peak_acceleration, peak_ground_acceleration, vector_resultant
from .units import AccelerationUnit, ArrayLike, to_gal
from .validation import as_acceleration_array, validate_sampling_rate


def calculate_measured_intensity(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    duration_s: float = 0.3,
    duration_policy: DurationSamplePolicy = "ceil",
    component_axis: int = -1,
    allow_fewer_components: bool = False,
    warn_nonstandard_rate: bool = True,
    workers: int | None = None,
    retain_intermediates: bool = True,
) -> MeasuredIntensityResult:
    """Calculate instrumental seismic intensity with the FFT reference method.

    The implementation follows the JMA sequence: Fourier transform each
    acceleration component, multiply by the published three-factor amplitude
    response, inverse transform, combine the filtered components by Euclidean
    norm, select the amplitude corresponding to 0.3 seconds of cumulative
    exceedance, and apply the logarithmic intensity conversion.

    Parameters
    ----------
    acceleration:
        Three acceleration components. One or two components can be enabled for
        controlled experiments with ``allow_fewer_components=True``.
    sampling_rate_hz:
        Sampling frequency. 100 Hz is the standard reference condition. Other
        positive rates are accepted with a warning and require independent
        validation.
    unit:
        ``"gal"``, ``"m/s^2"``, or ``"g"``.
    duration_s:
        Cumulative exceedance duration, normally 0.3 seconds.
    duration_policy:
        Rule used only when ``duration_s * sampling_rate_hz`` is not integral.
    retain_intermediates:
        Keep filtered waveforms, resultant acceleration, and filter response in
        the returned result. Disable this for lower peak memory consumption.

    Notes
    -----
    No mean removal, detrending, tapering, or zero padding is applied. These
    operations are available as explicit utilities because silently applying
    them would change the reference procedure.
    """
    total_started = time.perf_counter()
    rate = validate_sampling_rate(
        sampling_rate_hz,
        warn_nonstandard=warn_nonstandard_rate,
        stacklevel=2,
    )
    parsed_unit = AccelerationUnit.parse(unit)
    values = as_acceleration_array(
        acceleration,
        component_axis=component_axis,
        allow_fewer_components=allow_fewer_components,
    )
    values_gal = to_gal(values, parsed_unit, copy=False)
    samples = duration_sample_count(duration_s, rate, policy=duration_policy)

    filter_started = time.perf_counter()
    filtered, frequency, response = apply_jma_filter_fft(values_gal, rate, workers=workers)
    filter_elapsed = time.perf_counter() - filter_started

    threshold_started = time.perf_counter()
    resultant = vector_resultant(filtered)
    threshold = duration_threshold(resultant, samples)
    threshold_elapsed = time.perf_counter() - threshold_started

    raw = intensity_from_acceleration(threshold)
    reported = report_intensity(raw)
    scale = classify_intensity(reported)
    reference_conditions_met = (
        values.shape[1] == 3
        and math.isclose(rate, 100.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(duration_s, 0.3, rel_tol=0.0, abs_tol=1e-12)
        and duration_policy == "ceil"
    )
    timing = MeasuredIntensityTiming(
        fft_filter_s=filter_elapsed,
        duration_threshold_s=threshold_elapsed,
        total_s=time.perf_counter() - total_started,
    )

    return MeasuredIntensityResult(
        intensity_raw=raw,
        intensity=reported,
        scale=scale,
        threshold_acceleration_gal=threshold,
        duration_samples=samples,
        effective_duration_s=samples / rate,
        sampling_rate_hz=rate,
        sample_count=values.shape[0],
        component_count=values.shape[1],
        input_unit=parsed_unit,
        input_component_pga_gal=component_peak_acceleration(values_gal),
        input_pga_gal=peak_ground_acceleration(values_gal),
        filtered_component_pga_gal=component_peak_acceleration(filtered),
        filtered_pga_gal=float(np.max(resultant)),
        filtered_acceleration_gal=filtered if retain_intermediates else None,
        resultant_acceleration_gal=resultant if retain_intermediates else None,
        frequency_hz=frequency if retain_intermediates else None,
        filter_response=response if retain_intermediates else None,
        reference_conditions_met=reference_conditions_met,
        timing=timing,
    )


def measured_intensity(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    reported: bool = True,
    **kwargs: object,
) -> float:
    """Return only the scalar instrumental intensity for a complete record.

    Use :func:`calculate_measured_intensity` when filtered waveforms, threshold
    acceleration, PGA, or other diagnostics are needed.
    """
    result = calculate_measured_intensity(
        acceleration,
        sampling_rate_hz,
        unit=unit,
        retain_intermediates=False,
        **kwargs,
    )
    return result.intensity if reported else result.intensity_raw
