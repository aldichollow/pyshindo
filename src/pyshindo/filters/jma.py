"""Frequency-domain filter used for the reference intensity calculation.

Reference
---------
Japan Meteorological Agency, "Calculation of instrumental seismic intensity".
https://www.jma.go.jp/jma/kishou/know/jishin/kyoshin/kaisetsu/calc_sindo.html
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import fft

from ..units import AccelerationUnit, FloatArray, to_gal


@dataclass(frozen=True, slots=True)
class JMAFilterComponents:
    """The three published amplitude-response factors and their product."""

    frequency_hz: FloatArray
    period_effect: FloatArray
    high_cut: FloatArray
    low_cut: FloatArray
    combined: FloatArray


def jma_filter_components(frequency_hz: npt.ArrayLike) -> JMAFilterComponents:
    """Evaluate the published JMA intensity-filter amplitude response.

    Parameters
    ----------
    frequency_hz:
        Non-negative frequencies in hertz.

    Notes
    -----
    The combined response at zero frequency is defined as zero by continuity
    of the low-cut and period-effect product. The period-effect factor itself
    is returned as positive infinity at zero, matching its analytical form.
    """
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(~np.isfinite(frequency)) or np.any(frequency < 0.0):
        raise ValueError("frequency_hz must contain finite, non-negative values.")

    positive = frequency > 0.0
    period_effect = np.full_like(frequency, np.inf)
    period_effect[positive] = np.reciprocal(np.sqrt(frequency[positive]))

    squared = (frequency / 10.0) ** 2
    polynomial = 0.000155 * squared + 0.00134
    polynomial = polynomial * squared + 0.009664
    polynomial = polynomial * squared + 0.0557
    polynomial = polynomial * squared + 0.241
    polynomial = polynomial * squared + 0.694
    polynomial = polynomial * squared + 1.0
    high_cut = np.reciprocal(np.sqrt(polynomial))

    # -expm1(-x) is accurate when x is close to zero.
    low_cut = np.sqrt(-np.expm1(-((frequency / 0.5) ** 3)))
    combined = np.zeros_like(frequency)
    combined[positive] = (
        period_effect[positive] * high_cut[positive] * low_cut[positive]
    )

    return JMAFilterComponents(
        frequency_hz=frequency,
        period_effect=period_effect,
        high_cut=high_cut,
        low_cut=low_cut,
        combined=combined,
    )


def jma_filter_response(frequency_hz: npt.ArrayLike) -> FloatArray:
    """Return only the combined JMA intensity-filter amplitude response."""
    return jma_filter_components(frequency_hz).combined


@dataclass(frozen=True, slots=True)
class JMAFilterResult:
    """Output of one FFT pass of the published JMA intensity filter.

    A dataclass rather than a bare tuple so that a future addition here does
    not break unpacking at every call site.
    """

    filtered_acceleration_gal: FloatArray
    frequency_hz: FloatArray
    response: FloatArray


def apply_jma_filter_fft(
    acceleration: npt.ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    workers: int | None = None,
) -> JMAFilterResult:
    """Filter acceleration using an FFT, the published response, and an inverse FFT.

    Parameters
    ----------
    acceleration:
        An array shaped ``(samples, components)``. Converted to gal first, so
        the filtered output is always gal regardless of the input unit.
    sampling_rate_hz:
        Sampling frequency in hertz.
    unit:
        ``"gal"``, ``"m/s^2"``, or ``"g"``.
    workers:
        Optional number of FFT worker threads accepted by :mod:`scipy.fft`.

    Notes
    -----
    The transform length equals the record length. No detrending, tapering, or
    padding is applied implicitly because each would define a different signal
    processing procedure. Such preprocessing can be performed explicitly with
    :mod:`pyshindo.signal` when required by a data source.
    """
    parsed_unit = AccelerationUnit.parse(unit)
    values = np.asarray(acceleration, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("acceleration must have shape (samples, components).")
    if values.shape[0] == 0:
        raise ValueError("acceleration must contain at least one sample.")
    if not np.all(np.isfinite(values)):
        raise ValueError("acceleration contains non-finite values.")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and greater than zero.")
    values_gal = to_gal(values, parsed_unit, copy=False)

    sample_count = values_gal.shape[0]
    frequency = fft.rfftfreq(sample_count, d=1.0 / sampling_rate_hz)
    response = jma_filter_response(frequency)
    spectrum = fft.rfft(values_gal, axis=0, workers=workers)
    spectrum *= response[:, np.newaxis]
    filtered = fft.irfft(spectrum, n=sample_count, axis=0, workers=workers)
    return JMAFilterResult(
        filtered_acceleration_gal=np.ascontiguousarray(filtered, dtype=np.float64),
        frequency_hz=np.asarray(frequency, dtype=np.float64),
        response=response,
    )
