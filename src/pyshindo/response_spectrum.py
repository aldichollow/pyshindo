"""General elastic single-degree-of-freedom response spectrum: Sd, Sv, and PSA.

The relative displacement, relative velocity, and pseudo-acceleration
response of a damped single-degree-of-freedom oscillator, for any period
grid and any damping ratio -- the shared computation behind both
:mod:`pyshindo.long_period` (JMA's absolute velocity response, 5 percent
damping, the official 1.6-7.8 s grid) and :mod:`pyshindo.spectrum_intensity`
(Housner's relative velocity response, 20 percent damping, 0.1-2.5 s), with
neither of those two features' own conventions applied here.

Unlike both of them, this function does not choose a damping ratio or a
period grid -- there is no house convention to default to for a genuinely
general-purpose spectrum, so both are required arguments rather than
defaulting to either existing feature's choice.

This is the *relative* response only: the response of the oscillator's mass
relative to the ground it is mounted on, with no ground motion added and no
horizontal-component combination applied. :mod:`pyshindo.long_period` adds
ground velocity to get JMA's absolute velocity response and then combines
two horizontal components as a vector; neither step is generic enough to
belong here, and folding either in silently would mean this function was
quietly choosing between the relative and absolute conventions the two
existing features actually use, rather than leaving that choice explicit.
A caller wanting an absolute response spectrum can add
:func:`pyshindo.velocity.integrate_to_velocity` (already public) to
``sv_time_series_cm_s`` sample by sample -- set ``retain_time_series`` to get
that series -- and take the maximum of the sum; the peak of a sum is not the
sum of the peaks, so this cannot be done from ``sv_cm_s`` alone.

Pseudo-acceleration (``psa_gal = omega**2 * sd_cm``) is the standard
earthquake-engineering approximation to the peak absolute acceleration
response, exact only in the limit of zero damping. True absolute
acceleration response (relative acceleration plus ground acceleration) is
*not* implemented: it needs its own closed-form transfer function and a
primary-source check on its definition, neither of which has been done, so
it is left out rather than guessed at.

The oscillator solver itself is :mod:`pyshindo._spectral_response`, shared
with :mod:`pyshindo.long_period` and :mod:`pyshindo.spectrum_intensity`; see
that module's docstring for the primary sources behind the recurrence and
its coefficients.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._spectral_response import (
    design_oscillator_bank,
    relative_displacement_response,
    relative_velocity_response,
)
from .units import AccelerationUnit, ArrayLike, FloatArray, to_gal
from .validation import as_acceleration_array, validate_sampling_rate


@dataclass(frozen=True, slots=True)
class ResponseSpectrumTiming:
    """Wall-clock timing for one response spectrum calculation.

    ``response_s``: the oscillator bank design and response calculation.
    ``total_s``: the complete call. Measured with :func:`time.perf_counter`.
    """

    response_s: float
    total_s: float


@dataclass(frozen=True, slots=True)
class ResponseSpectrumResult:
    """Detailed output of the general elastic response spectrum calculation.

    ``sd_cm`` and ``sv_cm_s`` are the relative displacement and velocity
    response spectra themselves -- the peak response at each period, shaped
    ``(periods, components)`` -- and are always present. ``psa_gal`` is the
    pseudo-acceleration spectrum derived from ``sd_cm`` (see the module
    docstring for what it approximates and what it does not).
    ``sd_time_series_cm`` and ``sv_time_series_cm_s`` are the much larger
    full per-sample responses, ``(samples, periods, components)``, kept only
    when ``retain_time_series`` was set.
    """

    sd_cm: FloatArray
    sv_cm_s: FloatArray
    psa_gal: FloatArray
    periods_s: FloatArray
    damping_ratio: float
    sampling_rate_hz: float
    sample_count: int
    component_count: int
    sd_time_series_cm: FloatArray | None
    sv_time_series_cm_s: FloatArray | None
    timing: ResponseSpectrumTiming

    @property
    def record_duration_s(self) -> float:
        """Return sample count divided by sampling rate."""
        return self.sample_count / self.sampling_rate_hz


def calculate_response_spectrum(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    damping_ratio: float,
    periods_s: npt.ArrayLike,
    component_axis: int = -1,
    retain_time_series: bool = False,
) -> ResponseSpectrumResult:
    """Calculate the relative elastic response spectrum for each acceleration component.

    Parameters
    ----------
    acceleration:
        One to three acceleration components. No ground motion is added and
        no component combination is applied; every component is returned on
        its own, matching :func:`pyshindo.spectrum_intensity.calculate_spectrum_intensity`'s
        choice to leave combination to the caller.
    sampling_rate_hz:
        No published constants are tied to a specific rate here; the
        linear-acceleration-method solver is exact at any rate for a
        sufficiently smooth input.
    unit:
        ``"gal"``, ``"m/s^2"``, or ``"g"``. Results are always in cm (``sd_cm``),
        cm/s (``sv_cm_s``), and gal (``psa_gal``).
    damping_ratio:
        Required, not defaulted: JMA's long-period class uses 5 percent,
        Housner's SI value uses 20 percent, and a general-purpose function has
        no house convention of its own to fall back to.
    periods_s:
        Required, not defaulted, for the same reason as ``damping_ratio``.
    retain_time_series:
        Keep the full ``(samples, periods, components)`` per-sample relative
        displacement and velocity, in ``sd_time_series_cm`` and
        ``sv_time_series_cm_s``. Off by default because it is one array per
        period rather than one scalar. The peak-per-period spectra,
        ``sd_cm`` and ``sv_cm_s``, are always returned regardless of this flag.

    Notes
    -----
    No baseline correction, filtering, ground-motion addition, or component
    combination is applied.
    """
    total_started = time.perf_counter()
    rate = validate_sampling_rate(sampling_rate_hz, warn_nonstandard=False)
    parsed_unit = AccelerationUnit.parse(unit)
    values = as_acceleration_array(
        acceleration,
        component_axis=component_axis,
        warn_fewer_components=False,
    )
    values_gal = to_gal(values, parsed_unit, copy=False)
    periods = np.asarray(periods_s, dtype=np.float64)

    response_started = time.perf_counter()
    bank = design_oscillator_bank(periods, damping_ratio, rate)
    displacement_peaks, displacement_series = relative_displacement_response(
        bank, values_gal, collect=retain_time_series
    )
    velocity_peaks, velocity_series = relative_velocity_response(
        bank, values_gal, collect=retain_time_series
    )
    response_elapsed = time.perf_counter() - response_started

    omega = 2.0 * np.pi / periods
    psa_gal = (omega**2)[:, np.newaxis] * displacement_peaks

    return ResponseSpectrumResult(
        sd_cm=displacement_peaks,
        sv_cm_s=velocity_peaks,
        psa_gal=psa_gal,
        periods_s=periods,
        damping_ratio=float(damping_ratio),
        sampling_rate_hz=rate,
        sample_count=values_gal.shape[0],
        component_count=values_gal.shape[1],
        sd_time_series_cm=displacement_series,
        sv_time_series_cm_s=velocity_series,
        timing=ResponseSpectrumTiming(
            response_s=response_elapsed,
            total_s=time.perf_counter() - total_started,
        ),
    )
