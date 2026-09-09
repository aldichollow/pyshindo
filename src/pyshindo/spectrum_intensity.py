"""Housner's spectrum intensity (SI value): a relative-velocity response-spectrum index.

A different quantity from both instrumental seismic intensity and JMA's
long-period ground motion class. SI is the period-averaged relative velocity
response of a damped single-degree-of-freedom oscillator, over the period
range where ordinary buildings' natural periods fall:

    SI = (1 / 2.4) * integral[0.1, 2.5] Sv(T, h) dT

``Sv(T, h)`` is the *relative* velocity response spectrum at damping ratio
``h`` -- not the absolute velocity response :mod:`pyshindo.long_period` uses,
and not combined across horizontal components. Practical use (road-bridge
design manuals, gas-supply and elevator seismic shutoff) reports SI
separately per horizontal component; this module follows the same
convention and does not enforce it, matching how
:func:`pyshindo.velocity.peak_ground_velocity` leaves the same choice of
combining components to the caller.

The oscillator response itself is not specific to this definition -- it is
the same linear-acceleration-method solver
:mod:`pyshindo.long_period` uses, from :mod:`pyshindo._spectral_response`,
just without the ground-velocity addition or component combination that
turn a relative response into JMA's absolute one.

The published damping ratio for SI, h = 0.20, and the integration range and
2.4 s normalization are stated with a formula number in a public
prefectural road-bridge design manual, attributed to 大崎順彦: 鳥取県
道路橋梁設計マニュアル 3-6「スペクトル強度SI値」(式 3-11).
https://www.pref.tottori.lg.jp/secure/198289/dourokkyouryou03-6.pdf

The index itself originates with Housner, G.W. (1959), "Behavior of
Structures During Earthquakes," Journal of the Engineering Mechanics
Division, ASCE, 85(EM4), 109-129, building on the earlier response-spectrum
method of Housner, G.W. (1952), "Spectrum Intensities of Strong Motion
Earthquakes."

No standard integration grid is published for the 0.1-2.5 s range; the
default 121-point linear grid and trapezoidal rule used here were chosen by
checking convergence directly: 121 points agree with a 769-point grid to
about 2e-5 relative, which is well past the precision SI values are reported
to in practice (whole or tenths of cm/s).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.integrate import trapezoid

from ._spectral_response import design_oscillator_bank, relative_velocity_response
from .units import AccelerationUnit, ArrayLike, FloatArray, to_gal
from .validation import as_acceleration_array, validate_sampling_rate

LOWER_PERIOD_S: Final = 0.1
UPPER_PERIOD_S: Final = 2.5
INTEGRATION_WIDTH_S: Final = UPPER_PERIOD_S - LOWER_PERIOD_S

DEFAULT_DAMPING_RATIO: Final = 0.20
DEFAULT_PERIOD_COUNT: Final = 121


def default_periods_s(count: int = DEFAULT_PERIOD_COUNT) -> FloatArray:
    """Return the default 0.1-2.5 s linear period grid used to integrate Sv."""
    if count < 2:
        raise ValueError("count must be at least two.")
    return np.linspace(LOWER_PERIOD_S, UPPER_PERIOD_S, count)


@dataclass(frozen=True, slots=True)
class SpectrumIntensityTiming:
    """Wall-clock timing for one spectrum intensity calculation.

    ``response_s``: the oscillator bank design and response calculation.
    ``total_s``: the complete call. Measured with :func:`time.perf_counter`.
    """

    response_s: float
    total_s: float


@dataclass(frozen=True, slots=True)
class SpectrumIntensityResult:
    """Detailed output of the spectrum intensity calculation.

    ``sv_cm_s`` is the relative velocity response spectrum itself -- the
    peak response at each period, shaped ``(periods, components)`` -- and is
    always present; it is what ``si_cm_s`` integrates and what a
    response-spectrum plot needs. ``sv_time_series_cm_s`` is the much larger
    full per-sample response, ``(samples, periods, components)``, kept only
    when ``retain_spectrum`` was set.
    """

    si_cm_s: FloatArray
    sv_cm_s: FloatArray
    periods_s: FloatArray
    damping_ratio: float
    sampling_rate_hz: float
    sample_count: int
    component_count: int
    sv_time_series_cm_s: FloatArray | None
    timing: SpectrumIntensityTiming

    @property
    def record_duration_s(self) -> float:
        """Return sample count divided by sampling rate."""
        return self.sample_count / self.sampling_rate_hz


def calculate_spectrum_intensity(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    damping_ratio: float = DEFAULT_DAMPING_RATIO,
    periods_s: npt.ArrayLike | None = None,
    component_axis: int = -1,
    retain_spectrum: bool = False,
) -> SpectrumIntensityResult:
    """Calculate Housner's spectrum intensity (SI) for each acceleration component.

    Parameters
    ----------
    acceleration:
        One to three acceleration components. Practical use reports SI per
        horizontal component (pass the two horizontal columns explicitly,
        for example ``acceleration[:, :2]``, for NS/EW data); this function
        does not enforce that, the same choice
        :func:`pyshindo.velocity.peak_ground_velocity` leaves open.
    sampling_rate_hz:
        No published constants are tied to a specific rate here, unlike
        :mod:`pyshindo.long_period`; the linear-acceleration-method solver
        is exact at any rate for a sufficiently smooth input.
    unit:
        ``"gal"``, ``"m/s^2"``, or ``"g"``. SI is always returned in cm/s.
    damping_ratio:
        0.20 is the published value for SI specifically (see the module
        docstring); it is not a general-purpose structural damping value and
        is unrelated to the 5 percent used for JMA's long-period class.
    periods_s:
        Defaults to :func:`default_periods_s`, a 121-point linear grid over
        0.1-2.5 s. A custom grid must still span periods a caller wants
        integrated; SI outside that convention is not the published index.
    retain_spectrum:
        Keep the full ``(samples, periods, components)`` per-sample relative
        velocity response, in ``sv_time_series_cm_s``. Off by default because
        it is one array per period rather than one scalar. The peak-per-period
        spectrum, ``sv_cm_s``, is always returned regardless of this flag.

    Notes
    -----
    No baseline correction, filtering, or component combination is applied.
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
    periods = default_periods_s() if periods_s is None else np.asarray(periods_s, dtype=np.float64)

    response_started = time.perf_counter()
    bank = design_oscillator_bank(periods, damping_ratio, rate)
    peaks, series = relative_velocity_response(bank, values_gal, collect=retain_spectrum)
    response_elapsed = time.perf_counter() - response_started

    si = trapezoid(peaks, periods, axis=0) / INTEGRATION_WIDTH_S

    return SpectrumIntensityResult(
        si_cm_s=si,
        sv_cm_s=peaks,
        periods_s=periods,
        damping_ratio=float(damping_ratio),
        sampling_rate_hz=rate,
        sample_count=values_gal.shape[0],
        component_count=values_gal.shape[1],
        sv_time_series_cm_s=series,
        timing=SpectrumIntensityTiming(
            response_s=response_elapsed,
            total_s=time.perf_counter() - total_started,
        ),
    )
