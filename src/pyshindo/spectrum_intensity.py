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

from ._spectral_response import OscillatorBank, design_oscillator_bank, relative_velocity_response
from .exceptions import InvalidAccelerationError
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
    when ``retain_velocity_time_series`` was set.
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
    retain_velocity_time_series: bool = False,
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
    retain_velocity_time_series:
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
    peaks, series = relative_velocity_response(
        bank, values_gal, collect=retain_velocity_time_series
    )
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


@dataclass(frozen=True, slots=True)
class SpectrumIntensityUpdate:
    """Cumulative state after one streaming chunk or sample.

    The values are the exact complete-record ``Sv``/SI for the part of the
    record seen so far, not a rolling window -- see
    :class:`SpectrumIntensityEstimator` for why.
    """

    sample_index: int
    sample_count: int
    si_so_far_cm_s: FloatArray
    elapsed_s: float


class SpectrumIntensityEstimator:
    """Track Housner's spectrum intensity (SI) as samples arrive.

    ``Sv`` is a period-by-period running maximum of a relative-velocity
    response (see the module docstring), and SI integrates that maximum, so
    the streaming SI value is a cumulative maximum over everything seen so
    far, not a rolling window: it can only rise. This is exactly the pattern
    :class:`pyshindo.long_period.LongPeriodEstimator` uses for JMA's
    long-period class.

    Unlike :class:`pyshindo.realtime.RealtimeIntensityEstimator`'s 60-second
    window -- which translates Kunugi et al.'s own published real-time
    approximation filter into a streaming form -- neither JMA's long-period
    class nor this package's SI value has a published real-time
    specification. JMA's long-period class is itself only defined as a
    whole-record maximum; JMA has not published a streaming algorithm for
    it, so :class:`~pyshindo.long_period.LongPeriodEstimator`'s cumulative-
    maximum behavior is that package's own engineering choice for computing
    a batch-defined quantity incrementally, not a translation of an official
    real-time spec -- and the same is true here, one level further removed,
    since SI itself is not JMA's quantity to begin with.

    Memory is O(periods * components) regardless of how long the stream runs.
    The default 121-point grid is about 3.8 times :class:`LongPeriodEstimator`'s
    32 -- measured rather than assumed, this costs about 1.1 times as much per
    sample, not 3.8 times: the per-sample step is already vectorized across
    periods with NumPy, so per-call overhead dominates over the extra
    arithmetic far more than the period count alone would suggest.
    """

    def __init__(
        self,
        sampling_rate_hz: float = 100.0,
        *,
        unit: str | AccelerationUnit = AccelerationUnit.GAL,
        damping_ratio: float = DEFAULT_DAMPING_RATIO,
        periods_s: npt.ArrayLike | None = None,
    ) -> None:
        # No warning for a non-100 Hz rate, and no flag to ask for one: as
        # calculate_spectrum_intensity documents, no published constant here is
        # tied to a rate, and the solver is exact at any rate for a smooth
        # input. An earlier version warned from this constructor only, which
        # made the streaming and batch paths disagree about the same record.
        rate = validate_sampling_rate(sampling_rate_hz, warn_nonstandard=False)
        self._rate = rate
        self._dt = 1.0 / rate
        self._unit = AccelerationUnit.parse(unit)
        periods = (
            default_periods_s() if periods_s is None else np.array(periods_s, dtype=np.float64)
        )
        periods.setflags(write=False)
        self._periods = periods
        self._damping_ratio = float(damping_ratio)
        self._bank: OscillatorBank = design_oscillator_bank(periods, self._damping_ratio, rate)
        self._component_count: int | None = None
        self._displacement: FloatArray | None = None
        self._velocity: FloatArray | None = None
        self._previous_acceleration: FloatArray | None = None
        self._running_max: FloatArray | None = None
        self._sample_count = 0

    @property
    def sampling_rate_hz(self) -> float:
        """Return the configured sampling rate."""
        return self._rate

    @property
    def periods_s(self) -> FloatArray:
        """Return the period grid being evaluated."""
        return self._periods

    @property
    def damping_ratio(self) -> float:
        """Return the configured damping ratio."""
        return self._damping_ratio

    @property
    def sample_count(self) -> int:
        """Return the number of samples processed so far."""
        return self._sample_count

    @property
    def component_count(self) -> int | None:
        """Return the locked component count, or ``None`` before first input."""
        return self._component_count

    @property
    def sv_cm_s(self) -> FloatArray:
        """Return the per-period, per-component Sv maxima so far, in cm/s."""
        if self._running_max is None:
            return np.zeros((self._bank.period_count, 0), dtype=np.float64)
        return self._running_max.copy()

    @property
    def si_cm_s(self) -> FloatArray:
        """Return the SI value implied by everything processed so far, per component."""
        if self._running_max is None:
            return np.zeros(0, dtype=np.float64)
        return trapezoid(self._running_max, self._periods, axis=0) / INTEGRATION_WIDTH_S

    def _initialize_stream(self, component_count: int) -> None:
        if not 1 <= component_count <= 3:
            raise InvalidAccelerationError("A stream must contain one, two, or three components.")
        self._component_count = component_count
        shape = (self._bank.period_count, component_count)
        self._displacement = np.zeros(shape, dtype=np.float64)
        self._velocity = np.zeros(shape, dtype=np.float64)
        self._previous_acceleration = np.zeros(component_count, dtype=np.float64)
        self._running_max = np.zeros(shape, dtype=np.float64)

    def _prepare(self, acceleration: ArrayLike, component_axis: int) -> FloatArray:
        values = as_acceleration_array(
            acceleration,
            component_axis=component_axis,
            warn_fewer_components=False,
        )
        return to_gal(values, self._unit, copy=False)

    def _start_record(self, sample: FloatArray) -> None:
        assert self._displacement is not None
        assert self._velocity is not None
        assert self._previous_acceleration is not None
        assert self._running_max is not None
        self._displacement[:] = 0.0
        self._velocity[:] = -sample * self._dt
        self._previous_acceleration[:] = sample
        self._sample_count = 1
        np.maximum(self._running_max, np.abs(self._velocity), out=self._running_max)

    def _advance(self, values_gal: FloatArray) -> None:
        # values_gal always has at least one sample here: it comes from
        # as_acceleration_array via _prepare(), which itself rejects an
        # empty chunk before this is ever reached.
        samples = values_gal.shape[0]
        if self._component_count is None:
            self._initialize_stream(values_gal.shape[1])
        elif values_gal.shape[1] != self._component_count:
            raise ValueError(
                f"The stream was initialized with {self._component_count} components, but "
                f"this chunk contains {values_gal.shape[1]}."
            )
        assert self._displacement is not None
        assert self._velocity is not None
        assert self._previous_acceleration is not None
        assert self._running_max is not None

        start = 0
        if self._sample_count == 0:
            self._start_record(values_gal[0])
            start = 1
        if start >= samples:
            return

        bank = self._bank
        a11 = bank.a11[:, np.newaxis]
        a12 = bank.a12[:, np.newaxis]
        a21 = bank.a21[:, np.newaxis]
        a22 = bank.a22[:, np.newaxis]
        b11 = bank.b11[:, np.newaxis]
        b12 = bank.b12[:, np.newaxis]
        b21 = bank.b21[:, np.newaxis]
        b22 = bank.b22[:, np.newaxis]

        displacement = self._displacement
        velocity = self._velocity
        previous = self._previous_acceleration
        running_max = self._running_max
        for index in range(start, samples):
            current = values_gal[index]
            next_displacement = (
                a11 * displacement + a12 * velocity + b11 * previous + b12 * current
            )
            next_velocity = a21 * displacement + a22 * velocity + b21 * previous + b22 * current
            displacement[:] = next_displacement
            velocity[:] = next_velocity
            previous[:] = current
            np.maximum(running_max, np.abs(velocity), out=running_max)

        self._sample_count += samples - start

    def process(
        self, acceleration: ArrayLike, *, component_axis: int = -1
    ) -> SpectrumIntensityUpdate:
        """Process a chunk of acceleration and return the state so far."""
        started = time.perf_counter()
        values_gal = self._prepare(acceleration, component_axis)
        self._advance(values_gal)
        return self._update(time.perf_counter() - started)

    def process_sample(self, sample: ArrayLike) -> SpectrumIntensityUpdate:
        """Process one component vector (one to three values).

        Equivalent to :meth:`process` with a single row; provided so a caller
        driving the estimator one sample at a time does not have to reshape.
        """
        started = time.perf_counter()
        values = np.asarray(sample, dtype=np.float64)
        if values.ndim != 1 or not 1 <= values.size <= 3:
            raise InvalidAccelerationError(
                "process_sample expects one, two, or three component values."
            )
        values_gal = self._prepare(values[np.newaxis, :], -1)
        self._advance(values_gal)
        return self._update(time.perf_counter() - started)

    def _update(self, elapsed_s: float) -> SpectrumIntensityUpdate:
        return SpectrumIntensityUpdate(
            sample_index=self._sample_count - 1,
            sample_count=self._sample_count,
            si_so_far_cm_s=self.si_cm_s,
            elapsed_s=elapsed_s,
        )

    def result(self) -> SpectrumIntensityResult:
        """Return a complete-record result for everything processed so far.

        The same shape of answer :func:`calculate_spectrum_intensity` gives,
        so a streaming run can be reported or compared without a second pass.
        ``sv_time_series_cm_s`` is always ``None``: the estimator keeps only
        running maxima.
        """
        component_count = self._component_count or 0
        return SpectrumIntensityResult(
            si_cm_s=self.si_cm_s,
            sv_cm_s=self.sv_cm_s,
            periods_s=self._periods,
            damping_ratio=self._damping_ratio,
            sampling_rate_hz=self._rate,
            sample_count=self._sample_count,
            component_count=component_count,
            sv_time_series_cm_s=None,
            timing=SpectrumIntensityTiming(response_s=0.0, total_s=0.0),
        )
