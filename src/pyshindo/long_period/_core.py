"""Numerical kernel for the JMA long-period ground motion calculation.

Three pieces, each traceable to a JMA primary source:

1. the 20-second second-order high-pass recurrence applied to acceleration;
2. the single-degree-of-freedom oscillator bank solved by the linear
   acceleration method;
3. the ground velocity that turns a relative velocity response into an
   absolute one.

The published high-pass is stated as ``y_t = acc_t + a1*acc_(t-1) +
a2*acc_(t-2) - b1*y_(t-1) - b2*y_(t-2)``, with ``acc_HPF_t = G0 * y_t``.
Note the sign convention: ``b1`` is itself negative and the recurrence
subtracts it, so that term is effectively an addition, and the denominator
is already in SciPy's ``lfilter`` form ``[1, b1, b2]``.

JMA states that the ground velocity is the integral of the same filtered
acceleration, but not the discrete convention. The trapezoidal rule is used
here because under the first-order hold the oscillator recurrence already
assumes, it is the exact integral rather than an approximation, and because
it is what reproduces JMA's published response spectra; ``docs/long-period.md``
records that comparison.

References
----------
JMA, "長周期地震動に関する情報の作成に用いる絶対速度応答最大値の計算方法"
(別添資料3, 4th committee). Equation of motion, the linear-acceleration
recurrence, its A/B coefficients, the initial conditions, and the addition of
ground velocity. Based on 大崎順彦 (1994), 新・地震動のスペクトル解析入門.
https://www.data.jma.go.jp/eqev/data/study-panel/tyoshuki_joho_kentokai/kentokai4/sanko3.pdf

JMA, "絶対速度応答計算の改善について" (資料2, 7th committee). Why the earlier
5-second integrating filter distorted the absolute response, and the improved
20-second second-order high-pass applied to the acceleration record, with its
exact 100 Hz constants on page 13.
https://ds.data.jma.go.jp/svd/eqev/data/study-panel/tyoshuki_joho_kentokai/kentokai7/siryou2.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
from scipy import signal as scipy_signal
from scipy.integrate import cumulative_trapezoid

from ..units import FloatArray

REFERENCE_SAMPLING_RATE_HZ: Final = 100.0
"""The rate the published high-pass constants below are stated for."""

HORIZONTAL_COMPONENTS: Final = 2
"""The class is defined on the two horizontal components; there is no
one-component or three-component variant of it to generalize over."""

# Published 100 Hz constants (7th committee, 資料2, page 13).
_HPF_A1: Final = -2.0
_HPF_A2: Final = 1.0
_HPF_B1: Final = -1.995438545842
_HPF_B2: Final = 0.995448925627
_HPF_G0: Final = 0.997721867867

_HPF_PROTOTYPE_DAMPING: Final = 0.7071067811865476  # 1/sqrt(2)
_HPF_PROTOTYPE_OMEGA_N: Final = 0.322544346015  # rad/s, period 19.480066 s


@dataclass(frozen=True, slots=True)
class HighPassDesign:
    """Second-order high-pass recurrence in SciPy ``lfilter`` form.

    ``numerator`` and ``denominator`` produce the intermediate variable the
    JMA document calls ``y``; the filtered acceleration is ``gain * y``. The
    scaling is kept separate rather than folded into ``numerator`` so the
    stored state is exactly the document's ``y``.
    """

    numerator: FloatArray
    denominator: FloatArray
    gain: float
    sampling_rate_hz: float
    is_published_reference: bool


def design_high_pass(sampling_rate_hz: float) -> HighPassDesign:
    """Return the 20-second high-pass design for a sampling rate.

    At 100 Hz this is the published constant set, used verbatim.

    At any other rate the design is rebuilt from the analog prototype behind
    those constants. Taking the logarithm of the published poles gives a
    continuous-time pair with damping 1/sqrt(2) to the printed precision: the
    filter is a second-order Butterworth whose poles are the matched-z image
    ``z = exp(s*dt)`` of that prototype, not a bilinear transform, with the
    gain normalized to unity at Nyquist. Rebuilding the 100 Hz coefficients
    this way reproduces the published values to about 4e-9. The natural period
    lands at 19.48 s rather than the nominal 20 s; the committee material
    attributes the constants to 斎藤 (1978) without restating the corner
    definition, so the reference path uses the literals and only other rates
    use this reconstruction, marked as not the published reference.
    """
    rate = float(sampling_rate_hz)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and greater than zero.")
    numerator = np.array([1.0, _HPF_A1, _HPF_A2], dtype=np.float64)
    if rate == REFERENCE_SAMPLING_RATE_HZ:
        return HighPassDesign(
            numerator=numerator,
            denominator=np.array([1.0, _HPF_B1, _HPF_B2], dtype=np.float64),
            gain=_HPF_G0,
            sampling_rate_hz=rate,
            is_published_reference=True,
        )
    zeta = _HPF_PROTOTYPE_DAMPING
    pole_s = _HPF_PROTOTYPE_OMEGA_N * (-zeta + 1j * np.sqrt(1.0 - zeta * zeta))
    pole_z = np.exp(pole_s / rate)
    denominator = np.real(np.poly([pole_z, np.conj(pole_z)]))
    gain = float(
        abs(np.polyval(denominator, -1.0) / np.polyval(numerator, -1.0))
    )
    return HighPassDesign(
        numerator=numerator,
        denominator=denominator,
        gain=gain,
        sampling_rate_hz=rate,
        is_published_reference=False,
    )


def high_pass_initial_state(design: HighPassDesign) -> FloatArray:
    """Return zero filter state for the two horizontal channels."""
    order = max(len(design.numerator), len(design.denominator)) - 1
    return np.zeros((order, HORIZONTAL_COMPONENTS))


def apply_high_pass(
    design: HighPassDesign,
    acceleration: FloatArray,
    state: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Filter ``(samples, components)`` acceleration, returning output and new state.

    The same call serves batch and streaming use, so a chunked run and a
    single-sample run share one code path and cannot drift apart.
    """
    filtered, new_state = scipy_signal.lfilter(
        design.numerator,
        design.denominator,
        acceleration,
        axis=0,
        zi=state,
    )
    return np.ascontiguousarray(filtered * design.gain), new_state


@dataclass(frozen=True, slots=True)
class OscillatorBank:
    """Linear-acceleration recurrence coefficients for a set of periods.

    Each array holds one value per period. ``a11`` through ``b22`` are the
    published closed forms; they have been checked against an independent
    matrix-exponential first-order-hold discretization of the same equation of
    motion and agree to about 1e-14 across the official period grid, which
    also confirms that "線形加速度法" here means a first-order hold on the
    acceleration within each step.
    """

    periods_s: FloatArray
    damping_ratio: float
    sampling_rate_hz: float
    a11: FloatArray
    a12: FloatArray
    a21: FloatArray
    a22: FloatArray
    b11: FloatArray
    b12: FloatArray
    b21: FloatArray
    b22: FloatArray

    @property
    def period_count(self) -> int:
        """Return the number of periods in the bank."""
        return int(self.periods_s.size)


def design_oscillator_bank(
    periods_s: FloatArray,
    damping_ratio: float,
    sampling_rate_hz: float,
) -> OscillatorBank:
    """Build the recurrence coefficients for every period at once."""
    periods = np.asarray(periods_s, dtype=np.float64)
    if periods.ndim != 1 or periods.size == 0:
        raise ValueError("periods_s must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(periods)) or np.any(periods <= 0.0):
        raise ValueError("periods_s must be finite and greater than zero.")
    h = float(damping_ratio)
    if not 0.0 < h < 1.0:
        raise ValueError("damping_ratio must lie strictly between zero and one.")
    dt = 1.0 / float(sampling_rate_hz)

    w = 2.0 * np.pi / periods
    wd = w * np.sqrt(1.0 - h * h)
    e = np.exp(-h * w * dt)
    cos = np.cos(wd * dt)
    sin = np.sin(wd * dt)

    a11 = e * (cos + (h * w / wd) * sin)
    a12 = e * sin / wd
    a21 = -e * (w**2 / wd) * sin
    a22 = e * (cos - (h * w / wd) * sin)

    b11 = (
        e
        * (
            (1.0 / w**2 + 2.0 * h / (w**3 * dt)) * cos
            + (h / (w * wd) - (1.0 - 2.0 * h * h) / (w**2 * wd * dt)) * sin
        )
        - 2.0 * h / (w**3 * dt)
    )
    b12 = (
        e * (-(2.0 * h / (w**3 * dt)) * cos + ((1.0 - 2.0 * h * h) / (w**2 * wd * dt)) * sin)
        - 1.0 / w**2
        + 2.0 * h / (w**3 * dt)
    )
    b21 = (
        e * (-(1.0 / (w**2 * dt)) * cos - (h / (w * wd * dt) + 1.0 / wd) * sin)
        + 1.0 / (w**2 * dt)
    )
    b22 = e * ((1.0 / (w**2 * dt)) * cos + (h / (w * wd * dt)) * sin) - 1.0 / (w**2 * dt)

    return OscillatorBank(
        periods_s=periods,
        damping_ratio=h,
        sampling_rate_hz=float(sampling_rate_hz),
        a11=a11,
        a12=a12,
        a21=a21,
        a22=a22,
        b11=b11,
        b12=b12,
        b21=b21,
        b22=b22,
    )


class ResponseState:
    """Running oscillator, ground-velocity, and per-period maximum state.

    Holds exactly what the recurrence needs to continue: the displacement and
    velocity of every oscillator, the ground velocity, the previous filtered
    acceleration sample, and the running maximum of the horizontal absolute
    velocity for each period. Memory is O(periods), independent of record
    length, so a long record costs no more than a short one.

    The published recurrence advances from sample ``n`` to ``n+1`` using both
    ``A(n)`` and ``A(n+1)``, which is why the previous acceleration sample is
    part of the state: a streaming caller can only advance once the next
    sample has arrived.
    """

    __slots__ = (
        "_bank",
        "_dt",
        "displacement",
        "ground_velocity",
        "previous_acceleration",
        "running_max",
        "sample_count",
        "velocity",
    )

    def __init__(self, bank: OscillatorBank) -> None:
        self._bank = bank
        self._dt = 1.0 / bank.sampling_rate_hz
        shape = (bank.period_count, HORIZONTAL_COMPONENTS)
        self.displacement = np.zeros(shape, dtype=np.float64)
        self.velocity = np.zeros(shape, dtype=np.float64)
        self.ground_velocity = np.zeros(HORIZONTAL_COMPONENTS, dtype=np.float64)
        self.previous_acceleration = np.zeros(HORIZONTAL_COMPONENTS, dtype=np.float64)
        self.running_max = np.zeros(bank.period_count, dtype=np.float64)
        self.sample_count = 0


    def absolute_velocity_magnitude(self) -> FloatArray:
        """Return the horizontal absolute velocity magnitude for each period.

        Absolute velocity is the relative velocity response plus the ground
        velocity, per component. The horizontal components are then combined
        as a vector at this same instant -- JMA's "手法B", adopted in 2016
        over the earlier per-component maximum, because buildings are not
        aligned with north and east.
        """
        absolute = self.velocity + self.ground_velocity
        return np.hypot(absolute[:, 0], absolute[:, 1])

    def _start_record(self, sample: FloatArray) -> FloatArray:
        """Apply the published initialization for the first sample of a record.

        The JMA document initializes the bank as ``DIS(1) = 0`` and
        ``VEL(1) = -A(1)*dt`` -- the velocity is deliberately not zero. The
        term is small (one sample interval times the first filtered
        acceleration sample) and decays, but it is what the source specifies,
        so it is reproduced rather than simplified away.
        """
        self.displacement[:] = 0.0
        self.velocity[:] = -sample * self._dt
        self.ground_velocity[:] = 0.0
        self.previous_acceleration[:] = sample
        self.sample_count = 1
        magnitude = self.absolute_velocity_magnitude()
        np.maximum(self.running_max, magnitude, out=self.running_max)
        return magnitude

    def advance(
        self,
        filtered_acceleration: FloatArray,
        *,
        collect: bool = False,
    ) -> FloatArray | None:
        """Advance over a block of filtered acceleration.

        ``filtered_acceleration`` is ``(samples, components)`` and must
        already be high-pass filtered. Returns the per-sample horizontal
        absolute velocity for every period, shaped ``(samples, periods)``,
        when ``collect`` is set; otherwise only the running maxima are updated
        and ``None`` is returned, which keeps memory at O(periods).

        The recurrence is sequential, so the loop runs over samples and the
        periods and components are vectorized within each step, at roughly
        7 us per sample. Reusing scratch buffers instead of letting each step
        allocate measured no faster: the cost is NumPy call overhead on small
        arrays, not allocation.
        """
        bank = self._bank
        dt = self._dt
        samples = filtered_acceleration.shape[0]
        if samples == 0:
            return np.empty((0, bank.period_count), dtype=np.float64) if collect else None

        collected = np.empty((samples, bank.period_count), dtype=np.float64) if collect else None
        start = 0
        if self.sample_count == 0:
            magnitude = self._start_record(filtered_acceleration[0])
            if collected is not None:
                collected[0] = magnitude
            start = 1

        a11 = bank.a11[:, np.newaxis]
        a12 = bank.a12[:, np.newaxis]
        a21 = bank.a21[:, np.newaxis]
        a22 = bank.a22[:, np.newaxis]
        b11 = bank.b11[:, np.newaxis]
        b12 = bank.b12[:, np.newaxis]
        b21 = bank.b21[:, np.newaxis]
        b22 = bank.b22[:, np.newaxis]

        displacement = self.displacement
        velocity = self.velocity
        previous = self.previous_acceleration
        ground_velocity = self.ground_velocity
        half_dt = 0.5 * dt
        for index in range(start, samples):
            current = filtered_acceleration[index]
            ground_velocity += half_dt * (previous + current)
            next_displacement = (
                a11 * displacement + a12 * velocity + b11 * previous + b12 * current
            )
            next_velocity = a21 * displacement + a22 * velocity + b21 * previous + b22 * current
            displacement[:] = next_displacement
            velocity[:] = next_velocity
            previous[:] = current
            absolute = velocity + ground_velocity
            magnitude = np.hypot(absolute[:, 0], absolute[:, 1])
            np.maximum(self.running_max, magnitude, out=self.running_max)
            if collected is not None:
                collected[index] = magnitude

        self.sample_count += samples - start
        return collected


class OscillatorSolver(StrEnum):
    """Which numerical realization of the published recurrence to run.

    The two produce the same numbers to about 1e-12; they differ only in the
    order the arithmetic is performed, not in what is being solved.
    """

    FILTER = "filter"
    RECURRENCE = "recurrence"

    @classmethod
    def parse(cls, value: str | OscillatorSolver) -> OscillatorSolver:
        """Return a canonical solver choice."""
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            options = ", ".join(member.value for member in cls)
            raise ValueError(f"Unknown solver {value!r}. Expected one of: {options}.") from exc


def _transfer_functions(bank: OscillatorBank) -> tuple[FloatArray, FloatArray]:
    """Return one ``lfilter`` numerator and denominator per period.

    The published step ``x[n+1] = A x[n] + B0 u[n] + B1 u[n+1]`` is a state
    space realization of a linear time-invariant system, so it has an exact
    transfer function from filtered acceleration to relative velocity::

        V(z)     b22 z^2 + (b21 - a11 b22 + a21 b12) z + (a21 b11 - a11 b21)
        ---- = ---------------------------------------------------------------
        U(z)          z^2 - (a11 + a22) z + (a11 a22 - a12 a21)

    Nothing is reinterpreted here: the coefficients are derived from the same
    published closed forms, and running them reproduces the recurrence to about
    1e-12, the difference being floating-point ordering alone.
    """
    numerator = np.column_stack(
        [
            bank.b22,
            bank.b21 - bank.a11 * bank.b22 + bank.a21 * bank.b12,
            bank.a21 * bank.b11 - bank.a11 * bank.b21,
        ]
    )
    denominator = np.column_stack(
        [
            np.ones_like(bank.a11),
            -(bank.a11 + bank.a22),
            bank.a11 * bank.a22 - bank.a12 * bank.a21,
        ]
    )
    return numerator, denominator


def _seed_relative_velocity(bank: OscillatorBank, acceleration: FloatArray) -> FloatArray:
    """Return the relative velocity of the first two samples, shaped ``(2, periods, components)``.

    The published initialization is ``DIS(1) = 0`` and ``VEL(1) = -A(1)*dt``,
    which is not a state a difference equation can be started from directly.
    Producing these two samples from the recurrence itself and seeding the
    filter with them carries the initialization across exactly.
    """
    dt = 1.0 / bank.sampling_rate_hz
    first = -acceleration[0] * dt
    second = (
        bank.a22[:, np.newaxis] * first
        + bank.b21[:, np.newaxis] * acceleration[0]
        + bank.b22[:, np.newaxis] * acceleration[1]
    )
    return np.stack([np.broadcast_to(first, second.shape), second])


def filtered_response_maxima(
    bank: OscillatorBank,
    filtered_acceleration: FloatArray,
    *,
    collect: bool = False,
) -> tuple[FloatArray, FloatArray | None]:
    """Run the whole bank in compiled code and return the per-period maxima.

    Equivalent to driving :class:`ResponseState` over the same record, about
    13 times faster for a batch because the sequential step moves out of Python.
    Streaming keeps the recurrence: the periods are independent filters rather
    than a cascade, so a single sample would cost one ``lfilter`` call per
    period instead of one vectorized step.

    Memory stays at one period's worth of history unless ``collect`` is set.
    """
    samples = filtered_acceleration.shape[0]
    period_count = bank.period_count
    collected = np.empty((samples, period_count), dtype=np.float64) if collect else None
    running_max = np.zeros(period_count, dtype=np.float64)
    if samples == 0:
        return running_max, collected

    dt = 1.0 / bank.sampling_rate_hz
    ground_velocity = cumulative_trapezoid(filtered_acceleration, dx=dt, axis=0, initial=0)
    seed = _seed_relative_velocity(bank, filtered_acceleration) if samples > 1 else None
    numerator, denominator = _transfer_functions(bank)

    velocity = np.empty((samples, HORIZONTAL_COMPONENTS), dtype=np.float64)
    for index in range(period_count):
        if seed is None:
            velocity[0] = -filtered_acceleration[0] * dt
        else:
            velocity[:2] = seed[:, index, :]
            if samples > 2:
                state = np.column_stack(
                    [
                        scipy_signal.lfiltic(
                            numerator[index],
                            denominator[index],
                            velocity[1::-1, component],
                            filtered_acceleration[1::-1, component],
                        )
                        for component in range(HORIZONTAL_COMPONENTS)
                    ]
                )
                velocity[2:] = scipy_signal.lfilter(
                    numerator[index],
                    denominator[index],
                    filtered_acceleration[2:],
                    axis=0,
                    zi=state,
                )[0]
        absolute = velocity + ground_velocity
        magnitude = np.hypot(absolute[:, 0], absolute[:, 1])
        running_max[index] = magnitude.max()
        if collected is not None:
            collected[:, index] = magnitude
    return running_max, collected
