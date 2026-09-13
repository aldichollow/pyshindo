"""Single-degree-of-freedom elastic response, shared by every response-spectrum feature.

This is the linear-acceleration-method SDOF solver behind
:mod:`pyshindo.long_period` (JMA's absolute velocity response, two horizontal
components combined as a vector), :mod:`pyshindo.spectrum_intensity`
(Housner's relative velocity response, one value per component), and
:mod:`pyshindo.response_spectrum` (the general relative displacement,
velocity, and pseudo-acceleration spectrum, with neither ground motion nor
component combination applied). Nothing here is specific to any of them:
given a period, a damping ratio, and a sampling rate, it produces the
coefficients of the published recurrence and the exact transfer functions
equivalent to it, for either state variable (velocity or displacement). Each
feature adds its own acceleration filtering, ground-motion handling, and
component combination on top; see their own modules for that.

References
----------
JMA, "長周期地震動に関する情報の作成に用いる絶対速度応答最大値の計算方法"
(別添資料3, 4th committee). The equation of motion, the linear-acceleration
recurrence, its A/B coefficients, and the initial conditions. Based on
大崎順彦 (1994), 新・地震動のスペクトル解析入門.
https://www.data.jma.go.jp/eqev/data/study-panel/tyoshuki_joho_kentokai/kentokai4/sanko3.pdf
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

from .units import FloatArray


@dataclass(frozen=True, slots=True)
class OscillatorBank:
    """Linear-acceleration recurrence coefficients for a set of periods.

    Each array holds one value per period. ``a11`` through ``b22`` are the
    published closed forms; they have been checked against an independent
    matrix-exponential first-order-hold discretization of the same equation of
    motion and agree to about 1e-14 across the official long-period-class
    period grid, which also confirms that "線形加速度法" here means a
    first-order hold on the acceleration within each step.
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


def transfer_functions(bank: OscillatorBank) -> tuple[FloatArray, FloatArray]:
    """Return one ``lfilter`` numerator and denominator per period.

    The published step ``x[n+1] = A x[n] + B0 u[n] + B1 u[n+1]`` is a state
    space realization of a linear time-invariant system, so it has an exact
    transfer function from acceleration to relative velocity::

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


def seed_relative_velocity(bank: OscillatorBank, acceleration: FloatArray) -> FloatArray:
    """Return the relative velocity of the first two samples, shaped ``(2, periods, components)``.

    The published initialization is ``DIS(1) = 0`` and ``VEL(1) = -A(1)*dt``,
    which is not a state a difference equation can be started from directly.
    Producing these two samples from the recurrence itself and seeding the
    filter with them carries the initialization across exactly. Works for any
    number of acceleration components -- nothing here assumes two.
    """
    dt = 1.0 / bank.sampling_rate_hz
    first = -acceleration[0] * dt
    second = (
        bank.a22[:, np.newaxis] * first
        + bank.b21[:, np.newaxis] * acceleration[0]
        + bank.b22[:, np.newaxis] * acceleration[1]
    )
    return np.stack([np.broadcast_to(first, second.shape), second])


def displacement_transfer_functions(bank: OscillatorBank) -> tuple[FloatArray, FloatArray]:
    """Return one ``lfilter`` numerator and denominator per period, for relative displacement.

    Same state-space-to-transfer-function conversion as :func:`transfer_functions`,
    selecting the displacement element of the state vector (output ``C = [1, 0]``)
    instead of velocity (``C = [0, 1]``), so the denominator -- which comes from
    the state matrix alone, not the selected output -- is identical::

        D(z)     b12 z^2 + (b11 - a22 b12 + a12 b22) z + (a12 b21 - a22 b11)
        ---- = ---------------------------------------------------------------
        U(z)          z^2 - (a11 + a22) z + (a11 a22 - a12 a21)

    Verified against an independent step-by-step recurrence at about 1e-12
    relative, the same level as :func:`transfer_functions`.
    """
    numerator = np.column_stack(
        [
            bank.b12,
            bank.b11 - bank.a22 * bank.b12 + bank.a12 * bank.b22,
            bank.a12 * bank.b21 - bank.a22 * bank.b11,
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


def seed_relative_displacement(bank: OscillatorBank, acceleration: FloatArray) -> FloatArray:
    """Return the first two relative displacement samples, shaped ``(2, periods, components)``.

    ``DIS(1) = 0`` by the published initialization, for every period.
    ``DIS(2)`` follows from applying the recurrence once, using the same
    ``VEL(1) = -A(1)*dt`` used to seed :func:`seed_relative_velocity`.
    """
    dt = 1.0 / bank.sampling_rate_hz
    velocity_first = -acceleration[0] * dt
    second = (
        bank.a12[:, np.newaxis] * velocity_first
        + bank.b11[:, np.newaxis] * acceleration[0]
        + bank.b12[:, np.newaxis] * acceleration[1]
    )
    first = np.zeros_like(second)
    return np.stack([first, second])


def _run_transfer_function(
    bank: OscillatorBank,
    acceleration: FloatArray,
    numerator: FloatArray,
    denominator: FloatArray,
    single_sample_response: FloatArray,
    seed: FloatArray | None,
    *,
    collect: bool,
) -> tuple[FloatArray, FloatArray | None]:
    """Run one transfer function over every period, shared by the velocity and
    displacement response functions -- they differ only in which numerator,
    seed, and degenerate one-sample value they pass in.
    """
    samples = acceleration.shape[0]
    components = acceleration.shape[1]
    period_count = bank.period_count
    peaks = np.zeros((period_count, components), dtype=np.float64)
    series = (
        np.empty((samples, period_count, components), dtype=np.float64) if collect else None
    )
    if samples == 0:
        return peaks, series

    response = np.empty((samples, components), dtype=np.float64)
    for index in range(period_count):
        if seed is None:
            response[0] = single_sample_response
        else:
            response[:2] = seed[:, index, :]
            if samples > 2:
                state = np.column_stack(
                    [
                        scipy_signal.lfiltic(
                            numerator[index],
                            denominator[index],
                            response[1::-1, component],
                            acceleration[1::-1, component],
                        )
                        for component in range(components)
                    ]
                )
                response[2:] = scipy_signal.lfilter(
                    numerator[index],
                    denominator[index],
                    acceleration[2:],
                    axis=0,
                    zi=state,
                )[0]
        peaks[index] = np.max(np.abs(response), axis=0)
        if series is not None:
            series[:, index, :] = response
    return peaks, series


def relative_velocity_response(
    bank: OscillatorBank,
    acceleration: FloatArray,
    *,
    collect: bool = False,
) -> tuple[FloatArray, FloatArray | None]:
    """Return the peak relative velocity response of every period and component.

    ``acceleration`` is ``(samples, components)`` for any number of
    components -- unlike :mod:`pyshindo.long_period`, nothing here adds
    ground velocity or combines components, so this is the relative response
    alone. Returns ``(peaks, series)`` where ``peaks`` has shape
    ``(periods, components)``, and ``series`` is ``(samples, periods,
    components)`` when ``collect`` is set, else ``None``.

    Runs each period as one ``lfilter`` call, about an order of magnitude
    faster than stepping the recurrence sample by sample in Python; see
    :func:`pyshindo.long_period._core.filtered_response_maxima`, which uses
    the same technique.
    """
    samples = acceleration.shape[0]
    dt = 1.0 / bank.sampling_rate_hz
    seed = seed_relative_velocity(bank, acceleration) if samples > 1 else None
    numerator, denominator = transfer_functions(bank)
    single_sample = -acceleration[0] * dt if samples > 0 else np.empty(acceleration.shape[1])
    return _run_transfer_function(
        bank, acceleration, numerator, denominator, single_sample, seed, collect=collect
    )


def relative_displacement_response(
    bank: OscillatorBank,
    acceleration: FloatArray,
    *,
    collect: bool = False,
) -> tuple[FloatArray, FloatArray | None]:
    """Return the peak relative displacement response of every period and component.

    Same contract as :func:`relative_velocity_response`, for displacement
    instead of velocity: no ground motion added, no component combination,
    ``peaks`` shaped ``(periods, components)``.
    """
    samples = acceleration.shape[0]
    seed = seed_relative_displacement(bank, acceleration) if samples > 1 else None
    numerator, denominator = displacement_transfer_functions(bank)
    # DIS(1) = 0 by the published initialization, regardless of period.
    single_sample = np.zeros(acceleration.shape[1] if samples > 0 else 0)
    return _run_transfer_function(
        bank, acceleration, numerator, denominator, single_sample, seed, collect=collect
    )
