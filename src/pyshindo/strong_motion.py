"""JMA's own published velocity/displacement waveform derivation filters.

:mod:`pyshindo.velocity` derives velocity and displacement by plain
trapezoidal integration of acceleration, with no filtering imposed -- a
transparent default, but not what JMA itself uses for the velocity and
displacement waveforms it publishes. JMA's own description of how it derives
those waveforms (`速度波形・変位波形の求め方
<https://www.jma.go.jp/jma/kishou/know/jishin/kyoshin/kaisetsu/calc_wave.html>`_)
states that integrating raw acceleration directly produces spurious
long-period noise, so it instead applies one of two purpose-built recursive
filters, derived by the same method (斉藤, 1978) already cited for this
package's long-period high-pass filter (see
:func:`pyshindo.long_period.apply_ground_motion_high_pass`):

- Velocity: a third-order Butterworth high-pass with a 5-second cutoff.
- Displacement: a second-order filter reproducing the amplitude response of
  JMA's mechanical 1x strong-motion seismometer (natural period 6 s, damping
  ratio 0.55) -- not a literal double integration. This is why JMA's
  published displacement should be understood as ground motion within a
  defined, long-period-drift-suppressing passband, rather than the
  unbounded result of integrating acceleration twice.

Both filters take acceleration directly to their target quantity in one
pass; there is no separate integration step to call afterward.

Empirically, against the same 268-station corpus used elsewhere in this
package (two long-period ground motion observation events; see
``docs/validation.md``), :func:`apply_strong_motion_displacement_filter`
reproduces the published peak displacement in ``max.csv`` to a median
relative error of about 0.15 percent -- close to the peak-acceleration
level of agreement, and far tighter than the roughly 10-12 percent obtained
by high-pass filtering acceleration and then double-integrating it (see
:mod:`pyshindo.velocity`). :func:`apply_strong_motion_velocity_filter`,
by contrast, does *not* reproduce that same corpus's published peak velocity
as well (median error a few percent) as
:func:`pyshindo.long_period.apply_ground_motion_high_pass` followed by
:func:`pyshindo.velocity.integrate_to_velocity` does (median error about
0.01 percent, see ``docs/validation.md``) -- the long-period observation
page appears to compute its published velocity with the same 20-second
filter its own long-period class algorithm uses internally, not with this
general-purpose 5-second filter. This module's velocity filter is included
because JMA documents it as part of the same pair, and it is the documented
method for the general strong-motion waveform displays elsewhere on JMA's
site; it is not the recommended path to reproduce a long-period observation
page's published peak velocity.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from .units import AccelerationUnit, ArrayLike, FloatArray, to_gal
from .validation import STANDARD_SAMPLING_RATE_HZ, as_acceleration_array

# Coefficients as published, for 100 Hz acceleration in gal. JMA's page gives
# no method for other sampling rates, so these filters are restricted to 100 Hz
# rather than extrapolated.
_VELOCITY_GAIN = 0.004937561699
_VELOCITY_DENOMINATOR = (1.0, -2.974867761716, 2.950050339269, -0.975180618018)
_VELOCITY_NUMERATOR = tuple(_VELOCITY_GAIN * c for c in (1.0, -1.0, -1.0, 1.0))

_DISPLACEMENT_DENOMINATOR = (1.0, -1.988438073558305, 0.9885471048650272)
_DISPLACEMENT_NUMERATOR = (
    0.00002485615736514583,
    0.00004971231473029166,
    0.00002485615736514583,
)


def _require_reference_rate(sampling_rate_hz: float, filter_name: str) -> None:
    """Reject any rate other than 100 Hz: the published coefficients assume it."""
    rate = float(sampling_rate_hz)
    if not np.isclose(rate, STANDARD_SAMPLING_RATE_HZ, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"{filter_name} is only defined at {STANDARD_SAMPLING_RATE_HZ:g} Hz "
            f"(JMA's page gives no design method for other rates); "
            f"received {rate:g} Hz."
        )


def apply_strong_motion_velocity_filter(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    component_axis: int = -1,
) -> FloatArray:
    """Return velocity in cm/s via JMA's published recursive integration filter.

    This is JMA's own third-order, 5-second-cutoff Butterworth high-pass
    integration filter, not a plain trapezoidal integral -- see the module
    docstring for what it reproduces well (and does not).
    """
    _require_reference_rate(sampling_rate_hz, "apply_strong_motion_velocity_filter")
    parsed_unit = AccelerationUnit.parse(unit)
    values = as_acceleration_array(
        acceleration,
        component_axis=component_axis,
        warn_fewer_components=False,
    )
    values_gal = to_gal(values, parsed_unit, copy=False)
    velocity = scipy_signal.lfilter(_VELOCITY_NUMERATOR, _VELOCITY_DENOMINATOR, values_gal, axis=0)
    return np.ascontiguousarray(velocity, dtype=np.float64)


def apply_strong_motion_displacement_filter(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    component_axis: int = -1,
) -> FloatArray:
    """Return displacement in cm via JMA's mechanical-seismometer-emulation filter.

    This reproduces the amplitude response of JMA's mechanical 1x
    strong-motion seismometer (natural period 6 s, damping ratio 0.55)
    rather than integrating acceleration twice -- see the module docstring.
    It is the filter that reproduces the peak displacement published in a
    long-period ground motion observation page's ``max.csv``.
    """
    _require_reference_rate(sampling_rate_hz, "apply_strong_motion_displacement_filter")
    parsed_unit = AccelerationUnit.parse(unit)
    values = as_acceleration_array(
        acceleration,
        component_axis=component_axis,
        warn_fewer_components=False,
    )
    values_gal = to_gal(values, parsed_unit, copy=False)
    displacement = scipy_signal.lfilter(
        _DISPLACEMENT_NUMERATOR, _DISPLACEMENT_DENOMINATOR, values_gal, axis=0
    )
    return np.ascontiguousarray(displacement, dtype=np.float64)
