"""Velocity integration and peak ground velocity."""

from __future__ import annotations

import numpy as np
from scipy import integrate as scipy_integrate

from .signal import vector_resultant
from .units import AccelerationUnit, ArrayLike, FloatArray, to_gal
from .validation import as_acceleration_array, validate_sampling_rate


def integrate_to_velocity(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    component_axis: int = -1,
) -> FloatArray:
    """Integrate acceleration to velocity in cm/s (kine).

    Cumulative trapezoidal integration is applied independently to each
    component, starting from zero velocity. Input is converted to gal first,
    so the result is always cm/s regardless of the input unit.

    Notes
    -----
    No offset removal, detrending, or high-pass filtering is applied. A record
    with a nonzero mean -- including one still carrying a static gravity
    component on its vertical channel -- integrates into a velocity that drifts
    linearly, and any residual low-frequency error grows the same way. This is
    the arithmetic working correctly, not a defect: integration has no way to
    distinguish a real long-period signal from a baseline error.

    Baseline treatment is left to the caller because there is no single correct
    choice, and applying one silently would hide it. :func:`remove_offset` (with
    a pre-event ``baseline_samples`` interval) and :func:`detrend_acceleration`
    are the usual starting points; published strong-motion practice often uses a
    high-pass filter instead, with a corner frequency chosen for the instrument
    and the analysis.
    """
    rate = validate_sampling_rate(sampling_rate_hz, warn_nonstandard=False)
    parsed_unit = AccelerationUnit.parse(unit)
    values = as_acceleration_array(
        acceleration,
        component_axis=component_axis,
        warn_fewer_components=False,
    )
    values_gal = to_gal(values, parsed_unit, copy=False)
    velocity = scipy_integrate.cumulative_trapezoid(
        values_gal,
        dx=1.0 / rate,
        axis=0,
        initial=0,
    )
    return np.ascontiguousarray(velocity, dtype=np.float64)


def component_peak_velocity(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    component_axis: int = -1,
) -> FloatArray:
    """Return the maximum absolute velocity of each component in cm/s.

    This mirrors :func:`component_peak_acceleration`. See
    :func:`integrate_to_velocity` for the baseline caveat that applies to every
    velocity derived by integration.
    """
    velocity = integrate_to_velocity(
        acceleration,
        sampling_rate_hz,
        unit=unit,
        component_axis=component_axis,
    )
    return np.max(np.abs(velocity), axis=0)


def peak_ground_velocity(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    component_axis: int = -1,
) -> float:
    """Return the peak vector-resultant velocity (PGV) in cm/s.

    The resultant is taken over whichever components are supplied, matching
    :func:`peak_ground_acceleration`. With the usual three-component input this
    is the three-component resultant; pass only the two horizontal components to
    obtain the horizontal PGV that much of strong-motion practice reports
    instead. Neither convention is imposed here, because the choice belongs to
    the analysis rather than to the integration.

    See :func:`integrate_to_velocity` for the baseline caveat.
    """
    velocity = integrate_to_velocity(
        acceleration,
        sampling_rate_hz,
        unit=unit,
        component_axis=component_axis,
    )
    return float(np.max(vector_resultant(velocity)))
