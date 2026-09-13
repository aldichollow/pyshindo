"""Velocity and displacement integration, peak ground velocity, and peak ground displacement."""

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


def integrate_to_displacement(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    component_axis: int = -1,
) -> FloatArray:
    """Integrate acceleration to displacement in cm, via velocity.

    Two cumulative trapezoidal integrations -- the same method
    :func:`integrate_to_velocity` uses for the first of them -- starting
    from zero velocity and zero displacement. Input is converted to gal
    first, so the result is always cm regardless of the input unit.

    Notes
    -----
    Every baseline caveat in :func:`integrate_to_velocity` applies here
    twice over, and compounds: a nonzero-mean acceleration integrates into
    a velocity that drifts *linearly*, which then integrates into a
    displacement that drifts *quadratically*. A baseline error small enough
    to be a rounding footnote for peak velocity can dominate peak
    displacement outright. This is, again, the arithmetic working
    correctly on whatever signal it is given -- integration still cannot
    tell a real long-period displacement from an uncorrected baseline --
    but it means baseline treatment matters considerably more here than for
    velocity, not less.

    No baseline correction is applied. :func:`~pyshindo.signal.remove_offset`
    and :func:`~pyshindo.signal.detrend_acceleration` are the same starting
    points :func:`integrate_to_velocity` documents.

    JMA's own published displacement is not this: it is not an integration
    at all, but a filter reproducing the response of a specific instrument.
    See :func:`pyshindo.strong_motion.apply_strong_motion_displacement_filter`
    if matching that published value is the goal.
    """
    velocity = integrate_to_velocity(
        acceleration,
        sampling_rate_hz,
        unit=unit,
        component_axis=component_axis,
    )
    rate = validate_sampling_rate(sampling_rate_hz, warn_nonstandard=False)
    displacement = scipy_integrate.cumulative_trapezoid(
        velocity,
        dx=1.0 / rate,
        axis=0,
        initial=0,
    )
    return np.ascontiguousarray(displacement, dtype=np.float64)


def component_peak_displacement(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    component_axis: int = -1,
) -> FloatArray:
    """Return the maximum absolute displacement of each component in cm.

    This mirrors :func:`component_peak_velocity` and
    :func:`~pyshindo.signal.component_peak_acceleration`. See
    :func:`integrate_to_displacement` for the doubly compounded baseline
    caveat that applies to every displacement derived by integration.
    """
    displacement = integrate_to_displacement(
        acceleration,
        sampling_rate_hz,
        unit=unit,
        component_axis=component_axis,
    )
    return np.max(np.abs(displacement), axis=0)


def peak_ground_displacement(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    component_axis: int = -1,
) -> float:
    """Return the peak vector-resultant displacement (PGD) in cm.

    The resultant is taken over whichever components are supplied, matching
    :func:`peak_ground_velocity` and
    :func:`~pyshindo.signal.peak_ground_acceleration`: three components give
    the three-component resultant, two horizontals give the horizontal PGD.
    Neither convention is imposed here, for the same reason
    :func:`peak_ground_velocity` leaves it open.

    See :func:`integrate_to_displacement` for the baseline caveat -- it
    applies more severely here than for PGV.
    """
    displacement = integrate_to_displacement(
        acceleration,
        sampling_rate_hz,
        unit=unit,
        component_axis=component_axis,
    )
    return float(np.max(vector_resultant(displacement)))
