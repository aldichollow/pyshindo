"""Input validation for station coordinates and per-station values.

The same two checks every interpolation method needs before touching
geometry: are the stations real points, and are the values real numbers.
This plays the role :mod:`pyshindo.validation` plays for acceleration
records, but station coordinates are not acceleration data, so this module
does not reuse :func:`pyshindo.validation.as_acceleration_array` or
:class:`~pyshindo.exceptions.InvalidAccelerationError` -- doing so would
raise an acceleration-specific error for a mistake that has nothing to do
with acceleration, the same reasoning
:func:`pyshindo.long_period.calculation.as_horizontal_acceleration` already
follows for a different mismatch.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]


def validate_station_coordinates(
    latitudes_deg: npt.ArrayLike,
    longitudes_deg: npt.ArrayLike,
    *,
    minimum_count: int,
) -> tuple[FloatArray, FloatArray]:
    """Return validated, contiguous ``(latitudes, longitudes)`` in degrees.

    Requires at least ``minimum_count`` stations, every coordinate finite
    and within its physical range, and no two stations at the exact same
    point. A duplicate must be resolved by the caller -- averaging,
    jittering, or dropping one silently would be a modeling choice this
    function should not make on its own.
    """
    lat = np.asarray(latitudes_deg, dtype=np.float64)
    lon = np.asarray(longitudes_deg, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or lat.shape != lon.shape:
        raise ValueError(
            "latitudes_deg and longitudes_deg must each be one-dimensional and the "
            f"same length; received shapes {lat.shape} and {lon.shape}."
        )
    if lat.size < minimum_count:
        raise ValueError(
            f"At least {minimum_count} station(s) are required for this method; "
            f"received {lat.size}."
        )
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        raise ValueError("latitudes_deg and longitudes_deg must be finite.")
    if np.any(np.abs(lat) > 90.0):
        raise ValueError("latitudes_deg must lie within [-90, 90].")
    if np.any(np.abs(lon) > 180.0):
        raise ValueError("longitudes_deg must lie within [-180, 180].")
    coordinates = np.column_stack((lat, lon))
    if np.unique(coordinates, axis=0).shape[0] != lat.size:
        raise ValueError(
            "Duplicate station coordinates were found. Resolve them explicitly "
            "(drop, jitter, or average the values) before interpolating; this "
            "function will not choose for you."
        )
    return np.ascontiguousarray(lat), np.ascontiguousarray(lon)


def validate_values(values: npt.ArrayLike, station_count: int) -> FloatArray:
    """Return a validated, contiguous ``float64`` array of per-station values.

    A plan is built once for a fixed station layout, so a missing reading
    cannot simply be dropped here without invalidating the neighbor
    geometry the plan already precomputed; rebuild the plan for the active
    station subset instead, or use :func:`~pyshindo.spatial.interpolate_surface`
    for a one-shot call that only ever sees one frame in the first place.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (station_count,):
        raise ValueError(
            f"values must have shape ({station_count},) to match the station count "
            f"the plan was built for; received shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite.")
    return np.ascontiguousarray(array)
