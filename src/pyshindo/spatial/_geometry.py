"""Spherical distance and local-plane projection for station interpolation.

Internal to :mod:`pyshindo.spatial`. Longitude/latitude degrees are not
Cartesian coordinates: a k-d tree built directly on them would weight a
degree of longitude near a pole the same as one at the equator, and
Delaunay triangulation needs a genuine plane, not a pair of angles. Every
function in this subpackage that needs distance or a plane goes through
here, so the geometry stays in one place.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]

EARTH_RADIUS_KM: Final[float] = 6371.0088
"""IUGG mean Earth radius, in kilometers."""


def lonlat_to_unit_xyz(longitude_deg: npt.ArrayLike, latitude_deg: npt.ArrayLike) -> FloatArray:
    """Convert longitude/latitude in degrees to unit vectors on the sphere.

    Chord distance between two such vectors is a monotone function of
    great-circle distance (see :func:`chord_to_great_circle_km`), so a
    k-d tree built on these vectors returns the same nearest-neighbor
    ordering a true great-circle search would -- without the antimeridian
    seam and pole singularity a k-d tree built directly on bare
    longitude/latitude pairs would have.
    """
    lon = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def chord_to_great_circle_km(chord: npt.ArrayLike) -> FloatArray:
    """Convert unit-sphere chord length to great-circle distance in km.

    For two points on a unit sphere, ``chord = 2 * sin(central_angle / 2)``;
    inverting that gives the central angle, and arc length is
    ``EARTH_RADIUS_KM * central_angle``. The chord is clipped to ``[0, 1]``
    before the inverse sine because floating-point round-off can push an
    exact match's chord a hair outside that domain.
    """
    half_chord = np.clip(np.asarray(chord, dtype=np.float64) * 0.5, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(half_chord)


def great_circle_radius_to_chord(radius_km: float) -> float:
    """Return the unit-sphere chord length corresponding to a great-circle radius.

    The inverse of :func:`chord_to_great_circle_km`, for converting a
    search radius in kilometers into the ``distance_upper_bound`` a
    :class:`scipy.spatial.cKDTree` query built on
    :func:`lonlat_to_unit_xyz` expects.
    """
    return float(2.0 * math.sin(0.5 * radius_km / EARTH_RADIUS_KM))


def great_circle_distance_km(
    longitude1_deg: float, latitude1_deg: float, longitude2_deg: float, latitude2_deg: float
) -> float:
    """Return the great-circle distance in km between two points.

    Routed through the same chord calculation a k-d tree neighbor search
    uses, so a distance computed here and one returned by a plan's
    ``nearest_station_distance_km`` always agree exactly.
    """
    first = lonlat_to_unit_xyz(longitude1_deg, latitude1_deg)
    second = lonlat_to_unit_xyz(longitude2_deg, latitude2_deg)
    return float(chord_to_great_circle_km(np.linalg.norm(first - second)))


def aeqd_project_km(
    longitude_deg: npt.ArrayLike,
    latitude_deg: npt.ArrayLike,
    *,
    center_longitude_deg: float,
    center_latitude_deg: float,
) -> FloatArray:
    """Spherical azimuthal-equidistant projection, centered on the map.

    Every point's distance from the projection center is preserved exactly
    (hence "equidistant"); distances between two points that are both far
    from the center distort more, which is why
    :class:`~pyshindo.spatial.LinearConfig` bounds how far from its center
    the projection may be used (``maximum_extent_km``). Chosen over Web
    Mercator, which distorts area and would bias which stations a triangle
    edge favors near the poles, and over a general-purpose map-projection
    dependency, since a station interpolation surface needs exactly one
    projection, used only internally.
    """
    lon = np.deg2rad(np.asarray(longitude_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    lon0 = math.radians(center_longitude_deg)
    lat0 = math.radians(center_latitude_deg)
    delta_lon = (lon - lon0 + math.pi) % (2.0 * math.pi) - math.pi
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin0, cos0 = math.sin(lat0), math.cos(lat0)
    cos_central_angle = np.clip(sin0 * sin_lat + cos0 * cos_lat * np.cos(delta_lon), -1.0, 1.0)
    central_angle = np.arccos(cos_central_angle)
    sin_central_angle = np.sin(central_angle)
    # scale -> 1 as central_angle -> 0 (l'Hopital); handled explicitly to avoid 0/0 at
    # the projection center itself, which every station's own grid cell sits near.
    scale = np.ones_like(central_angle)
    away_from_center = central_angle > 1.0e-12
    scale[away_from_center] = (
        central_angle[away_from_center] / sin_central_angle[away_from_center]
    )
    x = EARTH_RADIUS_KM * scale * cos_lat * np.sin(delta_lon)
    y = EARTH_RADIUS_KM * scale * (cos0 * sin_lat - sin0 * cos_lat * np.cos(delta_lon))
    return np.column_stack((x, y))
