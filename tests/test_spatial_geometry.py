from __future__ import annotations

import math

import numpy as np
import pytest

from pyshindo.spatial import _geometry


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Independent great-circle distance, not sharing a code path with the module
    under test; see the identical helper and its docstring in test_spatial_idw.py."""
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def test_great_circle_distance_matches_independent_haversine() -> None:
    cases = [
        (35.0, 135.0, 35.0, 136.0),
        (0.0, 0.0, 0.0, 90.0),
        (89.0, 0.0, 89.0, 180.0),
        (-33.9, 151.2, 35.7, 139.7),  # Sydney to Tokyo, roughly
    ]
    for lat1, lon1, lat2, lon2 in cases:
        expected = _haversine_km(lat1, lon1, lat2, lon2)
        actual = _geometry.great_circle_distance_km(lon1, lat1, lon2, lat2)
        assert actual == pytest.approx(expected, rel=1e-9)


def test_chord_and_great_circle_round_trip() -> None:
    radii_km = np.array([0.0, 1.0, 100.0, 10000.0, 20000.0])
    chord = np.array([_geometry.great_circle_radius_to_chord(r) for r in radii_km])
    recovered = _geometry.chord_to_great_circle_km(chord)
    np.testing.assert_allclose(recovered, radii_km, atol=1e-6)


def test_lonlat_to_unit_xyz_has_unit_norm() -> None:
    rng = np.random.default_rng(1)
    lon = rng.uniform(-180.0, 180.0, 50)
    lat = rng.uniform(-90.0, 90.0, 50)
    xyz = _geometry.lonlat_to_unit_xyz(lon, lat)
    norms = np.linalg.norm(xyz, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_aeqd_projection_preserves_distance_from_its_own_center() -> None:
    # The defining property of an azimuthal-equidistant projection: every point's
    # projected distance from the origin equals its true great-circle distance from
    # the projection center.
    center_lon, center_lat = 135.0, 35.0
    rng = np.random.default_rng(2)
    lon = center_lon + rng.uniform(-10.0, 10.0, 40)
    lat = center_lat + rng.uniform(-10.0, 10.0, 40)

    xy = _geometry.aeqd_project_km(
        lon, lat, center_longitude_deg=center_lon, center_latitude_deg=center_lat
    )
    projected_distance = np.linalg.norm(xy, axis=1)
    true_distance = np.array(
        [
            _geometry.great_circle_distance_km(center_lon, center_lat, lo, la)
            for lo, la in zip(lon, lat, strict=True)
        ]
    )
    np.testing.assert_allclose(projected_distance, true_distance, atol=1e-6)


def test_aeqd_projection_maps_the_center_to_the_origin() -> None:
    xy = _geometry.aeqd_project_km(
        135.0, 35.0, center_longitude_deg=135.0, center_latitude_deg=35.0
    )
    np.testing.assert_allclose(xy, [[0.0, 0.0]], atol=1e-9)


def test_aeqd_projection_handles_the_antimeridian_seam() -> None:
    # A point just west of the antimeridian and one just east of it are physically
    # close; the projection must not treat them as roughly 360 degrees apart.
    center_lon, center_lat = 179.5, 0.0
    xy = _geometry.aeqd_project_km(
        [179.0, -179.0], [0.0, 0.0], center_longitude_deg=center_lon, center_latitude_deg=center_lat
    )
    distance_between = np.linalg.norm(xy[0] - xy[1])
    # The two points are 2 degrees of longitude apart at the equator (~222 km); a
    # projection that failed to wrap the seam would instead see them as roughly
    # 358 degrees apart, on the order of 39,000 km.
    assert distance_between < 300.0
