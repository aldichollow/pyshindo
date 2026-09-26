from __future__ import annotations

import math

import numpy as np
import pytest

from pyshindo.spatial import (
    GeographicBounds,
    IDWConfig,
    NearestConfig,
    SurfaceGrid,
    interpolate_surface,
)
from pyshindo.spatial.idw import build_idw_plan

RNG = np.random.default_rng(20260915)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance via the haversine formula.

    Deliberately independent of :mod:`pyshindo.spatial._geometry`, which
    instead routes distance through unit-sphere chord length -- a different
    enough numerical path that agreement between the two is a real check,
    not a tautology.
    """
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def _single_cell_grid(latitude_deg: float, longitude_deg: float) -> SurfaceGrid:
    bounds = GeographicBounds(
        west_deg=longitude_deg - 0.1,
        south_deg=latitude_deg - 0.1,
        east_deg=longitude_deg + 0.1,
        north_deg=latitude_deg + 0.1,
    )
    return SurfaceGrid.from_bounds(bounds, shape=(1, 1))


def test_two_station_idw_matches_an_independent_hand_calculation() -> None:
    grid = _single_cell_grid(35.0, 135.5)
    station_lat = np.array([35.0, 35.0])
    station_lon = np.array([135.0, 136.0])
    station_value = np.array([10.0, 20.0])
    target_lat, target_lon = float(grid.latitudes_deg[0]), float(grid.longitudes_deg[0])

    distance = np.array(
        [_haversine_km(target_lat, target_lon, station_lat[i], station_lon[i]) for i in range(2)]
    )
    weight = distance**-2.0
    weight /= weight.sum()
    expected = float(np.sum(weight * station_value))

    surface = interpolate_surface(
        station_lat,
        station_lon,
        station_value,
        grid=grid,
        config=IDWConfig(neighbors=2, power=2.0, max_distance_km=200.0, minimum_neighbors=2),
    )
    assert surface.support_mask[0, 0]
    assert surface.values[0, 0] == pytest.approx(expected, rel=1e-9)


def test_idw_reproduces_an_exact_station_value_at_its_own_grid_cell() -> None:
    # A grid cell within exact_tolerance_m of a station must return that station's
    # value exactly, overriding the distance weighting -- otherwise a cell almost on
    # top of a station would divide by a near-zero distance and let that one term
    # swamp the weighted average instead of simply reproducing the observation.
    grid = _single_cell_grid(35.0, 135.5)
    station_lat = np.array([35.0, 40.0, 30.0])
    station_lon = np.array([135.5, 140.0, 130.0])
    station_value = np.array([42.0, 999.0, -999.0])

    surface = interpolate_surface(
        station_lat,
        station_lon,
        station_value,
        grid=grid,
        config=IDWConfig(neighbors=3, power=2.0, max_distance_km=2000.0, minimum_neighbors=1),
    )
    assert surface.values[0, 0] == pytest.approx(42.0)
    assert surface.neighbor_count[0, 0] >= 1


def test_idw_is_invariant_to_station_order() -> None:
    grid = _single_cell_grid(35.0, 135.5)
    station_lat = RNG.uniform(33.0, 37.0, 12)
    station_lon = RNG.uniform(133.0, 138.0, 12)
    station_value = RNG.uniform(1.0, 100.0, 12)
    config = IDWConfig(neighbors=5, power=2.0, max_distance_km=500.0, minimum_neighbors=1)

    original = interpolate_surface(
        station_lat, station_lon, station_value, grid=grid, config=config
    )
    permutation = RNG.permutation(12)
    shuffled = interpolate_surface(
        station_lat[permutation],
        station_lon[permutation],
        station_value[permutation],
        grid=grid,
        config=config,
    )
    np.testing.assert_allclose(shuffled.values, original.values, equal_nan=True)


def test_log_transform_idw_is_scale_equivariant() -> None:
    # log(c * v) = log(c) + log(v); since IDW weights sum to 1, the constant log(c)
    # term passes through the weighted average unchanged, so
    # interpolate(c * values, transform="log") must equal
    # c * interpolate(values, transform="log") exactly (up to floating-point
    # rounding). This is a property of the transform, not of any particular
    # station geometry, so a random layout exercises it as well as a hand-picked one.
    grid = _single_cell_grid(35.0, 135.5)
    station_lat = RNG.uniform(33.0, 37.0, 8)
    station_lon = RNG.uniform(133.0, 138.0, 8)
    station_value = RNG.uniform(1.0, 100.0, 8)
    config = IDWConfig(neighbors=5, power=2.0, max_distance_km=500.0, minimum_neighbors=1)

    base = interpolate_surface(
        station_lat, station_lon, station_value, grid=grid, config=config, transform="log"
    )
    scale = 3.5
    scaled = interpolate_surface(
        station_lat,
        station_lon,
        station_value * scale,
        grid=grid,
        config=config,
        transform="log",
    )
    np.testing.assert_allclose(scaled.values, base.values * scale, rtol=1e-9)


def test_idw_support_radius_excludes_far_stations() -> None:
    grid = _single_cell_grid(35.0, 135.5)
    # One station right at the target cell, one far enough away that a 10 km radius
    # cannot reach it.
    station_lat = np.array([35.0, 45.0])
    station_lon = np.array([135.5, 145.0])

    plan = build_idw_plan(
        station_lat,
        station_lon,
        grid=grid,
        config=IDWConfig(neighbors=2, max_distance_km=10.0, minimum_neighbors=1),
    )
    assert plan.neighbor_count[0] == 1


def test_idw_with_a_single_neighbor_matches_nearest_assignment() -> None:
    # neighbors=1 is a real, reachable configuration -- scipy.spatial.cKDTree
    # drops the trailing axis when k == 1, and build_idw_plan reshapes it back
    # to (cell_count, 1); exercised here across several grid cells rather
    # than the single-cell grids the other tests use, since the reshape is
    # the thing under test.
    bounds = GeographicBounds(west_deg=134.0, south_deg=34.0, east_deg=136.0, north_deg=36.0)
    grid = SurfaceGrid.from_bounds(bounds, shape=(4, 4))
    station_lat = np.array([34.5, 35.5])
    station_lon = np.array([134.5, 135.5])
    station_value = np.array([10.0, 20.0])

    surface = interpolate_surface(
        station_lat,
        station_lon,
        station_value,
        grid=grid,
        method="idw",
        config=IDWConfig(neighbors=1, max_distance_km=500.0, minimum_neighbors=1),
    )
    nearest_surface = interpolate_surface(
        station_lat, station_lon, station_value, grid=grid,
        method="nearest", config=NearestConfig(allow_extrapolation=True),
    )
    assert np.all(surface.support_mask)
    np.testing.assert_array_equal(surface.values, nearest_surface.values)


def test_idw_requires_minimum_neighbors_for_support() -> None:
    grid = _single_cell_grid(35.0, 135.5)
    # Two stations within range, but far enough from the target that neither is an
    # exact match; minimum_neighbors=3 can never be satisfied by only 2 stations.
    station_lat = np.array([35.5, 34.5])
    station_lon = np.array([135.5, 135.5])
    station_value = np.array([1.0, 2.0])

    surface = interpolate_surface(
        station_lat,
        station_lon,
        station_value,
        grid=grid,
        config=IDWConfig(neighbors=4, max_distance_km=200.0, minimum_neighbors=3),
    )
    assert not surface.support_mask[0, 0]
    assert np.isnan(surface.values[0, 0])


def test_idw_config_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError, match="neighbors"):
        IDWConfig(neighbors=0, max_distance_km=10.0)
    with pytest.raises(ValueError, match="power"):
        IDWConfig(neighbors=4, power=0.0, max_distance_km=10.0)
    with pytest.raises(ValueError, match="power"):
        IDWConfig(neighbors=4, power=17.0, max_distance_km=10.0)
    with pytest.raises(ValueError, match="minimum_neighbors"):
        IDWConfig(neighbors=4, minimum_neighbors=5, max_distance_km=10.0)
    with pytest.raises(ValueError, match="max_distance_km"):
        IDWConfig(neighbors=4, max_distance_km=-1.0)
    with pytest.raises(ValueError, match="exact_tolerance_m"):
        IDWConfig(neighbors=4, max_distance_km=10.0, exact_tolerance_m=-1.0)


def test_suzuki_2017_pga_preset_matches_the_published_geometry() -> None:
    config = IDWConfig.suzuki_2017_pga()
    assert config.neighbors == 4
    assert config.power == 2.0
    assert config.max_distance_km == 50.0
    assert config.minimum_neighbors == 4
