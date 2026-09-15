from __future__ import annotations

import numpy as np
import pytest

from pyshindo.spatial import GeographicBounds, LinearConfig, SurfaceGrid, interpolate_surface
from pyshindo.spatial._geometry import aeqd_project_km
from pyshindo.spatial.linear import build_linear_plan

RNG = np.random.default_rng(20260915)


def _bounds() -> GeographicBounds:
    return GeographicBounds(west_deg=132.0, south_deg=33.0, east_deg=138.0, north_deg=39.0)


def test_linear_reconstructs_an_affine_function_exactly_inside_the_hull() -> None:
    # Piecewise-linear (barycentric) interpolation is exact for any affine function of
    # the coordinates it triangulates on -- the defining property of the method, not
    # an incidental one. Station values are assigned as a known plane in the same
    # azimuthal-equidistant projection the plan itself uses internally, so the
    # reconstructed surface can be checked against that plane directly at arbitrary
    # points inside the convex hull.
    bounds = _bounds()
    center_lon = 0.5 * (bounds.west_deg + bounds.east_deg)
    center_lat = 0.5 * (bounds.south_deg + bounds.north_deg)
    station_lat = RNG.uniform(bounds.south_deg + 0.2, bounds.north_deg - 0.2, 25)
    station_lon = RNG.uniform(bounds.west_deg + 0.2, bounds.east_deg - 0.2, 25)

    coefficient_x, coefficient_y, intercept = 0.7, -1.3, 5.0

    def plane(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        xy = aeqd_project_km(
            lon, lat, center_longitude_deg=center_lon, center_latitude_deg=center_lat
        )
        return coefficient_x * xy[:, 0] + coefficient_y * xy[:, 1] + intercept

    station_value = plane(station_lat, station_lon)
    grid = SurfaceGrid.from_bounds(bounds, shape=(20, 20))

    surface = interpolate_surface(
        station_lat, station_lon, station_value, grid=grid, method="linear", config=LinearConfig()
    )

    grid_lon, grid_lat = grid.flat_coordinates()
    expected = plane(grid_lat, grid_lon).reshape(grid.shape)
    supported = surface.support_mask
    assert supported.sum() > 0  # the test is vacuous otherwise
    np.testing.assert_allclose(surface.values[supported], expected[supported], atol=1e-6)


def test_linear_is_undefined_outside_the_convex_hull() -> None:
    bounds = _bounds()
    # Three stations clustered tightly in the middle of the grid leave most of the
    # grid, including every corner, outside their triangle.
    station_lat = np.array([35.9, 36.0, 36.1])
    station_lon = np.array([134.9, 135.1, 134.9])
    station_value = np.array([1.0, 2.0, 3.0])
    grid = SurfaceGrid.from_bounds(bounds, shape=(10, 10))

    surface = interpolate_surface(
        station_lat, station_lon, station_value, grid=grid, method="linear", config=LinearConfig()
    )
    assert not surface.support_mask[0, 0]  # southwest corner
    assert not surface.support_mask[-1, -1]  # northeast corner
    assert np.isnan(surface.values[0, 0])
    assert np.all(surface.neighbor_count[~surface.support_mask] == 0)
    assert np.all(surface.neighbor_count[surface.support_mask] == 3)


def test_linear_rejects_collinear_stations() -> None:
    bounds = _bounds()
    station_lat = np.array([35.0, 36.0, 37.0])
    station_lon = np.array([135.0, 135.0, 135.0])
    grid = SurfaceGrid.from_bounds(bounds, shape=(5, 5))
    with pytest.raises(ValueError, match="collinear|degenerate"):
        build_linear_plan(station_lat, station_lon, grid=grid, config=LinearConfig())


def test_linear_config_rejects_a_non_positive_maximum_extent() -> None:
    with pytest.raises(ValueError, match="maximum_extent_km"):
        LinearConfig(maximum_extent_km=0.0)


def test_linear_config_defaults_when_not_given() -> None:
    bounds = _bounds()
    station_lat = np.array([35.0, 36.0, 37.0])
    station_lon = np.array([135.0, 136.0, 134.0])
    grid = SurfaceGrid.from_bounds(bounds, shape=(5, 5))
    plan = build_linear_plan(station_lat, station_lon, grid=grid)
    assert plan.config.maximum_extent_km == pytest.approx(2000.0)


def test_linear_rejects_a_grid_extent_beyond_the_configured_limit() -> None:
    # A grid spanning most of the globe from its own center badly violates the local
    # planar-projection assumption; the method should say so rather than silently
    # returning a distorted triangulation.
    bounds = GeographicBounds(west_deg=-170.0, south_deg=-80.0, east_deg=170.0, north_deg=80.0)
    station_lat = np.array([0.0, 10.0, -10.0])
    station_lon = np.array([0.0, 10.0, -10.0])
    grid = SurfaceGrid.from_bounds(bounds, shape=(5, 5))
    with pytest.raises(ValueError, match="maximum_extent_km"):
        build_linear_plan(
            station_lat, station_lon, grid=grid, config=LinearConfig(maximum_extent_km=1000.0)
        )
