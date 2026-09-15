from __future__ import annotations

import numpy as np
import pytest

from pyshindo.spatial import GeographicBounds, NearestConfig, SurfaceGrid, interpolate_surface
from pyshindo.spatial._geometry import great_circle_distance_km
from pyshindo.spatial.nearest import build_nearest_plan

RNG = np.random.default_rng(20260915)


def _bounds() -> GeographicBounds:
    return GeographicBounds(west_deg=132.0, south_deg=33.0, east_deg=138.0, north_deg=39.0)


def test_nearest_matches_a_brute_force_search() -> None:
    bounds = _bounds()
    grid = SurfaceGrid.from_bounds(bounds, shape=(6, 6))
    station_lat = RNG.uniform(bounds.south_deg, bounds.north_deg, 15)
    station_lon = RNG.uniform(bounds.west_deg, bounds.east_deg, 15)
    station_value = np.arange(15, dtype=np.float64)

    surface = interpolate_surface(
        station_lat,
        station_lon,
        station_value,
        grid=grid,
        method="nearest",
        config=NearestConfig(allow_extrapolation=True),
    )

    grid_lon, grid_lat = grid.flat_coordinates()
    expected = np.empty(grid.cell_count)
    for cell in range(grid.cell_count):
        distances = [
            great_circle_distance_km(grid_lon[cell], grid_lat[cell], station_lon[i], station_lat[i])
            for i in range(15)
        ]
        expected[cell] = station_value[int(np.argmin(distances))]

    np.testing.assert_allclose(surface.values.reshape(-1), expected)


def test_nearest_preserves_class_like_integer_labels() -> None:
    # The one method meant to be honest about a metric with no continuous precursor:
    # a caller can pass class numbers directly, and every supported cell must equal
    # exactly one input value, never an average of two.
    bounds = _bounds()
    grid = SurfaceGrid.from_bounds(bounds, shape=(8, 8))
    station_lat = RNG.uniform(bounds.south_deg, bounds.north_deg, 10)
    station_lon = RNG.uniform(bounds.west_deg, bounds.east_deg, 10)
    classes = RNG.integers(0, 5, 10).astype(np.float64)

    surface = interpolate_surface(
        station_lat,
        station_lon,
        classes,
        grid=grid,
        method="nearest",
        config=NearestConfig(allow_extrapolation=True),
    )
    observed = np.unique(surface.values[surface.support_mask])
    assert set(observed).issubset(set(classes))


def test_nearest_support_radius_excludes_cells_too_far_from_any_station() -> None:
    bounds = _bounds()
    grid = SurfaceGrid.from_bounds(bounds, shape=(10, 10))
    # A single station near the southwest corner; a small radius should leave the
    # far (northeast) side of the grid unsupported.
    plan = build_nearest_plan(
        np.array([33.2]),
        np.array([132.2]),
        grid=grid,
        config=NearestConfig(max_distance_km=50.0),
    )
    assert not plan.support_mask.reshape(grid.shape)[-1, -1]
    assert plan.support_mask.reshape(grid.shape)[0, 0]


def test_nearest_requires_radius_or_explicit_extrapolation() -> None:
    with pytest.raises(ValueError, match="max_distance_km"):
        NearestConfig()
