from __future__ import annotations

import math

import numpy as np
import pytest

from pyshindo.spatial import GeographicBounds, SurfaceGrid


def test_bounds_reject_antimeridian_crossing() -> None:
    with pytest.raises(ValueError, match="antimeridian"):
        GeographicBounds(west_deg=170.0, south_deg=30.0, east_deg=-170.0, north_deg=40.0)


def test_bounds_reject_reversed_or_degenerate_extent() -> None:
    with pytest.raises(ValueError):
        GeographicBounds(west_deg=140.0, south_deg=30.0, east_deg=130.0, north_deg=40.0)
    with pytest.raises(ValueError):
        GeographicBounds(west_deg=130.0, south_deg=40.0, east_deg=140.0, north_deg=30.0)
    with pytest.raises(ValueError):
        # west == east is degenerate, not merely reversed.
        GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=130.0, north_deg=40.0)


def test_bounds_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        GeographicBounds(west_deg=float("nan"), south_deg=30.0, east_deg=140.0, north_deg=40.0)
    with pytest.raises(ValueError, match="finite"):
        GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=float("inf"), north_deg=40.0)


def test_bounds_reject_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        GeographicBounds(west_deg=-181.0, south_deg=30.0, east_deg=140.0, north_deg=40.0)
    with pytest.raises(ValueError):
        GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=140.0, north_deg=91.0)


def test_grid_from_explicit_shape_has_pixel_center_convention() -> None:
    bounds = GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=140.0, north_deg=40.0)
    grid = SurfaceGrid.from_bounds(bounds, shape=(2, 4))

    assert grid.shape == (2, 4)
    assert grid.cell_count == 8
    # 4 columns over a 10-degree span: centers at 131.25, 133.75, 136.25, 138.75.
    np.testing.assert_allclose(grid.longitudes_deg, [131.25, 133.75, 136.25, 138.75])
    # 2 rows over a 10-degree span: centers at 32.5, 37.5.
    np.testing.assert_allclose(grid.latitudes_deg, [32.5, 37.5])


def test_grid_shape_exceeding_max_cells_is_rejected_outright() -> None:
    bounds = GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=140.0, north_deg=40.0)
    with pytest.raises(ValueError, match="max_cells"):
        SurfaceGrid.from_bounds(bounds, shape=(1000, 1000), max_cells=100)


def test_grid_resolution_exceeding_max_cells_is_coarsened_not_rejected() -> None:
    # A resolution request has no single exact cell count to insist on, unlike an
    # explicit shape, so it is silently coarsened to fit instead of erroring.
    bounds = GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=140.0, north_deg=40.0)
    grid = SurfaceGrid.from_bounds(bounds, approximate_resolution_km=0.1, max_cells=500)
    assert grid.cell_count <= 500


def test_grid_rejects_a_non_positive_max_cells_or_shape() -> None:
    bounds = GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=140.0, north_deg=40.0)
    with pytest.raises(ValueError, match="max_cells"):
        SurfaceGrid.from_bounds(bounds, shape=(2, 2), max_cells=0)
    with pytest.raises(ValueError, match="positive values"):
        SurfaceGrid.from_bounds(bounds, shape=(0, 2))


def test_grid_rejects_a_non_positive_resolution() -> None:
    bounds = GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=140.0, north_deg=40.0)
    with pytest.raises(ValueError, match="approximate_resolution_km"):
        SurfaceGrid.from_bounds(bounds, approximate_resolution_km=0.0)


def test_grid_requires_exactly_one_of_shape_or_resolution() -> None:
    bounds = GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=140.0, north_deg=40.0)
    with pytest.raises(ValueError, match="exactly one"):
        SurfaceGrid.from_bounds(bounds)
    with pytest.raises(ValueError, match="exactly one"):
        SurfaceGrid.from_bounds(bounds, shape=(2, 2), approximate_resolution_km=10.0)


def test_grid_axes_must_be_one_dimensional_and_strictly_increasing() -> None:
    bounds = GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=140.0, north_deg=40.0)
    with pytest.raises(ValueError, match="longitudes_deg must be strictly increasing"):
        SurfaceGrid(bounds, np.array([135.0, 134.0, 136.0]), np.array([32.0, 35.0, 38.0]))
    with pytest.raises(ValueError, match="latitudes_deg must be strictly increasing"):
        SurfaceGrid(bounds, np.array([134.0, 135.0, 136.0]), np.array([32.0, 38.0, 35.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        SurfaceGrid(bounds, np.zeros((2, 2)), np.array([32.0, 35.0]))


def test_grid_axes_must_be_non_empty_and_finite() -> None:
    bounds = GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=140.0, north_deg=40.0)
    with pytest.raises(ValueError, match="at least one cell"):
        SurfaceGrid(bounds, np.array([]), np.array([]))
    with pytest.raises(ValueError, match="finite"):
        SurfaceGrid(bounds, np.array([134.0, np.nan]), np.array([32.0, 35.0]))


def test_grid_axes_are_read_only() -> None:
    bounds = GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=140.0, north_deg=40.0)
    grid = SurfaceGrid.from_bounds(bounds, shape=(2, 2))
    with pytest.raises(ValueError):
        grid.longitudes_deg[0] = 0.0


def test_image_corners_are_northwest_northeast_southeast_southwest() -> None:
    bounds = GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=140.0, north_deg=40.0)
    grid = SurfaceGrid.from_bounds(bounds, shape=(2, 2))
    assert grid.image_corners == (
        (130.0, 40.0),
        (140.0, 40.0),
        (140.0, 30.0),
        (130.0, 30.0),
    )


def test_flat_coordinates_are_row_major_matching_reshape() -> None:
    bounds = GeographicBounds(west_deg=130.0, south_deg=30.0, east_deg=140.0, north_deg=40.0)
    grid = SurfaceGrid.from_bounds(bounds, shape=(3, 5))
    lon, lat = grid.flat_coordinates()
    assert lon.shape == (15,)
    assert lat.shape == (15,)

    values = np.arange(15, dtype=np.float64)
    reshaped = values.reshape(grid.shape)
    # The value at flat index k must land at row k // nx, column k % nx.
    for k in (0, 4, 5, 9, 14):
        row, col = divmod(k, 5)
        assert reshaped[row, col] == values[k]
        assert lon[k] == grid.longitudes_deg[col]
        assert lat[k] == grid.latitudes_deg[row]


def test_resolution_grid_matches_a_hand_computed_cell_count() -> None:
    # At the equator, one degree of longitude and one degree of latitude are both
    # very close to EARTH_RADIUS_KM * pi / 180; picking a bounds box centered on the
    # equator makes the cos(latitude) longitude-scaling factor equal to 1, so the
    # expected cell count can be checked directly against that constant rather than
    # through the module's own derivation of it.
    km_per_degree = 6371.0088 * math.pi / 180.0
    bounds = GeographicBounds(west_deg=0.0, south_deg=-1.0, east_deg=1.0, north_deg=1.0)
    resolution_km = km_per_degree / 10.0  # 10 cells per degree
    grid = SurfaceGrid.from_bounds(bounds, approximate_resolution_km=resolution_km)
    assert grid.shape == (20, 10)
