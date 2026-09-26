from __future__ import annotations

import numpy as np
import pytest

from pyshindo.spatial import (
    GeographicBounds,
    IDWConfig,
    InterpolatedSurface,
    LinearConfig,
    NearestConfig,
    SpatialMethod,
    SurfaceGrid,
    SurfaceMetadata,
    ValueTransform,
    build_interpolation_plan,
    interpolate_surface,
)

RNG = np.random.default_rng(20260915)


def _bounds() -> GeographicBounds:
    return GeographicBounds(west_deg=132.0, south_deg=33.0, east_deg=138.0, north_deg=39.0)


def _stations(count: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = _bounds()
    lat = RNG.uniform(bounds.south_deg, bounds.north_deg, count)
    lon = RNG.uniform(bounds.west_deg, bounds.east_deg, count)
    value = RNG.uniform(1.0, 100.0, count)
    return lat, lon, value


def test_interpolate_surface_defaults_to_idw() -> None:
    lat, lon, value = _stations()
    grid = SurfaceGrid.from_bounds(_bounds(), shape=(10, 10))
    surface = interpolate_surface(
        lat, lon, value, grid=grid, config=IDWConfig(neighbors=4, max_distance_km=100.0)
    )
    assert surface.metadata.method is SpatialMethod.IDW


def test_interpolate_surface_defaults_transform_to_identity() -> None:
    lat, lon, value = _stations()
    grid = SurfaceGrid.from_bounds(_bounds(), shape=(10, 10))
    surface = interpolate_surface(
        lat, lon, value, grid=grid, config=IDWConfig(neighbors=4, max_distance_km=100.0)
    )
    assert surface.metadata.transform is ValueTransform.IDENTITY


def test_config_must_match_the_requested_method() -> None:
    lat, lon, _ = _stations()
    grid = SurfaceGrid.from_bounds(_bounds(), shape=(5, 5))
    with pytest.raises(TypeError, match="IDWConfig"):
        build_interpolation_plan(
            lat, lon, grid=grid, method="idw", config=NearestConfig(max_distance_km=10.0)
        )
    with pytest.raises(TypeError, match="LinearConfig"):
        build_interpolation_plan(
            lat,
            lon,
            grid=grid,
            method="linear",
            config=IDWConfig(neighbors=4, max_distance_km=10.0),
        )
    with pytest.raises(TypeError, match="NearestConfig"):
        build_interpolation_plan(
            lat, lon, grid=grid, method="nearest", config=LinearConfig()
        )


@pytest.mark.parametrize("method", ["idw", "linear", "nearest"])
def test_one_shot_and_reusable_plan_agree(method: str) -> None:
    lat, lon, value = _stations()
    grid = SurfaceGrid.from_bounds(_bounds(), shape=(10, 10))
    config: IDWConfig | LinearConfig | NearestConfig
    if method == "idw":
        config = IDWConfig(neighbors=4, max_distance_km=150.0, minimum_neighbors=1)
    elif method == "linear":
        config = LinearConfig()
    else:
        config = NearestConfig(max_distance_km=150.0)

    one_shot = interpolate_surface(lat, lon, value, grid=grid, method=method, config=config)
    plan = build_interpolation_plan(lat, lon, grid=grid, method=method, config=config)
    via_plan = plan.interpolate(value)

    np.testing.assert_array_equal(one_shot.support_mask, via_plan.support_mask)
    np.testing.assert_allclose(one_shot.values, via_plan.values, equal_nan=True)


def test_reused_plan_rebuilds_a_different_surface_per_frame() -> None:
    lat, lon, value = _stations()
    grid = SurfaceGrid.from_bounds(_bounds(), shape=(8, 8))
    plan = build_interpolation_plan(
        lat, lon, grid=grid, config=IDWConfig(neighbors=4, max_distance_km=150.0)
    )
    first = plan.interpolate(value)
    second = plan.interpolate(value * 2.0)
    supported = first.support_mask
    np.testing.assert_allclose(second.values[supported], first.values[supported] * 2.0)


def test_metadata_records_metric_unit_and_component_definition() -> None:
    lat, lon, value = _stations()
    grid = SurfaceGrid.from_bounds(_bounds(), shape=(6, 6))
    surface = interpolate_surface(
        lat,
        lon,
        value,
        grid=grid,
        config=IDWConfig(neighbors=4, max_distance_km=150.0),
        metric_name="pga",
        unit="gal",
        component_definition="horizontal_vector_peak",
    )
    assert surface.metadata.metric_name == "pga"
    assert surface.metadata.unit == "gal"
    assert surface.metadata.component_definition == "horizontal_vector_peak"
    assert surface.metadata.station_count == len(lat)


def test_interpolated_surface_rejects_arrays_shaped_unlike_its_grid() -> None:
    grid = SurfaceGrid.from_bounds(_bounds(), shape=(3, 4))
    metadata = SurfaceMetadata(
        method=SpatialMethod.IDW,
        transform=ValueTransform.IDENTITY,
        metric_name="generic",
        unit=None,
        component_definition=None,
        station_count=1,
        parameters={},
    )
    ok_shape = np.zeros(grid.shape)
    wrong_shape = np.zeros((4, 3))
    with pytest.raises(ValueError, match="values has shape"):
        InterpolatedSurface(
            grid=grid,
            values=wrong_shape,
            support_mask=ok_shape.astype(bool),
            nearest_station_distance_km=ok_shape,
            neighbor_count=ok_shape.astype(np.uint16),
            metadata=metadata,
        )


def test_sample_nearest_reads_back_the_value_at_a_supported_cell() -> None:
    lat, lon, value = _stations()
    grid = SurfaceGrid.from_bounds(_bounds(), shape=(10, 10))
    surface = interpolate_surface(
        lat, lon, value, grid=grid, config=IDWConfig(neighbors=4, max_distance_km=150.0)
    )
    supported_rows, supported_cols = np.nonzero(surface.support_mask)
    row, col = supported_rows[0], supported_cols[0]
    sample = surface.sample_nearest(
        latitude_deg=float(grid.latitudes_deg[row]), longitude_deg=float(grid.longitudes_deg[col])
    )
    assert sample == pytest.approx(float(surface.values[row, col]))
