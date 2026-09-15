"""Nearest-station (Voronoi-region) assignment.

The only method this package recommends for a metric that exists only as
class labels, with no underlying continuous value to interpolate -- see the
module-level docstring in :mod:`pyshindo.spatial`. Also useful as the
simplest, most transparent baseline in its own right: every grid cell takes
on the value of whichever station is physically closest to it, with no
smoothing and no distance weighting to justify.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ._geometry import chord_to_great_circle_km, lonlat_to_unit_xyz
from ._transform import forward_transform, inverse_transform
from .models import (
    InterpolatedSurface,
    NearestConfig,
    SpatialMethod,
    SurfaceGrid,
    SurfaceMetadata,
    ValueTransform,
)
from .validation import validate_station_coordinates, validate_values

type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class NearestPlan:
    """A precomputed nearest-station assignment, reusable across many frames."""

    grid: SurfaceGrid
    station_count: int
    nearest_index: npt.NDArray[np.int64]
    support_mask: npt.NDArray[np.bool_]
    nearest_distance_km: FloatArray
    config: NearestConfig

    def interpolate(
        self,
        values: npt.ArrayLike,
        *,
        transform: ValueTransform | str = ValueTransform.IDENTITY,
        metric_name: str = "generic",
        unit: str | None = None,
        component_definition: str | None = None,
    ) -> InterpolatedSurface:
        """Assign each supported grid cell its nearest station's value."""
        mode = transform if isinstance(transform, ValueTransform) else ValueTransform(transform)
        source = forward_transform(validate_values(values, self.station_count), mode)
        result = np.full(self.grid.cell_count, np.nan, dtype=np.float64)
        result[self.support_mask] = source[self.nearest_index[self.support_mask]]
        result = inverse_transform(result, mode)
        neighbor_count = self.support_mask.astype(np.uint16)
        return InterpolatedSurface(
            grid=self.grid,
            values=result.reshape(self.grid.shape),
            support_mask=self.support_mask.reshape(self.grid.shape),
            nearest_station_distance_km=self.nearest_distance_km.reshape(self.grid.shape),
            neighbor_count=neighbor_count.reshape(self.grid.shape),
            metadata=SurfaceMetadata(
                method=SpatialMethod.NEAREST,
                transform=mode,
                metric_name=metric_name,
                unit=unit,
                component_definition=component_definition,
                station_count=self.station_count,
                parameters={"max_distance_km": self.config.max_distance_km},
            ),
        )


def build_nearest_plan(
    latitudes_deg: npt.ArrayLike,
    longitudes_deg: npt.ArrayLike,
    *,
    grid: SurfaceGrid,
    config: NearestConfig,
) -> NearestPlan:
    """Precompute the nearest station for every grid cell."""
    lat, lon = validate_station_coordinates(latitudes_deg, longitudes_deg, minimum_count=1)
    # lonlat_to_unit_xyz always returns a fresh array with no other reference to
    # it, so there is nothing for cKDTree's own copy_data=True to protect against.
    tree = cKDTree(lonlat_to_unit_xyz(lon, lat))
    grid_lon, grid_lat = grid.flat_coordinates()
    # workers=-1 parallelizes the query across all available CPU cores; for a
    # few hundred thousand grid cells this is the dominant cost of building a
    # plan (profiled), and the result is identical to a single-threaded query.
    chord, index = tree.query(lonlat_to_unit_xyz(grid_lon, grid_lat), k=1, workers=-1)
    distance_km = chord_to_great_circle_km(chord)
    support_mask = (
        np.ones(distance_km.shape, dtype=bool)
        if config.max_distance_km is None
        else distance_km <= config.max_distance_km
    )
    return NearestPlan(
        grid=grid,
        station_count=lat.size,
        nearest_index=np.asarray(index, dtype=np.int64),
        support_mask=support_mask,
        nearest_distance_km=distance_km,
        config=config,
    )
