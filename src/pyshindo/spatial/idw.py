"""Local Shepard-style inverse-distance-weighted interpolation.

Donald Shepard, "A Two-Dimensional Interpolation Function for
Irregularly-Spaced Data," Proceedings of the 1968 ACM National Conference.
https://doi.org/10.1145/800186.810616

Each grid cell's value is a weighted average of its ``config.neighbors``
nearest stations, weighted by ``distance ** -config.power`` and normalized
to sum to one. A grid cell within ``config.exact_tolerance_m`` of a station
reproduces that station's value exactly, matching Shepard's original
exact-interpolation property -- otherwise a cell sitting almost on top of a
station would divide by a near-zero distance and let that one term
dominate the weighted average, rather than simply returning the station's
own observed value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from ._geometry import chord_to_great_circle_km, great_circle_radius_to_chord, lonlat_to_unit_xyz
from ._transform import forward_transform, inverse_transform
from .models import (
    IDWConfig,
    InterpolatedSurface,
    SpatialMethod,
    SurfaceGrid,
    SurfaceMetadata,
    ValueTransform,
)
from .validation import validate_station_coordinates, validate_values

type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class IDWPlan:
    """A precomputed IDW neighbor search, reusable across many frames."""

    grid: SurfaceGrid
    station_count: int
    neighbor_index: npt.NDArray[np.int64]
    neighbor_weight: FloatArray
    support_mask: npt.NDArray[np.bool_]
    exact_station_index: npt.NDArray[np.int64]
    nearest_distance_km: FloatArray
    neighbor_count: npt.NDArray[np.uint16]
    config: IDWConfig

    def interpolate(
        self,
        values: npt.ArrayLike,
        *,
        transform: ValueTransform | str = ValueTransform.IDENTITY,
        metric_name: str = "generic",
        unit: str | None = None,
        component_definition: str | None = None,
    ) -> InterpolatedSurface:
        """Interpolate one frame of station values onto this plan's grid."""
        mode = transform if isinstance(transform, ValueTransform) else ValueTransform(transform)
        source = forward_transform(validate_values(values, self.station_count), mode)
        result = np.full(self.grid.cell_count, np.nan, dtype=np.float64)
        rows = np.flatnonzero(self.support_mask)
        estimate = np.sum(
            self.neighbor_weight[rows] * source[self.neighbor_index[rows]], axis=1
        )
        exact = self.exact_station_index[rows] >= 0
        estimate[exact] = source[self.exact_station_index[rows][exact]]
        result[rows] = estimate
        result = inverse_transform(result, mode)
        return InterpolatedSurface(
            grid=self.grid,
            values=result.reshape(self.grid.shape),
            support_mask=self.support_mask.reshape(self.grid.shape),
            nearest_station_distance_km=self.nearest_distance_km.reshape(self.grid.shape),
            neighbor_count=self.neighbor_count.reshape(self.grid.shape),
            metadata=SurfaceMetadata(
                method=SpatialMethod.IDW,
                transform=mode,
                metric_name=metric_name,
                unit=unit,
                component_definition=component_definition,
                station_count=self.station_count,
                parameters={
                    "neighbors": self.config.neighbors,
                    "power": self.config.power,
                    "max_distance_km": self.config.max_distance_km,
                    "minimum_neighbors": self.config.minimum_neighbors,
                    "exact_tolerance_m": self.config.exact_tolerance_m,
                },
            ),
        )


def build_idw_plan(
    latitudes_deg: npt.ArrayLike,
    longitudes_deg: npt.ArrayLike,
    *,
    grid: SurfaceGrid,
    config: IDWConfig,
) -> IDWPlan:
    """Precompute IDW neighbor indices and weights for every grid cell."""
    lat, lon = validate_station_coordinates(latitudes_deg, longitudes_deg, minimum_count=1)
    tree = cKDTree(lonlat_to_unit_xyz(lon, lat), copy_data=True)
    k = min(config.neighbors, lat.size)
    upper_chord = (
        np.inf
        if config.max_distance_km is None
        else great_circle_radius_to_chord(config.max_distance_km)
    )
    grid_lon, grid_lat = grid.flat_coordinates()
    chord, index = tree.query(
        lonlat_to_unit_xyz(grid_lon, grid_lat), k=k, distance_upper_bound=upper_chord
    )
    # cKDTree.query drops the trailing axis when k == 1; restoring shape (M, k)
    # here lets every line below handle both cases identically.
    chord = np.asarray(chord, dtype=np.float64).reshape(grid.cell_count, k)
    index = np.asarray(index, dtype=np.int64).reshape(grid.cell_count, k)

    # Beyond distance_upper_bound (or beyond the last real station, when
    # k > station_count), SciPy fills chord with inf and index with station_count --
    # an intentionally out-of-bounds sentinel, not a real neighbor.
    valid = np.isfinite(chord) & (index < lat.size)
    neighbor_count = valid.sum(axis=1).astype(np.uint16)
    safe_index = np.where(valid, index, 0)
    distance_km = chord_to_great_circle_km(np.where(valid, chord, 0.0))
    nearest_distance_km = np.where(valid[:, 0], distance_km[:, 0], np.inf)

    exact = valid[:, 0] & (nearest_distance_km <= config.exact_tolerance_m / 1000.0)
    exact_station_index = np.where(exact, safe_index[:, 0], -1).astype(np.int64)
    support_mask = exact | (neighbor_count >= config.minimum_neighbors)

    weight = np.zeros_like(distance_km)
    weighted_rows = support_mask & ~exact
    if np.any(weighted_rows):
        row_distance = distance_km[weighted_rows]
        row_valid = valid[weighted_rows]
        with np.errstate(divide="ignore"):
            row_weight = np.where(row_valid, row_distance ** -config.power, 0.0)
        row_weight_sum = row_weight.sum(axis=1, keepdims=True)
        weight[weighted_rows] = row_weight / row_weight_sum
    weight[exact, 0] = 1.0

    return IDWPlan(
        grid=grid,
        station_count=lat.size,
        neighbor_index=safe_index,
        neighbor_weight=weight,
        support_mask=support_mask,
        exact_station_index=exact_station_index,
        nearest_distance_km=nearest_distance_km,
        neighbor_count=neighbor_count,
        config=config,
    )
