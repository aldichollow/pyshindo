"""Piecewise-linear interpolation on a Delaunay triangulation.

Projects the stations onto a local azimuthal-equidistant plane (see
:func:`pyshindo.spatial._geometry.aeqd_project_km`), triangulates them, and
linearly interpolates within each triangle using barycentric coordinates:
exact inside the station convex hull, undefined (unsupported) outside it.
The parameter-light alternative to :mod:`pyshindo.spatial.idw` -- no
radius, no neighbor count, no power to choose -- at the cost of a hard edge
at the hull boundary instead of IDW's smooth falloff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay, QhullError, cKDTree

from ._geometry import (
    aeqd_project_km,
    chord_to_great_circle_km,
    great_circle_distance_km,
    lonlat_to_unit_xyz,
)
from ._transform import forward_transform, inverse_transform
from .models import (
    InterpolatedSurface,
    LinearConfig,
    SpatialMethod,
    SurfaceGrid,
    SurfaceMetadata,
    ValueTransform,
)
from .validation import validate_station_coordinates, validate_values

type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LinearPlan:
    """A precomputed Delaunay triangulation, reusable across many frames."""

    grid: SurfaceGrid
    station_count: int
    vertex_index: npt.NDArray[np.int64]
    barycentric_weight: FloatArray
    support_mask: npt.NDArray[np.bool_]
    nearest_distance_km: FloatArray
    config: LinearConfig

    def interpolate(
        self,
        values: npt.ArrayLike,
        *,
        transform: ValueTransform | str = ValueTransform.IDENTITY,
        metric_name: str = "generic",
        unit: str | None = None,
        component_definition: str | None = None,
    ) -> InterpolatedSurface:
        """Linearly interpolate one frame of station values onto this plan's grid."""
        mode = transform if isinstance(transform, ValueTransform) else ValueTransform(transform)
        source = forward_transform(validate_values(values, self.station_count), mode)
        result = np.full(self.grid.cell_count, np.nan, dtype=np.float64)
        rows = np.flatnonzero(self.support_mask)
        result[rows] = np.sum(
            self.barycentric_weight[rows] * source[self.vertex_index[rows]], axis=1
        )
        result = inverse_transform(result, mode)
        neighbor_count = np.where(self.support_mask, 3, 0).astype(np.uint16)
        return InterpolatedSurface(
            grid=self.grid,
            values=result.reshape(self.grid.shape),
            support_mask=self.support_mask.reshape(self.grid.shape),
            nearest_station_distance_km=self.nearest_distance_km.reshape(self.grid.shape),
            neighbor_count=neighbor_count.reshape(self.grid.shape),
            metadata=SurfaceMetadata(
                method=SpatialMethod.LINEAR,
                transform=mode,
                metric_name=metric_name,
                unit=unit,
                component_definition=component_definition,
                station_count=self.station_count,
                parameters={"maximum_extent_km": self.config.maximum_extent_km},
            ),
        )


def build_linear_plan(
    latitudes_deg: npt.ArrayLike,
    longitudes_deg: npt.ArrayLike,
    *,
    grid: SurfaceGrid,
    config: LinearConfig | None = None,
) -> LinearPlan:
    """Triangulate the stations and precompute barycentric weights for every cell.

    Raises ``ValueError`` if the stations are collinear or otherwise too
    degenerate to triangulate, rather than propagating SciPy's Qhull error
    message, which does not mention stations, triangles, or this package.
    """
    if config is None:
        config = LinearConfig()
    lat, lon = validate_station_coordinates(latitudes_deg, longitudes_deg, minimum_count=3)
    center_longitude_deg = 0.5 * (grid.bounds.west_deg + grid.bounds.east_deg)
    center_latitude_deg = 0.5 * (grid.bounds.south_deg + grid.bounds.north_deg)
    extent_km = max(
        great_circle_distance_km(
            center_longitude_deg, center_latitude_deg, corner_longitude_deg, corner_latitude_deg
        )
        for corner_longitude_deg, corner_latitude_deg in grid.image_corners
    )
    if extent_km > config.maximum_extent_km:
        raise ValueError(
            f"The grid spans {extent_km:.0f} km from its center, beyond "
            f"config.maximum_extent_km={config.maximum_extent_km:.0f} km. The local "
            "planar projection this method relies on degrades over very large extents; "
            "use a smaller grid, raise maximum_extent_km deliberately, or use "
            "method='idw' instead."
        )

    station_xy = aeqd_project_km(
        lon,
        lat,
        center_longitude_deg=center_longitude_deg,
        center_latitude_deg=center_latitude_deg,
    )
    grid_lon, grid_lat = grid.flat_coordinates()
    grid_xy = aeqd_project_km(
        grid_lon,
        grid_lat,
        center_longitude_deg=center_longitude_deg,
        center_latitude_deg=center_latitude_deg,
    )
    try:
        triangulation = Delaunay(station_xy)
    except QhullError as exc:
        raise ValueError(
            "Could not triangulate the stations; they may be collinear or otherwise "
            "degenerate. method='idw' or method='nearest' do not require a "
            "triangulation and may work where this cannot."
        ) from exc

    simplex = triangulation.find_simplex(grid_xy)
    support_mask = simplex >= 0
    vertex_index = np.zeros((grid.cell_count, 3), dtype=np.int64)
    weight = np.zeros((grid.cell_count, 3), dtype=np.float64)
    rows = np.flatnonzero(support_mask)
    if rows.size:
        selected = simplex[rows]
        # scipy.spatial.Delaunay.transform[i] holds, for simplex i, the inverse
        # transform Tinv (rows :-1) and offset r (last row) such that the first
        # ndim barycentric coordinates are Tinv @ (point - r); the final
        # coordinate is 1 minus their sum. This is SciPy's own documented
        # recipe for barycentric coordinates, reused here rather than calling
        # LinearNDInterpolator per frame. Qhull fills transform with NaN for a
        # degenerate (zero-area) simplex; such a cell is excluded from
        # support_mask below rather than silently interpolating from NaN
        # weights, since points inside a real, non-degenerate hull should
        # never land in one.
        affine = triangulation.transform[selected]
        offset = grid_xy[rows] - affine[:, 2, :]
        first_two = np.einsum("nij,nj->ni", affine[:, :2, :], offset)
        weight[rows, :2] = first_two
        weight[rows, 2] = 1.0 - first_two.sum(axis=1)
        vertex_index[rows] = triangulation.simplices[selected]
        degenerate = ~np.all(np.isfinite(weight[rows]), axis=1)
        if np.any(degenerate):
            degenerate_rows = rows[degenerate]
            support_mask[degenerate_rows] = False
            weight[degenerate_rows] = 0.0
            vertex_index[degenerate_rows] = 0

    tree = cKDTree(lonlat_to_unit_xyz(lon, lat), copy_data=True)
    chord, _ = tree.query(lonlat_to_unit_xyz(grid_lon, grid_lat), k=1)
    nearest_distance_km = chord_to_great_circle_km(chord)

    return LinearPlan(
        grid=grid,
        station_count=lat.size,
        vertex_index=vertex_index,
        barycentric_weight=weight,
        support_mask=support_mask,
        nearest_distance_km=nearest_distance_km,
        config=config,
    )
