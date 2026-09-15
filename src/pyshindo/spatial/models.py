"""Shared types for observation-station spatial interpolation.

An :class:`InterpolatedSurface` is a station interpolation surface, not an
estimated ground-motion field: it spreads observed values across a grid by
geometry alone, with no source model, site amplification, or ground-motion
prediction equation involved. Treat it as what it is -- a smoothed view of
the same observations the existing marker maps in
:mod:`pyshindo.plotting.maps` already plot -- not as a substitute for JMA's
own estimated seismic intensity distribution or a ShakeMap.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ._geometry import EARTH_RADIUS_KM

type FloatArray = npt.NDArray[np.float64]
type ArrayLike = npt.ArrayLike

_KM_PER_DEGREE: float = EARTH_RADIUS_KM * math.pi / 180.0


class ValueTransform(StrEnum):
    """A monotone transform applied before interpolation and undone after.

    Averaging distance-weighted neighbors is only physically sensible in the
    space being averaged. Ground-motion amplitudes are often closer to
    log-normal than normal, so a log transform is common practice for
    quantities such as PGA -- but it is never applied silently here. See
    :mod:`pyshindo.spatial` for the recommended transform per metric.
    """

    IDENTITY = "identity"
    LOG = "log"


class SpatialMethod(StrEnum):
    """One of the three interpolation methods this package implements.

    ``IDW`` gives a smooth field with a small number of parameters to
    justify. ``LINEAR`` is parameter-light and exact inside the station
    convex hull, at the cost of a hard edge outside it. ``NEAREST`` makes no
    smoothness assumption at all and is the only one of the three honest to
    use when only class labels, not the underlying continuous value, are
    available -- see the module-level docstring in
    :mod:`pyshindo.spatial` for why classes and colors are never
    interpolated directly.
    """

    IDW = "idw"
    LINEAR = "linear"
    NEAREST = "nearest"


@dataclass(frozen=True, slots=True)
class GeographicBounds:
    """A rectangular west/south/east/north extent in degrees.

    Longitude must not wrap the antimeridian (``west_deg < east_deg`` is
    required, not merely ``west_deg != east_deg``): a wrapped grid would
    need a reversed longitude axis and a split image, which the rest of
    this subpackage and :mod:`pyshindo.plotting.surfaces` do not support.
    """

    west_deg: float
    south_deg: float
    east_deg: float
    north_deg: float

    def __post_init__(self) -> None:
        values = (self.west_deg, self.south_deg, self.east_deg, self.north_deg)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("GeographicBounds values must be finite.")
        if not -180.0 <= self.west_deg < self.east_deg <= 180.0:
            raise ValueError(
                "GeographicBounds requires -180 <= west_deg < east_deg <= 180 (a grid "
                f"crossing the antimeridian is not supported); received "
                f"west_deg={self.west_deg}, east_deg={self.east_deg}."
            )
        if not -90.0 <= self.south_deg < self.north_deg <= 90.0:
            raise ValueError(
                "GeographicBounds requires -90 <= south_deg < north_deg <= 90; received "
                f"south_deg={self.south_deg}, north_deg={self.north_deg}."
            )


@dataclass(frozen=True, slots=True)
class SurfaceGrid:
    """A regular grid of pixel centers over a :class:`GeographicBounds` extent.

    ``longitudes_deg`` and ``latitudes_deg`` are one-dimensional axes, not a
    meshgrid; every array this subpackage returns is shaped
    ``(latitudes_deg.size, longitudes_deg.size)`` -- row 0 is the
    southernmost row, matching how the axes are stored, not how a raster
    image is usually read. :mod:`pyshindo.plotting.surfaces` flips the
    array vertically when it builds an image from a surface; see its module
    docstring.
    """

    bounds: GeographicBounds
    longitudes_deg: FloatArray
    latitudes_deg: FloatArray

    def __post_init__(self) -> None:
        lon = np.array(self.longitudes_deg, dtype=np.float64, copy=True)
        lat = np.array(self.latitudes_deg, dtype=np.float64, copy=True)
        if lon.ndim != 1 or lat.ndim != 1:
            raise ValueError(
                "SurfaceGrid.longitudes_deg and latitudes_deg must be one-dimensional."
            )
        if lon.size < 1 or lat.size < 1:
            raise ValueError("SurfaceGrid requires at least one cell along each axis.")
        if not (np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))):
            raise ValueError("SurfaceGrid.longitudes_deg and latitudes_deg must be finite.")
        if lon.size > 1 and np.any(np.diff(lon) <= 0.0):
            raise ValueError("SurfaceGrid.longitudes_deg must be strictly increasing.")
        if lat.size > 1 and np.any(np.diff(lat) <= 0.0):
            raise ValueError("SurfaceGrid.latitudes_deg must be strictly increasing.")
        lon.setflags(write=False)
        lat.setflags(write=False)
        object.__setattr__(self, "longitudes_deg", lon)
        object.__setattr__(self, "latitudes_deg", lat)

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(ny, nx)``: row count first, matching every returned array."""
        return (self.latitudes_deg.size, self.longitudes_deg.size)

    @property
    def cell_count(self) -> int:
        """Return the total number of grid cells, ``ny * nx``."""
        return self.latitudes_deg.size * self.longitudes_deg.size

    @property
    def image_corners(self) -> tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ]:
        """Return the four outer-edge corners as (longitude, latitude) pairs.

        Ordered northwest, northeast, southeast, southwest: the order
        Plotly's ``layout.map.layers`` image-layer ``coordinates`` property
        expects.
        """
        bounds = self.bounds
        return (
            (bounds.west_deg, bounds.north_deg),
            (bounds.east_deg, bounds.north_deg),
            (bounds.east_deg, bounds.south_deg),
            (bounds.west_deg, bounds.south_deg),
        )

    def flat_coordinates(self) -> tuple[FloatArray, FloatArray]:
        """Return every cell center as flat ``(longitude, latitude)`` arrays.

        Row-major, matching ``values.reshape(-1)`` for an array shaped like
        :attr:`shape`.
        """
        lon = np.tile(self.longitudes_deg, self.latitudes_deg.size)
        lat = np.repeat(self.latitudes_deg, self.longitudes_deg.size)
        return lon, lat

    @classmethod
    def from_bounds(
        cls,
        bounds: GeographicBounds,
        *,
        shape: tuple[int, int] | None = None,
        approximate_resolution_km: float | None = None,
        max_cells: int = 750_000,
    ) -> SurfaceGrid:
        """Build a grid either from an exact cell count or an approximate resolution.

        Exactly one of ``shape`` (``(ny, nx)``) or ``approximate_resolution_km``
        must be given. An explicit ``shape`` that exceeds ``max_cells`` is
        rejected outright, since the caller asked for those exact
        dimensions. A resolution that would exceed ``max_cells`` is instead
        coarsened just enough to fit, since a resolution request has no
        single exact cell count to insist on in the first place.
        """
        if (shape is None) == (approximate_resolution_km is None):
            raise ValueError("Provide exactly one of shape or approximate_resolution_km.")
        if max_cells < 1:
            raise ValueError("max_cells must be at least 1.")

        if shape is not None:
            ny, nx = (int(value) for value in shape)
            if ny < 1 or nx < 1:
                raise ValueError(f"shape must contain positive values; received {shape}.")
            if ny * nx > max_cells:
                raise ValueError(
                    f"shape {(ny, nx)} requests {ny * nx} cells, more than "
                    f"max_cells={max_cells}. Reduce shape or raise max_cells explicitly."
                )
        else:
            resolution_km = float(approximate_resolution_km)  # type: ignore[arg-type]
            if not math.isfinite(resolution_km) or resolution_km <= 0.0:
                raise ValueError("approximate_resolution_km must be finite and positive.")
            mid_latitude_rad = math.radians(0.5 * (bounds.south_deg + bounds.north_deg))
            width_km = (bounds.east_deg - bounds.west_deg) * _KM_PER_DEGREE * math.cos(
                mid_latitude_rad
            )
            height_km = (bounds.north_deg - bounds.south_deg) * _KM_PER_DEGREE
            nx = max(1, round(width_km / resolution_km))
            ny = max(1, round(height_km / resolution_km))
            if ny * nx > max_cells:
                scale = math.sqrt(max_cells / (ny * nx))
                nx = max(1, round(nx * scale))
                ny = max(1, round(ny * scale))

        longitude_step = (bounds.east_deg - bounds.west_deg) / nx
        latitude_step = (bounds.north_deg - bounds.south_deg) / ny
        longitudes = bounds.west_deg + (np.arange(nx, dtype=np.float64) + 0.5) * longitude_step
        latitudes = bounds.south_deg + (np.arange(ny, dtype=np.float64) + 0.5) * latitude_step
        return cls(bounds, longitudes, latitudes)


def _require_radius_or_extrapolation(
    max_distance_km: float | None, allow_extrapolation: bool
) -> None:
    if max_distance_km is None:
        if not allow_extrapolation:
            raise ValueError(
                "max_distance_km is required unless allow_extrapolation=True. An "
                "unbounded search radius would let one distant station's value spread "
                "across an entire unsupported region with nothing marking it as such."
            )
    elif not math.isfinite(max_distance_km) or max_distance_km <= 0.0:
        raise ValueError("max_distance_km must be finite and positive.")


@dataclass(frozen=True, slots=True)
class IDWConfig:
    """Configuration for local Shepard-style inverse-distance weighting.

    Donald Shepard, "A Two-Dimensional Interpolation Function for
    Irregularly-Spaced Data," Proceedings of the 1968 ACM National
    Conference. https://doi.org/10.1145/800186.810616
    """

    neighbors: int
    power: float = 2.0
    max_distance_km: float | None = None
    minimum_neighbors: int = 1
    exact_tolerance_m: float = 1.0
    allow_extrapolation: bool = False

    def __post_init__(self) -> None:
        if self.neighbors < 1:
            raise ValueError("neighbors must be at least 1.")
        if not math.isfinite(self.power) or not 0.0 < self.power <= 16.0:
            # An upper bound, not a physical constant: distance ** -power can overflow
            # float64 for a very small distance and a very large power. 16 is generous
            # for any published IDW exponent (2 is by far the most common) while keeping
            # that overflow unreachable in practice.
            raise ValueError("power must be finite and lie in (0, 16].")
        if not 1 <= self.minimum_neighbors <= self.neighbors:
            raise ValueError("minimum_neighbors must lie in [1, neighbors].")
        if not math.isfinite(self.exact_tolerance_m) or self.exact_tolerance_m < 0.0:
            raise ValueError("exact_tolerance_m must be finite and non-negative.")
        _require_radius_or_extrapolation(self.max_distance_km, self.allow_extrapolation)

    @classmethod
    def suzuki_2017_pga(cls) -> IDWConfig:
        """Reproduce the PGA interpolation geometry in Suzuki et al. (2017).

        Wataru Suzuki et al., "Strong motions observed by K-NET and KiK-net
        during the 2016 Kumamoto earthquake sequence," Earth, Planets and
        Space 69, 19 (2017). https://doi.org/10.1186/s40623-017-0604-8

        The paper interpolates PGA from the four nearest stations within
        50 km, with inverse-distance-squared weights, displayed on a 1 km
        grid. The 1 km display resolution is a rendering choice -- pass it
        to :meth:`SurfaceGrid.from_bounds` separately -- not part of this
        interpolation-geometry preset, and the paper's ``identity``-space
        PGA is likewise a call-site choice (``transform=``), not stored
        here.

        ``minimum_neighbors=4`` reads "four neighboring stations" as a
        requirement rather than a typical count; the paper does not state
        what should happen with fewer nearby stations. That reading, and
        the :class:`IDWConfig` exact-node behavior, are this package's own
        decisions, not ones drawn from the publication. Treat the surface
        this preset produces as a reproduction of that paper's specific
        method for that specific event, not a general-purpose PGA default.
        """
        return cls(neighbors=4, power=2.0, max_distance_km=50.0, minimum_neighbors=4)


@dataclass(frozen=True, slots=True)
class NearestConfig:
    """Configuration for nearest-station (Voronoi-region) assignment.

    The only method this package recommends when a metric is available only
    as class labels, with no underlying continuous value to interpolate --
    see the module-level docstring in :mod:`pyshindo.spatial`.
    """

    max_distance_km: float | None = None
    allow_extrapolation: bool = False

    def __post_init__(self) -> None:
        _require_radius_or_extrapolation(self.max_distance_km, self.allow_extrapolation)


@dataclass(frozen=True, slots=True)
class LinearConfig:
    """Configuration for piecewise-linear interpolation on a Delaunay triangulation.

    Undefined outside the station convex hull -- there is no radius
    parameter to bound extrapolation because this method never
    extrapolates. ``maximum_extent_km`` instead bounds how far from its
    center the local planar projection the triangulation relies on may be
    used; see :func:`pyshindo.spatial._geometry.aeqd_project_km`.
    """

    maximum_extent_km: float = 2_000.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum_extent_km) or self.maximum_extent_km <= 0.0:
            raise ValueError("maximum_extent_km must be finite and positive.")


@dataclass(frozen=True, slots=True)
class SurfaceMetadata:
    """Everything needed to interpret an :class:`InterpolatedSurface` correctly.

    ``component_definition`` is a short, free-text provenance note such as
    ``"horizontal_vector_peak"`` or ``"three_component_vector_peak"`` -- not
    a closed set of choices, the same way
    :func:`pyshindo.velocity.peak_ground_velocity` leaves the choice of
    which components to combine to the caller rather than fixing one
    convention.
    """

    method: SpatialMethod
    transform: ValueTransform
    metric_name: str
    unit: str | None
    component_definition: str | None
    station_count: int
    parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class InterpolatedSurface:
    """The result of interpolating station observations onto a grid.

    ``values`` is NaN wherever :attr:`support_mask` is ``False``. This
    result carries no notion of land or sea: the field is numerically
    defined by station geometry alone, and whether the sea is drawn is a
    rendering choice made in :mod:`pyshindo.plotting.surfaces`, not a
    property of the interpolation. Seismic waves do not stop at the
    coastline, so a land mask is never used as a distance barrier during
    interpolation.
    """

    grid: SurfaceGrid
    values: FloatArray
    support_mask: npt.NDArray[np.bool_]
    nearest_station_distance_km: FloatArray
    neighbor_count: npt.NDArray[np.uint16]
    metadata: SurfaceMetadata

    def __post_init__(self) -> None:
        expected = self.grid.shape
        for name in ("values", "support_mask", "nearest_station_distance_km", "neighbor_count"):
            actual = np.shape(getattr(self, name))
            if actual != expected:
                raise ValueError(
                    f"{name} has shape {actual}, which does not match grid.shape {expected}."
                )

    def sample_nearest(self, latitude_deg: float, longitude_deg: float) -> float:
        """Return the value at the grid cell closest to one point.

        A convenience for point queries -- a map click, a spot check --
        against an already-computed surface. Does not interpolate again;
        it looks up the nearest existing cell, which may itself be
        unsupported (``NaN``).
        """
        longitude_index = int(np.argmin(np.abs(self.grid.longitudes_deg - longitude_deg)))
        latitude_index = int(np.argmin(np.abs(self.grid.latitudes_deg - latitude_deg)))
        return float(self.values[latitude_index, longitude_index])


class InterpolationPlan(Protocol):
    """Structural type shared by every plan this subpackage builds.

    A plan is precomputed station geometry -- a neighbor search, a
    triangulation, or a Voronoi assignment -- that stays valid for a fixed
    station layout and grid, so it can be reused across many frames of the
    same stations without repeating the geometry work each time. See
    :func:`pyshindo.spatial.build_interpolation_plan`.
    """

    @property
    def grid(self) -> SurfaceGrid: ...

    @property
    def station_count(self) -> int: ...

    def interpolate(
        self,
        values: ArrayLike,
        *,
        transform: ValueTransform | str = ValueTransform.IDENTITY,
        metric_name: str = "generic",
        unit: str | None = None,
        component_definition: str | None = None,
    ) -> InterpolatedSurface: ...
