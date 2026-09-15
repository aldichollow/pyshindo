"""Observation-station spatial interpolation.

This is a station interpolation surface, not an estimated ground-motion
field. It spreads observed values across a grid by geometry alone -- no
source model, no site amplification, no ground-motion prediction equation,
no uncertainty estimate. It is not a reproduction of JMA's own estimated
seismic intensity distribution and not a ShakeMap; both of those combine
station observations with a physical or statistical model of how shaking
propagates, which nothing here attempts.

Three methods are implemented:

- :data:`SpatialMethod.IDW` -- local Shepard-style inverse-distance
  weighting (:mod:`pyshindo.spatial.idw`). A smooth field with few
  parameters to justify; the default because it is the one most people mean
  by "interpolate between stations."
- :data:`SpatialMethod.LINEAR` -- piecewise-linear interpolation on a
  Delaunay triangulation (:mod:`pyshindo.spatial.linear`). Exact inside the
  station convex hull, undefined outside it; useful as a parameter-light
  comparison to IDW.
- :data:`SpatialMethod.NEAREST` -- nearest-station assignment
  (:mod:`pyshindo.spatial.nearest`). No smoothing at all, and the only
  method this package recommends when a metric is available only as class
  labels rather than a continuous value.

**Interpolate a continuous value, never a class number or a color.** If a
continuous precursor exists -- unrounded instrumental intensity
(``intensity_raw``), long-period ground-motion ``max_sva_cm_s``, PGA, PGV,
SI -- interpolate that, then classify or color the *result*:

.. code-block:: text

    station continuous values
      -> optional monotone transform (identity or log)
      -> spatial interpolation
      -> inverse transform
      -> optional reporting/rounding
      -> optional classification
      -> color mapping

Classifying first destroys information and performs arithmetic on ordinal
labels; interpolating colors is worse still, since a color table is a
presentation convention; not a metric space. If only class labels are
available, with no underlying continuous value, use ``method="nearest"``
only.

Everything here is NumPy/SciPy only and imports without Plotly, Shapely, or
Pillow. Rendering an :class:`InterpolatedSurface` as a map layer -- land
masking, color mapping, image encoding -- is
:mod:`pyshindo.plotting.surfaces`, an optional feature layered on top.

This subpackage has no default land region, no default bounding box, and no
implicit extrapolation: every geometry choice (a search radius, a maximum
triangulation extent, a grid resolution) is either explicit or clearly
documented as a visualization default, not a scientific one.
"""

from .idw import IDWPlan, build_idw_plan
from .interpolation import build_interpolation_plan, interpolate_surface
from .linear import LinearPlan, build_linear_plan
from .models import (
    GeographicBounds,
    IDWConfig,
    InterpolatedSurface,
    InterpolationPlan,
    LinearConfig,
    NearestConfig,
    SpatialMethod,
    SurfaceGrid,
    SurfaceMetadata,
    ValueTransform,
)
from .nearest import NearestPlan, build_nearest_plan

__all__ = [
    "GeographicBounds",
    "IDWConfig",
    "IDWPlan",
    "InterpolatedSurface",
    "InterpolationPlan",
    "LinearConfig",
    "LinearPlan",
    "NearestConfig",
    "NearestPlan",
    "SpatialMethod",
    "SurfaceGrid",
    "SurfaceMetadata",
    "ValueTransform",
    "build_idw_plan",
    "build_interpolation_plan",
    "build_linear_plan",
    "build_nearest_plan",
    "interpolate_surface",
]
