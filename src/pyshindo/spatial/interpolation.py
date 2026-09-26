"""The one-shot and reusable-plan entry points for station interpolation.

:func:`interpolate_surface` is the direct path for a single map.
:func:`build_interpolation_plan` only pays off when the same station
layout and grid are interpolated many times -- one frame per timestep of a
streaming or animated display, for example -- because it precomputes the
neighbor search or triangulation once and reuses it; for a single frame it
is the same computation with an extra step.
"""

from __future__ import annotations

import numpy.typing as npt

from .idw import build_idw_plan
from .linear import build_linear_plan
from .models import (
    IDWConfig,
    InterpolatedSurface,
    InterpolationPlan,
    LinearConfig,
    NearestConfig,
    SpatialMethod,
    SurfaceGrid,
    ValueTransform,
)
from .nearest import build_nearest_plan


def build_interpolation_plan(
    latitudes_deg: npt.ArrayLike,
    longitudes_deg: npt.ArrayLike,
    *,
    grid: SurfaceGrid,
    method: SpatialMethod | str = SpatialMethod.IDW,
    config: IDWConfig | LinearConfig | NearestConfig,
) -> InterpolationPlan:
    """Precompute station geometry for one method, to reuse across many frames.

    ``config`` must be the configuration type matching ``method`` --
    :class:`~pyshindo.spatial.IDWConfig` for ``"idw"``, and so on. A
    mismatched pair is rejected outright rather than silently picked apart
    for whichever fields the wrong method happens to share.
    """
    selected = method if isinstance(method, SpatialMethod) else SpatialMethod(method)
    if selected is SpatialMethod.IDW:
        if not isinstance(config, IDWConfig):
            raise TypeError(
                f"method='idw' requires an IDWConfig; received {type(config).__name__}."
            )
        return build_idw_plan(latitudes_deg, longitudes_deg, grid=grid, config=config)
    if selected is SpatialMethod.LINEAR:
        if not isinstance(config, LinearConfig):
            raise TypeError(
                f"method='linear' requires a LinearConfig; received {type(config).__name__}."
            )
        return build_linear_plan(latitudes_deg, longitudes_deg, grid=grid, config=config)
    if not isinstance(config, NearestConfig):
        raise TypeError(
            f"method='nearest' requires a NearestConfig; received {type(config).__name__}."
        )
    return build_nearest_plan(latitudes_deg, longitudes_deg, grid=grid, config=config)


def interpolate_surface(
    latitudes_deg: npt.ArrayLike,
    longitudes_deg: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    grid: SurfaceGrid,
    method: SpatialMethod | str = SpatialMethod.IDW,
    config: IDWConfig | LinearConfig | NearestConfig,
    transform: ValueTransform | str = ValueTransform.IDENTITY,
    metric_name: str = "generic",
    unit: str | None = None,
    component_definition: str | None = None,
) -> InterpolatedSurface:
    """Interpolate one frame of station values onto a grid.

    A thin convenience over :func:`build_interpolation_plan` followed by one
    ``.interpolate()`` call, for the common case of a single static map
    where reusing the plan does not matter.

    ``method`` defaults to ``"idw"`` as a visualization default, not a claim
    that inverse-distance weighting is the scientifically preferred method
    for every metric. ``config`` has no default: :class:`~pyshindo.spatial.IDWConfig`
    still has no default search radius, so a real choice about how far a
    station's influence should reach is still required. ``transform``
    defaults to ``"identity"`` -- interpolation changes nothing about the
    values themselves unless explicitly told to.
    """
    plan = build_interpolation_plan(
        latitudes_deg, longitudes_deg, grid=grid, method=method, config=config
    )
    return plan.interpolate(
        values,
        transform=transform,
        metric_name=metric_name,
        unit=unit,
        component_definition=component_definition,
    )
