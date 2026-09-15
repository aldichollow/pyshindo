"""Multi-station geographic distribution figures.

A different figure family from the rest of :mod:`pyshindo.plotting`: every
other figure here plots one record's time series or spectrum. These plot
many stations at once, each reduced to a single value, positioned by
latitude and longitude -- the shape :mod:`scripts.validate_official`
(outside the package) already computes station by station but has never had
anywhere to put on a map.

A thin adapter, like :mod:`pyshindo.obspy_interop`: every function here takes
parallel arrays of coordinates and values, not a specific data source. Feed
it JMA text-record metadata, station coordinates from an ObsPy inventory, or
a hand-built list -- what produced the values is not this module's concern.

Discrete quantities (seismic intensity, the long-period class) use the same
JMA-sourced color tables the rest of this package plots them with, one trace
per class so the legend reads as a proper key. A continuous quantity (SI
value, PGV, PGA) has no official class boundaries to shade by, so it is
one trace on a continuous colorscale with its own colorbar instead.

Uses Plotly's ``Scattermap`` (built-in MapLibre styles, no API token) rather
than the older, token-gated ``Scattermapbox``. The default basemap style,
``carto-positron``, is a muted grayscale tile set chosen so colored markers
stay legible; pass ``map_style`` to any figure function for a different
built-in style (for example ``"open-street-map"`` for the original colorful
tiles, or ``"carto-positron-nolabels"`` for an even quieter background).
Markers get a dark-gray halo -- ``Scattermap`` markers have no ``line``
(border) property, unlike ``Scatter``, so the halo is a second, larger,
gray-filled trace drawn immediately behind each marker trace instead.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from ..long_period.scale import LongPeriodClass
from ..scale import IntensityScale
from .theme import JMA_INTENSITY_COLORS, LONG_PERIOD_CLASS_COLORS, require_plotly

_MARKER_SIZE: Final = 9
_HALO_SIZE_MARGIN: Final = 3
_HALO_COLOR: Final = "#404040"
_FONT_FAMILY: Final = "Helvetica Neue, Helvetica, Arial, Noto Sans JP, sans-serif"
_DEFAULT_ZOOM: Final = 5.0
_ZOOM_MARGIN_DEG: Final = 0.3
_DEFAULT_MAP_STYLE: Final = "carto-positron"


def _validate_coordinates(
    latitudes_deg: npt.ArrayLike, longitudes_deg: npt.ArrayLike, count: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    lat = np.asarray(latitudes_deg, dtype=np.float64)
    lon = np.asarray(longitudes_deg, dtype=np.float64)
    if lat.shape != (count,) or lon.shape != (count,):
        raise ValueError(
            "latitudes_deg and longitudes_deg must each be one-dimensional and the same "
            f"length as the values; received {lat.shape} and {lon.shape} for {count} values."
        )
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("latitudes_deg and longitudes_deg must be finite.")
    if np.any(np.abs(lat) > 90.0):
        raise ValueError("latitudes_deg must lie within [-90, 90].")
    if np.any(np.abs(lon) > 180.0):
        raise ValueError("longitudes_deg must lie within [-180, 180].")
    return lat, lon


def _resolve_labels(labels: Sequence[str] | None, count: int) -> Sequence[str]:
    if labels is None:
        return [""] * count
    if len(labels) != count:
        raise ValueError(f"labels must contain {count} entries; received {len(labels)}.")
    return labels


def _validate_marker_size(marker_size: float) -> None:
    if not np.isfinite(marker_size) or marker_size <= 0.0:
        raise ValueError("marker_size must be finite and greater than zero.")


def _map_layout(
    lat: npt.NDArray[np.float64], lon: npt.NDArray[np.float64], title: str, map_style: str
) -> dict[str, Any]:
    """Return a Plotly layout centered and zoomed to fit every station."""
    if lat.size:
        center = {"lat": float(np.mean(lat)), "lon": float(np.mean(lon))}
        span = max(float(lat.max() - lat.min()), float(lon.max() - lon.min())) + _ZOOM_MARGIN_DEG
        zoom = _DEFAULT_ZOOM if span <= 0.0 else max(1.0, 8.0 - np.log2(span))
    else:
        center = {"lat": 36.0, "lon": 138.0}  # Japan, as a fallback for an empty figure
        zoom = _DEFAULT_ZOOM
    return {
        "title": {"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 16}},
        "font": {"family": _FONT_FAMILY, "size": 12, "color": "#1A1A1A"},
        "map": {"style": map_style, "center": center, "zoom": zoom},
        "margin": {"l": 0, "r": 0, "t": 48, "b": 0},
        "hoverlabel": {
            "bgcolor": "#FFFFFF",
            "bordercolor": "#C9D0DB",
            "font": {"family": _FONT_FAMILY, "size": 12, "color": "#1A1A1A"},
        },
    }


def _halo_trace(
    go: Any,
    lat: npt.NDArray[np.float64],
    lon: npt.NDArray[np.float64],
    *,
    marker_size: float,
    legendgroup: str | None = None,
) -> Any:
    """A dark-gray, oversized, non-interactive marker trace drawn behind the real one.

    Stands in for the border ``Scattermap`` markers cannot have (unlike
    ``Scatter``, they have no ``marker.line``), so colored markers read as
    outlined circles instead of blending into whatever the basemap shows
    underneath. Sized relative to the real marker (``marker_size +
    _HALO_SIZE_MARGIN``) so the halo scales along with it. Sharing
    ``legendgroup`` with the marker trace it backs makes Plotly hide both
    together when that trace's legend entry is toggled -- without it, this
    trace would stay on screen as an unlabeled dot.
    """
    return go.Scattermap(
        lat=lat,
        lon=lon,
        mode="markers",
        marker={"size": marker_size + _HALO_SIZE_MARGIN, "color": _HALO_COLOR, "opacity": 1.0},
        hoverinfo="skip",
        showlegend=False,
        legendgroup=legendgroup,
    )


def _discrete_class_figure(
    latitudes_deg: npt.ArrayLike,
    longitudes_deg: npt.ArrayLike,
    classes: Sequence[Any],
    *,
    colors: dict[Any, str],
    class_order: Sequence[Any],
    class_label: Any,
    labels: Sequence[str] | None,
    title: str,
    map_style: str,
    marker_size: float,
) -> Any:
    go, _, _ = require_plotly()
    _validate_marker_size(marker_size)
    count = len(classes)
    lat, lon = _validate_coordinates(latitudes_deg, longitudes_deg, count)
    resolved_labels = _resolve_labels(labels, count)

    figure = go.Figure()
    for scale in class_order:
        mask = [scale is value or scale == value for value in classes]
        if not any(mask):
            continue
        indices = np.flatnonzero(mask)
        class_lat, class_lon = lat[indices], lon[indices]
        group = str(class_label(scale))
        figure.add_trace(
            _halo_trace(go, class_lat, class_lon, marker_size=marker_size, legendgroup=group)
        )
        figure.add_trace(
            go.Scattermap(
                lat=class_lat,
                lon=class_lon,
                mode="markers",
                name=class_label(scale),
                legendgroup=group,
                marker={
                    "size": marker_size,
                    "color": colors[scale],
                    "opacity": 0.95,
                },
                text=[resolved_labels[int(i)] for i in indices],
                hovertemplate=(
                    f"%{{text}}<br>{class_label(scale)}<br>"
                    "%{lat:.4f}, %{lon:.4f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(**_map_layout(lat, lon, title, map_style))
    figure.update_layout(legend={"title": {"text": ""}, "font": {"size": 11}})
    return figure


def intensity_map_figure(
    latitudes_deg: npt.ArrayLike,
    longitudes_deg: npt.ArrayLike,
    intensities: Sequence[IntensityScale | str],
    *,
    labels: Sequence[str] | None = None,
    title: str = "Seismic intensity distribution",
    map_style: str = _DEFAULT_MAP_STYLE,
    marker_size: float = _MARKER_SIZE,
) -> Any:
    """Plot measured seismic intensity classes at their observation points.

    ``intensities`` are one :class:`~pyshindo.scale.IntensityScale` (or its
    string value, for example ``"5-"``) per station, in the same JMA colors
    every other intensity figure in this package uses. Each class is its own
    trace, so the legend reads as a proper key rather than a gradient.

    ``map_style`` selects the basemap; see the module docstring.
    ``marker_size`` shrinks markers for a dense map of many stations, or
    grows them for a sparse one; the gray halo scales along with it.
    """
    scales = [
        value if isinstance(value, IntensityScale) else IntensityScale(value)
        for value in intensities
    ]
    return _discrete_class_figure(
        latitudes_deg,
        longitudes_deg,
        scales,
        colors=dict(JMA_INTENSITY_COLORS),
        class_order=list(IntensityScale),
        class_label=lambda scale: f"震度{scale.japanese}",
        labels=labels,
        title=title,
        map_style=map_style,
        marker_size=marker_size,
    )


def long_period_class_map_figure(
    latitudes_deg: npt.ArrayLike,
    longitudes_deg: npt.ArrayLike,
    classes: Sequence[LongPeriodClass | str],
    *,
    labels: Sequence[str] | None = None,
    title: str = "Long-period ground motion class distribution",
    map_style: str = _DEFAULT_MAP_STYLE,
    marker_size: float = _MARKER_SIZE,
) -> Any:
    """Plot long-period ground motion classes at their observation points.

    ``classes`` are one :class:`~pyshindo.long_period.LongPeriodClass` (or its
    string value) per station, in the colors JMA uses on its own long-period
    observation pages.

    ``map_style`` selects the basemap; see the module docstring.
    ``marker_size`` shrinks markers for a dense map of many stations, or
    grows them for a sparse one; the gray halo scales along with it.
    """
    resolved = [
        value if isinstance(value, LongPeriodClass) else LongPeriodClass(value)
        for value in classes
    ]
    return _discrete_class_figure(
        latitudes_deg,
        longitudes_deg,
        resolved,
        colors=dict(LONG_PERIOD_CLASS_COLORS),
        class_order=list(LongPeriodClass),
        class_label=lambda scale: f"階級{scale.value}",
        labels=labels,
        title=title,
        map_style=map_style,
        marker_size=marker_size,
    )


def _log_color_values(
    value_label: str,
    array: npt.NDArray[np.float64],
    *,
    cmin: float | None,
    cmax: float | None,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Return log-space marker colors plus a colorbar labeled in real units.

    Plotly has no native log color axis for markers the way it does for an
    x/y axis, so the color channel is pre-transformed here and the colorbar
    ticks are placed back in real units afterward -- a handful of
    geometrically spaced values across the *displayed* range (``cmin``/
    ``cmax`` if given, the data's own min/max otherwise), not necessarily
    round numbers, since that range may span less than one decade.
    """
    if array.size and np.any(array <= 0.0):
        raise ValueError(
            "color_transform='log' requires every value to be strictly positive. "
            "PGA, PGV, SI, and Sva are non-negative by definition but a reading of "
            "exactly zero has no logarithm; use color_transform='identity', or "
            "exclude that station before plotting."
        )
    if (cmin is not None and cmin <= 0.0) or (cmax is not None and cmax <= 0.0):
        raise ValueError("cmin and cmax must be strictly positive under color_transform='log'.")
    color_values = np.log(array)
    low = cmin if cmin is not None else (float(array.min()) if array.size else None)
    high = cmax if cmax is not None else (float(array.max()) if array.size else None)
    if low is None or high is None:
        return color_values, {"title": {"text": value_label}}
    tick_values = [low] if low == high else np.geomspace(low, high, 6).tolist()
    colorbar = {
        "title": {"text": value_label},
        "tickvals": [math.log(value) for value in tick_values],
        "ticktext": [f"{value:.3g}" for value in tick_values],
    }
    return color_values, colorbar


def continuous_value_map_figure(
    latitudes_deg: npt.ArrayLike,
    longitudes_deg: npt.ArrayLike,
    values: npt.ArrayLike,
    *,
    value_label: str,
    labels: Sequence[str] | None = None,
    colorscale: str = "YlOrRd",
    color_transform: str = "identity",
    cmin: float | None = None,
    cmax: float | None = None,
    title: str = "Station distribution",
    map_style: str = _DEFAULT_MAP_STYLE,
    marker_size: float = _MARKER_SIZE,
) -> Any:
    """Plot a continuous per-station value (SI, PGV, PGA, ...) on a map.

    Unlike :func:`intensity_map_figure`/:func:`long_period_class_map_figure`,
    there are no official class boundaries to shade by for a quantity like
    SI value or PGV, so this is one trace on a continuous colorscale with its
    own colorbar rather than several discrete-class traces.

    ``color_transform`` chooses what space marker color is mapped in:
    ``"identity"`` (default, unchanged behavior) colors directly by
    ``values``; ``"log"`` colors by ``log(values)`` instead, restoring
    real-unit tick labels on the colorbar afterward. Never applied
    silently -- this is the same explicit-only convention
    :attr:`pyshindo.spatial.ValueTransform` uses for interpolation, extended
    to display: ground-motion amplitude is often closer to log-normal than
    normal, so a linear color scale compresses most stations into one end of
    the colorbar, but that is a real choice for the caller to make, not a
    default this function should guess. Every value must be strictly
    positive under ``"log"``.

    ``cmin``/``cmax`` left as ``None`` (default) auto-range to ``values``'
    own min/max, which is fine for a standalone map. Pass them explicitly
    -- in real units, even under ``color_transform="log"`` -- to fix the
    color range instead, the same way
    :func:`~pyshindo.plotting.surfaces.add_surface_layer`'s ``cmin``/``cmax``
    have no default at all: composing these markers with a surface layer
    built from the same colorscale needs both given the *same* explicit
    range, or the two will not agree about what a color means.

    ``map_style`` selects the basemap; see the module docstring.
    ``marker_size`` shrinks markers for a dense map of many stations, or
    grows them for a sparse one; the gray halo scales along with it.
    """
    go, _, _ = require_plotly()
    _validate_marker_size(marker_size)
    if (cmin is None) != (cmax is None):
        # Plotly only fixes the color range when both are set; passing just
        # one is silently ignored (falls back to full auto-ranging), which
        # would quietly discard the caller's explicit choice rather than
        # honor half of it.
        raise ValueError("cmin and cmax must be given together, or both left as None.")
    if cmin is not None and cmax is not None and cmin >= cmax:
        raise ValueError(f"cmin must be less than cmax; received {cmin}, {cmax}.")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    lat, lon = _validate_coordinates(latitudes_deg, longitudes_deg, array.size)
    resolved_labels = _resolve_labels(labels, array.size)

    if color_transform == "identity":
        color_values, colorbar = array, {"title": {"text": value_label}}
        marker_cmin, marker_cmax = cmin, cmax
    elif color_transform == "log":
        color_values, colorbar = _log_color_values(value_label, array, cmin=cmin, cmax=cmax)
        marker_cmin = math.log(cmin) if cmin is not None else None
        marker_cmax = math.log(cmax) if cmax is not None else None
    else:
        raise ValueError(
            f"Unsupported color_transform {color_transform!r}; choose 'identity' or 'log'."
        )

    marker: dict[str, Any] = {
        "size": marker_size,
        "color": color_values,
        "colorscale": colorscale,
        "showscale": True,
        "colorbar": colorbar,
        "opacity": 0.95,
    }
    if marker_cmin is not None:
        marker["cmin"] = marker_cmin
    if marker_cmax is not None:
        marker["cmax"] = marker_cmax

    figure = go.Figure()
    figure.add_trace(_halo_trace(go, lat, lon, marker_size=marker_size))
    figure.add_trace(
        go.Scattermap(
            lat=lat,
            lon=lon,
            mode="markers",
            marker=marker,
            text=resolved_labels,
            customdata=array,
            hovertemplate=f"%{{text}}<br>{value_label}: %{{customdata:.4g}}<br>"
            "%{lat:.4f}, %{lon:.4f}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(**_map_layout(lat, lon, title, map_style))
    return figure
