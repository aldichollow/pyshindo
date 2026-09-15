"""Render an :class:`~pyshindo.spatial.InterpolatedSurface` as a map layer.

Two masks decide what shows: :attr:`~pyshindo.spatial.InterpolatedSurface.support_mask`
(is this cell backed by nearby station geometry at all?) and a land mask (is
this cell land, for a land-only presentation?). They answer different
questions and are combined only here, at render time -- the interpolation
itself in :mod:`pyshindo.spatial` never uses the coastline as a distance
barrier, because seismic waves do not stop at the coastline.

The bundled land mask is a small pre-rasterized boolean grid
(``pyshindo/data/spatial/natural_earth_japan_land.npz``, built once by
``scripts/build_natural_earth_japan.py`` from Natural Earth's public-domain
1:10m land and minor-islands layers; see that script's docstring for
provenance). Land membership at render time is therefore a nearest-cell
array lookup, not a point-in-polygon test -- this module needs no geometry
library. It needs no image library either: the RGBA-to-PNG step is a small
hand-written encoder (:func:`zlib.compress` plus the fixed PNG chunk
framing), because the only thing this package ever encodes is one raster
per map. Together, rendering a surface adds no dependency beyond the
:mod:`pyshindo.plotting` package already requires (``pyshindo[plot]``).

Classes and colors are never interpolated -- see the module docstring in
:mod:`pyshindo.spatial`. This module mirrors that split:
:func:`add_surface_layer` colors a continuous field with a Plotly
colorscale; :func:`add_class_surface_layer` colors an already-classified
field (typically the output of a ``method="nearest"`` surface, or a
continuous surface classified by the caller after interpolation) by an
exact lookup table, the same JMA/long-period class colors
:mod:`pyshindo.plotting.maps` already uses for station markers.

The image layer is inserted below the existing traces
(``below="traces"``), so station markers and their halos stay on top and
observations continue to visually dominate the interpolated field
beneath them.
"""

from __future__ import annotations

import base64
import importlib.resources
import math
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from ..spatial import InterpolatedSurface, SurfaceGrid
from .theme import require_plotly

type FloatArray = npt.NDArray[np.float64]
type RGBAArray = npt.NDArray[np.uint8]
type ColorScale = str | Sequence[Any]
"""A named Plotly colorscale, or an explicit one: a list of colors, or a
list of ``[position, color]`` pairs -- Plotly's own two colorscale forms
(see ``plotly.colors.sample_colorscale``), for a discretized or otherwise
custom scale a name cannot express."""

_DEFAULT_LAND: Final = "natural_earth_japan_10m"


# --------------------------------------------------------------------------
# Land mask: a bundled raster and nearest-cell lookup, no geometry library.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _LandRaster:
    mask: npt.NDArray[np.bool_]
    west_deg: float
    south_deg: float
    resolution_deg: float


@lru_cache(maxsize=4)
def _load_land_raster(name: str) -> _LandRaster:
    if name != _DEFAULT_LAND:
        raise ValueError(
            f"Unknown land source {name!r}; the only bundled source is "
            f"{_DEFAULT_LAND!r}. Pass land=None to skip land masking entirely."
        )
    resource = importlib.resources.files("pyshindo") / "data" / "spatial" / (
        "natural_earth_japan_land.npz"
    )
    with importlib.resources.as_file(resource) as path, np.load(path) as data:
        ny, nx = (int(value) for value in data["shape"])
        mask = np.unpackbits(data["packed_mask"])[: ny * nx].reshape(ny, nx).astype(np.bool_)
        west_deg = float(data["west_deg"])
        south_deg = float(data["south_deg"])
        resolution_deg = float(data["resolution_deg"])
    return _LandRaster(
        mask=mask, west_deg=west_deg, south_deg=south_deg, resolution_deg=resolution_deg
    )


def land_mask_for_grid(
    grid: SurfaceGrid, *, land: str = _DEFAULT_LAND
) -> npt.NDArray[np.bool_]:
    """Return which cells of ``grid`` are land, from the bundled coastline raster.

    A nearest-cell lookup into a pre-rasterized raster (see the module
    docstring), independent of ``grid``'s own resolution. A ``grid`` finer
    than the bundled raster's own ~1.1 km cells does not gain coastline
    detail the raster does not have; see
    ``scripts/build_natural_earth_japan.py`` if sharper detail is ever
    needed for a specific use case.

    A cell outside the raster's own coverage (see its provenance JSON --
    Japan and its outlying territory, plus margin) is treated as not land,
    rather than reusing whichever edge cell happens to be nearest: nothing
    about ``GeographicBounds`` restricts a caller to Japan, and silently
    answering with the wrong region's coastline would be worse than
    answering "not land" plainly.
    """
    raster = _load_land_raster(land)
    column = np.round((grid.longitudes_deg - raster.west_deg) / raster.resolution_deg - 0.5)
    row = np.round((grid.latitudes_deg - raster.south_deg) / raster.resolution_deg - 0.5)
    column_in_bounds = (column >= 0) & (column < raster.mask.shape[1])
    row_in_bounds = (row >= 0) & (row < raster.mask.shape[0])
    column_index = np.clip(column.astype(np.intp), 0, raster.mask.shape[1] - 1)
    row_index = np.clip(row.astype(np.intp), 0, raster.mask.shape[0] - 1)
    land_lookup = raster.mask[np.ix_(row_index, column_index)]
    return land_lookup & row_in_bounds[:, None] & column_in_bounds[None, :]


def _visible_mask(surface: InterpolatedSurface, land: str | None) -> npt.NDArray[np.bool_]:
    if land is None:
        return surface.support_mask
    return surface.support_mask & land_mask_for_grid(surface.grid, land=land)


# --------------------------------------------------------------------------
# RGBA rendering: pure NumPy plus Plotly's own colorscale registry (no
# Shapely, no Pillow).
# --------------------------------------------------------------------------


def _freeze_colorscale(colorscale: ColorScale) -> str | tuple[Any, ...]:
    """Return a hashable equivalent of ``colorscale``, for use as a cache key.

    A named colorscale is already hashable. An explicit one is a list --
    unhashable, and therefore unusable as an ``lru_cache`` key directly --
    so each stop is frozen into a tuple (a bare color string stays a string,
    a ``[position, color]`` pair becomes a 2-tuple) and the whole thing
    becomes a tuple of stops.
    """
    if isinstance(colorscale, str):
        return colorscale
    return tuple(
        stop if isinstance(stop, str) else tuple(stop) for stop in colorscale
    )


@lru_cache(maxsize=32)
def _colorscale_lookup_table(colorscale: str | tuple[Any, ...], steps: int = 256) -> RGBAArray:
    """Sample a Plotly colorscale into a fixed-size RGB lookup table.

    Cached because building one costs a handful of milliseconds and every
    cell in a render reuses the same table -- a per-cell colorscale sample
    would be both slower and pointless. ``colorscale`` must already be in
    the frozen (hashable) form :func:`_freeze_colorscale` returns; it is
    converted back to plain lists here, since that is the form
    ``plotly.colors.sample_colorscale`` itself expects.
    """
    require_plotly()  # raises the package's own friendly message if Plotly is missing
    import plotly.colors as plotly_colors

    resolved: str | list[Any] = (
        colorscale
        if isinstance(colorscale, str)
        else [list(stop) if isinstance(stop, tuple) else stop for stop in colorscale]
    )
    samples = plotly_colors.sample_colorscale(
        resolved, np.linspace(0.0, 1.0, steps).tolist(), colortype="tuple"
    )
    return np.round(np.array(samples, dtype=np.float64) * 255.0).astype(np.uint8)


def render_surface_rgba(
    surface: InterpolatedSurface,
    *,
    colorscale: ColorScale = "YlOrRd",
    cmin: float,
    cmax: float,
    color_transform: str = "identity",
    opacity: float = 0.72,
    land: str | None = _DEFAULT_LAND,
) -> RGBAArray:
    """Render a continuous surface to an ``(ny, nx, 4)`` uint8 RGBA array.

    ``cmin``/``cmax`` have no default: an automatically chosen color range
    would change meaning between two maps of the same quantity, or between
    frames of an animation, with nothing on the map saying so.
    ``color_transform`` normalizes the *display* mapping independently of
    whatever transform :mod:`pyshindo.spatial` used during interpolation --
    a surface interpolated in log space can still be displayed on an
    identity color scale, or vice versa; the two are separate choices. Under
    ``color_transform="log"``, a cell whose value is zero or negative has no
    logarithm and is rendered transparent rather than raising, since a
    single such cell is a display edge case, not a reason to fail the whole
    render; it usually means the metric or transform was not the intended
    one for that surface.

    Cells outside :attr:`~pyshindo.spatial.InterpolatedSurface.support_mask`
    or outside land (unless ``land=None``) are rendered with alpha 0. A
    supported, visible cell whose value falls outside ``[cmin, cmax]`` is
    instead clipped to the nearest boundary color, the same convention
    Plotly's own marker colorscales use -- it stays visible, just pinned to
    one end of the scale, rather than disappearing.
    """
    if not (math.isfinite(cmin) and math.isfinite(cmax)) or cmin >= cmax:
        raise ValueError(f"cmin must be finite and less than cmax; received {cmin}, {cmax}.")
    if not 0.0 < opacity <= 1.0:
        raise ValueError(f"opacity must lie in (0, 1]; received {opacity}.")

    visible = _visible_mask(surface, land)
    if color_transform == "identity":
        display = surface.values
        low, high = cmin, cmax
    elif color_transform == "log":
        if cmin <= 0.0 or cmax <= 0.0:
            raise ValueError("color_transform='log' requires cmin and cmax to be positive.")
        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.log(surface.values)
        visible = visible & (surface.values > 0.0)
        low, high = math.log(cmin), math.log(cmax)
    else:
        raise ValueError(
            f"Unsupported color_transform {color_transform!r}; choose 'identity' or 'log'."
        )

    lookup_table = _colorscale_lookup_table(_freeze_colorscale(colorscale))
    normalized = np.full(display.shape, np.nan)
    finite = np.isfinite(display)
    normalized[finite] = np.clip((display[finite] - low) / (high - low), 0.0, 1.0)
    color_index = np.zeros(display.shape, dtype=np.intp)
    color_index[finite] = np.round(normalized[finite] * (len(lookup_table) - 1)).astype(np.intp)

    rgba = np.zeros((*surface.grid.shape, 4), dtype=np.uint8)
    rgba[..., :3] = lookup_table[color_index]
    rgba[..., 3] = np.where(visible & finite, round(255 * opacity), 0)
    return rgba


def render_class_surface_rgba(
    surface: InterpolatedSurface,
    *,
    colors: Mapping[float, str],
    opacity: float = 0.72,
    land: str | None = _DEFAULT_LAND,
) -> RGBAArray:
    """Render an already-classified surface to an ``(ny, nx, 4)`` uint8 RGBA array.

    ``colors`` maps each class's numeric code -- the same numbers
    :attr:`~pyshindo.spatial.InterpolatedSurface.values` holds -- to a
    ``"#RRGGBB"`` color. For the package's own long-period class colors,
    for example:

    .. code-block:: python

        from pyshindo.long_period import LongPeriodClass
        from pyshindo.plotting.theme import LONG_PERIOD_CLASS_COLORS

        colors = {float(cls.value): color for cls, color in LONG_PERIOD_CLASS_COLORS.items()}

    A cell whose value matches no key in ``colors`` is rendered transparent,
    the same as an unsupported cell, rather than guessing the nearest class.
    """
    if not colors:
        raise ValueError("colors must contain at least one class.")
    if not 0.0 < opacity <= 1.0:
        raise ValueError(f"opacity must lie in (0, 1]; received {opacity}.")

    visible = _visible_mask(surface, land)
    alpha = round(255 * opacity)
    rgba = np.zeros((*surface.grid.shape, 4), dtype=np.uint8)
    for class_value, hex_color in colors.items():
        red, green, blue = _hex_to_rgb(hex_color)
        matches = visible & np.isclose(surface.values, class_value)
        rgba[matches, 0] = red
        rgba[matches, 1] = green
        rgba[matches, 2] = blue
        rgba[matches, 3] = alpha
    return rgba


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    stripped = color.lstrip("#")
    if len(stripped) != 6:
        raise ValueError(f"Expected a 6-digit hex color like '#RRGGBB'; received {color!r}.")
    return int(stripped[0:2], 16), int(stripped[2:4], 16), int(stripped[4:6], 16)


# --------------------------------------------------------------------------
# PNG encoding: stdlib zlib and struct only, no Pillow. See tests/test_surfaces.py
# for a byte-level round trip against this same encoding, and this module's
# docstring for why this is hand-written at all.
# --------------------------------------------------------------------------

_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def encode_png(rgba: RGBAArray) -> bytes:
    """Encode an ``(height, width, 4)`` uint8 RGBA array as PNG bytes.

    Every scanline is prefixed with filter-type 0 (none), the simplest of
    PNG's five per-line filters and always valid regardless of image
    content; only compression ratio is traded away, not correctness or
    decoder compatibility. A single-image render does not need the smallest
    possible file.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"rgba must have shape (height, width, 4); received {rgba.shape}.")
    if rgba.dtype != np.uint8:
        raise ValueError(f"rgba must be uint8; received dtype {rgba.dtype}.")
    height, width, _ = rgba.shape

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    raw = np.empty((height, width * 4 + 1), dtype=np.uint8)
    raw[:, 0] = 0  # filter type 0 (none), once per scanline
    raw[:, 1:] = rgba.reshape(height, width * 4)
    idat = zlib.compress(raw.tobytes(), level=9)

    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


# --------------------------------------------------------------------------
# Plotly composition: insert the rendered raster as a map image layer.
# --------------------------------------------------------------------------


def _insert_image_layer(figure: Any, rgba: RGBAArray, grid: SurfaceGrid, *, below: str) -> None:
    # PNG rows run north -> south to match how raster images are conventionally
    # read; SurfaceGrid's own row 0 is the southernmost row (see its docstring),
    # so the array is flipped here, at the image boundary, rather than changing
    # pyshindo.spatial's row convention to suit one renderer.
    png_bytes = encode_png(np.ascontiguousarray(np.flipud(rgba)))
    data_uri = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"
    layer = {
        "sourcetype": "image",
        "source": data_uri,
        "coordinates": [list(corner) for corner in grid.image_corners],
        "opacity": 1.0,  # alpha is already encoded per pixel in the PNG itself
        "below": below,
    }
    existing_layers = list(figure.layout.map.layers or ())
    figure.update_layout(map_layers=[*existing_layers, layer])


def add_surface_layer(
    figure: Any,
    surface: InterpolatedSurface,
    *,
    colorscale: ColorScale = "YlOrRd",
    cmin: float,
    cmax: float,
    color_transform: str = "identity",
    opacity: float = 0.72,
    land: str | None = _DEFAULT_LAND,
    colorbar_title: str | None = None,
    colorbar_x: float | None = None,
    below: str = "traces",
) -> Any:
    """Add a continuous surface to a map figure as an image layer, in place.

    Compose under an existing station map:

    .. code-block:: python

        figure = continuous_value_map_figure(latitudes_deg, longitudes_deg, values,
                                              value_label="PGV [cm/s]")
        add_surface_layer(figure, surface, cmin=0.0, cmax=30.0)

    Inserted with ``below="traces"`` by default, so station markers and
    their halos stay drawn on top and remain the visually dominant
    layer; the interpolated surface is context underneath them, not a
    replacement for them. Returns ``figure`` for chaining, but mutates it
    in place, the same as Plotly's own ``update_layout``/``add_trace``.

    ``colorbar_title`` defaults to ``None`` -- no colorbar -- because a
    figure built with :func:`~pyshindo.plotting.maps.continuous_value_map_figure`
    already draws one for the markers, and the surface underneath them is
    context, not a second reading of the same quantity to label separately.
    Pass ``colorbar_title`` when the surface is shown with no marker map
    alongside it (nothing else on the figure gives its scale); Plotly does
    not know to avoid a second, unrelated colorbar's default position in
    that case, so also pass ``colorbar_x`` (Plotly's own default is
    approximately ``1.02``; try ``1.15`` or higher) if another one is
    already there.
    """
    go, _, _ = require_plotly()
    rgba = render_surface_rgba(
        surface,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
        color_transform=color_transform,
        opacity=opacity,
        land=land,
    )
    _insert_image_layer(figure, rgba, surface.grid, below=below)
    if colorbar_title is not None:
        # An image layer has no native colorbar; a near-invisible marker trace
        # carrying the same colorscale and limits stands in for one, the same
        # technique a hidden legend entry would use.
        figure.add_trace(
            go.Scattermap(
                lat=[None],
                lon=[None],
                mode="markers",
                marker={
                    "size": 0.1,
                    "color": [cmin],
                    "cmin": cmin,
                    "cmax": cmax,
                    "colorscale": colorscale,
                    "showscale": True,
                    "colorbar": {"title": {"text": colorbar_title}, "x": colorbar_x},
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return figure


def add_class_surface_layer(
    figure: Any,
    surface: InterpolatedSurface,
    *,
    colors: Mapping[float, str],
    opacity: float = 0.72,
    land: str | None = _DEFAULT_LAND,
    below: str = "traces",
) -> Any:
    """Add an already-classified surface to a map figure as an image layer, in place.

    No colorbar is added: the existing discrete-class station markers from
    :func:`pyshindo.plotting.maps.intensity_map_figure` or
    :func:`~pyshindo.plotting.maps.long_period_class_map_figure` already
    carry a proper legend for the same classes: this layer is a background
    wash in the same colors, not a second, redundant key.
    """
    require_plotly()
    rgba = render_class_surface_rgba(surface, colors=colors, opacity=opacity, land=land)
    _insert_image_layer(figure, rgba, surface.grid, below=below)
    return figure
