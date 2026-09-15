from __future__ import annotations

import struct
import zlib

import numpy as np
import plotly.graph_objects as go
import pytest

from pyshindo.plotting.surfaces import (
    add_class_surface_layer,
    add_surface_layer,
    encode_png,
    land_mask_for_grid,
    render_class_surface_rgba,
    render_surface_rgba,
)
from pyshindo.spatial import (
    GeographicBounds,
    IDWConfig,
    NearestConfig,
    SurfaceGrid,
    interpolate_surface,
)

RNG = np.random.default_rng(20260915)


def _bounds() -> GeographicBounds:
    # Central Honshu: a mix of land and sea within one small, fast-to-test grid.
    return GeographicBounds(west_deg=136.0, south_deg=34.0, east_deg=141.0, north_deg=37.0)


def _grid() -> SurfaceGrid:
    return SurfaceGrid.from_bounds(_bounds(), approximate_resolution_km=15.0)


def _stations(count: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = _bounds()
    lat = RNG.uniform(bounds.south_deg + 0.2, bounds.north_deg - 0.2, count)
    lon = RNG.uniform(bounds.west_deg + 0.2, bounds.east_deg - 0.2, count)
    value = RNG.uniform(1.0, 100.0, count)
    return lat, lon, value


def _continuous_surface():
    lat, lon, value = _stations()
    grid = _grid()
    return interpolate_surface(
        lat,
        lon,
        value,
        grid=grid,
        config=IDWConfig(neighbors=6, max_distance_km=100.0, minimum_neighbors=2),
    )


# --------------------------------------------------------------------------
# PNG encoding: a from-scratch stdlib decoder, independent of the encoder's
# own internal structure, so this checks byte-level format correctness (chunk
# framing, CRC32, zlib payload) rather than only round-tripping through code
# that shares assumptions with the encoder.
# --------------------------------------------------------------------------


def _decode_png(data: bytes) -> np.ndarray:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    chunks: dict[bytes, bytes] = {}
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])
        assert crc == zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF, f"bad CRC on {chunk_type!r}"
        chunks[chunk_type] = chunk_data
        offset += 12 + length
    assert b"IEND" in chunks

    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", chunks[b"IHDR"]
    )
    assert bit_depth == 8
    assert color_type == 6  # RGBA
    assert compression == 0
    assert filter_method == 0
    assert interlace == 0

    raw = zlib.decompress(chunks[b"IDAT"])
    stride = width * 4 + 1
    assert len(raw) == stride * height
    pixels = np.empty((height, width, 4), dtype=np.uint8)
    for row in range(height):
        scanline = raw[row * stride : (row + 1) * stride]
        assert scanline[0] == 0, "only filter type 0 is expected from this encoder"
        pixels[row] = np.frombuffer(scanline[1:], dtype=np.uint8).reshape(width, 4)
    return pixels


def test_png_round_trips_through_an_independent_decoder() -> None:
    rgba = RNG.integers(0, 256, size=(11, 17, 4), dtype=np.uint8)
    decoded = _decode_png(encode_png(rgba))
    np.testing.assert_array_equal(decoded, rgba)


def test_png_round_trips_a_single_pixel_and_an_odd_width() -> None:
    for shape in [(1, 1, 4), (3, 1, 4), (1, 5, 4)]:
        rgba = RNG.integers(0, 256, size=shape, dtype=np.uint8)
        np.testing.assert_array_equal(_decode_png(encode_png(rgba)), rgba)


def test_encode_png_rejects_wrong_shape_or_dtype() -> None:
    with pytest.raises(ValueError, match="shape"):
        encode_png(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        encode_png(np.zeros((4, 4, 4), dtype=np.float64))


# --------------------------------------------------------------------------
# Land mask
# --------------------------------------------------------------------------


def test_land_mask_identifies_known_land_and_sea_points() -> None:
    # A grid whose cells happen to land close to Tokyo (land) and a point well
    # out in the Pacific east of Japan (sea).
    bounds = GeographicBounds(west_deg=139.0, south_deg=34.0, east_deg=145.0, north_deg=36.0)
    grid = SurfaceGrid.from_bounds(bounds, shape=(2, 6))
    mask = land_mask_for_grid(grid)
    assert mask.shape == grid.shape
    # The westernmost column (near 139.5, close to Tokyo) should be land; the
    # easternmost column (near 144.5, open Pacific) should not be.
    assert mask[:, 0].any()
    assert not mask[:, -1].any()


def test_land_mask_treats_cells_outside_the_bundled_raster_as_not_land() -> None:
    # The bundled raster only covers Japan and its outlying territory (see
    # NATURAL_EARTH_PROVENANCE.json); nothing about GeographicBounds stops a
    # caller from building a grid elsewhere, and such cells must not silently
    # borrow whichever raster edge cell happens to be nearest.
    bounds = GeographicBounds(west_deg=-75.0, south_deg=39.0, east_deg=-73.0, north_deg=41.0)
    grid = SurfaceGrid.from_bounds(bounds, shape=(3, 3))
    mask = land_mask_for_grid(grid)
    assert not mask.any()


def test_land_mask_rejects_an_unknown_source_name() -> None:
    grid = _grid()
    with pytest.raises(ValueError, match="Unknown land source"):
        land_mask_for_grid(grid, land="not_a_real_source")


# --------------------------------------------------------------------------
# Continuous rendering
# --------------------------------------------------------------------------


def test_render_surface_rgba_has_correct_shape_and_alpha_channel() -> None:
    surface = _continuous_surface()
    rgba = render_surface_rgba(surface, cmin=0.0, cmax=100.0, land=None)
    assert rgba.shape == (*surface.grid.shape, 4)
    assert rgba.dtype == np.uint8
    # Alpha must be 0 exactly where unsupported, and the configured opacity
    # exactly where supported (land=None, so support alone decides visibility).
    expected_alpha = np.where(surface.support_mask, round(255 * 0.72), 0)
    np.testing.assert_array_equal(rgba[..., 3], expected_alpha)


def test_render_surface_rgba_land_mask_further_restricts_visibility() -> None:
    surface = _continuous_surface()
    without_land = render_surface_rgba(surface, cmin=0.0, cmax=100.0, land=None)
    with_land = render_surface_rgba(surface, cmin=0.0, cmax=100.0, land="natural_earth_japan_10m")
    visible_without = without_land[..., 3] > 0
    visible_with = with_land[..., 3] > 0
    # Land masking can only remove visibility, never add it.
    assert np.all(visible_with <= visible_without)


def test_render_surface_rgba_clips_out_of_range_values_to_boundary_color() -> None:
    surface = _continuous_surface()
    full_range = render_surface_rgba(
        surface, colorscale="YlOrRd", cmin=0.0, cmax=200.0, land=None
    )
    # Every supported cell's value is <= 100 (see _stations' uniform(1, 100)); a
    # narrower cmax forces some cells above range, which must clip to the top
    # color rather than turning transparent.
    narrow_range = render_surface_rgba(
        surface, colorscale="YlOrRd", cmin=0.0, cmax=50.0, land=None
    )
    supported = surface.support_mask
    assert np.array_equal(narrow_range[..., 3][supported], full_range[..., 3][supported])
    above_range = supported & (surface.values > 50.0)
    if above_range.any():
        top_color = narrow_range[above_range][0, :3]
        assert np.all(narrow_range[above_range][:, :3] == top_color)


def test_render_surface_rgba_rejects_invalid_color_range_or_opacity() -> None:
    surface = _continuous_surface()
    with pytest.raises(ValueError, match="cmin"):
        render_surface_rgba(surface, cmin=10.0, cmax=5.0)
    with pytest.raises(ValueError, match="opacity"):
        render_surface_rgba(surface, cmin=0.0, cmax=1.0, opacity=0.0)
    with pytest.raises(ValueError, match="Unsupported color_transform"):
        render_surface_rgba(surface, cmin=0.0, cmax=1.0, color_transform="sqrt")


def test_render_surface_rgba_log_transform_hides_non_positive_cells() -> None:
    lat, lon, _ = _stations(6)
    grid = _grid()
    # A value of exactly zero has no logarithm and must not crash the render.
    value = np.array([0.0, 1.0, 5.0, 10.0, 50.0, 90.0])
    surface = interpolate_surface(
        lat,
        lon,
        value,
        grid=grid,
        config=IDWConfig(neighbors=6, max_distance_km=200.0, minimum_neighbors=1),
    )
    rgba = render_surface_rgba(
        surface, cmin=0.1, cmax=100.0, color_transform="log", land=None
    )
    non_positive = surface.support_mask & (surface.values <= 0.0)
    assert np.all(rgba[..., 3][non_positive] == 0)


def test_render_surface_rgba_log_transform_requires_positive_range() -> None:
    surface = _continuous_surface()
    with pytest.raises(ValueError, match="positive"):
        render_surface_rgba(surface, cmin=-1.0, cmax=100.0, color_transform="log")


# --------------------------------------------------------------------------
# Discrete class rendering
# --------------------------------------------------------------------------


def _class_surface():
    lat, lon, _ = _stations()
    classes = RNG.integers(0, 5, lat.size).astype(np.float64)
    grid = _grid()
    return interpolate_surface(
        lat,
        lon,
        classes,
        grid=grid,
        method="nearest",
        config=NearestConfig(max_distance_km=150.0),
    )


_CLASS_COLORS = {0.0: "#D3D3D3", 1.0: "#0040FF", 2.0: "#FFE600", 3.0: "#FF2800", 4.0: "#A50021"}


def test_render_class_surface_rgba_uses_the_exact_color_per_class() -> None:
    surface = _class_surface()
    rgba = render_class_surface_rgba(surface, colors=_CLASS_COLORS, land=None)
    for class_value, hex_color in _CLASS_COLORS.items():
        expected = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
        matches = surface.support_mask & np.isclose(surface.values, class_value)
        if not matches.any():
            continue
        pixels = rgba[matches]
        assert np.all(pixels[:, 0] == expected[0])
        assert np.all(pixels[:, 1] == expected[1])
        assert np.all(pixels[:, 2] == expected[2])


def test_render_class_surface_rgba_rejects_empty_colors() -> None:
    surface = _class_surface()
    with pytest.raises(ValueError, match="colors"):
        render_class_surface_rgba(surface, colors={})


def test_render_class_surface_rgba_rejects_invalid_opacity() -> None:
    surface = _class_surface()
    with pytest.raises(ValueError, match="opacity"):
        render_class_surface_rgba(surface, colors=_CLASS_COLORS, opacity=1.5)


def test_render_class_surface_rgba_rejects_a_malformed_hex_color() -> None:
    surface = _class_surface()
    with pytest.raises(ValueError, match="RRGGBB"):
        render_class_surface_rgba(surface, colors={0.0: "#FFF"})


def test_render_class_surface_rgba_leaves_unmatched_classes_transparent() -> None:
    surface = _class_surface()
    # Deliberately omit class 4 from the palette.
    partial_colors = {k: v for k, v in _CLASS_COLORS.items() if k != 4.0}
    rgba = render_class_surface_rgba(surface, colors=partial_colors, land=None)
    unmatched = surface.support_mask & np.isclose(surface.values, 4.0)
    if unmatched.any():
        assert np.all(rgba[..., 3][unmatched] == 0)


# --------------------------------------------------------------------------
# Plotly composition
# --------------------------------------------------------------------------


def test_add_surface_layer_inserts_one_image_layer_below_traces() -> None:
    surface = _continuous_surface()
    figure = go.Figure()
    figure.add_trace(go.Scattermap(lat=[35.0], lon=[138.0], mode="markers"))
    figure.update_layout(map={"style": "carto-positron"})

    returned = add_surface_layer(figure, surface, cmin=0.0, cmax=100.0)
    assert returned is figure
    layers = figure.layout.map.layers
    assert len(layers) == 1
    layer = layers[0].to_plotly_json()
    assert layer["sourcetype"] == "image"
    assert layer["below"] == "traces"
    assert layer["source"].startswith("data:image/png;base64,")
    assert layer["coordinates"] == [
        list(corner) for corner in surface.grid.image_corners
    ]


def test_add_surface_layer_with_colorbar_title_adds_one_trace() -> None:
    surface = _continuous_surface()
    figure = go.Figure()
    figure.update_layout(map={"style": "carto-positron"})
    before = len(figure.data)

    add_surface_layer(figure, surface, cmin=0.0, cmax=100.0, colorbar_title="PGV [cm/s]")
    assert len(figure.data) == before + 1
    colorbar_trace = figure.data[-1]
    assert colorbar_trace.marker.showscale is True
    assert colorbar_trace.marker.colorbar.title.text == "PGV [cm/s]"


def test_add_surface_layer_colorbar_x_offsets_away_from_the_default_position() -> None:
    surface = _continuous_surface()
    figure = go.Figure()
    figure.update_layout(map={"style": "carto-positron"})

    add_surface_layer(
        figure, surface, cmin=0.0, cmax=100.0, colorbar_title="PGV [cm/s]", colorbar_x=1.15
    )
    assert figure.data[-1].marker.colorbar.x == 1.15


def test_add_surface_layer_without_colorbar_title_adds_no_trace() -> None:
    surface = _continuous_surface()
    figure = go.Figure()
    figure.update_layout(map={"style": "carto-positron"})
    before = len(figure.data)
    add_surface_layer(figure, surface, cmin=0.0, cmax=100.0)
    assert len(figure.data) == before


def test_add_class_surface_layer_inserts_a_layer_and_no_trace() -> None:
    surface = _class_surface()
    figure = go.Figure()
    figure.update_layout(map={"style": "carto-positron"})
    before = len(figure.data)

    add_class_surface_layer(figure, surface, colors=_CLASS_COLORS)
    assert len(figure.layout.map.layers) == 1
    assert len(figure.data) == before


def test_surface_layer_composes_below_multiple_existing_layers() -> None:
    surface = _continuous_surface()
    figure = go.Figure()
    figure.update_layout(
        map={"style": "carto-positron", "layers": [{"sourcetype": "raster", "source": []}]}
    )
    add_surface_layer(figure, surface, cmin=0.0, cmax=100.0)
    assert len(figure.layout.map.layers) == 2
