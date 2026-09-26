# %% Imports
# pyshindo.spatial has no optional dependency: NumPy and SciPy only. Rendering
# a surface as a map layer (pyshindo.plotting.surfaces) needs the optional
# extra: pip install "pyshindo[plot]"
import csv
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from pyshindo import calculate_measured_intensity
from pyshindo.io import download_jma_record, parse_jma_bytes
from pyshindo.plotting import (
    add_class_surface_layer,
    add_surface_layer,
    continuous_value_map_figure,
    intensity_map_figure,
    long_period_class_map_figure,
)
from pyshindo.plotting.theme import JMA_INTENSITY_COLORS, LONG_PERIOD_CLASS_COLORS
from pyshindo.scale import IntensityScale, classify_intensity_array
from pyshindo.spatial import (
    GeographicBounds,
    IDWConfig,
    NearestConfig,
    SurfaceGrid,
    interpolate_surface,
)

# %% Reuses the same real JMA long-period observation event, cache, and CSV
# columns as examples/10_station_map.py -- see that example for what each
# column means. No network is used if that example has already run once.
EVENT_ID = "20260823020050"  # 2026-08-23 02:00 茨城県南部 M5.9
EVENT_BASE = f"https://www.data.jma.go.jp/eew/data/ltpgm/{EVENT_ID}"
BASE = f"{EVENT_BASE}/data"
cache = Path(".cache/pyshindo")
cache.mkdir(parents=True, exist_ok=True)


def fetch(name: str) -> list[list[str]]:
    path = cache / f"{EVENT_ID}_{name}"
    if not path.exists():
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=120) as response:  # noqa: S310
            path.write_bytes(response.read())
    rows = list(csv.reader(path.read_bytes().decode("cp932").splitlines()))
    return [row for row in rows[1:] if row]


max_rows = fetch("max.csv")
level_rows = fetch("level.csv")


# %% The grid only needs to cover the stations actually being interpolated,
# with a little padding so the edge stations are not sitting right on the
# boundary -- built the same way for every metric below.
def grid_for_stations(
    lat: np.ndarray, lon: np.ndarray, *, resolution_km: float
) -> SurfaceGrid:
    bounds = GeographicBounds(
        west_deg=float(lon.min()) - 0.3,
        south_deg=float(lat.min()) - 0.3,
        east_deg=float(lon.max()) + 0.3,
        north_deg=float(lat.max()) + 0.3,
    )
    return SurfaceGrid.from_bounds(bounds, approximate_resolution_km=resolution_km)


# %% A discretized colorscale -- hard color bands at fixed breakpoints,
# instead of a smooth gradient -- for rendering a continuous surface so it
# reads like a classed map. Built from Plotly's own colorscale form, a list
# of ``[position, color]`` pairs: two entries placed a hair apart at each
# breakpoint give a hard edge instead of a blend. Used below for both the
# station markers and the interpolated surface from the *same* class
# boundaries and colors, so the two never disagree about what a color means.
def banded_colorscale(
    *,
    cmin: float,
    cmax: float,
    interior_breakpoints: list[float],
    colors: list[str],
    log: bool = False,
) -> list[list[float | str]]:
    """Build a hard-edged colorscale; one more color than interior_breakpoints."""
    if len(colors) != len(interior_breakpoints) + 1:
        raise ValueError("colors must have exactly one more entry than interior_breakpoints.")

    def position(value: float) -> float:
        if log:
            return (math.log(value) - math.log(cmin)) / (math.log(cmax) - math.log(cmin))
        return (value - cmin) / (cmax - cmin)

    edges = [cmin, *interior_breakpoints, cmax]
    eps = 1e-4
    scale: list[list[float | str]] = [[0.0, colors[0]]]
    for i in range(1, len(edges) - 1):
        pos = position(edges[i])
        scale.append([pos - eps, colors[i - 1]])
        scale.append([pos + eps, colors[i]])
    scale.append([1.0, colors[-1]])
    return scale


# %% Interpolate PGV: a continuous quantity, and the natural first case for
# IDW. max.csv publishes PGA/PGV per station; column 12 is the
# three-component resultant (see examples/10_station_map.py).
pgv_rows = [row for row in max_rows if row[12].strip()]
pgv_lat = np.array([float(row[2]) for row in pgv_rows])
pgv_lon = np.array([float(row[3]) for row in pgv_rows])
pgv_value = np.array([float(row[12]) for row in pgv_rows])
pgv_grid = grid_for_stations(pgv_lat, pgv_lon, resolution_km=5.0)

pgv_surface = interpolate_surface(
    pgv_lat,
    pgv_lon,
    pgv_value,
    grid=pgv_grid,
    config=IDWConfig(neighbors=6, max_distance_km=40.0, minimum_neighbors=2),
    metric_name="pgv",
    unit="cm/s",
    component_definition="three_component_vector_peak",
)
supported_fraction = pgv_surface.support_mask.mean()
print(
    f"PGV surface: {pgv_grid.shape[0]}x{pgv_grid.shape[1]} cells, "
    f"{supported_fraction:.1%} supported"
)
print(
    f"PGV range over supported cells: "
    f"{np.nanmin(pgv_surface.values):.2f} - {np.nanmax(pgv_surface.values):.2f} cm/s"
)

# %% Interpolating a continuous quantity in log space is a real choice, not a
# default: ground-motion amplitude is often closer to log-normal than normal.
# Compare the two at the same station-dense point near the middle of the array.
pgv_log_surface = interpolate_surface(
    pgv_lat,
    pgv_lon,
    pgv_value,
    grid=pgv_grid,
    config=IDWConfig(neighbors=6, max_distance_km=40.0, minimum_neighbors=2),
    transform="log",
    metric_name="pgv",
    unit="cm/s",
)
probe_lat, probe_lon = float(np.median(pgv_lat)), float(np.median(pgv_lon))
print(
    f"\nAt ({probe_lat:.2f}, {probe_lon:.2f}): "
    f"identity={pgv_surface.sample_nearest(probe_lat, probe_lon):.2f} cm/s, "
    f"log={pgv_log_surface.sample_nearest(probe_lat, probe_lon):.2f} cm/s"
)

# %% Measured seismic intensity is a third continuous quantity, but not one
# any published summary file hands over directly: max.csv's intensity column
# is the rounded, classified display value (see docs/api.md), not
# intensity_raw. This section derives the continuous precursor the same way
# examples/03_official_jma_record.py does for one station -- calculate it
# locally from the acceleration waveform -- for every station that reported
# one, then interpolates that.
archive_path = cache / f"{EVENT_ID}_acc.zip"
if not archive_path.exists():
    download_jma_record(
        f"{EVENT_BASE}/station/data/acc.zip", archive_path, max_bytes=64 * 1024 * 1024
    )
waveform_dir = cache / f"{EVENT_ID}_acc"
if not waveform_dir.exists():
    with zipfile.ZipFile(archive_path) as bundle:
        bundle.extractall(waveform_dir)

station_coordinates = {row[0]: (float(row[2]), float(row[3])) for row in max_rows}
station_names = {row[0]: row[1] for row in max_rows}
intensity_lat: list[float] = []
intensity_lon: list[float] = []
intensity_value: list[float] = []
intensity_labels: list[str] = []
for waveform_path in sorted(waveform_dir.rglob("*_acc.csv")):
    code = waveform_path.name[:5]
    if code not in station_coordinates:
        continue
    record = parse_jma_bytes(waveform_path.read_bytes(), source=str(waveform_path))
    result = calculate_measured_intensity(
        record.acceleration,
        record.metadata.sampling_rate_hz,
        unit="gal",
        retain_intermediates=False,
    )
    lat, lon = station_coordinates[code]
    intensity_lat.append(lat)
    intensity_lon.append(lon)
    intensity_value.append(result.intensity_raw)
    intensity_labels.append(station_names[code])

intensity_lat_array = np.array(intensity_lat)
intensity_lon_array = np.array(intensity_lon)
intensity_value_array = np.array(intensity_value)
intensity_grid = grid_for_stations(intensity_lat_array, intensity_lon_array, resolution_km=3.0)
intensity_surface = interpolate_surface(
    intensity_lat_array,
    intensity_lon_array,
    intensity_value_array,
    grid=intensity_grid,
    config=IDWConfig(neighbors=6, max_distance_km=40.0, minimum_neighbors=2),
    metric_name="instrumental_intensity_raw",
)
print(
    f"\nMeasured intensity (raw) computed for {len(intensity_value)} stations; "
    f"range {min(intensity_value):.2f} - {max(intensity_value):.2f}"
)
print(
    f"Interpolated (unrounded) at ({probe_lat:.2f}, {probe_lon:.2f}): "
    f"{intensity_surface.sample_nearest(probe_lat, probe_lon):.3f}"
)

# %% Rendered two ways from that same continuous precursor -- never
# interpolate a class number, see the pyshindo.spatial module docstring --
# but colored to agree with each other: station markers use JMA's own
# discrete intensity-class colors (classify_intensity_array only rounds
# intensity_raw for display, after interpolation already ran on the
# unrounded value), and the surface underneath is banded at those same
# official class boundaries (pyshindo.scale.INTENSITY_INTERVALS), so a
# marker's color and the surface color right under it always agree.
intensity_classes = classify_intensity_array(intensity_value_array)
intensity_colorscale = banded_colorscale(
    cmin=0.0,
    cmax=7.0,
    interior_breakpoints=[0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 5.5, 6.0, 6.5],
    colors=[JMA_INTENSITY_COLORS[scale] for scale in IntensityScale],
)

intensity_figure = intensity_map_figure(
    intensity_lat_array,
    intensity_lon_array,
    intensity_classes,
    labels=intensity_labels,
    title=f"Interpolated measured intensity surface (land only), event {EVENT_ID}",
)
add_surface_layer(
    intensity_figure,
    intensity_surface,
    colorscale=intensity_colorscale,
    cmin=0.0,
    cmax=7.0,
)
intensity_figure.show()

# %% Long-period ground motion class is categorical here: level.csv publishes
# only the class (0-4), not the max_sva_cm_s a continuous interpolation would
# need. This is exactly the situation method="nearest" exists for -- trying
# IDW or "linear" on class numbers directly would average ordinal labels,
# which is not a meaningful operation. Rendered with the package's own class
# colors -- the same ones the station markers already use, so the background
# wash and the legend agree.
class_lat = np.array([float(row[13]) for row in level_rows])
class_lon = np.array([float(row[14]) for row in level_rows])
class_value = np.array([float(row[5]) for row in level_rows])
class_names = [row[1] for row in level_rows]
class_grid = grid_for_stations(class_lat, class_lon, resolution_km=5.0)
class_surface = interpolate_surface(
    class_lat,
    class_lon,
    class_value,
    grid=class_grid,
    method="nearest",
    config=NearestConfig(max_distance_km=30.0),
    metric_name="long_period_class",
)
present = class_surface.values[class_surface.support_mask]
observed_classes = sorted(int(c) for c in np.unique(present))
print(f"\nNearest-assigned long-period classes present in the surface: {observed_classes}")

class_figure = long_period_class_map_figure(
    class_lat,
    class_lon,
    [row[5] for row in level_rows],
    labels=class_names,
    title=f"Long-period class surface (land only), event {EVENT_ID}",
)
add_class_surface_layer(
    class_figure,
    class_surface,
    colors={float(cls.value): color for cls, color in LONG_PERIOD_CLASS_COLORS.items()},
)
class_figure.show()

# %% PGA: interpolated with Suzuki et al. (2017)'s own IDW search geometry
# (IDWConfig.suzuki_2017_pga), averaged in log space -- ground-motion
# amplitude is closer to log-normal than normal -- and displayed on a
# log-scaled, banded colorscale shared by both the markers and the surface
# underneath them.
pga_rows = [row for row in max_rows if row[8].strip()]
pga_lat = np.array([float(row[2]) for row in pga_rows])
pga_lon = np.array([float(row[3]) for row in pga_rows])
pga_value = np.array([float(row[8]) for row in pga_rows])
pga_labels = [row[1] for row in pga_rows]
pga_grid = grid_for_stations(pga_lat, pga_lon, resolution_km=3.0)
pga_surface = interpolate_surface(
    pga_lat,
    pga_lon,
    pga_value,
    grid=pga_grid,
    config=IDWConfig.suzuki_2017_pga(),
    transform="log",
    metric_name="pga",
    unit="gal",
)
print(
    f"\nPGA range over supported cells: "
    f"{np.nanmin(pga_surface.values):.2f} - {np.nanmax(pga_surface.values):.2f} gal"
)

pga_colorscale = banded_colorscale(
    cmin=0.05,
    cmax=1000.0,
    interior_breakpoints=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500],
    colors=[
        "rgb(247,247,247)", "rgb(220,238,251)", "rgb(168,208,240)", "rgb(62,127,193)",
        "rgb(46,139,139)", "rgb(76,166,76)", "rgb(125,193,66)", "rgb(198,217,74)",
        "rgb(252,233,59)", "rgb(253,185,46)", "rgb(243,114,44)", "rgb(228,67,43)",
        "rgb(139,26,26)",
    ],  # fmt: skip
    log=True,
)

pga_figure = continuous_value_map_figure(
    pga_lat,
    pga_lon,
    pga_value,
    value_label="PGA [gal]",
    labels=pga_labels,
    colorscale=pga_colorscale,
    color_transform="log",
    # Left as None (the default), the markers would auto-range to this
    # event's own PGA min/max instead of the fixed 0.05-1000 range the
    # surface below uses -- matching cmin/cmax explicitly here is what
    # actually makes the marker fill and the surface color agree.
    cmin=0.05,
    cmax=1000.0,
    title=f"Interpolated PGA surface (land only), event {EVENT_ID}",
)
add_surface_layer(
    pga_figure,
    pga_surface,
    colorscale=pga_colorscale,
    cmin=0.05,
    cmax=1000.0,
    color_transform="log",
)
pga_figure.show()

# %% Every figure above used the bundled Natural Earth land mask (the
# default land="natural_earth_japan_10m"); pass land=None to see the raw
# support_mask extend the color out over the ocean instead, which is useful
# for sanity-checking the interpolation geometry itself but not usually the
# presentation you want. This is a station interpolation surface, not an
# estimated ground-motion field -- see the pyshindo.spatial module
# docstring for what that means and does not mean. In particular, every
# surface above is geometry only: distance and station density, nothing
# about terrain, soil, or geology in between two stations. Two points a
# short distance apart on the map can shake very differently in reality
# (a ridge versus a sediment-filled valley, for instance) in a way no
# amount of interpolation from surface stations alone can recover.

# %%
