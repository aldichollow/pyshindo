# %% Imports
# Requires the optional extra: pip install "pyshindo[plot]"
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
    long_period_class_map_figure,
)
from pyshindo.plotting.theme import LONG_PERIOD_CLASS_COLORS
from pyshindo.spatial import (
    GeographicBounds,
    IDWConfig,
    NearestConfig,
    SurfaceGrid,
    interpolate_surface,
)

# %% Same real event as examples/10_station_map.py and
# examples/12_station_surface_interpolation.py -- see those for what each
# column means. No network is used if either has already run once.
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


# %% A discretized colorscale -- hard color bands at fixed breakpoints,
# instead of a smooth gradient -- matching how published PGA/PGV maps are
# usually drawn (Suzuki et al. 2017 among them; see IDWConfig.suzuki_2017_pga
# below). Built from Plotly's own colorscale form, a list of
# ``[position, color]`` pairs: two entries placed a hair apart at each
# breakpoint give a hard edge instead of a blend. Reused for both the
# station markers and the interpolated surface, so the two never disagree
# about what a given color means.
def banded_colorscale(
    breakpoints: list[float], colors: list[str]
) -> tuple[list[list[float | str]], float, float]:
    """Build a log-spaced banded colorscale; return it with its (cmin, cmax)."""
    cmin, cmax = breakpoints[0], breakpoints[-1]
    positions = [
        (math.log(b) - math.log(cmin)) / (math.log(cmax) - math.log(cmin)) for b in breakpoints
    ]
    eps = 1e-4
    scale: list[list[float | str]] = [[0.0, colors[0]]]
    for i in range(1, len(positions) - 1):
        scale.append([positions[i] - eps, colors[i - 1]])
        scale.append([positions[i] + eps, colors[i]])
    scale.append([1.0, colors[-1]])
    return scale, cmin, cmax


PGA_BREAKPOINTS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
PGA_COLORS = [
    "rgb(247,247,247)", "rgb(220,238,251)", "rgb(168,208,240)", "rgb(62,127,193)",
    "rgb(46,139,139)", "rgb(76,166,76)", "rgb(125,193,66)", "rgb(198,217,74)",
    "rgb(252,233,59)", "rgb(253,185,46)", "rgb(243,114,44)", "rgb(228,67,43)",
    "rgb(139,26,26)",
]  # fmt: skip
pga_colorscale, pga_cmin, pga_cmax = banded_colorscale(PGA_BREAKPOINTS, PGA_COLORS)

# %% PGA: interpolated with Suzuki et al. (2017)'s own IDW search geometry
# (IDWConfig.suzuki_2017_pga), averaged in log space -- ground-motion
# amplitude is closer to log-normal than normal -- and displayed on the same
# log-scaled, banded colorscale for both the markers and the surface
# underneath them, so the two always agree.
pga_rows = [row for row in max_rows if row[8].strip()]
pga_lat = np.array([float(row[2]) for row in pga_rows])
pga_lon = np.array([float(row[3]) for row in pga_rows])
pga_value = np.array([float(row[8]) for row in pga_rows])
pga_labels = [row[1] for row in pga_rows]

pga_bounds = GeographicBounds(
    west_deg=float(pga_lon.min()) - 0.3,
    south_deg=float(pga_lat.min()) - 0.3,
    east_deg=float(pga_lon.max()) + 0.3,
    north_deg=float(pga_lat.max()) + 0.3,
)
pga_grid = SurfaceGrid.from_bounds(pga_bounds, approximate_resolution_km=3.0)
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

pga_figure = continuous_value_map_figure(
    pga_lat,
    pga_lon,
    pga_value,
    value_label="PGA [gal]",
    labels=pga_labels,
    colorscale=pga_colorscale,
    color_transform="log",
    title=f"Interpolated PGA surface (land only), event {EVENT_ID}",
)
add_surface_layer(
    pga_figure,
    pga_surface,
    colorscale=pga_colorscale,
    cmin=pga_cmin,
    cmax=pga_cmax,
    color_transform="log",
)
pga_figure.show()

# %% Measured seismic intensity (intensity_raw): not published directly --
# max.csv's intensity column is the rounded, classified display value (see
# docs/api.md), not intensity_raw -- so this derives the continuous
# precursor locally from each station's acceleration waveform, the same way
# examples/03_official_jma_record.py and examples/12_station_surface_interpolation.py
# do. Instrumental intensity is already a log-derived index by construction
# (see pyshindo.scale for the formula), so it is displayed on an identity,
# not log, scale here -- stacking a second log transform on top of one would
# not be a meaningful quantity.
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

intensity_bounds = GeographicBounds(
    west_deg=float(intensity_lon_array.min()) - 0.3,
    south_deg=float(intensity_lat_array.min()) - 0.3,
    east_deg=float(intensity_lon_array.max()) + 0.3,
    north_deg=float(intensity_lat_array.max()) + 0.3,
)
intensity_grid = SurfaceGrid.from_bounds(intensity_bounds, approximate_resolution_km=3.0)
intensity_surface = interpolate_surface(
    intensity_lat_array,
    intensity_lon_array,
    intensity_value_array,
    grid=intensity_grid,
    config=IDWConfig(neighbors=6, max_distance_km=40.0, minimum_neighbors=2),
    metric_name="instrumental_intensity_raw",
)

intensity_figure = continuous_value_map_figure(
    intensity_lat_array,
    intensity_lon_array,
    intensity_value_array,
    value_label="Measured intensity (unrounded)",
    labels=intensity_labels,
    title=f"Interpolated measured intensity surface (land only), event {EVENT_ID}",
)
add_surface_layer(
    intensity_figure,
    intensity_surface,
    cmin=0.0,
    cmax=float(intensity_value_array.max()),
)
intensity_figure.show()

# %% Long-period ground motion class: nearest-neighbor assignment (a class
# number is not a quantity IDW or linear interpolation should average), then
# rendered with the package's own class colors -- the same ones the station
# markers already use, so the background wash and the legend agree.
class_lat = np.array([float(row[13]) for row in level_rows])
class_lon = np.array([float(row[14]) for row in level_rows])
class_value = np.array([float(row[5]) for row in level_rows])
class_names = [row[1] for row in level_rows]

class_bounds = GeographicBounds(
    west_deg=float(class_lon.min()) - 0.3,
    south_deg=float(class_lat.min()) - 0.3,
    east_deg=float(class_lon.max()) + 0.3,
    north_deg=float(class_lat.max()) + 0.3,
)
class_grid = SurfaceGrid.from_bounds(class_bounds, approximate_resolution_km=5.0)
class_surface = interpolate_surface(
    class_lat,
    class_lon,
    class_value,
    grid=class_grid,
    method="nearest",
    config=NearestConfig(max_distance_km=30.0),
    metric_name="long_period_class",
)

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

# %% Every figure above used the bundled Natural Earth land mask (the
# default land="natural_earth_japan_10m"); pass land=None to see the raw
# support_mask extend the color out over the ocean instead, which is useful
# for sanity-checking the interpolation geometry itself but not usually the
# presentation you want.

# %%
