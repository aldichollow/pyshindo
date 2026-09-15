# %% Imports
# pyshindo.spatial has no optional dependency: NumPy and SciPy only.
import csv
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from pyshindo import calculate_measured_intensity
from pyshindo.io import download_jma_record, parse_jma_bytes
from pyshindo.spatial import (
    GeographicBounds,
    IDWConfig,
    NearestConfig,
    SurfaceGrid,
    build_interpolation_plan,
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

# %% Interpolate PGV: a continuous quantity, and the natural first case for
# IDW. max.csv publishes PGA/PGV per station; column 12 is the
# three-component resultant (see examples/10_station_map.py).
pgv_rows = [row for row in max_rows if row[12].strip()]
pgv_lat = np.array([float(row[2]) for row in pgv_rows])
pgv_lon = np.array([float(row[3]) for row in pgv_rows])
pgv_value = np.array([float(row[12]) for row in pgv_rows])

# The grid only needs to cover the stations actually being interpolated, with a
# little padding so the edge stations are not sitting right on the boundary.
bounds = GeographicBounds(
    west_deg=float(pgv_lon.min()) - 0.3,
    south_deg=float(pgv_lat.min()) - 0.3,
    east_deg=float(pgv_lon.max()) + 0.3,
    north_deg=float(pgv_lat.max()) + 0.3,
)
grid = SurfaceGrid.from_bounds(bounds, approximate_resolution_km=5.0)

pgv_surface = interpolate_surface(
    pgv_lat,
    pgv_lon,
    pgv_value,
    grid=grid,
    config=IDWConfig(neighbors=6, max_distance_km=40.0, minimum_neighbors=2),
    metric_name="pgv",
    unit="cm/s",
    component_definition="three_component_vector_peak",
)
supported_fraction = pgv_surface.support_mask.mean()
print(f"PGV surface: {grid.shape[0]}x{grid.shape[1]} cells, {supported_fraction:.1%} supported")
print(
    f"PGV range over supported cells: "
    f"{np.nanmin(pgv_surface.values):.2f} - {np.nanmax(pgv_surface.values):.2f} cm/s"
)

# %% Interpolating a continuous quantity in log space is a real choice, not a
# default: ground-motion amplitude is often closer to log-normal than normal.
# Compare the two at the same station-dense point near the middle of the array.
log_surface = interpolate_surface(
    pgv_lat,
    pgv_lon,
    pgv_value,
    grid=grid,
    config=IDWConfig(neighbors=6, max_distance_km=40.0, minimum_neighbors=2),
    transform="log",
    metric_name="pgv",
    unit="cm/s",
)
probe_lat, probe_lon = float(np.median(pgv_lat)), float(np.median(pgv_lon))
print(
    f"\nAt ({probe_lat:.2f}, {probe_lon:.2f}): "
    f"identity={pgv_surface.sample_nearest(probe_lat, probe_lon):.2f} cm/s, "
    f"log={log_surface.sample_nearest(probe_lat, probe_lon):.2f} cm/s"
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
intensity_lat: list[float] = []
intensity_lon: list[float] = []
intensity_value: list[float] = []
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

intensity_surface = interpolate_surface(
    np.array(intensity_lat),
    np.array(intensity_lon),
    np.array(intensity_value),
    grid=grid,
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

# %% Long-period ground motion class is categorical here: level.csv publishes
# only the class (0-4), not the max_sva_cm_s a continuous interpolation would
# need. This is exactly the situation method="nearest" exists for -- trying
# IDW or "linear" on class numbers directly would average ordinal labels,
# which is not a meaningful operation.
class_lat = np.array([float(row[13]) for row in level_rows])
class_lon = np.array([float(row[14]) for row in level_rows])
class_value = np.array([float(row[5]) for row in level_rows])

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
present = class_surface.values[class_surface.support_mask]
observed_classes = sorted(int(c) for c in np.unique(present))
print(f"\nNearest-assigned long-period classes present in the surface: {observed_classes}")

# %% A reusable plan pays off once the same stations and grid are interpolated
# more than once -- here, PGA reuses the PGV plan's geometry with no repeated
# neighbor search.
pga_value = np.array([float(row[8]) for row in pgv_rows])  # column 8: PGA resultant
plan = build_interpolation_plan(
    pgv_lat,
    pgv_lon,
    grid=grid,
    config=IDWConfig(neighbors=6, max_distance_km=40.0, minimum_neighbors=2),
)
pga_surface = plan.interpolate(pga_value, metric_name="pga", unit="gal")
print(
    f"\nPGA range over supported cells: "
    f"{np.nanmin(pga_surface.values):.2f} - {np.nanmax(pga_surface.values):.2f} gal"
)

# %% This is a station interpolation surface, not an estimated ground-motion
# field -- see the pyshindo.spatial module docstring. Rendering one of these
# as a map layer, with land-only masking, is pyshindo.plotting.surfaces.

# %%
