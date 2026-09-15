# %% Imports
# Requires the optional extra: pip install "pyshindo[plot]"
import csv
import urllib.request
from pathlib import Path

import numpy as np

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
BASE = f"https://www.data.jma.go.jp/eew/data/ltpgm/{EVENT_ID}/data"
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

# %% PGV: interpolate with IDW, then lay the rendered surface underneath the
# same station markers examples/10_station_map.py plots on their own. The
# color range (cmin/cmax) is a display choice, made here, not derived from
# the surface -- see render_surface_rgba's docstring for why.
pgv_rows = [row for row in max_rows if row[12].strip()]
pgv_lat = np.array([float(row[2]) for row in pgv_rows])
pgv_lon = np.array([float(row[3]) for row in pgv_rows])
pgv_value = np.array([float(row[12]) for row in pgv_rows])
pgv_labels = [row[1] for row in pgv_rows]

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
)

figure = continuous_value_map_figure(
    pgv_lat,
    pgv_lon,
    pgv_value,
    value_label="Published PGV [cm/s]",
    labels=pgv_labels,
    title=f"Interpolated PGV surface (land only), event {EVENT_ID}",
)
add_surface_layer(
    figure,
    pgv_surface,
    cmin=0.0,
    cmax=float(pgv_value.max()),
    colorbar_title="Interpolated PGV [cm/s]",
)
figure.show()

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

# %% Both figures used the bundled Natural Earth land mask (the default
# land="natural_earth_japan_10m"); pass land=None to see the raw
# support_mask extend the color out over the ocean instead, which is useful
# for sanity-checking the interpolation geometry itself but not usually the
# presentation you want.

# %%
