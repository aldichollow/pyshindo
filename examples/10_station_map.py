# %% Imports
# Requires the optional extra: pip install "pyshindo[plot]"
import csv
import urllib.request
from pathlib import Path

from pyshindo.plotting import continuous_value_map_figure, long_period_class_map_figure

# %% Every map figure here is a thin adapter: parallel latitude, longitude,
# and value arrays in, one figure out. Nothing in pyshindo.plotting.maps
# reads a CSV or knows what JMA's long-period observation page looks like --
# that is this example's job, not the library's.
#
# Records are downloaded rather than bundled: they are JMA's to distribute,
# and the terms of use should be read at the source.
EVENT_ID = "20260823020050"  # 2026-08-23 02:00 茨城県南部 M5.9, class 2
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


level_rows = fetch("level.csv")
max_rows = fetch("max.csv")

# %% Long-period ground motion class at every station that reported one
# level.csv: name, region, start time, duration, class, 7 band classes, lat, lon
latitudes = [float(row[13]) for row in level_rows]
longitudes = [float(row[14]) for row in level_rows]
classes = [row[5] for row in level_rows]
station_names = [row[1] for row in level_rows]

long_period_class_map_figure(
    latitudes,
    longitudes,
    classes,
    labels=station_names,
    title=f"Long-period ground motion class, event {EVENT_ID}",
).show()

# %% The same page's max.csv publishes peak velocity per station -- a
# continuous quantity, so it gets a colorscale instead of discrete classes
pgv_rows = [row for row in max_rows if row[12].strip()]  # skip stations with no published PGV
pgv_latitudes = [float(row[2]) for row in pgv_rows]
pgv_longitudes = [float(row[3]) for row in pgv_rows]
pgv_values = [float(row[12]) for row in pgv_rows]
pgv_labels = [row[1] for row in pgv_rows]

continuous_value_map_figure(
    pgv_latitudes,
    pgv_longitudes,
    pgv_values,
    value_label="Published PGV [cm/s]",
    labels=pgv_labels,
    title=f"Published peak velocity, event {EVENT_ID}",
).show()

# %% Filtering which stations to plot is the caller's job, not the figure
# function's -- pass only the stations you want. Here: class 1 and above,
# which zooms the automatic centering into the affected region instead of
# all of Japan (most stations nationwide report class 0 for a moderate quake).
affected = [(lat, lon, cls, name) for lat, lon, cls, name in zip(
    latitudes, longitudes, classes, station_names, strict=True
) if cls != "0"]
if affected:
    lat2, lon2, cls2, name2 = zip(*affected, strict=True)
    long_period_class_map_figure(
        lat2, lon2, cls2, labels=name2, title="Long-period class, affected stations only"
    ).show()

# %%
