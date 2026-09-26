"""Generate the bundled Japan-region land mask raster from Natural Earth.

Rendering a station interpolation surface as a land-only map layer needs to
know which grid cells are land. Rather than testing point-in-polygon
containment against a vector coastline at render time -- which would make
:mod:`pyshindo.plotting.surfaces` depend on Shapely for every user, just to
draw a map -- this script does that containment test once, here, and bundles
the result as a small boolean raster. At render time,
:mod:`pyshindo.plotting.surfaces` only ever does array indexing into that
raster: no geometry library, no polygon math, no Shapely import.

Run it as::

    uv run --with shapely python scripts/build_natural_earth_japan.py

Shapely is required to run this script but is never a runtime dependency of
the package -- ``uv run --with`` fetches it into a throwaway environment for
this one invocation and changes nothing in pyproject.toml. This script is a
maintainer tool, not part of the installed package or the test suite.

Data source
-----------
Natural Earth 1:10m Cultural/Physical Vectors, ``land`` and
``minor_islands`` layers, public domain
(https://www.naturalearthdata.com/about/terms-of-use/). Fetched as GeoJSON
from the project's own GitHub mirror
(https://github.com/nvkelso/natural-earth-vector), which republishes the
same public-domain shapefiles converted to GeoJSON; SHA-256 of each
downloaded file is recorded in the output provenance JSON alongside the
source URL, so the exact input this script ran against is auditable without
re-deriving it from Natural Earth's own periodically-updated release
archive.

Output
------

``src/pyshindo/data/spatial/natural_earth_japan_land.npz``
    A packed-bit boolean raster (``numpy.packbits``) over a fixed
    Japan-region bounding box, plus the grid bounds and resolution needed to
    reconstruct :class:`~pyshindo.spatial.SurfaceGrid`-style pixel centers
    from it.

``src/pyshindo/data/spatial/NATURAL_EARTH_PROVENANCE.json``
    Source URLs, SHA-256 of each downloaded GeoJSON, the crop bounds, the
    raster resolution, and this output file's own SHA-256.

Do not hand-edit either generated file; re-run this script instead.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Final

import numpy as np

LAND_URL: Final = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_10m_land.geojson"
)
MINOR_ISLANDS_URL: Final = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_10m_minor_islands.geojson"
)

# West/south/east/north, in degrees. Wide enough to include Japan's outlying
# territory -- Minamitorishima in the east (~153.98 E), Okinotorishima in the
# south (~20.42 N), and the Kuril-area northern extent (~45.6 N) -- with a
# margin so a station or grid cell near the edge is not clipped by the raster
# itself rather than by whatever bounds the caller actually requested.
CROP_WEST_DEG: Final = 122.0
CROP_SOUTH_DEG: Final = 19.0
CROP_EAST_DEG: Final = 155.0
CROP_NORTH_DEG: Final = 47.0

# ~1.1 km at this latitude range: fine enough that a national or regional map
# does not show a blocky coastline, coarse enough to keep both the raster
# build and the bundled file size modest. A caller wanting sharper coastline
# detail at a tight zoom is better served by a dedicated GIS layer than by
# this package's own visualization convenience.
RESOLUTION_DEG: Final = 0.01

OUTPUT_DIR: Final = Path(__file__).resolve().parent.parent / "src" / "pyshindo" / "data" / "spatial"
RASTER_PATH: Final = OUTPUT_DIR / "natural_earth_japan_land.npz"
PROVENANCE_PATH: Final = OUTPUT_DIR / "NATURAL_EARTH_PROVENANCE.json"

_CACHE_DIR: Final = Path(".cache/pyshindo/natural_earth")


def _download(url: str, destination: Path) -> str:
    """Download one file unless already cached; return its SHA-256 hex digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            destination.write_bytes(response.read())
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def _load_geometry(path: Path) -> Any:
    """Return the union of every polygon in one Natural Earth GeoJSON file."""
    import shapely

    with path.open("rb") as handle:
        collection = json.load(handle)
    geometries = [shapely.geometry.shape(feature["geometry"]) for feature in collection["features"]]
    return shapely.union_all(geometries)


def build_raster() -> None:
    """Download, union, and rasterize the Japan-region land mask."""
    land_path = _CACHE_DIR / "ne_10m_land.geojson"
    minor_islands_path = _CACHE_DIR / "ne_10m_minor_islands.geojson"
    land_sha256 = _download(LAND_URL, land_path)
    minor_islands_sha256 = _download(MINOR_ISLANDS_URL, minor_islands_path)

    import shapely

    print("Loading land polygons...")
    land = _load_geometry(land_path)
    minor_islands = _load_geometry(minor_islands_path)

    # Clipping to the crop box before unioning matters a great deal: the raw
    # global land polygon carries tens of thousands of vertices for coastline
    # this script will never rasterize a single point of, and every one of
    # them would otherwise be walked on every per-point containment test
    # below. Clipped and unioned once, the combined geometry has on the order
    # of 10,000 vertices instead.
    crop_box = shapely.geometry.box(CROP_WEST_DEG, CROP_SOUTH_DEG, CROP_EAST_DEG, CROP_NORTH_DEG)
    land = shapely.union_all(
        [shapely.intersection(land, crop_box), shapely.intersection(minor_islands, crop_box)]
    )
    # Builds an internal spatial index for repeated containment queries against
    # this one fixed geometry; without it, shapely.covers below still walks
    # every vertex on every one of the millions of per-point tests that follow.
    shapely.prepare(land)

    nx = round((CROP_EAST_DEG - CROP_WEST_DEG) / RESOLUTION_DEG)
    ny = round((CROP_NORTH_DEG - CROP_SOUTH_DEG) / RESOLUTION_DEG)
    longitudes = CROP_WEST_DEG + (np.arange(nx, dtype=np.float64) + 0.5) * RESOLUTION_DEG
    latitudes = CROP_SOUTH_DEG + (np.arange(ny, dtype=np.float64) + 0.5) * RESOLUTION_DEG
    print(f"Rasterizing {ny} x {nx} = {ny * nx:,} cells...")

    # shapely.covers (rather than .contains) includes boundary points, so a grid
    # cell centered exactly on the coastline is not spuriously left transparent.
    started = time.perf_counter()
    grid_lon, grid_lat = np.meshgrid(longitudes, latitudes)
    points = shapely.points(grid_lon.ravel(), grid_lat.ravel())
    mask = shapely.covers(land, points).reshape(ny, nx)
    elapsed_s = time.perf_counter() - started
    print(f"Rasterized in {elapsed_s:.2f} s; {mask.mean():.1%} of cells are land.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(mask, axis=None)
    np.savez_compressed(
        RASTER_PATH,
        packed_mask=packed,
        shape=np.array([ny, nx], dtype=np.int64),
        west_deg=np.float64(CROP_WEST_DEG),
        south_deg=np.float64(CROP_SOUTH_DEG),
        east_deg=np.float64(CROP_EAST_DEG),
        north_deg=np.float64(CROP_NORTH_DEG),
        resolution_deg=np.float64(RESOLUTION_DEG),
    )
    raster_sha256 = hashlib.sha256(RASTER_PATH.read_bytes()).hexdigest()

    provenance = {
        "source": "Natural Earth 1:10m Cultural/Physical Vectors, land + minor_islands",
        "license": "Public domain (https://www.naturalearthdata.com/about/terms-of-use/)",
        "fetched_from": {
            "land": {"url": LAND_URL, "sha256": land_sha256},
            "minor_islands": {"url": MINOR_ISLANDS_URL, "sha256": minor_islands_sha256},
        },
        "crop_bounds_deg": {
            "west": CROP_WEST_DEG,
            "south": CROP_SOUTH_DEG,
            "east": CROP_EAST_DEG,
            "north": CROP_NORTH_DEG,
        },
        "resolution_deg": RESOLUTION_DEG,
        "raster_shape": [ny, nx],
        "land_fraction": float(mask.mean()),
        "generation_script": "scripts/build_natural_earth_japan.py",
        "output_file": RASTER_PATH.name,
        "output_sha256": raster_sha256,
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {RASTER_PATH} and {PROVENANCE_PATH}")


if __name__ == "__main__":
    build_raster()
