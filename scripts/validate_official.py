"""Compare this package's output against JMA's own published values.

The Japan Meteorological Agency publishes, for each earthquake that triggered
long-period ground motion observation, both the raw waveforms and the values it
derived from them. That makes an end-to-end check possible without any
intermediate interpretation: read the same waveform, run the package, and
compare against what JMA reported for that station.

Five quantities are checked, from three published files per event:

    data/max.csv                     seismic intensity class, peak acceleration
                                     per component and three-component resultant,
                                     peak velocity per component and resultant
    data/level.csv                   overall long-period class, and the class for
                                     each of the seven one-second period bands
    station/data/velsp/*_velsp.csv   the absolute velocity response spectrum
                                     itself, at 5 percent damping

Run it as::

    python scripts/validate_official.py --event 20260823020050

It downloads about 20 MB on the first run and caches it. It is deliberately not
part of the test suite: it needs the network, and the data source keeps only a
few weeks of events, so a fixed event identifier stops resolving after a while.
The published results in docs/validation.md record what a run produced.
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np

from pyshindo import (
    calculate_measured_intensity,
    component_peak_velocity,
    peak_ground_velocity,
)
from pyshindo.io import download_jma_record, parse_jma_bytes
from pyshindo.long_period import calculate_long_period_class

BASE_URL: Final = "https://www.data.jma.go.jp/eew/data/ltpgm"
DEFAULT_EVENT_ID: Final = "20260823020050"
ENCODING: Final = "cp932"

# The waveform is published rounded to three decimals while JMA derived its own
# peaks from the unrounded signal, so recomputing a peak from the published file
# cannot reproduce the published number exactly. Agreement is therefore counted
# at a relative tolerance rather than to the last decimal place.
PEAK_RELATIVE_TOLERANCE: Final = 0.01

# Relative error is meaningless once the published value approaches the rounding
# step, so the peak velocity summary is restricted to stations that actually
# moved. Peak acceleration needs no such floor: every station has one.
SIGNIFICANT_PGV_CM_S: Final = 1.0

# max.csv writes a bare dash when no intensity was determined for a station.
NO_VALUE: Final = "-"

# Column positions in the two summary files, each of which carries one header row.
MAX_CODE, MAX_NAME, MAX_INTENSITY = 0, 1, 4
MAX_PGA: Final = slice(5, 9)
MAX_PGV: Final = slice(9, 13)
LEVEL_CODE, LEVEL_CLASS = 0, 5
LEVEL_BANDS: Final = slice(6, 13)

# velsp.csv holds one column per damping ratio and component; the long-period
# class is defined on the horizontal composite at 5 percent damping.
VELSP_COLUMN: Final = "0.050_hz"

# max.csv writes the two split intensity classes in Japanese; the package uses
# the ASCII notation from its own scale.
JMA_INTENSITY_NOTATION: Final = {"5弱": "5-", "5強": "5+", "6弱": "6-", "6強": "6+"}


@dataclass(frozen=True, slots=True)
class StationComparison:
    """Every published quantity for one station, ours beside JMA's."""

    code: str
    name: str
    intensity: float
    intensity_class: str
    official_intensity_class: str
    pga: np.ndarray
    official_pga: np.ndarray
    pgv: np.ndarray
    official_pgv: np.ndarray
    long_period_class: str
    official_long_period_class: str
    bands: tuple[str, ...]
    official_bands: tuple[str, ...]

    @property
    def has_official_intensity(self) -> bool:
        """Return whether JMA determined an intensity for this station."""
        return self.official_intensity_class.strip() not in {NO_VALUE, ""}

    @property
    def intensity_agrees(self) -> bool:
        """Return whether the reported intensity class matches."""
        official = self.official_intensity_class
        return self.intensity_class == JMA_INTENSITY_NOTATION.get(official, official)

    @property
    def pga_relative_error(self) -> np.ndarray:
        """Return the relative difference in peak acceleration, per component."""
        return np.abs(self.pga - self.official_pga) / self.official_pga

    @property
    def pga_agrees(self) -> bool:
        """Return whether every peak acceleration is within the relative tolerance."""
        return bool(np.all(self.pga_relative_error <= PEAK_RELATIVE_TOLERANCE))

    @property
    def pgv_relative_error(self) -> np.ndarray:
        """Return the relative difference in peak velocity, over the moving components."""
        significant = self.official_pgv >= SIGNIFICANT_PGV_CM_S
        if not np.any(significant):
            return np.empty(0)
        return (
            np.abs(self.pgv - self.official_pgv)[significant]
            / self.official_pgv[significant]
        )

    @property
    def long_period_agrees(self) -> bool:
        """Return whether the overall and all seven band classes match."""
        return (
            self.long_period_class == self.official_long_period_class
            and self.bands == self.official_bands
        )


@dataclass(frozen=True, slots=True)
class SpectrumComparison:
    """Absolute velocity response spectrum, ours beside JMA's."""

    code: str
    name: str
    periods_s: np.ndarray
    spectrum: np.ndarray
    official_spectrum: np.ndarray

    @property
    def relative_error_at_maximum(self) -> float:
        """Return the relative difference at the period carrying the maximum."""
        index = int(np.argmax(self.official_spectrum))
        official = self.official_spectrum[index]
        return float(abs(self.spectrum[index] - official) / official)

    @property
    def max_relative_error(self) -> float:
        """Return the largest relative difference over the official period grid."""
        error = np.abs(self.spectrum - self.official_spectrum) / self.official_spectrum
        return float(np.max(error))


def _fetch(url: str, destination: Path) -> Path:
    """Download one file unless it is already cached."""
    if not destination.exists():
        download_jma_record(url, destination, max_bytes=64 * 1024 * 1024)
    return destination


def _read_rows(path: Path) -> list[list[str]]:
    """Return the data rows of a JMA summary CSV, dropping its header line."""
    text = path.read_bytes().decode(ENCODING)
    return [row for row in list(csv.reader(text.splitlines()))[1:] if row]


def download_event(event_id: str, cache_dir: Path) -> tuple[Path, Path, Path]:
    """Download the summary files and the waveform archive for one event."""
    event_dir = cache_dir / event_id
    maximum = _fetch(f"{BASE_URL}/{event_id}/data/max.csv", event_dir / "data" / "max.csv")
    level = _fetch(f"{BASE_URL}/{event_id}/data/level.csv", event_dir / "data" / "level.csv")
    archive = _fetch(
        f"{BASE_URL}/{event_id}/station/data/acc.zip", event_dir / "station" / "acc.zip"
    )
    waveform_dir = event_dir / "station" / "acc"
    if not waveform_dir.exists():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(waveform_dir)
    return maximum, level, waveform_dir


@dataclass(frozen=True, slots=True)
class OfficialValues:
    """The published values for one station, as JMA reported them."""

    name: str
    intensity_class: str
    pga: np.ndarray
    pgv: np.ndarray
    long_period_class: str = ""
    bands: tuple[str, ...] = ()


def _floats(fields: list[str]) -> np.ndarray | None:
    """Parse a run of numeric fields, or return None if JMA left them blank."""
    if any(field.strip() == "" for field in fields):
        return None
    return np.array([float(field) for field in fields], dtype=np.float64)


def load_official(maximum_path: Path, level_path: Path) -> dict[str, OfficialValues]:
    """Return the published values for every station listed in both summary files."""
    published: dict[str, OfficialValues] = {}
    for row in _read_rows(maximum_path):
        pga = _floats(row[MAX_PGA])
        pgv = _floats(row[MAX_PGV])
        if pga is None or pgv is None:
            continue
        published[row[MAX_CODE]] = OfficialValues(
            name=row[MAX_NAME],
            intensity_class=row[MAX_INTENSITY],
            pga=pga,
            pgv=pgv,
        )
    complete: dict[str, OfficialValues] = {}
    for row in _read_rows(level_path):
        entry = published.get(row[LEVEL_CODE])
        if entry is not None:
            complete[row[LEVEL_CODE]] = replace(
                entry,
                long_period_class=row[LEVEL_CLASS],
                bands=tuple(row[LEVEL_BANDS]),
            )
    return complete


def _waveform_paths(waveform_dir: Path) -> dict[str, Path]:
    """Map each station code to its acceleration file."""
    return {path.name[:5]: path for path in sorted(waveform_dir.rglob("*_acc.csv"))}


def compare_stations(
    published: dict[str, OfficialValues], waveforms: dict[str, Path]
) -> Iterator[StationComparison]:
    """Compute every published quantity for each station and pair it with JMA's."""
    for code, path in waveforms.items():
        entry = published.get(code)
        if entry is None:
            continue
        record = parse_jma_bytes(path.read_bytes(), source=str(path))
        acceleration = record.acceleration
        rate = record.metadata.sampling_rate_hz
        intensity = calculate_measured_intensity(acceleration, rate, unit="gal")
        long_period = calculate_long_period_class(acceleration[:, :2], rate, unit="gal")
        yield StationComparison(
            code=code,
            name=entry.name,
            intensity=intensity.intensity,
            intensity_class=intensity.scale.value,
            official_intensity_class=entry.intensity_class,
            pga=np.append(intensity.input_component_pga_gal, intensity.input_pga_gal),
            official_pga=entry.pga,
            pgv=np.append(
                component_peak_velocity(acceleration, rate, unit="gal"),
                peak_ground_velocity(acceleration, rate, unit="gal"),
            ),
            official_pgv=entry.pgv,
            long_period_class=long_period.long_period_class.value,
            official_long_period_class=entry.long_period_class,
            bands=tuple(band.long_period_class.value for band in long_period.bands),
            official_bands=entry.bands,
        )


def compare_spectrum(
    event_id: str, cache_dir: Path, code: str, name: str, waveform: Path
) -> SpectrumComparison:
    """Compare the computed response spectrum against JMA's published one."""
    stem = waveform.name.removesuffix("_acc.csv")
    path = _fetch(
        f"{BASE_URL}/{event_id}/station/data/velsp/{stem}_velsp.csv",
        cache_dir / event_id / "station" / "velsp" / f"{stem}_velsp.csv",
    )
    text = path.read_bytes().decode(ENCODING).splitlines()
    header_index = next(i for i, line in enumerate(text) if line.startswith("period,"))
    table = list(csv.DictReader(text[header_index:]))
    official = {float(row["period"]): float(row[VELSP_COLUMN]) for row in table}

    record = parse_jma_bytes(waveform.read_bytes(), source=str(waveform))
    result = calculate_long_period_class(
        record.acceleration[:, :2], record.metadata.sampling_rate_hz, unit="gal"
    )
    # JMA tabulates a coarser period grid than the class definition uses, so
    # compare only the periods it actually published.
    periods = result.periods_s
    keep = [index for index, period in enumerate(periods) if round(float(period), 3) in official]
    shared = periods[keep]
    ours = result.sva_cm_s[keep]
    theirs = np.array([official[round(float(period), 3)] for period in shared])
    return SpectrumComparison(code, name, shared, ours, theirs)


def report(comparisons: list[StationComparison], spectra: list[SpectrumComparison]) -> str:
    """Render the comparison as a Markdown report."""
    lines: list[str] = []
    total = len(comparisons)
    rated = [c for c in comparisons if c.has_official_intensity]
    intensity_ok = sum(c.intensity_agrees for c in rated)
    pga_ok = sum(c.pga_agrees for c in comparisons)
    pga_error = np.concatenate([c.pga_relative_error for c in comparisons]) * 100.0
    long_period_ok = sum(c.long_period_agrees for c in comparisons)
    pgv_error = np.concatenate([c.pgv_relative_error for c in comparisons]) * 100.0

    lines.append(f"Stations compared: {total}\n")
    lines.append("| Quantity | Source | Agreement |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Seismic intensity class | max.csv | {intensity_ok}/{len(rated)} "
        f"({total - len(rated)} without a published intensity) |"
    )
    lines.append(
        f"| Peak acceleration, 3 components and resultant | max.csv | "
        f"median {np.median(pga_error):.1e} percent, {pga_ok}/{total} stations within "
        f"{PEAK_RELATIVE_TOLERANCE * 100:.0f} percent |"
    )
    lines.append(
        f"| Long-period class, overall and 7 bands | level.csv | {long_period_ok}/{total} |"
    )
    if spectra:
        worst = max(s.max_relative_error for s in spectra)
        peak = max(s.relative_error_at_maximum for s in spectra)
        lines.append(
            f"| Absolute velocity response spectrum | velsp.csv | "
            f"{len(spectra)} stations, {peak:.1e} at the maximum, {worst:.1e} worst period |"
        )
    lines.append(
        f"| Peak velocity, components at or above {SIGNIFICANT_PGV_CM_S:.0f} cm/s | max.csv | "
        f"median {np.median(pgv_error):.1f} percent, 95th percentile "
        f"{np.percentile(pgv_error, 95):.1f} percent |"
    )

    disagreeing = [
        c for c in rated if not (c.intensity_agrees and c.long_period_agrees)
    ]
    if disagreeing:
        lines.append("\nStations where a class differs:\n")
        lines.append("| Station | Intensity (ours / JMA) | Long-period (ours / JMA) |")
        lines.append("|---|---|---|")
        for c in disagreeing:
            lines.append(
                f"| {c.code} {c.name} | {c.intensity_class} / {c.official_intensity_class} "
                f"| {c.long_period_class} / {c.official_long_period_class} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the comparison and print a Markdown report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event", default=DEFAULT_EVENT_ID, help="JMA event identifier")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/pyshindo/validation"))
    parser.add_argument(
        "--spectra", type=int, default=5, help="how many stations to compare spectra for"
    )
    parser.add_argument("--output", type=Path, help="write the report here as well as to stdout")
    args = parser.parse_args(argv)

    maximum_path, level_path, waveform_dir = download_event(args.event, args.cache_dir)
    published = load_official(maximum_path, level_path)
    waveforms = _waveform_paths(waveform_dir)
    comparisons = list(compare_stations(published, waveforms))
    if not comparisons:
        print(f"No station of event {args.event} had both a waveform and published values.")
        return 1

    strongest = sorted(comparisons, key=lambda c: -c.intensity)[: args.spectra]
    spectra = [
        compare_spectrum(args.event, args.cache_dir, c.code, c.name, waveforms[c.code])
        for c in strongest
    ]

    text = f"Event {args.event}\n\n" + report(comparisons, spectra)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
