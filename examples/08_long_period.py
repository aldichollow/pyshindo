# %% Imports
# Plotting requires the optional extra: pip install "pyshindo[plot]"
import csv
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from pyshindo import synthetic_three_component_motion
from pyshindo.io import parse_jma_text
from pyshindo.long_period import (
    OFFICIAL_PERIODS_S,
    LongPeriodEstimator,
    calculate_long_period_class,
)
from pyshindo.plotting import long_period_spectrum_figure

# %% A synthetic record, scaled up until it reaches a reportable class
sampling_rate_hz = 100.0
acceleration_gal = (
    synthetic_three_component_motion(sampling_rate_hz=sampling_rate_hz, duration_s=60.0) * 12.0
)
# The class is defined on the two horizontal components only. The vertical is
# not missing here, it is simply not part of this quantity.
horizontal_gal = np.ascontiguousarray(acceleration_gal[:, :2])

result = calculate_long_period_class(horizontal_gal, sampling_rate_hz, unit="gal")

print(f"Long-period ground motion class: {result.long_period_class}")
print(f"Maximum Sva:                     {result.max_sva_cm_s:.4f} cm/s")
print(f"Critical period:                 {result.critical_period_s:.1f} s")
print(f"Reference conditions met:        {result.reference_conditions_met}")

# %% The per-band classes JMA reports alongside the overall one
for band in result.bands:
    low, high = band.period_range_s
    print(
        f"  {band.japanese_label:5s} ({low:.1f}-{high:.1f} s): "
        f"{band.max_sva_cm_s:8.4f} cm/s  ->  class {band.long_period_class}"
    )

# %% The spectrum, drawn against the class thresholds
long_period_spectrum_figure(result).show()

# %% Streaming: the class can only grow as more of the record arrives
# JMA's class is the maximum over the whole record, so a running estimate is a
# cumulative maximum, not a rolling window.
estimator = LongPeriodEstimator(sampling_rate_hz, unit="gal")
for start in range(0, horizontal_gal.shape[0], 1000):
    update = estimator.process(horizontal_gal[start : start + 1000])
    print(
        f"  t={update.sample_count / sampling_rate_hz:6.1f} s  "
        f"Sva so far {update.max_sva_so_far_cm_s:8.4f} cm/s  "
        f"class {update.class_so_far}"
    )
# The estimator steps the published recurrence, which is also what the batch
# calculation does under solver="recurrence" -- same arithmetic, not merely
# close. The default batch solver runs the same equation as an IIR filter per
# period, which is far faster and agrees to floating-point rounding.
np.testing.assert_array_equal(
    estimator.sva_cm_s,
    calculate_long_period_class(
        horizontal_gal, sampling_rate_hz, unit="gal", solver="recurrence"
    ).sva_cm_s,
)
np.testing.assert_allclose(estimator.sva_cm_s, result.sva_cm_s, rtol=1e-10)
print("streaming matches the reference solver exactly, and the fast solver to 1e-10")

# %% Verify against an officially published JMA record
# JMA publishes, for each event, the acceleration waveforms and the absolute
# velocity response spectra it computed from them. The "0.050_hz" column of
# the velsp file is exactly what this package computes: 5 percent damping,
# horizontal vector composite, one value per period.
#
# Records are downloaded rather than bundled: they are JMA's to distribute,
# and the terms of use should be read at the source.
EVENT_ID = "20260823020050"  # 2026-08-23 02:00 茨城県南部 M5.9, class 2
STATION = "47052"  # 浦安市日の出
OBSERVED_AT = "20260823020040"
BASE = f"https://www.data.jma.go.jp/eew/data/ltpgm/{EVENT_ID}/station/data"
cache = Path(".cache/pyshindo")
cache.mkdir(parents=True, exist_ok=True)


def fetch(url: str, path: Path) -> bytes:
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            path.write_bytes(response.read())
    return path.read_bytes()


# The waveforms come as one zip holding every station in the event.
archive = fetch(f"{BASE}/acc.zip", cache / f"{EVENT_ID}_acc.zip")
with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
    name = next(n for n in bundle.namelist() if n.endswith(f"{STATION}{OBSERVED_AT}_acc.csv"))
    record = parse_jma_text(bundle.read(name).decode("cp932"))

official_bytes = fetch(
    f"{BASE}/velsp/{STATION}{OBSERVED_AT}_velsp.csv",
    cache / f"{EVENT_ID}_{STATION}_velsp.csv",
)
rows = list(csv.reader(io.StringIO(official_bytes.decode("cp932"))))
header = next(index for index, row in enumerate(rows) if row and row[0].strip() == "period")
column = [name.strip() for name in rows[header]].index("0.050_hz")
published = {round(float(row[0]), 1): float(row[column]) for row in rows[header + 1 :] if row}
official_sva = np.array([published[round(float(p), 1)] for p in OFFICIAL_PERIODS_S])

# %% Compare
verified = calculate_long_period_class(
    record.acceleration[:, :2],
    record.metadata.sampling_rate_hz,
    unit=record.metadata.unit,
)
relative = np.abs(verified.sva_cm_s - official_sva) / official_sva

print(f"\nStation {STATION}, {record.acceleration.shape[0]} samples")
print(f"  JMA published maximum Sva: {official_sva.max():.4f} cm/s")
print(f"  pyshindo maximum Sva:      {verified.sva_cm_s.max():.6f} cm/s")
print(f"  relative difference at the maximum: {relative[np.argmax(official_sva)]:.2e}")
print(f"  largest relative difference over all 32 periods: {relative.max():.2e}")
print(f"  class: JMA reports 2, pyshindo gives {verified.long_period_class}")

# The remaining disagreement is consistent with the four-decimal rounding of
# the published spectrum and the three-decimal rounding of the published
# waveform, not with a difference in method.

# %%
