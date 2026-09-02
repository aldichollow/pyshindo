# %% Imports
# Plotting requires the optional extra: pip install "pyshindo[plot]"
from pathlib import Path

from pyshindo import calculate_measured_intensity
from pyshindo.io import download_jma_record, read_jma_record
from pyshindo.plotting import acceleration_figure, measured_result_figure

# %% Select the official JMA example record
# The JMA calculation page identifies this Yonago record as measured intensity 5.1.
record_url = (
    "https://ds.data.jma.go.jp/eqev/data/kyoshin/jishin/"
    "001006_tottori-seibu/dat/AA06EA01.csv"
)
cache_path = Path(".cache/pyshindo/AA06EA01.csv")

# %% Download once and preserve the source URL in metadata
if not cache_path.exists():
    download_jma_record(record_url, cache_path)
record = read_jma_record(cache_path, source=record_url)

print(f"sampling rate: {record.metadata.sampling_rate_hz} Hz, unit: {record.metadata.unit}")
print(f"components: {record.metadata.component_names}, shape: {record.acceleration.shape}")

# %% Recalculate the published example
result = calculate_measured_intensity(
    record.acceleration,
    record.metadata.sampling_rate_hz,
    unit=record.metadata.unit,
)

print(f"Calculated raw intensity:      {result.intensity_raw:.6f}")
print(f"Calculated reported intensity: {result.intensity:.1f}")
print("Published reported intensity:  5.1")
assert result.intensity == 5.1

# %% Interactive inspection
input_figure = acceleration_figure(
    record.acceleration,
    record.metadata.sampling_rate_hz,
    component_names=record.metadata.component_names,
    title="JMA strong-motion record: 2000 Tottori-ken Seibu (Yonago)",
)
input_figure.show()

result_figure = measured_result_figure(
    result,
    component_names=record.metadata.component_names,
)
result_figure.show()
