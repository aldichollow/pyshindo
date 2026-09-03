# %% Imports
# Plotting requires the optional extra: pip install "pyshindo[plot]"
import numpy as np

from pyshindo import (
    RealtimeFilter,
    RealtimeIntensityEstimator,
    calculate_realtime_intensity,
    compare_intensity_methods,
    scale_acceleration_to_intensity,
    synthetic_three_component_motion,
)
from pyshindo.plotting import intensity_comparison_figure, realtime_result_figure

# %% Prepare a 90-second stream with pre-event and post-event data
sampling_rate_hz = 100.0
acceleration_gal = synthetic_three_component_motion(
    sampling_rate_hz=sampling_rate_hz,
    duration_s=90.0,
    center_s=24.0,
    width_s=3.2,
    amplitudes_gal=(80.0, 55.0, 30.0),
    frequencies_hz=(0.9, 2.0, 4.5),
)
acceleration_gal, _ = scale_acceleration_to_intensity(
    acceleration_gal,
    target_intensity_raw=4.95,
    sampling_rate_hz=sampling_rate_hz,
)

# %% Compare complete-record and real-time results
comparison = compare_intensity_methods(
    acceleration_gal,
    sampling_rate_hz,
    unit="gal",
    realtime_options={"filter_name": RealtimeFilter.AUTO},
)

print(f"FFT raw intensity:       {comparison.measured.intensity_raw:.6f}")
print(f"Real-time raw maximum:   {comparison.realtime.approximate_intensity_raw:.6f}")
print(f"Raw difference:          {comparison.raw_difference:+.6f}")
print(f"Reported-class agreement: {comparison.scale_agreement}")

comparison_figure = intensity_comparison_figure(comparison)
comparison_figure.show()

# %% Inspect the complete real-time trace
realtime = calculate_realtime_intensity(
    acceleration_gal,
    sampling_rate_hz,
    unit="gal",
)
trace_figure = realtime_result_figure(realtime)
trace_figure.show()

# %% Re-run the same stream using irregular packet sizes
estimator = RealtimeIntensityEstimator(
    sampling_rate_hz,
    unit="gal",
)
rng = np.random.default_rng(42)
position = 0
streamed_raw: list[np.ndarray] = []

while position < acceleration_gal.shape[0]:
    packet_size = int(rng.integers(1, 81))
    packet = acceleration_gal[position : position + packet_size]
    output = estimator.process(packet)
    streamed_raw.append(output.intensity_raw)
    position += packet.shape[0]

streamed_raw_array = np.concatenate(streamed_raw)
np.testing.assert_allclose(streamed_raw_array, realtime.intensity_raw, equal_nan=True)
print("Irregular chunk replay matches complete-chunk replay.")

# %%
