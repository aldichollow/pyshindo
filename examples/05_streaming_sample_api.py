# %% Imports
import numpy as np

from pyshindo import RealtimeIntensityEstimator, synthetic_three_component_motion

# %% Build a short stream
sampling_rate_hz = 100.0
acceleration_gal = synthetic_three_component_motion(
    sampling_rate_hz=sampling_rate_hz,
    duration_s=5.0,
    center_s=2.5,
    width_s=0.7,
)

# %% Process one sensor sample at a time
estimator = RealtimeIntensityEstimator(sampling_rate_hz, unit="gal")
latest = None

for sample in acceleration_gal:
    latest = estimator.process_sample(sample)

if latest is None:
    raise RuntimeError("The demonstration stream was empty.")

print(f"Last sample index:       {latest.sample_index}")
print(f"Last time:               {latest.time_s:.3f} s")
print(f"Last raw intensity:      {latest.intensity_raw:.6f}")
print(f"Maximum raw intensity:   {estimator.approximate_intensity_raw:.6f}")
print(f"Last filtered resultant: {latest.resultant_acceleration_gal:.6f} gal")
assert np.isfinite(estimator.approximate_intensity_raw)

# %%
