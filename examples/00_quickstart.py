# %% Imports
import numpy as np

from pyshindo import (
    RealtimeFilter,
    calculate_measured_intensity,
    calculate_realtime_intensity,
    peak_ground_velocity,
    remove_offset,
    scale_acceleration_to_intensity,
    synthetic_three_component_motion,
)
from pyshindo.long_period import calculate_long_period_class

# %% Create a deterministic demonstration record
sampling_rate_hz = 100.0
acceleration_gal = synthetic_three_component_motion(
    sampling_rate_hz=sampling_rate_hz,
    duration_s=30.0,
)
acceleration_gal, scale_factor = scale_acceleration_to_intensity(
    acceleration_gal,
    target_intensity_raw=4.8,
    sampling_rate_hz=sampling_rate_hz,
)

# %% Complete-record FFT reference
measured = calculate_measured_intensity(
    acceleration_gal,
    sampling_rate_hz,
    unit="gal",
)

print(f"Amplitude scale factor: {scale_factor:.6f}")
print(f"FFT raw intensity:       {measured.intensity_raw:.6f}")
print(f"FFT reported intensity:  {measured.intensity:.1f}")
print(f"FFT class:               {measured.scale.japanese}")
print(f"Threshold:               {measured.threshold_acceleration_gal:.6f} gal")

# %% Real-time replay
realtime = calculate_realtime_intensity(
    acceleration_gal,
    sampling_rate_hz,
    unit="gal",
    filter_name=RealtimeFilter.AUTO,
)

print(f"Real-time filter:         {realtime.filter_name}")
print(f"Real-time raw maximum:    {realtime.approximate_intensity_raw:.6f}")
print(f"Real-time reported max:   {realtime.approximate_intensity:.1f}")
raw_difference = measured.intensity_raw - realtime.approximate_intensity_raw
print(f"Raw method difference:    {raw_difference:+.6f}")

# %% Peak ground velocity
# Integration turns any baseline offset into a drift, so the correction is an
# explicit choice rather than something applied silently. See examples/06.
pgv = peak_ground_velocity(remove_offset(acceleration_gal), sampling_rate_hz, unit="gal")
print(f"PGV:                      {pgv:.3f} cm/s")

# %% Long-period ground motion class
# A different JMA quantity, defined on the two horizontal components only.
long_period = calculate_long_period_class(
    np.ascontiguousarray(acceleration_gal[:, :2]),
    sampling_rate_hz,
    unit="gal",
)
print(f"Long-period class:        {long_period.long_period_class}")
print(f"Maximum Sva:              {long_period.max_sva_cm_s:.3f} cm/s")
print(f"at period:                {long_period.critical_period_s:.1f} s")

# %%
