# %% Imports
# Plotting requires the optional extra: pip install "pyshindo[plot]"
import numpy as np

from pyshindo import (
    amplitude_duration_curve,
    calculate_measured_intensity,
    scale_acceleration_to_intensity,
    synthetic_three_component_motion,
    time_axis,
)
from pyshindo.plotting import (
    acceleration_figure,
    amplitude_duration_figure,
    jma_filter_components_figure,
    measured_result_figure,
)

# %% Generate a reproducible three-component motion
sampling_rate_hz = 100.0
acceleration_gal = synthetic_three_component_motion(
    sampling_rate_hz=sampling_rate_hz,
    duration_s=30.0,
    center_s=13.0,
    width_s=2.4,
    amplitudes_gal=(90.0, 62.0, 34.0),
    frequencies_hz=(1.1, 2.4, 5.8),
)
acceleration_gal, _ = scale_acceleration_to_intensity(
    acceleration_gal,
    target_intensity_raw=5.15,
    sampling_rate_hz=sampling_rate_hz,
)
time_s = time_axis(acceleration_gal.shape[0], sampling_rate_hz)

# %% Inspect the original acceleration
input_figure = acceleration_figure(
    acceleration_gal,
    sampling_rate_hz,
    time_s=time_s,
    title="Synthetic three-component acceleration",
)
input_figure.show()

# %% Inspect the three published frequency-response factors
response_figure = jma_filter_components_figure(
    minimum_hz=0.01,
    maximum_hz=50.0,
)
response_figure.show()

# %% Run the complete-record FFT calculation
result = calculate_measured_intensity(
    acceleration_gal,
    sampling_rate_hz,
    unit="gal",
)

print(f"Raw intensity:      {result.intensity_raw:.6f}")
print(f"Reported intensity: {result.intensity:.1f}")
print(f"Intensity class:    {result.scale.japanese}")
print(f"Threshold:          {result.threshold_acceleration_gal:.6f} gal")
print(f"Input vector PGA:   {result.input_pga_gal:.6f} gal")
print(f"Filtered PGA:       {result.filtered_pga_gal:.6f} gal")

# %% Plot the filtered components, resultant, and threshold
result_figure = measured_result_figure(result, time_s=time_s)
result_figure.show()

# %% Inspect the amplitude-duration order statistic
if result.resultant_acceleration_gal is None:
    raise RuntimeError("Intermediates were not retained.")
curve = amplitude_duration_curve(result.resultant_acceleration_gal, sampling_rate_hz)
threshold_index = result.duration_samples - 1
assert np.isclose(curve.amplitude[threshold_index], result.threshold_acceleration_gal)

duration_figure = amplitude_duration_figure(
    result.resultant_acceleration_gal,
    sampling_rate_hz,
)
duration_figure.show()
