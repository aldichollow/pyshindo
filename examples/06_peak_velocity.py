# %% Imports
# Plotting requires the optional extra: pip install "pyshindo[plot]"
import numpy as np

from pyshindo import (
    component_peak_velocity,
    detrend_acceleration,
    integrate_to_velocity,
    peak_ground_acceleration,
    peak_ground_velocity,
    remove_offset,
    synthetic_three_component_motion,
    time_axis,
)
from pyshindo.long_period import apply_ground_motion_high_pass
from pyshindo.plotting import acceleration_figure

# %% Generate a record and add a small instrument offset
sampling_rate_hz = 100.0
acceleration_gal = synthetic_three_component_motion(
    sampling_rate_hz=sampling_rate_hz,
    duration_s=30.0,
)
# A real accelerometer rarely sits at exactly zero. A 0.5 gal offset is small
# next to the peak amplitude but it is what dominates the integrated velocity.
offset_gal = 0.5
drifting_gal = acceleration_gal + offset_gal
time_s = time_axis(drifting_gal.shape[0], sampling_rate_hz)

# %% PGA is barely affected by the offset; PGV is not
print(f"PGA without offset: {peak_ground_acceleration(acceleration_gal):.3f} gal")
print(f"PGA with offset:    {peak_ground_acceleration(drifting_gal):.3f} gal")
print(f"PGV without offset: {peak_ground_velocity(acceleration_gal, sampling_rate_hz):.3f} cm/s")
print(f"PGV with offset:    {peak_ground_velocity(drifting_gal, sampling_rate_hz):.3f} cm/s")

# %% Integration turns a constant offset into a linear ramp
# This is arithmetic working correctly, not a defect: integration cannot tell a
# baseline error apart from real long-period motion, so the correction has to
# be an explicit choice by the caller.
raw_velocity = integrate_to_velocity(drifting_gal, sampling_rate_hz)
corrected_velocity = integrate_to_velocity(
    remove_offset(drifting_gal),
    sampling_rate_hz,
)
final_drift = offset_gal * time_s[-1]
print(f"Expected drift after {time_s[-1]:.0f} s: {final_drift:.3f} cm/s")
print(f"Observed final velocity:      {np.max(np.abs(raw_velocity[-1])):.3f} cm/s")

# %% Compare baseline treatments
# apply_ground_motion_high_pass is the 20-second high-pass the long-period
# ground motion class uses, generalized to any component count. It is a
# specific published filter, not a general-purpose recommendation -- but it
# also happens to reproduce JMA's own published peak velocity for a
# long-period observation record to about 0.01 percent, where the other
# treatments here do not get closer than a percent or two. See
# docs/validation.md for that finding, verified against 268 real stations.
for label, prepared in (
    ("raw (no correction)", drifting_gal),
    ("remove_offset", remove_offset(drifting_gal)),
    ("detrend_acceleration", detrend_acceleration(drifting_gal)),
    (
        "apply_ground_motion_high_pass",
        apply_ground_motion_high_pass(drifting_gal, sampling_rate_hz),
    ),
):
    pgv = peak_ground_velocity(prepared, sampling_rate_hz)
    print(f"{label:29s} PGV = {pgv:7.3f} cm/s")

# %% Per-component peaks, in the same shape as component_peak_acceleration
peaks = component_peak_velocity(remove_offset(drifting_gal), sampling_rate_hz)
for name, value in zip(("NS", "EW", "UD"), peaks, strict=True):
    print(f"Peak {name} velocity: {value:.3f} cm/s")

# %% Inspect the uncorrected and corrected velocity traces
acceleration_figure(
    raw_velocity,
    sampling_rate_hz,
    time_s=time_s,
    title="Velocity from uncorrected acceleration (drifts)",
    unit_label="cm/s",
).show()

acceleration_figure(
    corrected_velocity,
    sampling_rate_hz,
    time_s=time_s,
    title="Velocity after remove_offset",
    unit_label="cm/s",
).show()

# %%
