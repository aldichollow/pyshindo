# %% Imports
# Plotting requires the optional extra: pip install "pyshindo[plot]"
import numpy as np

from pyshindo import (
    component_peak_displacement,
    component_peak_velocity,
    detrend_acceleration,
    integrate_to_displacement,
    integrate_to_velocity,
    peak_ground_acceleration,
    peak_ground_displacement,
    peak_ground_velocity,
    remove_offset,
    synthetic_three_component_motion,
    time_axis,
)
from pyshindo.long_period import apply_ground_motion_high_pass
from pyshindo.plotting import acceleration_figure
from pyshindo.signal import vector_resultant
from pyshindo.strong_motion import apply_strong_motion_displacement_filter

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

# %% PGA is barely affected by the offset; PGV is more sensitive, and PGD more so still
print(f"PGA without offset: {peak_ground_acceleration(acceleration_gal):.3f} gal")
print(f"PGA with offset:    {peak_ground_acceleration(drifting_gal):.3f} gal")
print(f"PGV without offset: {peak_ground_velocity(acceleration_gal, sampling_rate_hz):.3f} cm/s")
print(f"PGV with offset:    {peak_ground_velocity(drifting_gal, sampling_rate_hz):.3f} cm/s")
print(
    f"PGD without offset: {peak_ground_displacement(acceleration_gal, sampling_rate_hz):.3f} cm"
)
print(f"PGD with offset:    {peak_ground_displacement(drifting_gal, sampling_rate_hz):.3f} cm")

# %% Integration turns a constant offset into a linear velocity ramp, and --
# integrating that ramp again -- a quadratic displacement drift. This is
# arithmetic working correctly, not a defect: integration cannot tell a
# baseline error apart from real long-period motion, so the correction has to
# be an explicit choice by the caller. Displacement compounds the effect: a
# baseline this small is barely visible in PGV but dominates PGD.
raw_velocity = integrate_to_velocity(drifting_gal, sampling_rate_hz)
raw_displacement = integrate_to_displacement(drifting_gal, sampling_rate_hz)
corrected_velocity = integrate_to_velocity(
    remove_offset(drifting_gal),
    sampling_rate_hz,
)
corrected_displacement = integrate_to_displacement(
    remove_offset(drifting_gal),
    sampling_rate_hz,
)
expected_velocity_drift = offset_gal * time_s[-1]
expected_displacement_drift = 0.5 * offset_gal * time_s[-1] ** 2
print(f"\nExpected velocity drift after {time_s[-1]:.0f} s:     {expected_velocity_drift:.3f} cm/s")
print(f"Observed final velocity:                {np.max(np.abs(raw_velocity[-1])):.3f} cm/s")
print(f"Expected displacement drift after {time_s[-1]:.0f} s: {expected_displacement_drift:.3f} cm")
print(f"Observed final displacement:            {np.max(np.abs(raw_displacement[-1])):.3f} cm")

# %% Compare baseline treatments for the generic double integration
# apply_ground_motion_high_pass is the 20-second high-pass the long-period
# ground motion class uses, generalized to any component count. It is a
# specific published filter, not a general-purpose recommendation -- but it
# also happens to reproduce JMA's own published peak velocity for a
# long-period observation record to about 0.01 percent, where the other
# treatments here do not get closer than a percent or two (see
# docs/validation.md). None of these treatments get PGD nearly that close,
# because double integration is not how JMA derives displacement at all --
# see the next cell.
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
    pgd = peak_ground_displacement(prepared, sampling_rate_hz)
    print(f"{label:29s} PGV = {pgv:7.3f} cm/s   PGD = {pgd:7.3f} cm")

# %% JMA's own published displacement is not a double integration at all: it
# is a filter reproducing the amplitude response of its mechanical 1x
# strong-motion seismometer (natural period 6 s, damping 0.55), applied
# directly to acceleration. That is a bounded resonant response, not an
# accumulator, so it settles to a finite value under a sustained offset
# instead of drifting without bound -- no remove_offset/detrend_acceleration
# is needed first. Against a real 268-station corpus this reproduces JMA's
# published peak displacement to a median relative error of about 0.15
# percent, roughly two orders of magnitude closer than any treatment above
# (see docs/validation.md).
jma_displacement = apply_strong_motion_displacement_filter(drifting_gal, sampling_rate_hz)
pgd_jma_filter = float(np.max(vector_resultant(jma_displacement)))
print(f"\nPGD via apply_strong_motion_displacement_filter: {pgd_jma_filter:7.3f} cm")

# %% Per-component peaks, in the same shape as component_peak_acceleration
peaks = component_peak_velocity(remove_offset(drifting_gal), sampling_rate_hz)
for name, value in zip(("NS", "EW", "UD"), peaks, strict=True):
    print(f"Peak {name} velocity: {value:.3f} cm/s")

displacement_peaks = component_peak_displacement(remove_offset(drifting_gal), sampling_rate_hz)
for name, value in zip(("NS", "EW", "UD"), displacement_peaks, strict=True):
    print(f"Peak {name} displacement: {value:.3f} cm")

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

# %% Displacement drifts the same way, only faster -- compare the vertical
# scale here against the velocity traces above for the same 0.5 gal offset
acceleration_figure(
    raw_displacement,
    sampling_rate_hz,
    time_s=time_s,
    title="Displacement from uncorrected acceleration (drifts quadratically)",
    unit_label="cm",
).show()

acceleration_figure(
    corrected_displacement,
    sampling_rate_hz,
    time_s=time_s,
    title="Displacement after remove_offset",
    unit_label="cm",
).show()

# %%
