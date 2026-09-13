# %% Imports
import numpy as np

from pyshindo import (
    calculate_response_spectrum,
    calculate_spectrum_intensity,
    synthetic_three_component_motion,
)
from pyshindo.long_period import (
    OFFICIAL_DAMPING_RATIO,
    OFFICIAL_PERIODS_S,
    apply_ground_motion_high_pass,
    calculate_long_period_class,
)
from pyshindo.spectrum_intensity import DEFAULT_DAMPING_RATIO, default_periods_s
from pyshindo.velocity import integrate_to_velocity

# %% calculate_response_spectrum is the general elastic response -- Sd, Sv,
# and pseudo-acceleration -- with neither a damping ratio nor a period grid
# defaulted for you: unlike pyshindo.long_period (5%, the official 32-point
# grid) and pyshindo.spectrum_intensity (20%, 0.1-2.5 s), a general-purpose
# function has no house convention of its own to fall back to.
sampling_rate_hz = 100.0
acceleration_gal = synthetic_three_component_motion(
    sampling_rate_hz=sampling_rate_hz, duration_s=30.0
)
periods_s = np.geomspace(0.05, 5.0, 60)

result = calculate_response_spectrum(
    acceleration_gal, sampling_rate_hz, damping_ratio=0.05, periods_s=periods_s
)
print(f"Sd shape: {result.sd_cm.shape}  (periods, components)")
print(f"Sv shape: {result.sv_cm_s.shape}")
print(f"PSA shape: {result.psa_gal.shape}")
print(f"PSA (NS) at T={periods_s[0]:.2f}s: {result.psa_gal[0, 0]:.3f} gal")

# %% PSA = omega^2 * Sd exactly -- it is derived from Sd, not independently
# computed, so this holds to machine precision regardless of the record
omega = 2.0 * np.pi / periods_s
np.testing.assert_allclose(result.psa_gal, (omega**2)[:, np.newaxis] * result.sd_cm)

# %% This is the same shared solver pyshindo.long_period and
# pyshindo.spectrum_intensity already use -- reusing their own damping ratio
# and period grid here reproduces their Sv exactly (both are relative
# response, uncombined, so this is a direct comparison, not merely a similar
# number)
si_result = calculate_spectrum_intensity(acceleration_gal, sampling_rate_hz)
si_shaped = calculate_response_spectrum(
    acceleration_gal,
    sampling_rate_hz,
    damping_ratio=DEFAULT_DAMPING_RATIO,
    periods_s=default_periods_s(),
)
np.testing.assert_array_equal(si_shaped.sv_cm_s, si_result.sv_cm_s)
print("\nSv from calculate_response_spectrum matches calculate_spectrum_intensity exactly")

# %% This function returns the *relative* response only -- no ground motion
# added, no component combination. pyshindo.long_period adds ground velocity
# and combines two horizontal components on top of the same relative
# response; reproducing that here means retaining the full time series and
# doing both steps yourself, since the peak of a sum is not the sum of peaks.
# Both steps use the same 20-second-high-pass-filtered acceleration -- that
# filter is part of JMA's definition, not something this general function
# applies on its own.
horizontal_gal = np.ascontiguousarray(acceleration_gal[:, :2])
filtered = apply_ground_motion_high_pass(horizontal_gal, sampling_rate_hz)
detailed = calculate_response_spectrum(
    filtered,
    sampling_rate_hz,
    damping_ratio=OFFICIAL_DAMPING_RATIO,
    periods_s=OFFICIAL_PERIODS_S,
    retain_velocity_time_series=True,  # only Sv is needed here, not Sd
)
ground_velocity = integrate_to_velocity(filtered, sampling_rate_hz)
absolute_velocity = detailed.sv_time_series_cm_s + ground_velocity[:, np.newaxis, :]
horizontal_composite = np.hypot(absolute_velocity[..., 0], absolute_velocity[..., 1])
sva_reconstructed = horizontal_composite.max(axis=0)

reference = calculate_long_period_class(horizontal_gal, sampling_rate_hz)
np.testing.assert_allclose(sva_reconstructed, reference.sva_cm_s, rtol=1e-9)
print("Reconstructed absolute velocity response matches pyshindo.long_period's own Sva")

# %%
