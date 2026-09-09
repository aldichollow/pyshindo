# %% Imports
import numpy as np

from pyshindo import (
    calculate_spectrum_intensity,
    peak_ground_velocity,
    synthetic_three_component_motion,
)
from pyshindo.spectrum_intensity import default_periods_s

# %% SI is per component, not combined -- unlike peak_ground_velocity's
# resultant, or JMA's long-period class, which vector-combines two horizontals
sampling_rate_hz = 100.0
acceleration_gal = synthetic_three_component_motion(
    sampling_rate_hz=sampling_rate_hz,
    duration_s=30.0,
)
result = calculate_spectrum_intensity(acceleration_gal, sampling_rate_hz, unit="gal")

for name, si in zip(("NS", "EW", "UD"), result.si_cm_s, strict=True):
    print(f"SI {name}: {si:6.3f} cm/s")

# %% SI correlates with PGV -- both are velocity-scale ground-motion measures
# -- but is not the same number: SI weights the 0.1-2.5 s band that ordinary
# buildings resonate in, rather than the single instant of peak velocity.
pgv = peak_ground_velocity(acceleration_gal, sampling_rate_hz, unit="gal")
print(f"\nPGV (3-component resultant): {pgv:.3f} cm/s")
print("SI values above, for comparison, are not combined the same way")

# %% The published damping ratio is 0.20, specific to SI -- not the 5 percent
# used for JMA's long-period class, and not a general structural value
lightly_damped = calculate_spectrum_intensity(
    acceleration_gal, sampling_rate_hz, unit="gal", damping_ratio=0.05
)
print(f"\nSI (NS), h=0.20: {result.si_cm_s[0]:.3f} cm/s")
print(f"SI (NS), h=0.05: {lightly_damped.si_cm_s[0]:.3f} cm/s (lighter damping, larger response)")

# %% Only horizontal components, matching the conventional use of SI
# (road-bridge design, gas and elevator seismic shutoff)
horizontal = calculate_spectrum_intensity(
    np.ascontiguousarray(acceleration_gal[:, :2]), sampling_rate_hz, unit="gal"
)
print(f"\nHorizontal-only SI (NS, EW): {horizontal.si_cm_s}")

# %% retain_spectrum=True keeps the full response spectrum, Sv(T), not just
# the integrated SI value -- the same curve the integral averages over 2.4 s
detailed = calculate_spectrum_intensity(
    np.ascontiguousarray(acceleration_gal[:, :1]),
    sampling_rate_hz,
    unit="gal",
    retain_spectrum=True,
)
peak_index = int(np.argmax(detailed.sv_cm_s[:, 0]))
peak_period_s = detailed.periods_s[peak_index]
print(f"\nPeak Sv: {detailed.sv_cm_s[peak_index, 0]:.3f} cm/s at T={peak_period_s:.2f} s")
print(f"Integrated over 0.1-2.5 s, that curve averages to SI = {detailed.si_cm_s[0]:.3f} cm/s")

# %% The 0.1-2.5 s integration range has no published discretization; the
# default 121-point linear grid was chosen by checking convergence directly
coarse = calculate_spectrum_intensity(
    acceleration_gal[:, :1], sampling_rate_hz, unit="gal", periods_s=default_periods_s(25)
)
fine = calculate_spectrum_intensity(
    acceleration_gal[:, :1], sampling_rate_hz, unit="gal", periods_s=default_periods_s(769)
)
default_grid = calculate_spectrum_intensity(acceleration_gal[:, :1], sampling_rate_hz, unit="gal")
print(f"\nSI (NS) at 25 periods:  {coarse.si_cm_s[0]:.6f} cm/s")
print(f"SI (NS) at 121 periods: {default_grid.si_cm_s[0]:.6f} cm/s (the default)")
print(f"SI (NS) at 769 periods: {fine.si_cm_s[0]:.6f} cm/s")

# %%
