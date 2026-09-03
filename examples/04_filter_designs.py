# %% Imports
# Plotting requires the optional extra: pip install "pyshindo[plot]"
import numpy as np

from pyshindo import (
    RealtimeFilter,
    design_realtime_filter,
    jma_filter_response,
    kunugi_2012_analog_amplitude,
    lowrate_gamma_stability_margins,
    published_lowrate_gamma_set,
    realtime_filter_response,
)
from pyshindo.plotting import filter_response_figure, filter_stages_figure

# %% Inspect the standard 100-Hz improved design
standard_design = design_realtime_filter(
    100.0,
    filter_name=RealtimeFilter.KUNUGI_2012,
)
print(standard_design.name)
print(standard_design.sos)
print(f"Maximum pole radius: {standard_design.max_pole_radius:.12f}")

# %% Inspect the named analog factors that make up each of the three designs
for design in (
    design_realtime_filter(100.0, filter_name=RealtimeFilter.KUNUGI_2008),
    standard_design,
    design_realtime_filter(50.0, filter_name=RealtimeFilter.JP7681907_LOWRATE),
):
    print(f"--- {design.name} @ {design.sampling_rate_hz:g} Hz ---")
    for stage in design.stages:
        print(f"  {stage.name:20s} f={stage.characteristic_frequency_hz}")
    filter_stages_figure(design).show()

# %% Reproduce the analog amplitude-ratio range
frequency_hz = np.geomspace(0.1, 50.0, 10_000)
ratio = kunugi_2012_analog_amplitude(frequency_hz) / jma_filter_response(frequency_hz)
print(f"Analog ratio minimum: {ratio.min():.6f}")
print(f"Analog ratio maximum: {ratio.max():.6f}")

# %% Compare digital responses
figure_100_hz = filter_response_figure(
    100.0,
    filter_names=(
        RealtimeFilter.KUNUGI_2008,
        RealtimeFilter.KUNUGI_2012,
    ),
)
figure_100_hz.show()

# %% Inspect a generalized 50-Hz design
lowrate_design = design_realtime_filter(
    50.0,
    filter_name=RealtimeFilter.JP7681907_LOWRATE,
)
gamma_set = published_lowrate_gamma_set(50.0)
margins = lowrate_gamma_stability_margins(50.0, gamma_set)
print(lowrate_design.name)
print(gamma_set)
print(margins)
print(f"Maximum pole radius: {lowrate_design.max_pole_radius:.12f}")

frequency_50_hz = np.geomspace(0.1, 24.9, 2000)
response_50_hz = realtime_filter_response(lowrate_design, frequency_50_hz)
print(f"Digital gain at 1 Hz: {np.interp(1.0, frequency_50_hz, response_50_hz.amplitude):.6f}")

figure_50_hz = filter_response_figure(
    50.0,
    filter_names=(RealtimeFilter.AUTO,),
)
figure_50_hz.show()

# %%
