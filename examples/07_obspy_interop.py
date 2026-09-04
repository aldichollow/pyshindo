# %% Imports
# Requires the optional extra: pip install "pyshindo[obspy]"
import obspy

from pyshindo import (
    calculate_measured_intensity,
    peak_ground_velocity,
    remove_offset,
    synthetic_three_component_motion,
)
from pyshindo.obspy_interop import from_obspy_stream

# %% Build a stream that stands in for a file ObsPy would read
# In real use this is `obspy.read("...")` for K-NET, KiK-net, miniSEED, SAC, and
# the rest of ObsPy's readers. A stream is assembled here instead so the example
# runs offline and the data are unambiguously acceleration in gal -- ObsPy's own
# bundled example record is seismometer velocity, which would be the wrong
# physical quantity to feed an intensity calculation.
sampling_rate_hz = 100.0
acceleration_gal = synthetic_three_component_motion(
    sampling_rate_hz=sampling_rate_hz,
    duration_s=30.0,
)
start_time = obspy.UTCDateTime(2026, 9, 4, 0, 0, 0)
stream = obspy.Stream(
    traces=[
        obspy.Trace(
            data=acceleration_gal[:, index],
            header={
                "network": "XX",
                "station": "DEMO",
                "location": "",
                "channel": channel,
                "sampling_rate": sampling_rate_hz,
                "starttime": start_time,
            },
        )
        # Deliberately not in NS/EW/UD order: the converter sorts the columns
        # using the SEED orientation code, the last character of each channel.
        for index, channel in ((2, "HNZ"), (0, "HNN"), (1, "HNE"))
    ]
)
print(stream)

# %% Convert to the arrays pyshindo expects
# `unit` is required, not detected: SEED and the formats around it carry no
# dependable physical-unit field, so ObsPy returns whatever the reader produced.
# Stating it records the assertion rather than guessing at it.
record = from_obspy_stream(stream, unit="gal")

print(f"components:    {record.metadata.component_names}")
print(f"channels:      {record.metadata.channel_codes}")
print(f"station:       {record.metadata.network}.{record.metadata.station}")
print(f"sampling rate: {record.metadata.sampling_rate_hz} Hz")
print(f"shape:         {record.acceleration.shape}, duration {record.duration_s:.1f} s")

# %% Feed the converted record straight into the calculations
result = calculate_measured_intensity(
    record.acceleration,
    record.metadata.sampling_rate_hz,
    unit=record.metadata.unit,
)
print(f"Measured intensity: {result.intensity:.1f} (震度{result.scale.japanese})")

pgv = peak_ground_velocity(
    remove_offset(record.acceleration),
    record.metadata.sampling_rate_hz,
)
print(f"PGV:                {pgv:.3f} cm/s")

# %% What the converter refuses to do
# Nothing is resampled, trimmed, merged, rotated, or rescaled. Traces that
# disagree are rejected instead of reconciled, because aligning them changes the
# record and should be a deliberate call to ObsPy first.
misaligned = stream.copy()
misaligned[0].stats.sampling_rate = 50.0
try:
    from_obspy_stream(misaligned, unit="gal")
except ValueError as error:
    print(f"Rejected as expected: {error}")

# %% Horizontal-only records are still useful
horizontal = from_obspy_stream(
    stream.select(channel="HN[NE]"),
    unit="gal",
)
print(f"horizontal components: {horizontal.metadata.component_names}")
horizontal_pgv = peak_ground_velocity(
    remove_offset(horizontal.acceleration),
    horizontal.metadata.sampling_rate_hz,
)
print(f"horizontal PGV:        {horizontal_pgv:.3f} cm/s")

# %%
