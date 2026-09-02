"""Deterministic synthetic acceleration records for examples and tests."""

from __future__ import annotations

import math

import numpy as np

from .measured import measured_intensity
from .units import AccelerationUnit, ArrayLike, FloatArray
from .validation import as_acceleration_array, validate_sampling_rate


def synthetic_three_component_motion(
    sampling_rate_hz: float = 100.0,
    duration_s: float = 30.0,
    *,
    center_s: float | None = None,
    width_s: float = 2.5,
    amplitudes_gal: tuple[float, float, float] = (90.0, 62.0, 34.0),
    frequencies_hz: tuple[float, float, float] = (1.1, 2.4, 5.8),
    phases_rad: tuple[float, float, float] = (0.0, 0.35, 1.0),
    noise_std_gal: float = 0.02,
    seed: int = 2026,
) -> FloatArray:
    """Generate a smooth three-component burst with optional white noise.

    This signal is intended for demonstrations, regression tests, and API
    exploration. It is not a physical ground-motion simulator and must not be
    used as a substitute for validation against observed records.
    """
    rate = validate_sampling_rate(sampling_rate_hz, warn_nonstandard=False)
    duration_s = float(duration_s)
    width_s = float(width_s)
    noise_std_gal = float(noise_std_gal)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and greater than zero.")
    if not math.isfinite(width_s) or width_s <= 0.0:
        raise ValueError("width_s must be finite and greater than zero.")
    if not math.isfinite(noise_std_gal) or noise_std_gal < 0.0:
        raise ValueError("noise_std_gal must be finite and non-negative.")
    if len(amplitudes_gal) != 3 or len(frequencies_hz) != 3 or len(phases_rad) != 3:
        raise ValueError("Amplitude, frequency, and phase tuples must each contain three values.")
    parameters = (*amplitudes_gal, *frequencies_hz, *phases_rad)
    if any(not math.isfinite(value) for value in parameters):
        raise ValueError("Synthetic-motion parameters must be finite.")
    if any(value <= 0.0 for value in frequencies_hz):
        raise ValueError("frequencies_hz must be positive.")

    sample_count = max(1, int(round(duration_s * rate)))
    time_s = np.arange(sample_count, dtype=np.float64) / rate
    center = duration_s * 0.45 if center_s is None else float(center_s)
    if not math.isfinite(center):
        raise ValueError("center_s must be finite.")
    envelope = np.exp(-0.5 * ((time_s - center) / width_s) ** 2)
    values = np.column_stack(
        [
            amplitude
            * envelope
            * np.sin(2.0 * np.pi * frequency * time_s + phase)
            for amplitude, frequency, phase in zip(
                amplitudes_gal,
                frequencies_hz,
                phases_rad,
                strict=True,
            )
        ]
    )
    if noise_std_gal > 0.0:
        generator = np.random.default_rng(seed)
        values += generator.normal(scale=noise_std_gal, size=values.shape)
    return np.ascontiguousarray(values, dtype=np.float64)


def scale_acceleration_to_intensity(
    acceleration: ArrayLike,
    target_intensity_raw: float,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    component_axis: int = -1,
) -> tuple[FloatArray, float]:
    """Scale a complete record to a target raw FFT-reference intensity.

    Instrumental intensity changes by ``2 * log10(scale)`` under positive linear
    amplitude scaling. The function therefore needs only one reference
    calculation. It returns the scaled data in the input unit and layout, plus
    the applied dimensionless factor.
    """
    target = float(target_intensity_raw)
    if not math.isfinite(target):
        raise ValueError("target_intensity_raw must be finite.")
    original = np.asarray(acceleration)
    values = as_acceleration_array(
        acceleration,
        component_axis=component_axis,
        allow_fewer_components=False,
        warn_fewer_components=False,
    )
    current = measured_intensity(
        values,
        sampling_rate_hz,
        unit=unit,
        reported=False,
        component_axis=-1,
    )
    if not math.isfinite(current):
        raise ValueError("The input record has no positive intensity threshold to scale.")
    factor = 10.0 ** ((target - current) / 2.0)
    scaled = np.ascontiguousarray(values * factor, dtype=np.float64)
    if original.ndim == 1:
        return scaled[:, 0], factor
    restored = np.moveaxis(scaled, -1, component_axis)
    return np.ascontiguousarray(restored), factor
