"""Convenience functions for comparing reference and real-time intensity."""

from __future__ import annotations

from collections.abc import Mapping

from .measured import calculate_measured_intensity
from .models import IntensityComparisonResult
from .realtime import calculate_realtime_intensity
from .units import AccelerationUnit, ArrayLike

_COMMON_OPTION_NAMES = frozenset(
    {"acceleration", "sampling_rate_hz", "unit", "component_axis", "allow_fewer_components"}
)


def _validate_algorithm_options(
    name: str,
    options: Mapping[str, object] | None,
) -> dict[str, object]:
    if options is None:
        return {}
    forbidden = _COMMON_OPTION_NAMES.intersection(options)
    if forbidden:
        joined = ", ".join(sorted(forbidden))
        raise ValueError(f"{name} must not override common input option(s): {joined}.")
    return dict(options)


def compare_intensity_methods(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    component_axis: int = -1,
    allow_fewer_components: bool = False,
    measured_options: Mapping[str, object] | None = None,
    realtime_options: Mapping[str, object] | None = None,
) -> IntensityComparisonResult:
    """Calculate the FFT reference and real-time approximation for one record.

    ``raw_difference`` is defined as measured intensity minus the maximum
    real-time intensity. Algorithm-specific options may be supplied through the
    two mappings; common input conventions cannot be overridden there.
    """
    measured_kwargs: dict[str, object] = {
        "component_axis": component_axis,
        "allow_fewer_components": allow_fewer_components,
        "retain_intermediates": False,
    }
    realtime_kwargs: dict[str, object] = {
        "component_axis": component_axis,
        "allow_fewer_components": allow_fewer_components,
        "retain_filtered": False,
    }
    measured_kwargs.update(_validate_algorithm_options("measured_options", measured_options))
    realtime_kwargs.update(_validate_algorithm_options("realtime_options", realtime_options))

    measured = calculate_measured_intensity(
        acceleration,
        sampling_rate_hz,
        unit=unit,
        **measured_kwargs,
    )
    realtime = calculate_realtime_intensity(
        acceleration,
        sampling_rate_hz,
        unit=unit,
        **realtime_kwargs,
    )
    return IntensityComparisonResult(measured=measured, realtime=realtime)
