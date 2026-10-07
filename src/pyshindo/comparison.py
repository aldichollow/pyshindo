"""Compare intensity methods for one record, or any two results for equality."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .measured import calculate_measured_intensity
from .models import ComparisonReport, FieldDifference, IntensityComparisonResult
from .realtime import calculate_realtime_intensity
from .units import AccelerationUnit, ArrayLike

_COMMON_OPTION_NAMES = frozenset(
    {"acceleration", "sampling_rate_hz", "unit", "component_axis", "allow_fewer_components"}
)


def _validate_algorithm_options(
    name: str,
    options: Mapping[str, Any] | None,
) -> dict[str, Any]:
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
    measured_options: Mapping[str, Any] | None = None,
    realtime_options: Mapping[str, Any] | None = None,
) -> IntensityComparisonResult:
    """Calculate the FFT reference and real-time approximation for one record.

    ``raw_difference`` is defined as measured intensity minus the maximum
    real-time intensity. Algorithm-specific options may be supplied through the
    two mappings; common input conventions cannot be overridden there.
    """
    measured_kwargs: dict[str, Any] = {
        "component_axis": component_axis,
        "allow_fewer_components": allow_fewer_components,
        "retain_intermediates": False,
    }
    realtime_kwargs: dict[str, Any] = {
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


def compare_results(
    a: object, b: object, *, rtol: float = 1e-9, atol: float = 1e-12
) -> ComparisonReport:
    """Field-by-field equality check between two results of the same type.

    Every result type in this package that holds a NumPy array field
    disables the dataclass-generated ``__eq__`` entirely (``eq=False``),
    because ``a.field == b.field`` for an array field is itself an array,
    not a bool, and raises rather than compares -- this is the intended way
    to compare two of them instead. A numeric field (a float, or an array of
    any numeric dtype) is compared within ``rtol``/``atol`` via
    :func:`numpy.allclose`, with two ``NaN`` entries at the same position
    treated as agreeing rather than as a mismatch; everything else (an enum,
    a string, a nested dataclass, a tuple of dataclasses) falls back to
    ``==``. A field declared ``compare=False`` -- currently just ``timing``
    on every result that has one, since wall-clock timing is never the same
    between two runs even given identical input -- is skipped, honoring the
    same field metadata the standard library's own generated ``__eq__``
    would have.
    """
    if type(a) is not type(b):
        raise TypeError(
            "compare_results requires two instances of the same type; "
            f"received {type(a).__name__} and {type(b).__name__}."
        )
    if not dataclasses.is_dataclass(a) or isinstance(a, type):
        raise TypeError("compare_results requires two dataclass instances, not types.")

    differences: list[FieldDifference] = []
    for result_field in dataclasses.fields(a):
        if result_field.compare is False:
            continue
        difference = _compare_field(
            result_field.name,
            getattr(a, result_field.name),
            getattr(b, result_field.name),
            rtol=rtol,
            atol=atol,
        )
        if difference is not None:
            differences.append(difference)
    return ComparisonReport(equal=not differences, differences=tuple(differences))


def _compare_field(
    name: str, value_a: object, value_b: object, *, rtol: float, atol: float
) -> FieldDifference | None:
    if isinstance(value_a, np.ndarray) or isinstance(value_b, np.ndarray):
        return _compare_array_field(name, value_a, value_b, rtol=rtol, atol=atol)
    if isinstance(value_a, float) and isinstance(value_b, float):
        return _compare_float_field(name, value_a, value_b, rtol=rtol, atol=atol)
    if dataclasses.is_dataclass(value_a) and type(value_a) is type(value_b):
        nested = compare_results(value_a, value_b, rtol=rtol, atol=atol)
        return None if nested.equal else FieldDifference(name, f"nested mismatch -- {nested}")
    if value_a != value_b:
        return FieldDifference(name, f"{value_a!r} != {value_b!r}")
    return None


def _compare_float_field(
    name: str, value_a: float, value_b: float, *, rtol: float, atol: float
) -> FieldDifference | None:
    if math.isnan(value_a) and math.isnan(value_b):
        return None
    if math.isclose(value_a, value_b, rel_tol=rtol, abs_tol=atol):
        return None
    absolute_error = abs(value_a - value_b)
    relative_error = absolute_error / abs(value_b) if value_b else math.inf
    return FieldDifference(
        name,
        f"{value_a!r} != {value_b!r}",
        max_absolute_error=absolute_error,
        max_relative_error=relative_error,
    )


def _compare_array_field(
    name: str, value_a: object, value_b: object, *, rtol: float, atol: float
) -> FieldDifference | None:
    if value_a is None or value_b is None:
        if value_a is value_b:
            return None
        return FieldDifference(name, f"one side is None: {value_a!r} vs {value_b!r}")
    array_a, array_b = np.asarray(value_a), np.asarray(value_b)
    if array_a.shape != array_b.shape:
        return FieldDifference(name, f"shape mismatch: {array_a.shape} vs {array_b.shape}")
    if array_a.dtype.kind not in "fc":  # not float or complex: compare exactly (int, bool, ...)
        if np.array_equal(array_a, array_b):
            return None
        return FieldDifference(name, "array values differ (exact)")

    finite_a, finite_b = np.isfinite(array_a), np.isfinite(array_b)
    if not np.array_equal(finite_a, finite_b):
        return FieldDifference(name, "finite/non-finite pattern differs (NaN or Inf mismatch)")
    if np.allclose(array_a[finite_a], array_b[finite_b], rtol=rtol, atol=atol):
        return None
    absolute_errors = np.abs(array_a[finite_a] - array_b[finite_b])
    denominator = np.abs(array_b[finite_b])
    relative_errors = absolute_errors / np.where(denominator == 0, 1.0, denominator)
    return FieldDifference(
        name,
        "array values differ",
        max_absolute_error=float(absolute_errors.max()) if absolute_errors.size else 0.0,
        max_relative_error=float(relative_errors.max()) if relative_errors.size else 0.0,
    )
