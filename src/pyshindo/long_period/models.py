"""Result models for the long-period ground motion calculation.

Kept in the subpackage rather than the package-wide :mod:`pyshindo.models`,
which already carries filter designs, intensity results, and record I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..units import FloatArray
from .scale import LongPeriodClass


@dataclass(frozen=True, slots=True)
class LongPeriodTiming:
    """Wall-clock timing for one long-period calculation.

    ``high_pass_s``: the 20-second high-pass pass over the record.
    ``response_s``: the oscillator bank and ground-velocity recursion.
    ``total_s``: the complete call. Measured with :func:`time.perf_counter`.
    """

    high_pass_s: float
    response_s: float
    total_s: float


@dataclass(frozen=True, slots=True)
class LongPeriodBandResult:
    """Class for one of the seven period bands JMA reports separately."""

    band_second: int
    period_range_s: tuple[float, float]
    max_sva_cm_s: float
    long_period_class: LongPeriodClass

    @property
    def japanese_label(self) -> str:
        """Return the band name as JMA writes it, for example ``3秒台``."""
        return f"{self.band_second}秒台"


@dataclass(frozen=True, slots=True)
class LongPeriodResult:
    """Detailed output of the long-period ground motion class calculation."""

    sva_cm_s: FloatArray
    periods_s: FloatArray
    max_sva_cm_s: float
    critical_period_s: float
    long_period_class: LongPeriodClass
    bands: tuple[LongPeriodBandResult, ...]
    damping_ratio: float
    sampling_rate_hz: float
    sample_count: int
    component_count: int
    high_pass_applied: bool
    reference_conditions_met: bool
    absolute_velocity_cm_s: FloatArray | None
    timing: LongPeriodTiming

    @property
    def record_duration_s(self) -> float:
        """Return sample count divided by sampling rate."""
        return self.sample_count / self.sampling_rate_hz


    def band(self, band_second: int) -> LongPeriodBandResult:
        """Return the result for one period band, keyed by its integer second."""
        for entry in self.bands:
            if entry.band_second == band_second:
                return entry
        raise ValueError(f"No period band {band_second} in this result.")


@dataclass(frozen=True, slots=True)
class LongPeriodUpdate:
    """Cumulative state after one streaming chunk or sample.

    The values are the exact complete-record statistics for the part of the
    record seen so far, not a rolling window: JMA's class is the maximum over
    the whole record, so a streaming estimate can only grow.
    """

    sample_index: int
    sample_count: int
    max_sva_so_far_cm_s: float
    class_so_far: LongPeriodClass
    elapsed_s: float
