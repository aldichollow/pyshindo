"""Stateful, chunk-invariant long-period ground motion estimation."""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

from ..exceptions import InvalidAccelerationError
from ..units import AccelerationUnit, ArrayLike, FloatArray, to_gal
from ..validation import validate_sampling_rate
from ._core import (
    ResponseState,
    apply_high_pass,
    design_high_pass,
    design_oscillator_bank,
    high_pass_initial_state,
)
from .calculation import (
    as_horizontal_acceleration,
    meets_reference_conditions,
    resolve_periods,
    summarize_bands,
    warn_if_non_reference_rate,
)
from .models import LongPeriodResult, LongPeriodTiming, LongPeriodUpdate
from .scale import (
    OFFICIAL_DAMPING_RATIO,
    LongPeriodClass,
    classify_long_period,
)


class LongPeriodEstimator:
    """Track the long-period ground motion class as samples arrive.

    The JMA class is the maximum absolute velocity response over the whole
    record, so the streaming value is a cumulative maximum over everything
    seen so far, not a rolling window. It can only rise. This is the same
    quantity the batch calculation returns for that prefix of the record, and
    the two agree exactly: chunked and sample-by-sample processing run through
    the identical filter and recurrence state.

    Memory is O(periods) regardless of how long the stream runs.
    """

    def __init__(
        self,
        sampling_rate_hz: float = 100.0,
        *,
        unit: str | AccelerationUnit = AccelerationUnit.GAL,
        damping_ratio: float = OFFICIAL_DAMPING_RATIO,
        periods_s: npt.ArrayLike | None = None,
        high_pass: bool = True,
        warn_nonstandard_rate: bool = True,
    ) -> None:
        rate = validate_sampling_rate(sampling_rate_hz, warn_nonstandard=False)
        if warn_nonstandard_rate:
            warn_if_non_reference_rate(rate, stacklevel=3)
        self._rate = rate
        self._unit = AccelerationUnit.parse(unit)
        self._periods = resolve_periods(periods_s)
        self._damping_ratio = float(damping_ratio)
        self._high_pass = high_pass
        self._design = design_high_pass(rate) if high_pass else None
        self._filter_state = (
            high_pass_initial_state(self._design) if self._design is not None else None
        )
        bank = design_oscillator_bank(self._periods, self._damping_ratio, rate)
        self._response = ResponseState(bank)

    @property
    def sampling_rate_hz(self) -> float:
        """Return the configured sampling rate."""
        return self._rate

    @property
    def periods_s(self) -> FloatArray:
        """Return the period grid being evaluated."""
        return self._periods

    @property
    def sample_count(self) -> int:
        """Return the number of samples processed so far."""
        return self._response.sample_count

    @property
    def sva_cm_s(self) -> FloatArray:
        """Return the per-period Sva maxima so far, in cm/s."""
        return self._response.running_max.copy()

    @property
    def max_sva_cm_s(self) -> float:
        """Return the largest Sva over all periods so far, in cm/s."""
        if self._response.sample_count == 0:
            return 0.0
        return float(np.max(self._response.running_max))

    @property
    def long_period_class(self) -> LongPeriodClass:
        """Return the class implied by everything processed so far."""
        return classify_long_period(self.max_sva_cm_s)

    def _prepare(self, acceleration: ArrayLike, component_axis: int) -> FloatArray:
        values = as_horizontal_acceleration(acceleration, component_axis=component_axis)
        return to_gal(values, self._unit, copy=False)

    def _advance(self, values_gal: FloatArray) -> None:
        if self._design is not None and self._filter_state is not None:
            filtered, self._filter_state = apply_high_pass(
                self._design, values_gal, self._filter_state
            )
        else:
            filtered = values_gal
        self._response.advance(filtered, collect=False)

    def process(self, acceleration: ArrayLike, *, component_axis: int = -1) -> LongPeriodUpdate:
        """Process a chunk of horizontal acceleration and return the state so far."""
        started = time.perf_counter()
        values_gal = self._prepare(acceleration, component_axis)
        self._advance(values_gal)
        return self._update(time.perf_counter() - started)

    def process_sample(self, sample: ArrayLike) -> LongPeriodUpdate:
        """Process one two-component sample.

        Equivalent to :meth:`process` with a single row; provided so a caller
        driving the estimator one sample at a time does not have to reshape.
        """
        started = time.perf_counter()
        values = np.asarray(sample, dtype=np.float64)
        if values.shape != (2,):
            raise InvalidAccelerationError(
                f"sample must hold two horizontal components; received shape {values.shape}."
            )
        values_gal = self._prepare(values[np.newaxis, :], -1)
        self._advance(values_gal)
        return self._update(time.perf_counter() - started)

    def _update(self, elapsed_s: float) -> LongPeriodUpdate:
        maximum = self.max_sva_cm_s
        return LongPeriodUpdate(
            sample_index=self._response.sample_count - 1,
            sample_count=self._response.sample_count,
            max_sva_so_far_cm_s=maximum,
            class_so_far=classify_long_period(maximum),
            elapsed_s=elapsed_s,
        )

    def result(self) -> LongPeriodResult:
        """Return a complete-record result for everything processed so far.

        The same shape of answer :func:`calculate_long_period_class` gives, so
        a streaming run can be reported or compared without a second pass.
        ``absolute_velocity_cm_s`` is always ``None``: the estimator keeps
        only running maxima.
        """
        sva = self.sva_cm_s
        critical_index = int(np.argmax(sva)) if sva.size else 0
        maximum = float(sva[critical_index]) if sva.size else 0.0
        reference_conditions_met = meets_reference_conditions(
            high_pass=self._high_pass,
            published_high_pass=(
                self._design is not None and self._design.is_published_reference
            ),
            sampling_rate_hz=self._rate,
            periods_s=self._periods,
            damping_ratio=self._damping_ratio,
            component_count=2,
        )
        return LongPeriodResult(
            sva_cm_s=sva,
            periods_s=self._periods,
            max_sva_cm_s=maximum,
            critical_period_s=float(self._periods[critical_index]),
            long_period_class=classify_long_period(maximum),
            bands=summarize_bands(sva, self._periods),
            damping_ratio=self._damping_ratio,
            sampling_rate_hz=self._rate,
            sample_count=self._response.sample_count,
            component_count=2,
            high_pass_applied=self._high_pass,
            reference_conditions_met=reference_conditions_met,
            absolute_velocity_cm_s=None,
            timing=LongPeriodTiming(high_pass_s=0.0, response_s=0.0, total_s=0.0),
        )
