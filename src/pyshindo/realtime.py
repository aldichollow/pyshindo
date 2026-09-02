"""Stateful and batch implementations of real-time seismic intensity.

The causal filter is applied independently to each acceleration component. The
filtered components are combined by Euclidean norm, and an exact rolling order
statistic selects the amplitude representing the configured cumulative duration
(0.3 seconds by default) within the latest 60 seconds.

References
----------
Kunugi, T. et al. (2008), https://doi.org/10.4294/zisin.60.243
Kunugi, T. et al. (2013), https://doi.org/10.4294/zisin.65.223
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Final

import numpy as np
from scipy.signal import sosfilt

from ._order import RollingKthLargest
from .duration import DurationSamplePolicy, duration_sample_count
from .exceptions import (
    InvalidAccelerationError,
    MissingComponentWarning,
    NonstandardSamplingRateWarning,
)
from .filters.realtime import (
    Kunugi2008Parameters,
    Kunugi2012Parameters,
    LowRateGammaPolicy,
    LowRateGammaSet,
    RealtimeFilter,
    design_realtime_filter,
)
from .models import (
    RealtimeChunk,
    RealtimeChunkTiming,
    RealtimeIntensityResult,
    RealtimeSample,
    RecursiveFilterDesign,
)
from .scale import (
    classify_intensity,
    intensity_from_acceleration,
    intensity_series_from_acceleration,
    report_intensity,
    report_intensity_array,
)
from .signal import component_peak_acceleration, peak_ground_acceleration, vector_resultant
from .units import AccelerationUnit, ArrayLike, FloatArray, conversion_factor, to_gal
from .validation import as_acceleration_array, validate_sampling_rate

_DEFAULT_WINDOW_S: Final = 60.0
_DEFAULT_DURATION_S: Final = 0.3


class RealtimeIntensityEstimator:
    """Incrementally calculate real-time seismic intensity from acceleration.

    Recursive-filter state and the rolling order statistic are preserved across
    arbitrary chunk boundaries. Processing one sample at a time or replaying the
    same samples in larger chunks gives the same algorithmic sequence. Larger
    chunks are recommended when the input transport permits them because SciPy
    performs recursive filtering in compiled code.

    ``process_sample`` uses an in-place state update specialized for one sample,
    avoiding a SciPy call and most temporary arrays. This is useful for callback-
    driven acquisition. The object is stateful and is not safe for concurrent
    calls; use one estimator per independent stream.
    """

    def __init__(
        self,
        sampling_rate_hz: float = 100.0,
        *,
        unit: str | AccelerationUnit = AccelerationUnit.GAL,
        filter_name: str | RealtimeFilter = RealtimeFilter.AUTO,
        parameters: Kunugi2008Parameters | Kunugi2012Parameters | None = None,
        lowrate_gamma_policy: str | LowRateGammaPolicy = LowRateGammaPolicy.PIECEWISE,
        lowrate_gammas: LowRateGammaSet | None = None,
        filter_design: RecursiveFilterDesign | None = None,
        window_s: float = _DEFAULT_WINDOW_S,
        duration_s: float = _DEFAULT_DURATION_S,
        duration_policy: DurationSamplePolicy = "ceil",
        component_axis: int = -1,
        allow_fewer_components: bool = False,
        warn_nonstandard_rate: bool = True,
    ) -> None:
        self.sampling_rate_hz = validate_sampling_rate(
            sampling_rate_hz,
            warn_nonstandard=warn_nonstandard_rate,
            stacklevel=2,
        )
        self.input_unit = AccelerationUnit.parse(unit)
        self.component_axis = component_axis
        self.allow_fewer_components = allow_fewer_components
        self.duration_samples = duration_sample_count(
            duration_s,
            self.sampling_rate_hz,
            policy=duration_policy,
        )
        self.window_samples = duration_sample_count(
            window_s,
            self.sampling_rate_hz,
            policy=duration_policy,
        )
        if self.window_samples < self.duration_samples:
            raise ValueError("window_s must contain at least duration_s samples.")

        if filter_design is None:
            self.filter_design = design_realtime_filter(
                self.sampling_rate_hz,
                filter_name=filter_name,
                parameters=parameters,
                lowrate_gamma_policy=lowrate_gamma_policy,
                lowrate_gammas=lowrate_gammas,
            )
        else:
            if not math.isclose(
                filter_design.sampling_rate_hz,
                self.sampling_rate_hz,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "filter_design.sampling_rate_hz does not match sampling_rate_hz."
                )
            if not filter_design.stable:
                raise ValueError("filter_design must be stable.")
            self.filter_design = filter_design

        highest_frequency = max(self.filter_design.characteristic_frequencies_hz)
        if self.filter_design.nyquist_hz <= highest_frequency:
            warnings.warn(
                "The Nyquist frequency is not above every characteristic frequency in the "
                "selected approximation filter. The recursive design is stable, but its "
                "amplitude response requires application-specific validation.",
                NonstandardSamplingRateWarning,
                stacklevel=2,
            )

        self._rank = RollingKthLargest(self.window_samples, self.duration_samples)
        self._zi: np.ndarray | None = None
        self._component_count: int | None = None
        self._sample_count = 0
        self._record_max = -np.inf
        self._has_intensity = False

        # Cached in plain Python types for process_sample(): avoids re-deriving the unit
        # factor and re-reading numpy scalars from self.filter_design.sos on every sample.
        self._gal_factor: float = conversion_factor(self.input_unit, AccelerationUnit.GAL)
        self._sos_py: tuple[tuple[float, float, float, float, float], ...] = tuple(
            (row[0], row[1], row[2], row[4], row[5]) for row in self.filter_design.sos.tolist()
        )

    @property
    def sample_count(self) -> int:
        """Return the total number of processed samples."""
        return self._sample_count

    @property
    def component_count(self) -> int | None:
        """Return the locked component count, or ``None`` before first input."""
        return self._component_count

    @property
    def current_threshold_acceleration_gal(self) -> float | None:
        """Return the current rolling duration threshold, if available."""
        return self._rank.threshold

    @property
    def approximate_intensity_raw(self) -> float:
        """Return the maximum real-time intensity observed since reset."""
        return self._record_max if self._has_intensity else np.nan

    @property
    def approximate_intensity(self) -> float:
        """Return the reported one-decimal maximum intensity observed since reset."""
        return report_intensity(self.approximate_intensity_raw)

    @property
    def filter_state(self) -> np.ndarray | None:
        """Return a defensive copy of the recursive-filter state."""
        return None if self._zi is None else self._zi.copy()

    def reset(self) -> None:
        """Reset filter, rolling-window, index, and maximum state."""
        self._rank.clear()
        self._zi = None
        self._component_count = None
        self._sample_count = 0
        self._record_max = -np.inf
        self._has_intensity = False

    def _initialize_stream(self, component_count: int) -> None:
        if not 1 <= component_count <= 3:
            raise InvalidAccelerationError("A stream must contain one, two, or three components.")
        if component_count < 3:
            if not self.allow_fewer_components:
                raise InvalidAccelerationError(
                    "Three orthogonal acceleration components are required for this calculation."
                )
            warnings.warn(
                f"Only {component_count} acceleration component(s) were supplied. The resulting "
                "intensity is not the standard three-component value.",
                MissingComponentWarning,
                stacklevel=3,
            )
        self._component_count = component_count
        self._zi = np.zeros(
            (self.filter_design.sos.shape[0], 2, component_count),
            dtype=np.float64,
        )

    def _update_record_max(self, intensity_raw: float) -> float:
        if not self._has_intensity or intensity_raw > self._record_max:
            self._record_max = intensity_raw
        self._has_intensity = True
        return self._record_max

    def process(self, acceleration: ArrayLike) -> RealtimeChunk:
        """Process one chunk and return sample-aligned intermediate outputs."""
        total_started = time.perf_counter()
        first_chunk = self._component_count is None
        values = as_acceleration_array(
            acceleration,
            component_axis=self.component_axis,
            allow_fewer_components=self.allow_fewer_components,
            warn_fewer_components=False,
        )
        if first_chunk:
            self._initialize_stream(values.shape[1])
        elif values.shape[1] != self._component_count:
            raise ValueError(
                f"The stream was initialized with {self._component_count} components, but "
                f"this chunk contains {values.shape[1]}."
            )

        values_gal = values if self._gal_factor == 1.0 else values * self._gal_factor
        if self._zi is None:
            raise RuntimeError("Filter state was not initialized.")
        filter_started = time.perf_counter()
        filtered, self._zi = sosfilt(
            self.filter_design.sos,
            values_gal,
            axis=0,
            zi=self._zi,
        )
        filtered = np.ascontiguousarray(filtered, dtype=np.float64)
        resultant = vector_resultant(filtered)
        filter_elapsed = time.perf_counter() - filter_started

        order_statistic_started = time.perf_counter()
        thresholds = np.full(resultant.shape, np.nan, dtype=np.float64)
        update = self._rank.update
        for index, value in enumerate(resultant):
            threshold = update(float(value))
            if threshold is not None:
                thresholds[index] = threshold
        order_statistic_elapsed = time.perf_counter() - order_statistic_started

        intensity_raw = intensity_series_from_acceleration(thresholds)
        intensity = report_intensity_array(intensity_raw)
        valid = ~np.isnan(intensity_raw)
        record_max = np.full(intensity_raw.shape, np.nan, dtype=np.float64)
        if np.any(valid):
            candidates = np.where(valid, intensity_raw, -np.inf)
            cumulative = np.maximum.accumulate(candidates)
            if self._has_intensity:
                cumulative = np.maximum(cumulative, self._record_max)
            record_max[valid] = cumulative[valid]
            self._record_max = float(cumulative[valid][-1])
            self._has_intensity = True

        start = self._sample_count
        stop = start + values.shape[0]
        sample_index = np.arange(start, stop, dtype=np.int64)
        time_s = sample_index.astype(np.float64) / self.sampling_rate_hz
        self._sample_count = stop
        timing = RealtimeChunkTiming(
            filter_s=filter_elapsed,
            order_statistic_s=order_statistic_elapsed,
            total_s=time.perf_counter() - total_started,
        )

        return RealtimeChunk(
            sample_index=sample_index,
            time_s=time_s,
            filtered_acceleration_gal=filtered,
            resultant_acceleration_gal=resultant,
            threshold_acceleration_gal=thresholds,
            intensity_raw=intensity_raw,
            intensity=intensity,
            record_max_intensity_raw=record_max,
            timing=timing,
        )

    def process_sample(self, acceleration: ArrayLike) -> RealtimeSample:
        """Process exactly one component vector with a low-allocation state update.

        Runs the filter cascade in plain Python floats rather than NumPy
        arrays -- NumPy's per-call dispatch overhead dominates at this size.
        ``self._zi`` is written back in place every call, so interleaving
        this with :meth:`process` stays correct.
        """
        total_started = time.perf_counter()
        values = np.asarray(acceleration, dtype=np.float64)
        if values.ndim != 1:
            raise InvalidAccelerationError(
                "process_sample expects a one-dimensional component vector."
            )
        size = values.size
        if size == 0 or size > 3:
            raise InvalidAccelerationError(
                "process_sample expects one, two, or three component values."
            )
        # A plain Python loop over <=3 floats avoids NumPy's per-call reduction
        # overhead, which dominates at this size (see the docstring above).
        raw = values.tolist()
        if not all(math.isfinite(x) for x in raw):
            raise InvalidAccelerationError("acceleration contains a non-finite value.")
        if self._component_count is None:
            self._initialize_stream(size)
        elif size != self._component_count:
            raise ValueError(
                f"The stream was initialized with {self._component_count} components, but "
                f"this sample contains {size}."
            )
        if self._zi is None:
            raise RuntimeError("Filter state was not initialized.")

        gal_factor = self._gal_factor
        current = raw if gal_factor == 1.0 else [x * gal_factor for x in raw]
        component_count = size
        # A single tolist() on the whole cascade state, followed by a single
        # array() back, replaces one numpy view + tolist() + slice-write per
        # section (12 numpy calls for a 6-section design) with plain list
        # operations -- cheaper at this element count for the same reason the
        # cascade itself runs on Python floats. self._zi is already replaced
        # wholesale on every process() call, so identity is not relied upon.
        zi_list = self._zi.tolist()
        for section_index, (b0, b1, b2, a1, a2) in enumerate(self._sos_py):
            section_state = zi_list[section_index]
            s0 = section_state[0]
            s1 = section_state[1]
            next_current = [0.0] * component_count
            next_s0 = [0.0] * component_count
            next_s1 = [0.0] * component_count
            for c in range(component_count):
                x = current[c]
                output = b0 * x + s0[c]
                next_s0[c] = b1 * x - a1 * output + s1[c]
                next_s1[c] = b2 * x - a2 * output
                next_current[c] = output
            section_state[0] = next_s0
            section_state[1] = next_s1
            current = next_current
        self._zi = np.array(zi_list, dtype=np.float64)

        filtered = np.array(current, dtype=np.float64)
        if component_count == 1:
            resultant = abs(current[0])
        elif component_count == 2:
            resultant = math.sqrt(current[0] * current[0] + current[1] * current[1])
        else:
            resultant = math.sqrt(
                current[0] * current[0] + current[1] * current[1] + current[2] * current[2]
            )
        threshold = self._rank.update(resultant)
        if threshold is None:
            intensity_raw = None
            intensity = None
            scale = None
            record_max = None
        else:
            intensity_raw = intensity_from_acceleration(threshold)
            intensity = report_intensity(intensity_raw)
            scale = classify_intensity(intensity)
            record_max = self._update_record_max(intensity_raw)

        sample_index = self._sample_count
        time_s = sample_index / self.sampling_rate_hz
        self._sample_count += 1
        elapsed_s = time.perf_counter() - total_started
        return RealtimeSample(
            sample_index=sample_index,
            time_s=time_s,
            filtered_acceleration_gal=filtered,
            resultant_acceleration_gal=resultant,
            threshold_acceleration_gal=threshold,
            intensity_raw=intensity_raw,
            intensity=intensity,
            scale=scale,
            record_max_intensity_raw=record_max,
            elapsed_s=elapsed_s,
        )


def calculate_realtime_intensity(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    filter_name: str | RealtimeFilter = RealtimeFilter.AUTO,
    parameters: Kunugi2008Parameters | Kunugi2012Parameters | None = None,
    lowrate_gamma_policy: str | LowRateGammaPolicy = LowRateGammaPolicy.PIECEWISE,
    lowrate_gammas: LowRateGammaSet | None = None,
    filter_design: RecursiveFilterDesign | None = None,
    window_s: float = _DEFAULT_WINDOW_S,
    duration_s: float = _DEFAULT_DURATION_S,
    duration_policy: DurationSamplePolicy = "ceil",
    component_axis: int = -1,
    allow_fewer_components: bool = False,
    warn_nonstandard_rate: bool = True,
    retain_filtered: bool = True,
) -> RealtimeIntensityResult:
    """Replay a complete record through the stateful real-time algorithm."""
    estimator = RealtimeIntensityEstimator(
        sampling_rate_hz,
        unit=unit,
        filter_name=filter_name,
        parameters=parameters,
        lowrate_gamma_policy=lowrate_gamma_policy,
        lowrate_gammas=lowrate_gammas,
        filter_design=filter_design,
        window_s=window_s,
        duration_s=duration_s,
        duration_policy=duration_policy,
        component_axis=component_axis,
        allow_fewer_components=allow_fewer_components,
        warn_nonstandard_rate=warn_nonstandard_rate,
    )
    chunk = estimator.process(acceleration)
    approximate_raw = estimator.approximate_intensity_raw
    approximate = estimator.approximate_intensity
    approximate_scale = None if np.isnan(approximate) else classify_intensity(approximate)

    values = as_acceleration_array(
        acceleration,
        component_axis=component_axis,
        allow_fewer_components=allow_fewer_components,
        warn_fewer_components=False,
    )
    values_gal = to_gal(values, unit, copy=False)
    return RealtimeIntensityResult(
        intensity_raw=chunk.intensity_raw,
        intensity=chunk.intensity,
        threshold_acceleration_gal=chunk.threshold_acceleration_gal,
        resultant_acceleration_gal=chunk.resultant_acceleration_gal,
        filtered_acceleration_gal=chunk.filtered_acceleration_gal if retain_filtered else None,
        record_max_intensity_raw=chunk.record_max_intensity_raw,
        sampling_rate_hz=estimator.sampling_rate_hz,
        window_samples=estimator.window_samples,
        duration_samples=estimator.duration_samples,
        filter_name=estimator.filter_design.name,
        input_component_pga_gal=component_peak_acceleration(values_gal),
        input_pga_gal=peak_ground_acceleration(values_gal),
        filtered_component_pga_gal=component_peak_acceleration(
            chunk.filtered_acceleration_gal
        ),
        filtered_pga_gal=float(np.max(chunk.resultant_acceleration_gal)),
        timing=chunk.timing,
        approximate_intensity_raw=approximate_raw,
        approximate_intensity=approximate,
        approximate_scale=approximate_scale,
    )


def realtime_intensity(
    acceleration: ArrayLike,
    sampling_rate_hz: float = 100.0,
    *,
    unit: str | AccelerationUnit = AccelerationUnit.GAL,
    reported: bool = False,
    **kwargs: object,
) -> FloatArray:
    """Return only the sample-aligned real-time intensity series."""
    result = calculate_realtime_intensity(
        acceleration,
        sampling_rate_hz,
        unit=unit,
        retain_filtered=False,
        **kwargs,
    )
    return result.intensity if reported else result.intensity_raw
