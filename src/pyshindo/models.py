"""Result and filter-design data models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ._immutable import frozen_array, frozen_mapping
from .scale import IntensityScale
from .signal import time_axis
from .units import AccelerationUnit, FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class FrequencyResponse:
    """Complex filter response sampled at physical frequencies.

    Equality is disabled (``eq=False``): the auto-generated comparison
    would do ``a.frequency_hz == b.frequency_hz``, itself an array rather
    than a bool. Use :func:`pyshindo.comparison.compare_results` instead.
    """

    frequency_hz: FloatArray
    response: npt.NDArray[np.complex128]

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequency_hz", frozen_array(self.frequency_hz))
        object.__setattr__(self, "response", frozen_array(self.response))

    @property
    def amplitude(self) -> FloatArray:
        """Return the response magnitude."""
        return np.abs(self.response)

    @property
    def phase_rad(self) -> FloatArray:
        """Return the unwrapped response phase in radians."""
        return np.unwrap(np.angle(self.response))


@dataclass(frozen=True, slots=True, eq=False)
class FilterStage:
    """One named, individually inspectable component of a filter cascade.

    ``sos`` is a single second-order-section row (shape ``(6,)``), in the same
    normalized form as one row of :attr:`RecursiveFilterDesign.sos`: first-order
    stages simply have zero for their second-order coefficients. Cascading every
    stage of a design, in order, reproduces that design's combined response --
    stages exist so each named component can be inspected or plotted on its own,
    not as an alternative way to filter data.

    Equality is disabled (``eq=False``) because ``sos`` is an array; use
    :func:`pyshindo.comparison.compare_results` instead.
    """

    name: str
    characteristic_frequency_hz: float | None
    sos: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "sos", frozen_array(self.sos, dtype=np.float64))


@dataclass(frozen=True, slots=True, eq=False)
class RecursiveFilterDesign:
    """Second-order-section representation of a published approximation filter.

    A design is immutable configuration: ``sos`` is read-only, and
    ``parameters`` is a read-only :class:`~types.MappingProxyType`.
    :func:`scipy.signal.sosfilt` does need a mutable buffer, but nothing in
    this package ever calls it with ``design.sos`` directly --
    :class:`~pyshindo.realtime.RealtimeIntensityEstimator` always takes its
    own private, independently writable copy first. Equality is disabled
    (``eq=False``) because ``sos`` is an array; use
    :func:`pyshindo.comparison.compare_results` instead.
    """

    name: str
    sampling_rate_hz: float
    sos: FloatArray
    parameters: Mapping[str, float]
    characteristic_frequencies_hz: tuple[float, ...]
    max_pole_radius: float
    stable: bool
    stages: tuple[FilterStage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sos", frozen_array(self.sos, dtype=np.float64))
        object.__setattr__(self, "parameters", frozen_mapping(self.parameters))

    @property
    def nyquist_hz(self) -> float:
        """Return the Nyquist frequency of the design."""
        return self.sampling_rate_hz / 2.0

    @property
    def section_count(self) -> int:
        """Return the number of second-order sections."""
        return int(self.sos.shape[0])

    @property
    def stability_margin(self) -> float:
        """Return ``1 - max_pole_radius``; positive values indicate stability."""
        return 1.0 - self.max_pole_radius


@dataclass(frozen=True, slots=True, eq=False)
class AmplitudeDurationCurve:
    """Amplitude sorted from high to low and its cumulative exceedance duration.

    Equality is disabled (``eq=False``) because both fields are arrays; use
    :func:`pyshindo.comparison.compare_results` instead.
    """

    amplitude: FloatArray
    exceedance_duration_s: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "amplitude", frozen_array(self.amplitude, dtype=np.float64))
        object.__setattr__(
            self,
            "exceedance_duration_s",
            frozen_array(self.exceedance_duration_s, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class MeasuredIntensityTiming:
    """Wall-clock timing for one :func:`calculate_measured_intensity` call.

    ``fft_filter_s``: the FFT/response/inverse-FFT stage. ``duration_threshold_s``:
    the vector resultant and 0.3-second order-statistic selection. ``total_s``:
    the complete call. Measured with :func:`time.perf_counter`.
    """

    fft_filter_s: float
    duration_threshold_s: float
    total_s: float


@dataclass(frozen=True, slots=True, eq=False)
class MeasuredIntensityResult:
    """Detailed output of the frequency-domain reference calculation.

    Equality is disabled (``eq=False``) because several fields are arrays;
    use :func:`pyshindo.comparison.compare_results` instead. ``timing`` is
    excluded from that comparison (``compare=False``) since wall-clock
    timing is never the same between two runs even given identical input.
    """

    intensity_raw: float
    intensity: float
    scale: IntensityScale
    threshold_acceleration_gal: float
    duration_samples: int
    effective_duration_s: float
    sampling_rate_hz: float
    sample_count: int
    component_count: int
    input_unit: AccelerationUnit
    input_component_pga_gal: FloatArray
    input_pga_gal: float
    filtered_component_pga_gal: FloatArray
    filtered_pga_gal: float
    filtered_acceleration_gal: FloatArray | None
    resultant_acceleration_gal: FloatArray | None
    frequency_hz: FloatArray | None
    filter_response: FloatArray | None
    reference_conditions_met: bool
    timing: MeasuredIntensityTiming = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "input_component_pga_gal", frozen_array(self.input_component_pga_gal)
        )
        object.__setattr__(
            self, "filtered_component_pga_gal", frozen_array(self.filtered_component_pga_gal)
        )
        for optional_field in (
            "filtered_acceleration_gal",
            "resultant_acceleration_gal",
            "frequency_hz",
            "filter_response",
        ):
            value = getattr(self, optional_field)
            if value is not None:
                object.__setattr__(self, optional_field, frozen_array(value))

    @property
    def record_duration_s(self) -> float:
        """Return sample count divided by sampling rate."""
        return self.sample_count / self.sampling_rate_hz


@dataclass(frozen=True, slots=True)
class RealtimeChunkTiming:
    """Wall-clock timing for one :meth:`RealtimeIntensityEstimator.process` call.

    ``filter_s``: the compiled ``sosfilt`` pass. ``order_statistic_s``: the
    per-sample rolling-threshold loop. ``reporting_s``: converting the
    threshold series to intensity and applying the decimal display rounding
    (:func:`~pyshindo.scale.report_intensity_array`) -- the Decimal-based
    two-step rounding this involves runs per sample in Python and, for a
    large chunk, can cost as much as ``order_statistic_s`` itself, so it is
    broken out here rather than left invisible inside the gap between the
    other three fields and ``total_s``. ``total_s``: the complete call.
    """

    filter_s: float
    order_statistic_s: float
    reporting_s: float
    total_s: float


@dataclass(frozen=True, slots=True, eq=False)
class RealtimeIntensityResult:
    """Detailed output of a batch replay of the real-time algorithm.

    Equality is disabled (``eq=False``) because several fields are arrays;
    use :func:`pyshindo.comparison.compare_results` instead. ``timing`` is
    excluded from that comparison (``compare=False``) since wall-clock
    timing is never the same between two runs even given identical input.
    """

    intensity_raw: FloatArray
    intensity: FloatArray
    threshold_acceleration_gal: FloatArray
    resultant_acceleration_gal: FloatArray
    filtered_acceleration_gal: FloatArray | None
    record_max_intensity_raw: FloatArray
    sampling_rate_hz: float
    window_samples: int
    duration_samples: int
    filter_name: str
    input_component_pga_gal: FloatArray
    input_pga_gal: float
    filtered_component_pga_gal: FloatArray
    filtered_pga_gal: float
    approximate_intensity_raw: float
    approximate_intensity: float
    approximate_scale: IntensityScale | None
    timing: RealtimeChunkTiming = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        for required_field in (
            "intensity_raw",
            "intensity",
            "threshold_acceleration_gal",
            "resultant_acceleration_gal",
            "record_max_intensity_raw",
            "input_component_pga_gal",
            "filtered_component_pga_gal",
        ):
            object.__setattr__(self, required_field, frozen_array(getattr(self, required_field)))
        if self.filtered_acceleration_gal is not None:
            object.__setattr__(
                self, "filtered_acceleration_gal", frozen_array(self.filtered_acceleration_gal)
            )

    @property
    def sample_count(self) -> int:
        """Return the number of output samples."""
        return int(self.intensity_raw.size)

    @property
    def record_duration_s(self) -> float:
        """Return sample count divided by sampling rate."""
        return self.sample_count / self.sampling_rate_hz

    @property
    def window_s(self) -> float:
        """Return the effective rolling-window duration in seconds."""
        return self.window_samples / self.sampling_rate_hz

    @property
    def effective_duration_s(self) -> float:
        """Return the sample-domain cumulative-duration condition in seconds."""
        return self.duration_samples / self.sampling_rate_hz

    @property
    def first_valid_sample_index(self) -> int | None:
        """Return the first sample with a defined duration threshold."""
        valid = np.flatnonzero(~np.isnan(self.intensity_raw))
        return None if valid.size == 0 else int(valid[0])

    @property
    def first_valid_time_s(self) -> float | None:
        """Return time of the first defined real-time intensity value."""
        index = self.first_valid_sample_index
        return None if index is None else index / self.sampling_rate_hz

    @property
    def peak_sample_index(self) -> int | None:
        """Return the sample at which the record maximum is first reached."""
        if np.all(np.isnan(self.intensity_raw)):
            return None
        return int(np.nanargmax(self.intensity_raw))

    @property
    def peak_time_s(self) -> float | None:
        """Return the time at which the record maximum is first reached."""
        index = self.peak_sample_index
        return None if index is None else index / self.sampling_rate_hz

    @property
    def peak_threshold_acceleration_gal(self) -> float:
        """Return the duration threshold at the peak real-time intensity."""
        index = self.peak_sample_index
        return np.nan if index is None else float(self.threshold_acceleration_gal[index])


@dataclass(frozen=True, slots=True, eq=False)
class RealtimeChunk:
    """Outputs produced from one streaming input chunk.

    Equality is disabled (``eq=False``) because several fields are arrays;
    use :func:`pyshindo.comparison.compare_results` instead. ``timing`` is
    excluded from that comparison (``compare=False``) since wall-clock
    timing is never the same between two runs even given identical input.
    """

    sample_index: npt.NDArray[np.int64]
    time_s: FloatArray
    filtered_acceleration_gal: FloatArray
    resultant_acceleration_gal: FloatArray
    threshold_acceleration_gal: FloatArray
    intensity_raw: FloatArray
    intensity: FloatArray
    record_max_intensity_raw: FloatArray
    timing: RealtimeChunkTiming = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_index", frozen_array(self.sample_index, dtype=np.int64))
        for required_field in (
            "time_s",
            "filtered_acceleration_gal",
            "resultant_acceleration_gal",
            "threshold_acceleration_gal",
            "intensity_raw",
            "intensity",
            "record_max_intensity_raw",
        ):
            object.__setattr__(self, required_field, frozen_array(getattr(self, required_field)))


@dataclass(frozen=True, slots=True, eq=False)
class RealtimeSample:
    """Low-allocation scalar output produced for one streaming sample.

    Equality is disabled (``eq=False``) because ``filtered_acceleration_gal``
    is an array; use :func:`pyshindo.comparison.compare_results` instead.
    """

    sample_index: int
    time_s: float
    filtered_acceleration_gal: FloatArray
    resultant_acceleration_gal: float
    threshold_acceleration_gal: float | None
    intensity_raw: float | None
    intensity: float | None
    scale: IntensityScale | None
    record_max_intensity_raw: float | None
    elapsed_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "filtered_acceleration_gal", frozen_array(self.filtered_acceleration_gal)
        )


@dataclass(frozen=True, slots=True)
class IntensityComparisonResult:
    """Frequency-domain reference and real-time approximation for one record."""

    measured: MeasuredIntensityResult
    realtime: RealtimeIntensityResult

    @property
    def raw_difference(self) -> float:
        """Return measured intensity minus approximate real-time intensity."""
        return self.measured.intensity_raw - self.realtime.approximate_intensity_raw

    @property
    def reported_difference(self) -> float:
        """Return reported measured intensity minus reported approximation."""
        return self.measured.intensity - self.realtime.approximate_intensity

    @property
    def absolute_raw_difference(self) -> float:
        """Return the absolute raw difference between the two methods."""
        return abs(self.raw_difference)

    @property
    def scale_agreement(self) -> bool:
        """Return whether both reported values map to the same intensity class."""
        return self.measured.scale is self.realtime.approximate_scale


@dataclass(frozen=True, slots=True)
class JMARecordMetadata:
    """Header fields from a JMA strong-motion text record."""

    station_code: str
    latitude_deg: float
    longitude_deg: float
    sampling_rate_hz: float
    unit: str
    start_time: datetime
    component_names: tuple[str, ...]
    source: str | None = None

    @property
    def acceleration_unit(self) -> AccelerationUnit:
        """Parse the declared unit into :class:`AccelerationUnit`."""
        return AccelerationUnit.parse(self.unit)


@dataclass(frozen=True, slots=True, eq=False)
class JMARecord:
    """A parsed JMA acceleration record.

    Equality is disabled (``eq=False``) because ``acceleration`` is an
    array; use :func:`pyshindo.comparison.compare_results` instead.
    """

    metadata: JMARecordMetadata
    acceleration: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "acceleration", frozen_array(self.acceleration))

    @property
    def time_s(self) -> FloatArray:
        """Return a zero-based time axis in seconds."""
        return time_axis(self.acceleration.shape[0], self.metadata.sampling_rate_hz)

    @property
    def duration_s(self) -> float:
        """Return sample count divided by sampling rate."""
        return self.acceleration.shape[0] / self.metadata.sampling_rate_hz


@dataclass(frozen=True, slots=True)
class DownloadedRecord:
    """Path and provenance returned by a data download helper."""

    path: Path
    url: str
    byte_count: int
    sha256: str
    headers: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", frozen_mapping(self.headers))


@dataclass(frozen=True, slots=True)
class ObsPyRecordMetadata:
    """Station and timing fields carried over from an ObsPy stream.

    ``unit`` is not read from the stream: SEED and its common exchange formats
    carry no reliable physical-unit field, so the caller states the unit when
    converting and it is recorded here unchanged. ``component_names`` are
    derived from the SEED orientation codes in ``channel_codes``; see
    :func:`pyshindo.obspy_interop.from_obspy_stream`.
    """

    network: str
    station_code: str
    location: str
    channel_codes: tuple[str, ...]
    component_names: tuple[str, ...]
    sampling_rate_hz: float
    start_time: datetime
    unit: str

    @property
    def acceleration_unit(self) -> AccelerationUnit:
        """Parse the declared unit into :class:`AccelerationUnit`."""
        return AccelerationUnit.parse(self.unit)


@dataclass(frozen=True, slots=True, eq=False)
class ObsPyRecord:
    """An acceleration record converted from an ObsPy stream.

    Equality is disabled (``eq=False``) because ``acceleration`` is an
    array; use :func:`pyshindo.comparison.compare_results` instead.
    """

    metadata: ObsPyRecordMetadata
    acceleration: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "acceleration", frozen_array(self.acceleration))

    @property
    def time_s(self) -> FloatArray:
        """Return a zero-based time axis in seconds."""
        return time_axis(self.acceleration.shape[0], self.metadata.sampling_rate_hz)

    @property
    def duration_s(self) -> float:
        """Return sample count divided by sampling rate."""
        return self.acceleration.shape[0] / self.metadata.sampling_rate_hz


@dataclass(frozen=True, slots=True)
class FieldDifference:
    """One field where two results compared by :func:`~pyshindo.comparison.compare_results`
    disagree.
    """

    field_name: str
    reason: str
    max_absolute_error: float | None = None
    max_relative_error: float | None = None


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """The outcome of :func:`pyshindo.comparison.compare_results`.

    Truthy exactly when ``equal`` is, so ``if compare_results(a, b):`` reads
    naturally; ``str()`` lists every differing field, for an assertion
    message or a printed diagnostic.
    """

    equal: bool
    differences: tuple[FieldDifference, ...]

    def __bool__(self) -> bool:
        return self.equal

    def __str__(self) -> str:
        if self.equal:
            return "ComparisonReport: equal"
        lines = [f"ComparisonReport: {len(self.differences)} field(s) differ"]
        for difference in self.differences:
            lines.append(f"  {difference.field_name}: {difference.reason}")
        return "\n".join(lines)
