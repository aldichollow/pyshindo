"""Result and filter-design data models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .scale import IntensityScale
from .signal import time_axis
from .units import AccelerationUnit, FloatArray


@dataclass(frozen=True, slots=True)
class FrequencyResponse:
    """Complex filter response sampled at physical frequencies."""

    frequency_hz: FloatArray
    response: npt.NDArray[np.complex128]

    @property
    def amplitude(self) -> FloatArray:
        """Return the response magnitude."""
        return np.abs(self.response)

    @property
    def phase_rad(self) -> FloatArray:
        """Return the unwrapped response phase in radians."""
        return np.unwrap(np.angle(self.response))


@dataclass(frozen=True, slots=True)
class FilterStage:
    """One named, individually inspectable component of a filter cascade.

    ``sos`` is a single second-order-section row (shape ``(6,)``), in the same
    normalized form as one row of :attr:`RecursiveFilterDesign.sos`: first-order
    stages simply have zero for their second-order coefficients. Cascading every
    stage of a design, in order, reproduces that design's combined response --
    stages exist so each named component can be inspected or plotted on its own,
    not as an alternative way to filter data.
    """

    name: str
    characteristic_frequency_hz: float | None
    sos: FloatArray


@dataclass(frozen=True, slots=True)
class RecursiveFilterDesign:
    """Second-order-section representation of a published approximation filter.

    ``sos`` stays writable: :func:`scipy.signal.sosfilt` requires a mutable
    buffer even though it only reads it. A design is meant to be treated as
    immutable configuration all the same -- mutating ``sos`` after handing a
    design to :class:`~pyshindo.realtime.RealtimeIntensityEstimator` is
    unsupported, since the estimator takes its own private copy precisely to
    stay correct if a caller does this.
    """

    name: str
    sampling_rate_hz: float
    sos: FloatArray
    parameters: dict[str, float]
    characteristic_frequencies_hz: tuple[float, ...]
    max_pole_radius: float
    stable: bool
    stages: tuple[FilterStage, ...]

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


@dataclass(frozen=True, slots=True)
class AmplitudeDurationCurve:
    """Amplitude sorted from high to low and its cumulative exceedance duration."""

    amplitude: FloatArray
    exceedance_duration_s: FloatArray


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


@dataclass(frozen=True, slots=True)
class MeasuredIntensityResult:
    """Detailed output of the frequency-domain reference calculation."""

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
    timing: MeasuredIntensityTiming

    @property
    def record_duration_s(self) -> float:
        """Return sample count divided by sampling rate."""
        return self.sample_count / self.sampling_rate_hz


@dataclass(frozen=True, slots=True)
class RealtimeChunkTiming:
    """Wall-clock timing for one :meth:`RealtimeIntensityEstimator.process` call.

    ``filter_s``: the compiled ``sosfilt`` pass. ``order_statistic_s``: the
    per-sample rolling-threshold loop (usually the larger share for long
    chunks). ``total_s``: the complete call.
    """

    filter_s: float
    order_statistic_s: float
    total_s: float


@dataclass(frozen=True, slots=True)
class RealtimeIntensityResult:
    """Detailed output of a batch replay of the real-time algorithm."""

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
    timing: RealtimeChunkTiming

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


@dataclass(frozen=True, slots=True)
class RealtimeChunk:
    """Outputs produced from one streaming input chunk."""

    sample_index: npt.NDArray[np.int64]
    time_s: FloatArray
    filtered_acceleration_gal: FloatArray
    resultant_acceleration_gal: FloatArray
    threshold_acceleration_gal: FloatArray
    intensity_raw: FloatArray
    intensity: FloatArray
    record_max_intensity_raw: FloatArray
    timing: RealtimeChunkTiming


@dataclass(frozen=True, slots=True)
class RealtimeSample:
    """Low-allocation scalar output produced for one streaming sample."""

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


@dataclass(frozen=True, slots=True)
class JMARecord:
    """A parsed JMA acceleration record."""

    metadata: JMARecordMetadata
    acceleration: FloatArray

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
    headers: dict[str, Any]


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
    station: str
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


@dataclass(frozen=True, slots=True)
class ObsPyRecord:
    """An acceleration record converted from an ObsPy stream."""

    metadata: ObsPyRecordMetadata
    acceleration: FloatArray

    @property
    def time_s(self) -> FloatArray:
        """Return a zero-based time axis in seconds."""
        return time_axis(self.acceleration.shape[0], self.metadata.sampling_rate_hz)

    @property
    def duration_s(self) -> float:
        """Return sample count divided by sampling rate."""
        return self.acceleration.shape[0] / self.metadata.sampling_rate_hz
