"""Conversion from ObsPy streams into the arrays this package expects.

ObsPy already reads the strong-motion formats this package does not
(K-NET/KiK-net, miniSEED, SAC, and the rest of ``obspy.read``), and already
does instrument response removal. Rather than reimplement any of that, this
module is a thin adapter: it takes a stream whose traces are *already in
acceleration units* and returns the ``(samples, components)`` array,
sampling rate, and station metadata used everywhere else in
:mod:`pyshindo`. It never resamples, trims, merges, rotates, or rescales --
each of those would change the record, and ObsPy exposes them explicitly
(:meth:`~obspy.core.stream.Stream.resample`,
:meth:`~obspy.core.stream.Stream.trim`,
:meth:`~obspy.core.stream.Stream.merge`,
:meth:`~obspy.core.stream.Stream.remove_response`) for the caller to apply
first.

Columns come out in horizontal, horizontal, vertical order, taken from the
SEED orientation code. ``"1"``/``"2"`` are SEED's codes for two orthogonal
horizontals at an unspecified azimuth, so they keep the labels ``"H1"``/
``"H2"`` rather than being renamed as if they pointed north and east.
Nothing is lost by that: the calculations combine the components with a
Euclidean norm, which is invariant to rotation within the horizontal plane.

Requires the optional dependency group: ``pip install "pyshindo[obspy]"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from .exceptions import DataFormatError
from .models import ObsPyRecord, ObsPyRecordMetadata
from .units import AccelerationUnit
from .validation import as_acceleration_array

_ORIENTATION_CODES: Final = {
    "N": (0, "NS"),
    "1": (0, "H1"),
    "E": (1, "EW"),
    "2": (1, "H2"),
    "Z": (2, "UD"),
    "3": (2, "UD"),
}
# Some readers put the component name in the channel field outright rather
# than a SEED code; ObsPy's K-NET reader is the common case.
_CHANNEL_NAMES: Final = {
    "NS": (0, "NS"),
    "EW": (1, "EW"),
    "UD": (2, "UD"),
}


def require_obspy() -> Any:
    """Import ObsPy lazily and return the module."""
    try:
        import obspy
    except ImportError as exc:
        raise ImportError(
            "ObsPy interoperability requires the optional dependency group: "
            'pip install "pyshindo[obspy]".'
        ) from exc
    return obspy


def _classify_channel(channel: str) -> tuple[int, str]:
    """Return the ordering rank and component label for a channel code."""
    code = channel.strip().upper()
    if code in _CHANNEL_NAMES:
        return _CHANNEL_NAMES[code]
    if code and code[-1] in _ORIENTATION_CODES:
        return _ORIENTATION_CODES[code[-1]]
    raise DataFormatError(
        f"Cannot determine the component orientation of channel {channel!r}. "
        "Expected a SEED orientation code (N/E/Z, 1/2, or 3) as the last "
        "character, or a channel named NS/EW/UD. Pass channel_order to state "
        "the order explicitly."
    )


def _component_label(channel: str) -> str:
    """Return a display label for a channel, falling back to the code itself.

    Used only when ``channel_order`` already fixed the column order, so an
    orientation code this module does not recognize is no longer an error --
    the caller has stated the order, and the raw code is a truthful label.
    """
    try:
        return _classify_channel(channel)[1]
    except DataFormatError:
        return channel.strip().upper()


def _selected_traces(stream: Any, channel_order: Sequence[str] | None) -> list[Any]:
    """Return the traces to convert, ordered horizontal-horizontal-vertical."""
    traces = list(stream)
    if not traces:
        raise DataFormatError("The stream contains no traces.")
    if channel_order is not None:
        wanted = [code.strip().upper() for code in channel_order]
        if len(set(wanted)) != len(wanted):
            raise DataFormatError("channel_order must not repeat a channel code.")
        by_channel = {str(trace.stats.channel).strip().upper(): trace for trace in traces}
        missing = [code for code in wanted if code not in by_channel]
        if missing:
            available = ", ".join(sorted(by_channel)) or "none"
            raise DataFormatError(
                f"channel_order requested {missing} but the stream has: {available}."
            )
        return [by_channel[code] for code in wanted]

    if len(traces) > 3:
        raise DataFormatError(
            f"The stream contains {len(traces)} traces. Select one station's three "
            "components first, for example stream.select(station='...', channel='HN?')."
        )
    ranks = [_classify_channel(str(trace.stats.channel))[0] for trace in traces]
    if len(set(ranks)) != len(ranks):
        channels = ", ".join(str(trace.stats.channel) for trace in traces)
        raise DataFormatError(
            f"The stream has more than one trace for the same orientation: {channels}."
        )
    # Sort by key alone: sorting (key, trace) pairs would fall through to
    # comparing Trace objects, which raises TypeError instead of anything useful.
    return sorted(traces, key=lambda trace: _classify_channel(str(trace.stats.channel))[0])


def _validate_consistency(traces: Sequence[Any]) -> None:
    """Reject streams whose traces do not describe one aligned recording."""
    first = traces[0].stats
    sample_interval = 1.0 / float(first.sampling_rate)
    for trace in traces:
        stats = trace.stats
        if np.ma.isMaskedArray(trace.data):
            raise DataFormatError(
                f"Trace {stats.channel} contains masked gaps. Fill or merge them first, "
                "for example stream.merge(fill_value='interpolate')."
            )
        if not np.isclose(float(stats.sampling_rate), float(first.sampling_rate), rtol=1e-9):
            raise DataFormatError(
                "All traces must share one sampling rate; found "
                f"{first.sampling_rate} Hz and {stats.sampling_rate} Hz. "
                "Resample explicitly before converting."
            )
        if int(stats.npts) != int(first.npts):
            raise DataFormatError(
                "All traces must have the same number of samples; found "
                f"{first.npts} and {stats.npts}. Trim explicitly before converting, "
                "for example stream.trim(starttime, endtime)."
            )
        # Half a sample: anything larger would misalign the components against
        # each other, which silently changes the vector resultant.
        if abs(float(stats.starttime - first.starttime)) > 0.5 * sample_interval:
            raise DataFormatError(
                "All traces must start at the same time; found "
                f"{first.starttime} and {stats.starttime}. Trim explicitly before "
                "converting."
            )
        for field in ("network", "station", "location"):
            if str(getattr(stats, field)) != str(getattr(first, field)):
                raise DataFormatError(
                    f"All traces must come from one station; {field} differs "
                    f"({getattr(first, field)!r} and {getattr(stats, field)!r}). "
                    "Select one station first."
                )


def from_obspy_stream(
    stream: Any,
    *,
    unit: str | AccelerationUnit,
    channel_order: Sequence[str] | None = None,
    allow_fewer_components: bool = True,
) -> ObsPyRecord:
    """Convert an ObsPy stream of acceleration traces into an :class:`ObsPyRecord`.

    Parameters
    ----------
    stream:
        An :class:`obspy.core.stream.Stream` holding up to three traces from a
        single station, already converted to acceleration. Traces are ordered
        horizontal, horizontal, vertical from their SEED orientation codes.
    unit:
        The acceleration unit of the trace data: ``"gal"``, ``"m/s^2"``, or
        ``"g"``. This is required rather than detected because SEED and the
        formats around it carry no dependable physical-unit field -- ObsPy
        returns whatever the reader produced, which for a response-removed
        stream is whatever ``output=`` asked for. Stating it here records the
        caller's assertion in the metadata instead of guessing.
    channel_order:
        Explicit channel codes, in the order they should become columns. Use
        this when the orientation cannot be read from the channel names.
    allow_fewer_components:
        Permit one or two components. The standard intensity calculation needs
        all three, and will reject fewer itself; two horizontals are still
        useful for a horizontal PGV.

    Returns
    -------
    ObsPyRecord
        ``record.acceleration`` is ``(samples, components)`` float64 in the
        stated unit, and ``record.metadata`` carries the station fields, the
        original channel codes, and the derived component labels.

    Notes
    -----
    The stream is only read, never modified. Traces that disagree on sampling
    rate, length, start time, or station are rejected rather than reconciled,
    because aligning them is a change to the record that the caller should make
    deliberately.
    """
    require_obspy()
    parsed_unit = AccelerationUnit.parse(unit)
    traces = _selected_traces(stream, channel_order)
    _validate_consistency(traces)

    columns = [np.asarray(trace.data, dtype=np.float64) for trace in traces]
    values = as_acceleration_array(
        np.column_stack(columns),
        allow_fewer_components=allow_fewer_components,
    )
    stats = traces[0].stats
    metadata = ObsPyRecordMetadata(
        network=str(stats.network),
        station=str(stats.station),
        location=str(stats.location),
        channel_codes=tuple(str(trace.stats.channel) for trace in traces),
        component_names=tuple(_component_label(str(trace.stats.channel)) for trace in traces),
        sampling_rate_hz=float(stats.sampling_rate),
        start_time=stats.starttime.datetime,
        unit=parsed_unit.value,
    )
    return ObsPyRecord(metadata=metadata, acceleration=values)
