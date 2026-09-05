from __future__ import annotations

import numpy as np
import pytest

from pyshindo import calculate_measured_intensity, peak_ground_velocity
from pyshindo.exceptions import DataFormatError, MissingComponentWarning
from pyshindo.obspy_interop import from_obspy_stream

obspy = pytest.importorskip("obspy")

SAMPLING_RATE_HZ = 100.0


def build_stream(
    channels: tuple[str, ...] = ("HNN", "HNE", "HNZ"),
    *,
    sample_count: int = 300,
    station: str = "TEST",
    **stats_overrides: object,
) -> object:
    """Build a synthetic three-component acceleration stream."""
    rng = np.random.default_rng(20260904)
    traces = []
    for index, channel in enumerate(channels):
        stats = {
            "network": "XX",
            "station": station,
            "location": "",
            "channel": channel,
            "sampling_rate": SAMPLING_RATE_HZ,
            "starttime": obspy.UTCDateTime(2026, 9, 4, 0, 0, 0),
        }
        stats.update(stats_overrides.get(channel, {}))  # type: ignore[arg-type]
        data = rng.normal(0.0, 10.0 * (index + 1), size=sample_count)
        traces.append(obspy.Trace(data=data, header=stats))
    return obspy.Stream(traces=traces)


def test_traces_are_ordered_and_labeled_from_seed_orientation_codes() -> None:
    # Deliberately out of order: the converter must sort to NS, EW, UD.
    stream = build_stream(("HNZ", "HNE", "HNN"))

    record = from_obspy_stream(stream, unit="gal")

    assert record.metadata.component_names == ("NS", "EW", "UD")
    assert record.metadata.channel_codes == ("HNN", "HNE", "HNZ")
    assert record.acceleration.shape == (300, 3)
    assert record.metadata.sampling_rate_hz == SAMPLING_RATE_HZ
    assert record.metadata.station == "TEST"
    assert record.metadata.unit == "gal"
    assert record.duration_s == pytest.approx(3.0)

    # Column order must follow the sorted channels, not the input order.
    by_channel = {str(trace.stats.channel): trace.data for trace in stream}
    for column, channel in enumerate(record.metadata.channel_codes):
        np.testing.assert_allclose(record.acceleration[:, column], by_channel[channel])


def test_unnamed_horizontals_keep_honest_h1_h2_labels() -> None:
    # SEED "1"/"2" are orthogonal horizontals at an unspecified azimuth, so
    # they must not be relabeled as if they were north and east.
    record = from_obspy_stream(build_stream(("BH1", "BH2", "BHZ")), unit="gal")

    assert record.metadata.component_names == ("H1", "H2", "UD")


def test_component_named_channels_are_recognized() -> None:
    record = from_obspy_stream(build_stream(("NS", "EW", "UD")), unit="gal")

    assert record.metadata.component_names == ("NS", "EW", "UD")


def test_unit_is_recorded_and_parsed_but_never_rescales_the_data() -> None:
    stream = build_stream()
    original = stream[0].data.copy()

    record = from_obspy_stream(stream, unit="m/s^2")

    assert record.metadata.acceleration_unit.value == "m/s^2"
    # The adapter converts nothing: the caller states the unit downstream.
    column = record.metadata.channel_codes.index(str(stream[0].stats.channel))
    np.testing.assert_allclose(record.acceleration[:, column], original)


def test_channel_order_overrides_detection_and_tolerates_unknown_codes() -> None:
    record = from_obspy_stream(
        build_stream(("ODD", "WEIRD", "HNZ")),
        unit="gal",
        channel_order=("WEIRD", "ODD", "HNZ"),
    )

    assert record.metadata.channel_codes == ("WEIRD", "ODD", "HNZ")
    assert record.metadata.component_names == ("WEIRD", "ODD", "UD")


def test_two_horizontal_components_are_allowed_with_a_warning() -> None:
    stream = build_stream(("HNN", "HNE"))

    with pytest.warns(MissingComponentWarning):
        record = from_obspy_stream(stream, unit="gal")

    assert record.acceleration.shape == (300, 2)
    # A horizontal-only PGV is exactly what a two-component record supports.
    assert peak_ground_velocity(record.acceleration, SAMPLING_RATE_HZ) > 0.0


def test_converted_record_feeds_the_intensity_calculation() -> None:
    record = from_obspy_stream(build_stream(), unit="gal")

    result = calculate_measured_intensity(
        record.acceleration,
        record.metadata.sampling_rate_hz,
        unit=record.metadata.unit,
    )

    assert np.isfinite(result.intensity_raw)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"HNE": {"sampling_rate": 50.0}}, "sampling rate"),
        ({"HNE": {"starttime": obspy.UTCDateTime(2026, 9, 4, 0, 0, 5)}}, "start at the same time"),
        ({"HNE": {"station": "OTHER"}}, "one station"),
    ],
)
def test_inconsistent_traces_are_rejected(overrides: dict, message: str) -> None:
    stream = build_stream(**overrides)

    with pytest.raises(DataFormatError, match=message):
        from_obspy_stream(stream, unit="gal")


def test_a_fraction_of_a_sample_of_offset_is_also_rejected() -> None:
    # 4.9 ms at 100 Hz: under the old half-sample tolerance this was silently
    # accepted, even though it already distorts the vector resultant.
    stream = build_stream(
        HNE={"starttime": obspy.UTCDateTime(2026, 9, 4, 0, 0, 0, 4900)},
    )
    with pytest.raises(DataFormatError, match="start at the same time"):
        from_obspy_stream(stream, unit="gal")


def test_zero_sampling_rate_is_rejected_with_a_clear_message() -> None:
    zero_rate = {"sampling_rate": 0.0}
    stream = build_stream(HNN=zero_rate, HNE=zero_rate, HNZ=zero_rate)
    with pytest.raises(DataFormatError, match="sampling rate"):
        from_obspy_stream(stream, unit="gal")


def test_empty_channel_order_is_rejected_with_a_clear_message() -> None:
    stream = build_stream()
    with pytest.raises(DataFormatError, match="channel_order must not be empty"):
        from_obspy_stream(stream, unit="gal", channel_order=())


def test_two_traces_sharing_a_channel_code_are_rejected() -> None:
    stream = build_stream(("HNN", "HNN", "HNZ"))
    with pytest.raises(DataFormatError, match="more than one trace with channel code"):
        from_obspy_stream(stream, unit="gal", channel_order=("HNN", "HNZ"))


def test_mismatched_lengths_are_rejected() -> None:
    stream = build_stream()
    stream[1].data = stream[1].data[:-10]

    with pytest.raises(DataFormatError, match="same number of samples"):
        from_obspy_stream(stream, unit="gal")


def test_duplicate_orientation_is_rejected_without_comparing_traces() -> None:
    # Two traces of the same orientation must produce the package's own error,
    # not a TypeError from sorting Trace objects against each other.
    stream = build_stream(("HNN", "HN1", "HNZ"))

    with pytest.raises(DataFormatError, match="same orientation"):
        from_obspy_stream(stream, unit="gal")


def test_masked_gaps_are_rejected() -> None:
    stream = build_stream()
    stream[0].data = np.ma.masked_array(stream[0].data, mask=np.zeros(300, dtype=bool))
    stream[0].data.mask[10:20] = True

    with pytest.raises(DataFormatError, match="masked gaps"):
        from_obspy_stream(stream, unit="gal")


def test_too_many_traces_ask_for_an_explicit_selection() -> None:
    stream = build_stream() + build_stream(station="OTHER")

    with pytest.raises(DataFormatError, match="stream.select"):
        from_obspy_stream(stream, unit="gal")


def test_empty_stream_is_rejected() -> None:
    with pytest.raises(DataFormatError, match="no traces"):
        from_obspy_stream(obspy.Stream(), unit="gal")


def test_channel_order_reports_missing_channels() -> None:
    with pytest.raises(DataFormatError, match="channel_order requested"):
        from_obspy_stream(build_stream(), unit="gal", channel_order=("HNN", "NOPE"))
