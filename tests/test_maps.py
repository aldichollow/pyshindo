from __future__ import annotations

import pytest

from pyshindo.long_period import LongPeriodClass
from pyshindo.plotting import (
    continuous_value_map_figure,
    intensity_map_figure,
    long_period_class_map_figure,
)
from pyshindo.scale import IntensityScale

LAT = [35.6, 35.8, 36.0, 35.9]
LON = [139.7, 139.9, 140.1, 139.95]


def test_intensity_map_groups_one_trace_per_present_class() -> None:
    figure = intensity_map_figure(
        LAT, LON, [IntensityScale.FIVE_LOWER, "3", IntensityScale.TWO, "3"]
    )
    # Two stations share class "3", so it is one named trace with two points, not two.
    # Each named trace also gets an unnamed halo trace drawn behind it.
    by_name = {trace.name: trace for trace in figure.data if trace.name}
    assert set(by_name) == {"震度2", "震度3", "震度5弱"}
    assert len(by_name["震度3"].lat) == 2
    assert len(figure.data) == 2 * len(by_name)


def test_intensity_map_accepts_strings_and_enum_members_interchangeably() -> None:
    from_strings = intensity_map_figure(LAT[:2], LON[:2], ["2", "3"])
    from_enum = intensity_map_figure(
        LAT[:2], LON[:2], [IntensityScale.TWO, IntensityScale.THREE]
    )
    string_names = {t.name for t in from_strings.data if t.name}
    enum_names = {t.name for t in from_enum.data if t.name}
    assert string_names == enum_names


def test_intensity_map_uses_the_jma_color_guide() -> None:
    from pyshindo.plotting.theme import JMA_INTENSITY_COLORS

    figure = intensity_map_figure(LAT[:1], LON[:1], [IntensityScale.SEVEN])
    named = next(trace for trace in figure.data if trace.name)
    assert named.marker.color == JMA_INTENSITY_COLORS[IntensityScale.SEVEN]


def test_intensity_map_markers_have_a_white_halo_behind_them() -> None:
    figure = intensity_map_figure(LAT[:1], LON[:1], [IntensityScale.SEVEN])
    halo = next(trace for trace in figure.data if not trace.name)
    assert halo.marker.color == "#FFFFFF"
    assert halo.showlegend is False


def test_halo_trace_shares_a_legendgroup_with_its_class_so_toggling_hides_both() -> None:
    figure = intensity_map_figure(LAT[:1], LON[:1], [IntensityScale.SEVEN])
    halo = next(trace for trace in figure.data if not trace.name)
    named = next(trace for trace in figure.data if trace.name)
    assert halo.legendgroup == named.legendgroup
    assert named.legendgroup


def test_long_period_class_map_groups_one_trace_per_present_class() -> None:
    figure = long_period_class_map_figure(
        LAT, LON, [LongPeriodClass.ZERO, "1", LongPeriodClass.ZERO, "2"]
    )
    by_name = {trace.name: trace for trace in figure.data if trace.name}
    assert set(by_name) == {"階級0", "階級1", "階級2"}
    assert len(by_name["階級0"].lat) == 2


def test_continuous_value_map_is_a_single_colorscaled_trace() -> None:
    figure = continuous_value_map_figure(LAT, LON, [1.0, 5.0, 12.0, 3.0], value_label="SI [cm/s]")
    # One halo trace plus the colorscaled data trace.
    assert len(figure.data) == 2
    marker = figure.data[-1].marker
    assert marker.showscale is True
    assert marker.colorbar.title.text == "SI [cm/s]"
    assert list(marker.color) == [1.0, 5.0, 12.0, 3.0]


def test_labels_appear_in_hover_text() -> None:
    figure = continuous_value_map_figure(
        LAT[:2], LON[:2], [1.0, 2.0], value_label="x", labels=["Station A", "Station B"]
    )
    assert list(figure.data[-1].text) == ["Station A", "Station B"]


def test_map_is_centered_and_zoomed_to_fit_every_station() -> None:
    figure = continuous_value_map_figure(LAT, LON, [1.0, 2.0, 3.0, 4.0], value_label="x")
    center = figure.layout.map.center
    assert center.lat == pytest.approx(sum(LAT) / len(LAT))
    assert center.lon == pytest.approx(sum(LON) / len(LON))
    assert figure.layout.map.zoom > 0


def test_a_single_station_still_produces_a_valid_figure() -> None:
    figure = continuous_value_map_figure([35.6], [139.7], [10.0], value_label="x")
    assert figure.layout.map.zoom > 0


def test_an_empty_station_list_still_produces_a_valid_figure() -> None:
    figure = continuous_value_map_figure([], [], [], value_label="x")
    assert len(figure.data) == 2  # the halo trace plus the (empty) data trace
    assert figure.layout.map.zoom > 0
    assert figure.layout.map.center.lat == pytest.approx(36.0)


def test_stations_at_the_same_point_do_not_error_on_zero_span() -> None:
    figure = continuous_value_map_figure([35.6, 35.6], [139.7, 139.7], [1.0, 2.0], value_label="x")
    assert figure.layout.map.zoom > 0


def test_mismatched_coordinate_and_value_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        continuous_value_map_figure(LAT, LON[:-1], [1.0, 2.0, 3.0, 4.0], value_label="x")


def test_mismatched_label_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="labels must contain"):
        continuous_value_map_figure(LAT, LON, [1.0, 2.0, 3.0, 4.0], value_label="x", labels=["A"])


def test_out_of_range_latitude_is_rejected() -> None:
    with pytest.raises(ValueError, match="latitudes_deg"):
        continuous_value_map_figure([95.0], [139.7], [1.0], value_label="x")


def test_non_finite_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        continuous_value_map_figure([float("nan")], [139.7], [1.0], value_label="x")


def test_continuous_values_must_be_one_dimensional() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        continuous_value_map_figure(LAT, LON, [[1.0, 2.0]] * 4, value_label="x")


def test_unknown_intensity_class_is_rejected() -> None:
    with pytest.raises(ValueError):
        intensity_map_figure(LAT[:1], LON[:1], ["not-a-class"])


def test_default_basemap_style_is_a_muted_grayscale_style() -> None:
    figure = continuous_value_map_figure(LAT, LON, [1.0, 2.0, 3.0, 4.0], value_label="x")
    assert figure.layout.map.style == "carto-positron"


def test_map_style_is_overridable() -> None:
    figure = continuous_value_map_figure(
        LAT, LON, [1.0, 2.0, 3.0, 4.0], value_label="x", map_style="open-street-map"
    )
    assert figure.layout.map.style == "open-street-map"


def test_hover_label_text_is_readable_against_its_white_background() -> None:
    figure = continuous_value_map_figure(LAT, LON, [1.0, 2.0, 3.0, 4.0], value_label="x")
    assert figure.layout.hoverlabel.bgcolor == "#FFFFFF"
    assert figure.layout.hoverlabel.font.color == "#1A1A1A"
