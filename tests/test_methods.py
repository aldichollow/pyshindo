from __future__ import annotations

import pytest

from pyshindo.methods import (
    MethodDescriptor,
    MethodStatus,
    available_method_ids,
    method_descriptor,
)


def test_available_method_ids_is_sorted_and_nonempty() -> None:
    ids = available_method_ids()
    assert len(ids) > 0
    assert list(ids) == sorted(ids)


@pytest.mark.parametrize("method_id", available_method_ids())
def test_every_registered_descriptor_is_internally_consistent(method_id: str) -> None:
    descriptor = method_descriptor(method_id)
    assert isinstance(descriptor, MethodDescriptor)
    assert descriptor.method_id == method_id
    assert descriptor.method_name  # non-empty
    assert isinstance(descriptor.status, MethodStatus)
    assert isinstance(descriptor.primary_references, tuple)
    assert isinstance(descriptor.deviations, tuple)
    assert isinstance(descriptor.package_decisions, tuple)
    # A REFERENCE or PUBLISHED_APPROXIMATION method should cite something; a
    # GENERAL_ENGINEERING_METHOD may legitimately cite nothing seismology-specific.
    if descriptor.status in (MethodStatus.REFERENCE, MethodStatus.PUBLISHED_APPROXIMATION):
        assert descriptor.primary_references


def test_method_descriptor_rejects_an_unknown_id_with_a_readable_message() -> None:
    with pytest.raises(ValueError, match="Unknown method_id 'not_a_real_method'"):
        method_descriptor("not_a_real_method")


def test_method_descriptor_error_lists_known_ids() -> None:
    with pytest.raises(ValueError, match="jma_measured_intensity"):
        method_descriptor("nonexistent")


def test_jma_measured_intensity_is_classified_as_reference() -> None:
    descriptor = method_descriptor("jma_measured_intensity")
    assert descriptor.status is MethodStatus.REFERENCE


def test_kunugi_filters_are_classified_as_published_approximations() -> None:
    for method_id in (
        "kunugi_realtime_intensity_2008",
        "kunugi_realtime_intensity_2012",
        "kunugi_lowrate_realtime_intensity",
    ):
        assert method_descriptor(method_id).status is MethodStatus.PUBLISHED_APPROXIMATION


def test_suzuki_preset_is_classified_as_a_package_extension_not_reference() -> None:
    # It reproduces one paper's specific geometry choice within a general
    # method (IDW); it is not itself JMA's or anyone else's reference method.
    descriptor = method_descriptor("suzuki_2017_pga_idw_preset")
    assert descriptor.status is MethodStatus.PACKAGE_EXTENSION


def test_general_response_spectrum_is_classified_as_general_engineering_method() -> None:
    descriptor = method_descriptor("general_elastic_response_spectrum")
    assert descriptor.status is MethodStatus.GENERAL_ENGINEERING_METHOD
    assert descriptor.primary_references == ()


def test_method_descriptor_instances_are_frozen() -> None:
    descriptor = method_descriptor("jma_measured_intensity")
    with pytest.raises(AttributeError):
        descriptor.method_name = "something else"  # type: ignore[misc]
