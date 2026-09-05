from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyshindo.filters import design_realtime_filter
from pyshindo.realtime import RealtimeIntensityEstimator, calculate_realtime_intensity


def _record() -> np.ndarray:
    generator = np.random.default_rng(8)
    time = np.arange(12_000, dtype=np.float64) / 100.0
    burst = np.exp(-0.5 * ((time - 25.0) / 3.0) ** 2)
    deterministic = np.column_stack(
        [
            110.0 * burst * np.sin(2 * np.pi * 1.1 * time),
            70.0 * burst * np.sin(2 * np.pi * 2.4 * time + 0.3),
            35.0 * burst * np.sin(2 * np.pi * 6.0 * time + 0.8),
        ]
    )
    return deterministic + generator.normal(scale=0.02, size=deterministic.shape)


def test_chunking_does_not_change_realtime_output() -> None:
    values = _record()
    whole = RealtimeIntensityEstimator().process(values)

    estimator = RealtimeIntensityEstimator()
    chunks = []
    boundaries = [1, 7, 31, 100, 999, 1024, 4096, values.shape[0]]
    start = 0
    for stop in boundaries:
        if stop <= start:
            continue
        chunks.append(estimator.process(values[start:stop]))
        start = stop
    if start < values.shape[0]:
        chunks.append(estimator.process(values[start:]))

    filtered = np.concatenate([chunk.filtered_acceleration_gal for chunk in chunks], axis=0)
    threshold = np.concatenate([chunk.threshold_acceleration_gal for chunk in chunks])
    intensity = np.concatenate([chunk.intensity_raw for chunk in chunks])
    assert np.array_equal(filtered, whole.filtered_acceleration_gal)
    assert np.array_equal(threshold, whole.threshold_acceleration_gal, equal_nan=True)
    assert np.array_equal(intensity, whole.intensity_raw, equal_nan=True)


def test_batch_result_reports_maximum() -> None:
    result = calculate_realtime_intensity(_record())
    assert np.isfinite(result.approximate_intensity_raw)
    assert result.approximate_intensity_raw == pytest.approx(np.nanmax(result.intensity_raw))
    assert result.approximate_scale is not None
    assert result.duration_samples == 30
    assert result.window_samples == 6000


def test_nonstandard_stable_rate_warns_but_calculates() -> None:
    values = np.zeros((800, 3))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = calculate_realtime_intensity(values, sampling_rate_hz=80.0)
    assert result.intensity_raw.shape == (800,)
    assert captured


def test_mutating_a_shared_filter_design_after_construction_is_isolated() -> None:
    design = design_realtime_filter(100.0, filter_name="kunugi-2012")
    reference = RealtimeIntensityEstimator(
        100.0, filter_design=design_realtime_filter(100.0, filter_name="kunugi-2012")
    )
    estimator = RealtimeIntensityEstimator(100.0, filter_design=design)
    values = _record()[:500]

    reference_output = np.array(
        [reference.process_sample(sample).filtered_acceleration_gal for sample in values]
    )
    design.sos[0, 0] *= 3.0  # mutate the caller's copy after the estimator was built
    chunk_output = estimator.process(values.copy()).filtered_acceleration_gal

    np.testing.assert_allclose(chunk_output, reference_output, rtol=1e-9, atol=1e-9)


def test_single_sample_path_matches_chunk_path() -> None:
    values = _record()[:1000]
    chunk = RealtimeIntensityEstimator().process(values)

    estimator = RealtimeIntensityEstimator()
    filtered = []
    resultant = []
    threshold = []
    intensity = []
    for sample in values:
        output = estimator.process_sample(sample)
        filtered.append(output.filtered_acceleration_gal)
        resultant.append(output.resultant_acceleration_gal)
        threshold.append(
            np.nan
            if output.threshold_acceleration_gal is None
            else output.threshold_acceleration_gal
        )
        intensity.append(np.nan if output.intensity_raw is None else output.intensity_raw)

    np.testing.assert_allclose(
        np.asarray(filtered),
        chunk.filtered_acceleration_gal,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(resultant),
        chunk.resultant_acceleration_gal,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(threshold),
        chunk.threshold_acceleration_gal,
        equal_nan=True,
    )
    np.testing.assert_allclose(np.asarray(intensity), chunk.intensity_raw, equal_nan=True)


def test_chunk_and_sample_timing_are_reported() -> None:
    values = _record()[:500]
    chunk = RealtimeIntensityEstimator().process(values)
    assert chunk.timing.filter_s >= 0.0
    assert chunk.timing.order_statistic_s >= 0.0
    assert chunk.timing.total_s >= chunk.timing.filter_s + chunk.timing.order_statistic_s

    estimator = RealtimeIntensityEstimator()
    sample = estimator.process_sample(values[0])
    assert sample.elapsed_s >= 0.0


def test_interleaved_chunk_and_sample_calls_match_pure_chunk_path() -> None:
    """process() and process_sample() share one authoritative filter state
    (self._zi), so freely mixing them on one estimator must stay exact."""
    values = _record()[:1000]
    whole = RealtimeIntensityEstimator().process(values)

    estimator = RealtimeIntensityEstimator()
    filtered: list[np.ndarray] = []
    resultant: list[float] = []
    threshold: list[float] = []
    intensity: list[float] = []
    segments = [values[0:1], values[1:50], values[50:51], values[51:400], values[400:1000]]
    use_sample_api = False
    for segment in segments:
        if use_sample_api:
            for sample in segment:
                output = estimator.process_sample(sample)
                filtered.append(output.filtered_acceleration_gal)
                resultant.append(output.resultant_acceleration_gal)
                threshold.append(
                    np.nan
                    if output.threshold_acceleration_gal is None
                    else output.threshold_acceleration_gal
                )
                intensity.append(np.nan if output.intensity_raw is None else output.intensity_raw)
        else:
            chunk = estimator.process(segment)
            filtered.extend(chunk.filtered_acceleration_gal)
            resultant.extend(chunk.resultant_acceleration_gal)
            threshold.extend(chunk.threshold_acceleration_gal)
            intensity.extend(chunk.intensity_raw)
        use_sample_api = not use_sample_api

    np.testing.assert_allclose(
        np.asarray(filtered), whole.filtered_acceleration_gal, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(resultant), whole.resultant_acceleration_gal, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(threshold), whole.threshold_acceleration_gal, equal_nan=True
    )
    np.testing.assert_allclose(np.asarray(intensity), whole.intensity_raw, equal_nan=True)


def test_comparison_reports_scale_agreement_property() -> None:
    from pyshindo import compare_intensity_methods

    comparison = compare_intensity_methods(_record())
    assert isinstance(comparison.scale_agreement, bool)
    assert comparison.absolute_raw_difference == abs(comparison.raw_difference)
