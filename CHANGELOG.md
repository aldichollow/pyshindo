# Changelog

## 0.3.2 - 2026-10-07

Added:

- `pyshindo.methods` -- `method_descriptor`/`available_method_ids` classify each algorithm as a reference method, a published approximation, a package extension, or a general engineering method, with its primary references and deviations.
- `compare_results` -- field-by-field comparison for any two result objects (or filter designs), honoring `rtol`/`atol` and skipping non-comparable fields such as timing.
- `CITATION.cff` and `docs/scope.md` (what the package covers, what it deliberately does not, and its actual API-stability practice).

Changed (breaking):

- Every result and filter-design dataclass with an array or dict field now freezes it read-only after construction; mutating one in place (for example `result.some_array[0] = x`) now raises `ValueError` instead of silently succeeding.
- The same dataclasses' auto-generated `==` (which raised `ValueError` for any array field) is disabled; use `compare_results` instead.

Fixed:

- `report_intensity_array` is substantially faster on large arrays, with no change in output.

## 0.3.1 - 2026-09-26

Added:

- `pyshindo.spatial` -- interpolates station observations (PGV, PGA, measured intensity, or any continuous value) onto a grid with inverse-distance weighting, Delaunay-linear, or nearest-neighbor methods. NumPy/SciPy only, no new dependency. Neighbor searches run across all available CPU cores.
- `pyshindo.plotting.surfaces` -- renders an interpolated surface as a land-only map layer under station markers, using a bundled Natural Earth coastline raster. No new dependency; Shapely is used only by a maintainer-only build script, never at runtime.
- `color_transform`, `cmin`, and `cmax` on `continuous_value_map_figure`, for a log-scaled or fixed-range marker colorbar.

Changed:

- Map figure markers are slightly smaller, with a dark gray halo instead of white.

## 0.3.0 - 2026-09-14

First release published to PyPI, with the public API reviewed for consistency. See [`docs/migration.md`](docs/migration.md) for the upgrade path -- every breaking change is listed there with before/after code.

Renamed (breaking):

- `intensity_from_acceleration`, `acceleration_from_intensity`, `intensity_series_from_acceleration` -> `..._threshold_acceleration` forms. They take a 0.3-second threshold acceleration, not a waveform.
- `calculate_spectrum_intensity`'s `retain_spectrum` -> `retain_velocity_time_series`; it never controlled `sv_cm_s`, which is always returned.
- `published_lowrate_gamma_set`'s `policy` -> `lowrate_gamma_policy`, matching `design_realtime_filter`.
- `ObsPyRecordMetadata.station` -> `station_code`; `LongPeriodResult.absolute_velocity_cm_s` -> `absolute_velocity_time_series_cm_s`; `LongPeriodUpdate.class_so_far` -> `long_period_class_so_far`.

Changed (breaking):

- `realtime_intensity`'s `reported` now defaults to `True`, matching `measured_intensity`. The two returned different quantities under the same keyword.
- `peak_ground_acceleration` and `component_peak_acceleration` take `unit=` and always return gal, matching the PGV and PGD pairs.
- `apply_jma_filter_fft` returns `JMAFilterResult` instead of a bare tuple, takes `unit=`, and defaults `sampling_rate_hz` to 100.0.
- `scale_acceleration_to_intensity`'s `target_intensity_raw` is keyword-only, so the second positional argument is `sampling_rate_hz` as everywhere else.
- `SpectrumIntensityEstimator` no longer warns on a non-100 Hz rate, and its `warn_nonstandard_rate` argument is gone; the batch function never warned.
- `RealtimeIntensityEstimator`'s configuration attributes are read-only; assigning to them silently desynced the filter design from the rate.

Added:

- Every exception and warning class, `pyshindo.long_period`, `MeasuredIntensityTiming`, `RealtimeChunkTiming`, `report_intensity_array`, and `intensity_series_from_threshold_acceleration` are reachable from `pyshindo` directly.
- `RealtimeIntensityEstimator.approximate_scale`, the running intensity class.

Fixed:

- `NonstandardSamplingRateWarning`, `MissingComponentWarning`, and `FractionalDurationWarning` now point at the caller's line instead of pyshindo's own source.

Docs and packaging:

- `docs/migration.md`; documented why `RealtimeIntensityEstimator` has no `result()`.
- `Typing :: Typed` classifier; README links are absolute so they resolve on PyPI.

## 0.2.2 - 2026-09-13

- Added: peak ground displacement -- `integrate_to_displacement`, `component_peak_displacement`, `peak_ground_displacement`, alongside the existing velocity trio.
- Added: `pyshindo.strong_motion` -- JMA's own published velocity/displacement waveform filters. `apply_strong_motion_displacement_filter` reproduces JMA's mechanical 1x strong-motion seismometer response directly from acceleration (no integration involved) and matches published peak displacement to ~0.15% median error, the recommended way to reproduce a long-period observation page's PGD.
- Added: `calculate_response_spectrum` also returns pseudo-velocity (PSV); its displacement and velocity time series are now independent `retain_*` flags.
- Added: `RealtimeChunkTiming.reporting_s` exposes the display-rounding cost.
- Added: `marker_size` on the map figures.
- Added: `ClippingReport.__str__` summarizes instead of dumping every interval.
- Fixed: removed two unreachable branches in `RollingKthLargest._rebalance`.

## 0.2.1 - 2026-09-10

- Added: Housner's spectrum intensity (SI value) -- `calculate_spectrum_intensity`, plus a streaming estimator.
- Added: a general elastic response spectrum -- `calculate_response_spectrum` (Sd, Sv, PSA).
- Added: `detect_clipping`, a diagnostic-only check for saturated samples.
- Added: multi-station distribution maps -- `pyshindo.plotting.maps`.
- Added: `apply_obspy_calibration`, for readers (K-NET/KiK-net among them) that leave data in raw counts.
- Added: `classify_intensity_array` and `intensity_interval` are now exported.
- Fixed: `scale_acceleration_to_intensity`'s `allow_fewer_components` was ignored internally.
- Fixed: map figures didn't validate longitude range, and their legend toggle left a marker's halo behind.
- Docs: corrected `RealtimeIntensityEstimator.process_sample`'s docstring.
- Tests/CI: closed coverage gaps; CI now runs Python 3.12 and 3.13.

## 0.2.0 - 2026-09-05

- Added: the JMA long-period ground motion class (`pyshindo.long_period`), plus a streaming estimator.
- Added: velocity by integration and peak ground velocity (`integrate_to_velocity`, `peak_ground_velocity`).
- Added: `apply_ground_motion_high_pass` reproduces JMA's own published peak velocity.
- Added: optional ObsPy interoperability (`pyshindo.obspy_interop.from_obspy_stream`).
- Added: `scripts/validate_official.py` compares results against JMA's published values for a whole event.
- Fixed: mutable shared state in period grids and in `RealtimeIntensityEstimator`'s filter design.
- Fixed: `acceleration_figure` silently dropped channels past the third.
- Packaging: PEP 639 license metadata, `MANIFEST.in`, `project.urls`.

## 0.1.0 - 2026-09-02

- Frequency-domain reference calculation of instrumental seismic intensity.
- Causal real-time approximation filters (2008, 2012, and the generalized low-rate design).
- Stateful real-time estimation with an exact rolling order statistic.
- JMA strong-motion text parsing and single-file downloading.
- Optional Plotly figures.
