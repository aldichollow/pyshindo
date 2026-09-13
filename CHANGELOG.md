# Changelog

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
