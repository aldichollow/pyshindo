# Changelog

## 0.2.0 - 2026-09-05

- JMA long-period ground motion class (`pyshindo.long_period`): the published 20-second second-order high-pass, a bank of 32 damped oscillators over the official 1.6-7.8 s grid solved by the linear acceleration method, ground velocity by trapezoidal integration, the per-sample horizontal vector composite adopted by JMA in 2016, and the overall plus per-period-band classes. `LongPeriodEstimator` produces bit-identical results for streaming input. Every published constant is used verbatim at 100 Hz and regression-tested as a literal; other sampling rates re-derive the high-pass from the analog prototype behind those constants and are flagged as non-reference. Verified against JMA's own published absolute velocity response spectra, agreeing to about 2e-6 at the record maximum.
- `pyshindo.plotting.long_period_spectrum_figure` draws the response spectrum against the class thresholds.
- Velocity by cumulative trapezoidal integration (`integrate_to_velocity`) and peak ground velocity (`component_peak_velocity`, `peak_ground_velocity`), always returned in cm/s. No baseline correction is applied implicitly; the drift that follows from integrating an uncorrected record is documented rather than hidden.
- Optional ObsPy interoperability (`pyshindo[obspy]`, `pyshindo.obspy_interop.from_obspy_stream`): converts a stream already in acceleration units into this package's arrays and metadata, making K-NET, KiK-net, miniSEED, SAC, and everything else `obspy.read` handles usable without reimplementing a reader. The adapter never resamples, trims, merges, rotates, or rescales, and rejects traces that disagree instead of reconciling them. The acceleration unit is a required argument rather than a guess, because SEED carries no dependable unit field.

## 0.1.0 - 2026-09-02

- Frequency-domain reference calculation of instrumental seismic intensity (JMA FFT method), with per-call timing on the result.
- Original (2008) and improved (2012) causal approximation filters, plus the generalized low-sampling-rate filter from JP7681907B2.
- Stateful chunk and single-sample real-time estimation with an exact rolling order statistic; results are invariant to chunk boundaries.
- JMA strong-motion text parsing and opt-in single-file downloading; validated end to end against the official 2000 Tottori-ken Seibu (Yonago) record, reproducing the published measured intensity of 5.1.
- Optional Plotly figures (`pyshindo[plot]`), including a named per-stage breakdown of each causal filter's analog factors (`RecursiveFilterDesign.stages`, `filter_stage_response`, `filter_stages_figure`) for inspecting each component's own frequency response, not just the combined result. A shared theme keeps color, line weight, and layout consistent across every figure, including a stacked one-row-per-channel view of acceleration (`acceleration_figure`, also used for the filtered channels in `measured_result_figure`).
- Passes `mypy` cleanly; `pyshindo[dev]` includes it alongside `pytest` and `ruff`.
