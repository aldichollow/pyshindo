# Changelog

## 0.1.0 - 2026-09-02

- Frequency-domain reference calculation of instrumental seismic intensity (JMA FFT method), with per-call timing on the result.
- Original (2008) and improved (2012) causal approximation filters, plus the generalized low-sampling-rate filter from JP7681907B2.
- Stateful chunk and single-sample real-time estimation with an exact rolling order statistic; results are invariant to chunk boundaries.
- JMA strong-motion text parsing and opt-in single-file downloading; validated end to end against the official 2000 Tottori-ken Seibu (Yonago) record, reproducing the published measured intensity of 5.1.
- Optional Plotly figures (`pyshindo[plot]`), including a named per-stage breakdown of each causal filter's analog factors (`RecursiveFilterDesign.stages`, `filter_stage_response`, `filter_stages_figure`) for inspecting each component's own frequency response, not just the combined result. A shared theme keeps color, line weight, and layout consistent across every figure, including a stacked one-row-per-channel view of acceleration (`acceleration_figure`, also used for the filtered channels in `measured_result_figure`).
- Passes `mypy` cleanly; `pyshindo[dev]` includes it alongside `pytest` and `ruff`.
