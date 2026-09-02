# Changelog

## 0.1.0 - 2026-09-02

- Frequency-domain reference calculation of instrumental seismic intensity (JMA FFT method), with per-call timing on the result.
- Original (2008) and improved (2012) causal approximation filters, plus the generalized low-sampling-rate filter from JP7681907B2.
- Stateful chunk and single-sample real-time estimation with an exact rolling order statistic; results are invariant to chunk boundaries.
- JMA strong-motion text parsing and opt-in single-file downloading; validated end to end against the official 2000 Tottori-ken Seibu (Yonago) record, reproducing the published measured intensity of 5.1.
- Optional Plotly figures (`pyshindo[plot]`).
