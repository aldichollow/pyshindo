# pyshindo

## 概要

`pyshindo` は加速度記録から気象庁の計測震度を計算するPythonパッケージです。記録全体を使うFFT参照計算(計測震度)と、逐次入力向けの因果的リアルタイム近似を明確に分離しているのが特徴です。気象庁の公開計算式・Kunugi論文・関連特許(JP4229337B2/JP5946067B2/JP7681907B2)を根拠に、係数を式から都度導出する実装になっています(固定係数表の丸写しではありません)。リアルタイム側は直近60秒の閾値をヒストグラム丸めなしの厳密な順序統計量で保持し、逐次入力(`process_sample`)と一括入力(`process`)のどちらでも同じ結果になるよう作られています。詳細なアルゴリズム解説は日本語で [`docs/algorithm.md`](docs/algorithm.md) にあります。

本パッケージは個人が趣味として開発しているものです。計算結果の正確性・完全性を保証するものではありませんので、ご利用は自己判断・自己責任でお願いします。

`pyshindo` is a small Python package for calculating Japanese instrumental seismic intensity from acceleration records. It keeps the complete-record FFT calculation separate from causal real-time approximations, so the meaning of both results remains explicit.

The package targets Python 3.12 or later. It is a research and engineering reference implementation, not a certified seismic intensity meter, earthquake early-warning service, or safety controller.

## What is implemented

- The published JMA frequency-domain calculation: FFT per component, the three-factor intensity response, inverse FFT, three-component resultant, the 0.3-second cumulative-duration threshold, and the official decimal treatment.
- The original 2008 causal approximation filter.
- The improved 2012 causal approximation filter.
- The generalized low-sampling-rate filter disclosed in JP7681907B2.
- Exact rolling order statistics for a 60-second real-time window without discretizing intensity into fixed-width bins.
- Stateful chunk and single-sample APIs whose results are invariant to chunk boundaries.
- Unit conversion, sampling diagnostics, PGA, preprocessing helpers, response inspection, JMA text-record parsing, and optional Plotly figures.
- Built-in wall-clock timing: every result carries a `timing` field (or, for `process_sample`, `elapsed_s`) measured with `time.perf_counter`, so callers can inspect calculation cost without wrapping their own timer.

Relevant real-time algorithms are associated with patent documents. Read [PATENTS.md](PATENTS.md) before distribution or operational use. The MIT license covers copyright in this source code and is not a patent-clearance opinion.

## Installation

From a checkout:

```bash
python -m pip install -e .
```

Install the optional Plotly figures:

```bash
python -m pip install -e ".[plot]"
```

## Complete-record FFT calculation

```python
from pyshindo import calculate_measured_intensity

result = calculate_measured_intensity(
    acceleration,              # shape: (samples, 3)
    sampling_rate_hz=100.0,
    unit="m/s^2",
)

print(result.intensity_raw)                 # Unrounded continuous value
print(result.intensity)                     # Official one-decimal treatment
print(result.scale.japanese)                # Example: "5弱"
print(result.threshold_acceleration_gal)    # 0.3-second threshold
print(result.filtered_pga_gal)
print(result.timing.total_s)                # wall-clock time for this call
```

`result.filtered_acceleration_gal`, `result.resultant_acceleration_gal`, the frequency vector, and the applied response are retained by default. Set `retain_intermediates=False` for lower memory use.

For a scalar-only call:

```python
from pyshindo import measured_intensity

intensity = measured_intensity(acceleration, 100.0, unit="gal")
```

## Real-time calculation

```python
from pyshindo import RealtimeIntensityEstimator

estimator = RealtimeIntensityEstimator(
    sampling_rate_hz=100.0,
    unit="gal",
)

for chunk in acceleration_chunks:
    output = estimator.process(chunk)
    latest = output.intensity_raw[-1]
    print(output.timing.filter_s, output.timing.order_statistic_s)
```

The estimator filters every sample, preserves recursive state, and maintains the exact 30th-largest value in the latest 60 seconds at 100 Hz. The first valid output appears with sample 30; preceding values are `NaN`. A zero threshold maps to negative infinity, as required by the logarithmic conversion.

For one complete-record replay:

```python
from pyshindo import calculate_realtime_intensity

trace = calculate_realtime_intensity(acceleration, 100.0, unit="gal")
print(trace.approximate_intensity_raw)
print(trace.approximate_intensity)
```

## Sampling rates other than 100 Hz

The FFT calculation accepts any positive sampling rate and evaluates the published response at the corresponding FFT frequencies. A warning is emitted because comparability still depends on the source bandwidth, anti-aliasing, record preparation, and validation data.

The default real-time selection is `RealtimeFilter.AUTO`:

- at 80 Hz or above, the improved 2012 filter is used;
- below 80 Hz, the generalized low-rate design is used;
- below 1 Hz, no published gamma table is available and an error is raised.

The selected design is recorded in `result.filter_name`. An explicit 2012 request is checked for pole stability and fails instead of returning a diverging sequence. Batch resampling is available through `resample_acceleration`, but is never performed implicitly.

## Data and figures

`pyshindo.io` parses the seven-line JMA strong-motion text header and can download one explicitly selected URL. No observed waveform is bundled. See [`docs/data.md`](docs/data.md).

Plotly figures use a restrained package theme. Intensity colors 1 through 7 follow the JMA web color guide; the guide does not assign intensity 0 a color, so the neutral intensity-0 background is identified as a package choice.

## Documentation

- [Algorithm guide (Japanese)](docs/algorithm.md)
- [API reference (Japanese)](docs/api.md)
- [Observed data I/O (Japanese)](docs/data.md)

## Primary references

- Japan Meteorological Agency, "Calculation of instrumental seismic intensity."
- Kunugi, Aoi, and Nakamura (2008), *A real-time processing method of seismic intensity*, DOI: 10.4294/zisin.60.243.
- Kunugi, Aoi, and Nakamura (2013), *An improved approximation filter for the real-time calculation of seismic intensity*, DOI: 10.4294/zisin.65.223.
- JP4229337B2 / JP5946067B2 / JP7681907B2 -- see [PATENTS.md](PATENTS.md).

---

This is a personal, hobby-scale project maintained by one individual, not a company or research group. It comes with no warranty of accuracy, completeness, or fitness for any particular purpose -- use your own judgment, especially for anything safety-related.
