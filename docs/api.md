# APIリファレンス

`pyshindo` トップレベル名前空間の主な関数・クラスの早見表です。各関数のパラメータの詳細はdocstring(`help(...)`)を参照してください。値の意味やアルゴリズムの背景は[`docs/algorithm.md`](algorithm.md) を参照してください。

## 最小実装例

自前の加速度データが無くてもこのまま動きます。

```python
from pyshindo import calculate_measured_intensity, synthetic_three_component_motion

# 形状 (サンプル数, 3) の3成分加速度。実データがあれば代わりにそれを渡してください。
acceleration_gal = synthetic_three_component_motion(sampling_rate_hz=100.0, duration_s=30.0)

result = calculate_measured_intensity(acceleration_gal, sampling_rate_hz=100.0, unit="gal")

print(result.intensity)          # 4.8 (気象庁の1桁表示値)
print(result.scale.japanese)     # "5弱"
```

より詳しい使用例は[`examples/`](../examples/)を参照してください。

## 計測震度(FFT参照計算)

```python
calculate_measured_intensity(acceleration, sampling_rate_hz=100.0, *, unit="gal",
                              duration_s=0.3, duration_policy="ceil",
                              component_axis=-1, allow_fewer_components=False,
                              retain_intermediates=True) -> MeasuredIntensityResult
measured_intensity(acceleration, sampling_rate_hz=100.0, *, unit="gal",
                    reported=True) -> float
```

- `acceleration`: 形状 `(サンプル数, 3)` の3成分加速度。`unit` は `"gal"` / `"m/s^2"` / `"g"`。
- `measured_intensity` はスカラー値だけが必要なときの簡易版。
- `MeasuredIntensityResult` の主なフィールド: `intensity_raw`(丸め前連続値)、`intensity`(気象庁の1桁表示値)、`scale`(`IntensityScale`)、`threshold_acceleration_gal`(0.3秒閾値)、`filtered_acceleration_gal` / `resultant_acceleration_gal`(フィルタ後波形、`retain_intermediates=False`で省略可)、`timing`(`MeasuredIntensityTiming`: 各段階の実測時間)。

## リアルタイム震度(因果近似)

```python
RealtimeIntensityEstimator(sampling_rate_hz=100.0, *, unit="gal",
                            filter_name=RealtimeFilter.AUTO, window_s=60.0,
                            duration_s=0.3, allow_fewer_components=False)
    .process(acceleration) -> RealtimeChunk       # チャンク一括
    .process_sample(acceleration) -> RealtimeSample  # 1サンプルずつ
    .reset() -> None
    .approximate_intensity_raw / .approximate_intensity  # 記録内の最大値
    .current_threshold_acceleration_gal
    .filter_state

calculate_realtime_intensity(acceleration, sampling_rate_hz=100.0, *, unit="gal",
                              filter_name=RealtimeFilter.AUTO) -> RealtimeIntensityResult
realtime_intensity(acceleration, sampling_rate_hz=100.0, *, unit="gal") -> ndarray
```

`process()` と `process_sample()` は同一インスタンス上で自由に混在できます。チャンクの分割位置に結果は依存しません。`RealtimeChunk.timing` / `RealtimeSample.elapsed_s` に実測所要時間が入ります。

`RealtimeFilter`: `AUTO`(既定、100 Hz以上でkunugi2012・80 Hz未満でjp7681907-lowrateへ自動切替) / `KUNUGI_2008` / `KUNUGI_2012` / `JP7681907_LOWRATE`。

## 両者の比較

```python
compare_intensity_methods(acceleration, sampling_rate_hz=100.0, *, unit="gal",
                           measured_options=None, realtime_options=None)
    -> IntensityComparisonResult
```

`.raw_difference` / `.reported_difference` / `.scale_agreement` でFFT参照値とリアルタイム最大値のずれを確認できます。

## フィルタ設計の検査

```python
design_realtime_filter(sampling_rate_hz=100.0, *, filter_name=RealtimeFilter.AUTO,
                        lowrate_gamma_policy=LowRateGammaPolicy.PIECEWISE,
                        check_stability=True) -> RecursiveFilterDesign
realtime_filter_response(design, frequency_hz=None) -> FrequencyResponse
filter_stage_response(design, stage, frequency_hz=None) -> FrequencyResponse  # 1因子だけの特性
jma_filter_response(frequency_hz) -> ndarray          # FFT参照フィルタの振幅応答
jma_filter_components(frequency_hz) -> JMAFilterComponents  # 周期効果/ハイカット/ローカット別
kunugi_2012_analog_amplitude(frequency_hz) -> ndarray  # 2012フィルタの連続時間近似
published_lowrate_gamma_set(sampling_rate_hz, *, policy=...) -> LowRateGammaSet
lowrate_stability_lower_bounds(sampling_rate_hz) -> tuple[float, ...]
lowrate_gamma_stability_margins(sampling_rate_hz, gammas) -> tuple[float, ...]
```

`RecursiveFilterDesign` は正規化されたSOS係数・極半径・安定性フラグを持ちます。`LowRateGammaPolicy`: `PIECEWISE`(既定、JP7681907B2の区分的テーブル) / `CONSTANT_ACCURATE`(γ=1/12、低レートでは不安定になり得る) / `CONSTANT_STABLE`(γ=1/4、任意周波数で安定)。

`RecursiveFilterDesign.stages: tuple[FilterStage, ...]` には、結合済みSOSを構成する前の個別の解析的因子(名前・特性周波数・単体SOS)が入っています。`filter_stage_response()`で1因子ずつ、`pyshindo.plotting.filter_stages_figure(design)`でまとめて可視化できます。全因子を独立にカスケードした結果は結合済みの`.sos`と一致します。

## 震度値と震度階級

```python
intensity_from_acceleration(threshold_gal) -> float          # a0 -> 連続値
acceleration_from_intensity(intensity) -> float              # 逆変換
report_intensity(value) -> float                              # 気象庁の十進丸め処理
classify_intensity(value) -> IntensityScale                   # 0〜7 / 5弱〜6強
intensity_label(value, *, language="ja") -> str                # "震度5弱" 等
INTENSITY_INTERVALS: dict[IntensityScale, tuple[float, float]]
```

## 継続時間・順序統計

```python
duration_sample_count(duration_s, sampling_rate_hz, *, policy="ceil") -> int
duration_threshold(resultant_acceleration_gal, sample_count) -> float
duration_threshold_at(resultant_acceleration_gal, sampling_rate_hz, *, duration_s=0.3) -> float
exceedance_duration(amplitude, threshold, sampling_rate_hz) -> float
amplitude_duration_curve(amplitude, sampling_rate_hz) -> AmplitudeDurationCurve
```

## 信号処理ユーティリティ

```python
vector_resultant(acceleration) -> ndarray             # 3成分合成
component_peak_acceleration(acceleration) -> ndarray   # 成分別PGA
peak_ground_acceleration(acceleration) -> float        # 合成PGA
time_axis(sample_count, sampling_rate_hz) -> ndarray
remove_offset(acceleration, *, baseline_samples=None) -> ndarray
detrend_acceleration(acceleration, *, mode="linear") -> ndarray
cosine_taper(acceleration, *, fraction=0.05) -> ndarray
resample_acceleration(acceleration, original_rate_hz, target_rate_hz=100.0) -> ndarray
sampling_diagnostics(timestamps_s) -> SamplingDiagnostics
```

前処理系(`remove_offset` / `detrend_acceleration` / `cosine_taper` / `resample_acceleration`)は計測震度の定義に暗黙には含まれないため、常に明示的に呼び出す必要があります。

## 単位変換

```python
to_gal(values, unit) -> ndarray
convert_acceleration(values, from_unit, to_unit) -> ndarray
AccelerationUnit: "gal" / "m/s^2" / "g"
STANDARD_GRAVITY_MPS2 = 9.80665
```

## 合成データ

```python
synthetic_three_component_motion(sampling_rate_hz=100.0, duration_s=30.0, ...) -> ndarray
scale_acceleration_to_intensity(acceleration, target_intensity_raw, sampling_rate_hz=100.0)
    -> tuple[ndarray, float]
```

テスト・デモ用の決定論的な3成分波形生成と、目標の生震度値に合わせた振幅スケーリング。物理的な地震動シミュレータではありません。

## `pyshindo.io`(観測データ)

```python
from pyshindo.io import parse_jma_text, parse_jma_bytes, read_jma_record, download_jma_record
```

詳細は [`docs/data.md`](data.md) を参照してください。

## `pyshindo.plotting`(可視化、要 `pip install pyshindo[plot]`)

```python
from pyshindo.plotting import (
    acceleration_figure, jma_filter_components_figure, filter_response_figure,
    filter_stages_figure, measured_result_figure, amplitude_duration_figure,
    realtime_result_figure, intensity_comparison_figure,
)
```

いずれもPlotlyの `Figure` を返すだけで、数値結果には関与しません。使用例は[`examples/`](../examples/) を参照してください。
