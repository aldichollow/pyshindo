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
    .approximate_scale -> IntensityScale | None          # 上記の階級(入力前はNone)
    .current_threshold_acceleration_gal
    .filter_state

calculate_realtime_intensity(acceleration, sampling_rate_hz=100.0, *, unit="gal",
                              filter_name=RealtimeFilter.AUTO) -> RealtimeIntensityResult
realtime_intensity(acceleration, sampling_rate_hz=100.0, *, unit="gal",
                    reported=True) -> ndarray
```

`process()` と `process_sample()` は同一インスタンス上で自由に混在できます。チャンクの分割位置に結果は依存しません。`RealtimeChunk.timing` / `RealtimeSample.elapsed_s` に実測所要時間が入ります。

`LongPeriodEstimator`・`SpectrumIntensityEstimator`と違い、このクラスには`result()`がありません。あの2つがまとめるのは周期ごとの最大値という固定長の要約で、ストリーミング状態がそのまま完全な答えを保持しています。一方`RealtimeIntensityResult`は記録全長にわたるサンプル単位の時系列なので、`result()`を用意すると処理済みの全サンプルを保持することになり、無限に続くライブ入力を一定メモリで扱えるという本クラスの性質が失われます。各`process()`が返す`RealtimeChunk`をどう保持・破棄するかは呼び出し側の選択です。記録全体が最初からメモリ上にあるなら`calculate_realtime_intensity`を使ってください。

`RealtimeFilter`: `AUTO`(既定、80 Hz以上でkunugi2012・80 Hz未満でjp7681907-lowrateへ自動切替) / `KUNUGI_2008` / `KUNUGI_2012` / `JP7681907_LOWRATE`。

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
published_lowrate_gamma_set(sampling_rate_hz, *, lowrate_gamma_policy=...) -> LowRateGammaSet
lowrate_stability_lower_bounds(sampling_rate_hz) -> tuple[float, ...]
lowrate_gamma_stability_margins(sampling_rate_hz, gammas) -> tuple[float, ...]
```

`RecursiveFilterDesign` は正規化されたSOS係数・極半径・安定性フラグを持ちます。`LowRateGammaPolicy`: `PIECEWISE`(既定、JP7681907B2の区分的テーブル) / `CONSTANT_ACCURATE`(γ=1/12、低レートでは不安定になり得る) / `CONSTANT_STABLE`(γ=1/4、任意周波数で安定)。

`RecursiveFilterDesign.stages: tuple[FilterStage, ...]` には、結合済みSOSを構成する前の個別の解析的因子(名前・特性周波数・単体SOS)が入っています。`filter_stage_response()`で1因子ずつ、`pyshindo.plotting.filter_stages_figure(design)`でまとめて可視化できます。全因子を独立にカスケードした結果は結合済みの`.sos`と一致します。

## 震度値と震度階級

```python
intensity_from_threshold_acceleration(threshold_gal) -> float  # a0 -> 連続値
threshold_acceleration_from_intensity(intensity) -> float      # 逆変換
report_intensity(value) -> float                              # 気象庁の十進丸め処理
classify_intensity(value) -> IntensityScale                   # 0〜7 / 5弱〜6強
classify_intensity_array(values) -> ndarray[str]               # classify_intensityの配列版
intensity_interval(scale) -> tuple[float, float]                # 階級の下限(含む)・上限(含まない)
intensity_label(value, *, language="ja") -> str                # "震度5弱" 等
INTENSITY_INTERVALS: dict[IntensityScale, tuple[float, float]]
```

`intensity_from_threshold_acceleration`が受け取るのは、0.3秒継続の判定で選ばれた**閾値加速度1個**であって加速度波形ではありません。記録全体から計測震度を求めたい場合は`measured_intensity`(または`calculate_measured_intensity`)を使ってください。

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
component_peak_acceleration(acceleration, *, unit="gal") -> ndarray   # 成分別PGA(常にgal)
peak_ground_acceleration(acceleration, *, unit="gal") -> float        # 合成PGA(常にgal)
time_axis(sample_count, sampling_rate_hz) -> ndarray
remove_offset(acceleration, *, baseline_samples=None) -> ndarray
detrend_acceleration(acceleration, *, mode="linear") -> ndarray
cosine_taper(acceleration, *, fraction=0.05) -> ndarray
resample_acceleration(acceleration, original_rate_hz, target_rate_hz=100.0) -> ndarray
sampling_diagnostics(timestamps_s) -> SamplingDiagnostics
detect_clipping(acceleration, *, max_range_gal=None, range_tolerance=0.001,
                repeat_threshold=3, extreme_fraction=0.9) -> ClippingReport
```

前処理系(`remove_offset` / `detrend_acceleration` / `cosine_taper` / `resample_acceleration`)は計測震度の定義に暗黙には含まれないため、常に明示的に呼び出す必要があります。

`detect_clipping`は診断専用です。値の補正・削除は一切行わず、他のどの計算関数からも自動的には呼ばれません。検知方式は2種類で、両方を独立に評価します: `max_range_gal`(既知のデジタイザのフルスケール、未指定ならこの方式は無効)付近に張り付いたサンプルと、`repeat_threshold`回以上連続する完全に同一の値のうち、その成分自身の最大振幅の`extreme_fraction`倍以上の区間にあるもの(静穏なノイズフロアでの量子化による偶然の一致を誤検知しないための制約)。実データ268観測点(`docs/validation.md`と同じ検証コーパス)で確認したところクリッピングはゼロでしたが、境界的な誤検知が2件あり、いずれも振幅0.5 gal未満の非常に静かな観測点で、公表波形が小数3桁に丸められていることによる滑らかなピーク付近での値の停滞が原因でした(クリッピングではありません)。詳細はdocstringを参照してください。

## 速度・変位・PGV・PGD

```python
integrate_to_velocity(acceleration, sampling_rate_hz=100.0, *, unit="gal") -> ndarray
component_peak_velocity(acceleration, sampling_rate_hz=100.0, *, unit="gal") -> ndarray
peak_ground_velocity(acceleration, sampling_rate_hz=100.0, *, unit="gal") -> float

integrate_to_displacement(acceleration, sampling_rate_hz=100.0, *, unit="gal") -> ndarray
component_peak_displacement(acceleration, sampling_rate_hz=100.0, *, unit="gal") -> ndarray
peak_ground_displacement(acceleration, sampling_rate_hz=100.0, *, unit="gal") -> float
```

加速度を台形則で累積積分して速度を得ます。変位は同じ台形則でその速度をもう一度積分したものです。入力単位に関わらず内部でgalへ変換するため、戻り値は速度が常にcm/s(カイン)、変位が常にcmです。

ベースライン処理(オフセット除去・トレンド除去・ハイパスフィルタ)は一切自動適用しません。積分は「本物の長周期成分」と「ベースラインの誤差」を区別できないため、平均がゼロでない記録(上下動に重力成分が残っている場合を含む)を積分すると速度は直線的にドリフトします。これは演算が正しく働いた結果であって不具合ではありません。どの補正が適切かは記録と目的によって変わるので、`remove_offset`(事前区間を `baseline_samples` で指定)や `detrend_acceleration` を明示的に呼び出してください。強震観測の実務ではハイパスフィルタを用いることも一般的です。

**変位はこのドリフトが二重に効きます。** 速度の直線的なドリフトをもう一度積分すると、変位は**二次関数的に**ドリフトします。実際に測定したところ、記録長を2倍にすると変位側のドリフトはおよそ4倍になりました(速度は2倍のまま)。PGVでは無視できる程度の基線誤差が、PGDでは支配的になり得るということです。

PGA・PGV・PGDはいずれも入力単位に関わらず内部でgalへ変換してから計算するため、戻り値の単位はそれぞれ常にgal・cm/s・cmです。`peak_ground_velocity`・`peak_ground_displacement`はどちらも渡された成分の合成値を返します(`peak_ground_acceleration`と同じ規約)。3成分を渡せば3成分合成、水平2成分だけを渡せば水平PGV/PGDになります。どちらを採るかは解析側の選択なので、これらの関数側では固定していません。

気象庁「長周期地震動の観測結果」ページが`max.csv`で公表している最大速度は、この既定(補正なしの台形積分)とは一致しません。代わりに、長周期地震動階級の計算で使っている20秒ハイパスを加速度に先に適用してから積分すると、268観測点で相対誤差の中央値0.01%程度まで一致します。このハイパスは`pyshindo.long_period.apply_ground_motion_high_pass`として公開しています:

```python
from pyshindo.long_period import apply_ground_motion_high_pass

filtered = apply_ground_motion_high_pass(acceleration, sampling_rate_hz=100.0, unit="gal")
pgv = peak_ground_velocity(filtered, sampling_rate_hz=100.0, unit="gal")
```

この関係は一次資料が明記したものではなく、実証的に確認した経験則です。詳細と検証結果は [`docs/validation.md`](validation.md) を参照してください。

**`max.csv`の最大変位は、そもそも積分では計算されていません。** 気象庁は「[速度波形・変位波形の求め方](https://www.jma.go.jp/jma/kishou/know/jishin/kyoshin/kaisetsu/calc_wave.html)」で、変位波形を加速度の二重積分ではなく、気象庁の機械式1倍強震計(固有周期6秒、減衰定数0.55)の振幅特性を再現する専用フィルタで直接算出すると明記しています。これは`pyshindo.strong_motion.apply_strong_motion_displacement_filter`として実装しており、`integrate_to_displacement`と違って積分を経由せず、加速度から直接変位を返します(100 Hzでのみ定義された係数のため、他のサンプリング周波数では`ValueError`になります):

```python
from pyshindo.strong_motion import apply_strong_motion_displacement_filter
from pyshindo.signal import vector_resultant

displacement = apply_strong_motion_displacement_filter(acceleration, sampling_rate_hz=100.0, unit="gal")
pgd = float(np.max(vector_resultant(displacement)))
```

268観測点での相対誤差の中央値は約0.15%です。同じページは速度波形についても専用フィルタ(カットオフ5秒の3次バターワースハイパス、`apply_strong_motion_velocity_filter`として同モジュールに実装)を定義していますが、こちらは長周期地震動観測結果ページの最大速度の再現には向きません(`apply_ground_motion_high_pass`の方が良く一致します)。詳細は[`docs/validation.md`](validation.md)を参照してください。

使用例は [`examples/06_peak_velocity.py`](../examples/06_peak_velocity.py) にあります。

## スペクトル強度(SI値)

```python
calculate_spectrum_intensity(acceleration, sampling_rate_hz=100.0, *, unit="gal",
                              damping_ratio=0.20, periods_s=None,
                              component_axis=-1, retain_velocity_time_series=False)
    -> SpectrumIntensityResult

result.si_cm_s        # 成分ごとのSI値 [cm/s]、shape (成分数,)
result.sv_cm_s         # 周期ごとの相対速度応答スペクトル、shape (周期数, 成分数)
result.periods_s       # 既定は0.1〜2.5秒を121分割した等間隔グリッド
result.sv_time_series_cm_s  # サンプル単位の応答。retain_velocity_time_series=True のときだけ
```

`sv_cm_s`(周期ごとの最大値)はフラグに関係なく常に返ります。`retain_velocity_time_series`が制御するのは、それよりはるかに大きい`(サンプル数, 周期数, 成分数)`の時系列`sv_time_series_cm_s`だけです。

Housnerのスペクトル強度(SI値): `SI = (1/2.4) * ∫[0.1, 2.5] Sv(T, h=0.20) dT`。`Sv`は**相対**速度応答スペクトルで、長周期地震動階級が使う**絶対**速度応答スペクトルとは別物です。水平2成分は合成しません(`peak_ground_velocity`と同じく、どの成分を渡すかは呼び出し側の選択です。実務上の慣行はNS/EW水平2成分を個別に扱うことなので、水平だけが欲しい場合は`acceleration[:, :2]`)。

1自由度系の応答計算自体は`pyshindo.long_period`が使っているのと同じ線形加速度法のソルバー(`pyshindo._spectral_response`)を共有しています。地動速度の加算や成分合成といったJMA長周期地震動階級固有の処理を行わない点だけが違います。

SI値には気象庁の震度階級のような公式の離散階級は存在しないため、連続値(cm/s)をそのまま返します。積分区間(0.1〜2.5秒)のグリッド分割数・積分則は一次資料に明記がなく、収束性を独立に検証して既定値(線形121分割、台形則)を選んでいます。121分割は769分割との相対誤差が2e-05程度で、SI値が実務で報告される精度(cm/s単位で整数〜小数第1位)より十分細かい水準です。

この式・減衰比0.20・水平成分ごとの報告は、鳥取県道路橋梁設計マニュアル3-6「スペクトル強度SI値」(式3-11、大崎順彦による)に式番号付きで明記されています。実データでの確認: 同マニュアルの図3-18は2000年鳥取県西部地震のSI値を観測点ごとに示しており、対応するK-NET/KiK-net記録(米子TTR008、日野TTRH02地表センサー)を実際にダウンロードして`calculate_spectrum_intensity`にかけると、図の値と1%以内で一致します(米子NS 47.59 対 47.12、EW 66.05 対 65.44、日野NS 113.7 対 113 cm/s)。

使用例は [`examples/09_spectrum_intensity.py`](../examples/09_spectrum_intensity.py) にあります。

### ストリーミング版

```python
SpectrumIntensityEstimator(sampling_rate_hz=100.0, *, unit="gal",
                            damping_ratio=0.20, periods_s=None)
    .process(acceleration) -> SpectrumIntensityUpdate       # チャンク一括
    .process_sample(acceleration) -> SpectrumIntensityUpdate  # 1サンプルずつ
    .si_cm_s / .sv_cm_s   # ここまでの累積値
    .result() -> SpectrumIntensityResult
```

`pyshindo.long_period.LongPeriodEstimator`と同じ「記録開始からの累積最大値」という挙動です(ローリングウィンドウではありません)。ただし、長周期地震動階級・SI値のどちらも気象庁が公表した公式のリアルタイム計算方法があるわけではなく、この累積最大値という解釈はどちらもこのパッケージ独自の設計判断である点に注意してください(公式のリアルタイム近似フィルタが存在する計測震度の`RealtimeIntensityEstimator`とはこの点が異なります)。バッチ計算(`calculate_spectrum_intensity`の既定ソルバー)とは演算順序が異なるため浮動小数点の丸め水準で一致し、ビット単位の一致ではありません。既定の121点グリッドは長周期地震動階級の32点の約3.8倍ですが、1サンプルあたりの処理コストは実測で約1.1倍にとどまります(周期方向の演算が既にNumPyでベクトル化されているため)。

## 弾性応答スペクトル(Sd/Sv/PSV/PSA)

```python
calculate_response_spectrum(acceleration, sampling_rate_hz=100.0, *, unit="gal",
                             damping_ratio, periods_s, component_axis=-1,
                             retain_displacement_time_series=False,
                             retain_velocity_time_series=False)
    -> ResponseSpectrumResult

result.sd_cm     # 変位応答スペクトル [cm]、shape (周期数, 成分数)
result.sv_cm_s    # 速度応答スペクトル(真の相対速度応答)[cm/s]
result.psv_cm_s   # 擬似速度応答スペクトル [cm/s] = omega * sd_cm
result.psa_gal    # 擬似加速度応答スペクトル [gal] = omega^2 * sd_cm
```

長周期地震動階級・SI値の両方が内部で共有している1自由度系ソルバー(`pyshindo._spectral_response`)を、そのままの形で公開する汎用関数です。`damping_ratio`・`periods_s`にはどちらも既定値がありません -- 長周期地震動階級(減衰5%、公式32点グリッド)・SI値(減衰20%、0.1〜2.5秒)のどちらの慣行を既定にしても「汎用」という位置づけと矛盾するため、呼び出し側が必ず明示します。

**相対応答のみ**です。地動速度・地動変位の加算、成分合成はいずれも行いません(この相対応答にJMA固有の処理を足して絶対応答にする部分は`long_period`側が担います)。絶対応答スペクトルが欲しい場合は、`retain_velocity_time_series=True`で得られる`sv_time_series_cm_s`に、既存の`integrate_to_velocity`(公開関数)で求めた地動速度をサンプルごとに足してから最大値を取ってください(和の最大値は最大値の和と一致しないため、`sv_cm_s`だけからは絶対応答を再構成できません)。`examples/11_response_spectrum.py`で、この手順が`pyshindo.long_period`自身のSvaと一致することを確認しています。変位・速度の時系列(`sd_time_series_cm`・`sv_time_series_cm_s`)はそれぞれ独立したフラグで保持するかどうかを選べます -- 片方しか使わない場合に、使わない方まで計算・保持するコストを避けるためです。

`psv_cm_s`(擬似速度、PSV = ω×Sd)・`psa_gal`(擬似加速度、PSA = ω²×Sd)はどちらも`sd_cm`から追加コストなしで導出できるため常に返します。いずれも絶対応答(それぞれ真の相対速度応答・絶対加速度応答)への工学的近似で、減衰ゼロの極限でのみ厳密に一致します。建築基準のような耐震工学の実務ではSd/PSV/PSAの3点セットで応答スペクトルを語ることが多く、この対応が取れるようにしています。真の絶対加速度応答(相対加速度+地動加速度)は、独自の伝達関数導出と一次資料での定義確認が必要なため実装していません。

使用例は [`examples/11_response_spectrum.py`](../examples/11_response_spectrum.py) にあります。

## 単位変換

```python
to_gal(values, unit) -> ndarray
convert_acceleration(values, from_unit, to_unit) -> ndarray
AccelerationUnit: "gal" / "m/s^2" / "g"
STANDARD_GRAVITY_MPS2 = 9.80665
```

## 例外と警告

```python
PyShindoError                      # このパッケージ固有の例外の基底クラス
├─ InvalidAccelerationError        # 加速度データの形状・値が不正(ValueErrorでもある)
├─ InsufficientDataError           # 記録が要求された計算に対して短すぎる
├─ UnstableFilterError             # 要求された漸化式フィルタが数値的に不安定
└─ DataFormatError                 # 入力記録が想定フォーマットと一致しない

PyShindoWarning                    # このパッケージ固有の警告の基底クラス
├─ NonstandardSamplingRateWarning  # 100 Hz以外のサンプリング周波数で計算している
├─ NonstandardProcessingWarning    # 任意の前処理が参照計算を変更している
├─ MissingComponentWarning         # 3成分未満の加速度を使っている
└─ FractionalDurationWarning       # 継続時間が整数サンプル数にならない
```

いずれも`pyshindo`直下と`pyshindo.exceptions`の両方から参照できます。例外側は`ValueError`も継承しているため、`except ValueError`で書かれた既存のコードでも捕捉されます。このパッケージ由来かどうかで区別したい場合は`except PyShindoError`を使ってください。

警告は`warnings`モジュールの標準的な仕組みで制御できます:

```python
import warnings
from pyshindo import NonstandardSamplingRateWarning

warnings.filterwarnings("ignore", category=NonstandardSamplingRateWarning)
```

## 合成データ

```python
synthetic_three_component_motion(sampling_rate_hz=100.0, duration_s=30.0, ...) -> ndarray
scale_acceleration_to_intensity(acceleration, sampling_rate_hz=100.0, *,
                                 target_intensity_raw) -> tuple[ndarray, float]
```

テスト・デモ用の決定論的な3成分波形生成と、目標の生震度値に合わせた振幅スケーリング。物理的な地震動シミュレータではありません。

## `pyshindo.io`(観測データ)

```python
from pyshindo.io import parse_jma_text, parse_jma_bytes, read_jma_record, download_jma_record
```

詳細は [`docs/data.md`](data.md) を参照してください。

## `pyshindo.long_period`(長周期地震動階級)

```python
from pyshindo.long_period import (
    calculate_long_period_class, LongPeriodEstimator, LongPeriodClass,
)

result = calculate_long_period_class(horizontal_gal, 100.0, unit="gal")
result.long_period_class     # LongPeriodClass("2") など
result.max_sva_cm_s          # 全周期での絶対速度応答の最大値 [cm/s]
result.critical_period_s     # 最大値を与えた周期
result.sva_cm_s              # 周期ごとのSva (32,)
result.bands                 # 周期帯別(1秒台〜7秒台)の最大Svaと階級
```

水平2成分のみを入力します(上下動は使いません。3成分を渡すとエラーになります)。逐次処理は `LongPeriodEstimator` です。

`solver` で振動子バンクの解き方を選べます。いずれも解いている式は同じで、演算の順序だけが違います。

- `"filter"`(既定): 周期ごとの2次IIRフィルタとしてまとめて解きます。逐次ループがコンパイル済みコードに移るため約13倍高速です。
- `"recurrence"`: 気象庁の資料どおりの状態空間漸化式をPythonで1サンプルずつ進めます。`LongPeriodEstimator` が使っているのはこちらなので、一括結果を逐次結果とビット単位で一致させたい場合に選びます。

既定の `"filter"` と `"recurrence"` の差は浮動小数点の丸めのみで、相対1e-12程度です。通常のデータでは階級判定に影響しませんが、Svaが階級の閾値からちょうど丸め幅程度しか離れていない場合、まれにどちらの階級になるかが1ソルバー分だけ変わり得ます(これは実装の不具合ではなく、閾値ぴったりの値を機械精度で扱う以上避けられない挙動です)。逐次処理との厳密な一致が必要な場合は `solver="recurrence"` を使ってください。

この計算の内部で使っている20秒ハイパスは `apply_ground_motion_high_pass` として単独でも呼べます(水平2成分に限らず、任意の成分数に使えます)。気象庁が公表している最大速度の再現に使えることを確認しています(上記「速度・PGV」の節、[`docs/validation.md`](validation.md))。

アルゴリズム・一次資料・公式値との照合結果は [`docs/long-period.md`](long-period.md)、公式値との照合の全体像は [`docs/validation.md`](validation.md) を参照してください。

## `pyshindo.obspy_interop`(ObsPy連携、要 `pip install pyshindo[obspy]`)

```python
from pyshindo.obspy_interop import apply_obspy_calibration, from_obspy_stream

record = from_obspy_stream(stream, *, unit, channel_order=None, allow_fewer_components=True)
record.acceleration            # (サンプル数, 成分数) のfloat64配列
record.metadata                # ObsPyRecordMetadata
```

ObsPyの `Stream` を本パッケージが扱う配列へ変換するだけの薄いアダプタです。詳細は [`docs/data.md`](data.md) を参照してください。

`apply_obspy_calibration(stream)` は、K-NET/KiK-netなど一部のObsPyリーダーが`trace.data`を生カウント値のまま残し、物理量換算係数を`trace.stats.calib`に別途持たせている場合に、それを掛けて`calib`を1.0に戻したコピーを返します。`from_obspy_stream`はデータが既に物理量であることを前提とするため、その手前で通してください。

## `pyshindo.spatial`(観測点間の空間補間、追加インストール不要)

```python
from pyshindo.spatial import (
    GeographicBounds, SurfaceGrid, IDWConfig, LinearConfig, NearestConfig,
    interpolate_surface, build_interpolation_plan,
)

grid = SurfaceGrid.from_bounds(GeographicBounds(west_deg, south_deg, east_deg, north_deg),
                                approximate_resolution_km=5.0)
surface = interpolate_surface(latitudes_deg, longitudes_deg, values, grid=grid,
                               method="idw", config=IDWConfig(neighbors=8, max_distance_km=...),
                               transform="identity", metric_name="pga", unit="gal")
surface.values          # grid.shape、支持範囲外はNaN
surface.support_mask    # 補間の根拠となる観測点があったセルだけTrue
```

観測点の値をNumPy/SciPyだけで空間補間します(Plotly等は不要)。`method`は`"idw"`(局所IDW)・`"linear"`(Delaunay三角形分割上の線形補間)・`"nearest"`(最近傍)。階級・色は補間せず、連続量(計測震度なら`intensity_raw`、長周期地震動なら`max_sva_cm_s`)を補間してから分類してください。階級しか値がない場合は`"nearest"`のみを使います。`config`の探索半径は既定を持たず、`allow_extrapolation=True`を明示しない限り必須です。使用例は[`examples/12_station_surface_interpolation.py`](../examples/12_station_surface_interpolation.py)。

## `pyshindo.plotting`(可視化、要 `pip install pyshindo[plot]`)

```python
from pyshindo.plotting import (
    acceleration_figure, jma_filter_components_figure, filter_response_figure,
    filter_stages_figure, measured_result_figure, amplitude_duration_figure,
    realtime_result_figure, intensity_comparison_figure,
    long_period_spectrum_figure,
)
```

いずれもPlotlyの `Figure` を返すだけで、数値結果には関与しません。使用例は[`examples/`](../examples/) を参照してください。

### 複数観測点の分布図

```python
from pyshindo.plotting import (
    intensity_map_figure, long_period_class_map_figure, continuous_value_map_figure,
)

intensity_map_figure(latitudes_deg, longitudes_deg, intensities, *, labels=None, title=...,
                      map_style="carto-positron")
long_period_class_map_figure(latitudes_deg, longitudes_deg, classes, *, labels=None, title=...,
                              map_style="carto-positron")
continuous_value_map_figure(latitudes_deg, longitudes_deg, values, *, value_label, labels=None,
                             colorscale="YlOrRd", color_transform="identity", title=...,
                             map_style="carto-positron")
```

これも薄いアダプタです。緯度・経度・値の並列配列を渡すだけで、`pyshindo.io`・`pyshindo.obspy_interop`・その他どのデータ源から来た値かは関知しません。震度・長周期地震動階級は既存の図と同じJMA配色で階級ごとに1トレースに分け、SI値・PGVのような公式階級のない連続値は連続カラースケール+カラーバーの1トレースになります。`color_transform="log"`でマーカー色をlog(values)にできます(既定は`"identity"`で変更なし、`pyshindo.spatial`の`transform`と同じく暗黙には適用されません)。PGA/PGVのような対数正規分布に近い量は、線形のままだと震源近傍以外の差が潰れて見えるため。Plotlyの`Scattermap`(トークン不要の組み込みMapLibreスタイル)を使用します。既定の背景地図`"carto-positron"`はマーカーの色が沈まないよう抑えたグレースケールで、`map_style`引数で`"open-street-map"`(元のカラフルなOSMタイル)や`"carto-positron-nolabels"`などPlotlyの組み込みスタイルに切り替えられます。`Scattermap`のマーカーには`Scatter`と違って枠線(`marker.line`)がないため、各マーカーの背後に濃いグレーの縁取り用の非表示トレースを重ねています。使用例は[`examples/10_station_map.py`](../examples/10_station_map.py)を参照してください。

### 補間サーフェスのレイヤー描画

```python
from pyshindo.plotting import add_surface_layer, add_class_surface_layer

add_surface_layer(figure, surface, *, cmin, cmax, colorscale="YlOrRd",
                   color_transform="identity", land="natural_earth_japan_10m",
                   colorbar_title=None, colorbar_x=None)
add_class_surface_layer(figure, surface, *, colors, land="natural_earth_japan_10m")
```

`pyshindo.spatial.interpolate_surface`の出力を、既存の観測点分布図(上記)の下に画像レイヤーとして重ねます。`land`(既定で陸地のみ表示、`None`で無効化)は同梱のラスタ(Natural Earth 1:10m、`scripts/build_natural_earth_japan.py`で生成)によるもので、Shapely等の追加依存は不要です。`colorscale`にはPlotly組み込み名だけでなく、`[position, color]`のリスト(離散的なバンド配色を作る場合など)も渡せます。使用例は[`examples/12_station_surface_interpolation.py`](../examples/12_station_surface_interpolation.py)。
