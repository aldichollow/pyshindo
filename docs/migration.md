# 移行ガイド

破壊的変更のあるバージョンごとに、旧APIから新APIへの対応表を記録します。

## 0.2.2 から 0.3.0 へ

PyPIでの公開を機に、公開APIの命名と一貫性を全面的に見直しました。PyPIは同じバージョン番号のファイルを上書きできないため、破壊的変更はこのタイミングにまとめています。

変更は大きく3種類です。

- **A. 名前が変わったもの** — 機械的に置換できます。
- **B. 戻り値・既定値が変わったもの** — 置換では検出できないため、該当箇所を目視で確認してください。
- **C. 追加されたもの** — 既存コードへの影響はありません。

---

### A. 名前が変わったもの

#### A-1. 閾値加速度と震度の変換関数

これらが受け取るのは加速度**波形**ではなく、0.3秒継続の判定で選ばれた閾値加速度**1個**(スカラー)です。旧名は`measured_intensity`など波形を受け取る関数と同じ`_from_acceleration`という語尾だったため、何を渡すべきか名前から判断できませんでした。

| 0.2.2まで | 0.3.0から |
| --- | --- |
| `intensity_from_acceleration(threshold_gal)` | `intensity_from_threshold_acceleration(threshold_gal)` |
| `acceleration_from_intensity(intensity)` | `threshold_acceleration_from_intensity(intensity)` |
| `intensity_series_from_acceleration(values_gal)` | `intensity_series_from_threshold_acceleration(values_gal)` |

引数・戻り値・計算内容は変わっていません。

置換する場合は、この順序で実行してください(順序を変えると部分一致で壊れます):

```bash
grep -rl 'from_acceleration\|acceleration_from_intensity' --include='*.py' . | xargs sed -i '' \
  -e 's/intensity_series_from_acceleration/intensity_series_from_threshold_acceleration/g' \
  -e 's/acceleration_from_intensity/threshold_acceleration_from_intensity/g' \
  -e 's/intensity_from_acceleration/intensity_from_threshold_acceleration/g'
```

(Linuxの`sed`では`-i ''`ではなく`-i`。実行後は`git diff`で意図しない箇所が変わっていないか確認してください。)

#### A-2. キーワード引数名

| 対象 | 0.2.2まで | 0.3.0から |
| --- | --- | --- |
| `calculate_spectrum_intensity` | `retain_spectrum=` | `retain_velocity_time_series=` |
| `published_lowrate_gamma_set` | `policy=` | `lowrate_gamma_policy=` |

`retain_spectrum`は名前に反して、スペクトル(`sv_cm_s`)ではなく時系列(`sv_time_series_cm_s`)を保持するフラグでした。`sv_cm_s`はこのフラグに関係なく常に返されます。同じものを制御する`calculate_response_spectrum`側の名前に揃えました。

`policy`は`design_realtime_filter`の`lowrate_gamma_policy`と同じ`LowRateGammaPolicy`を取りながら別名でした。

#### A-3. データクラスのフィールド名

| 対象 | 0.2.2まで | 0.3.0から |
| --- | --- | --- |
| `ObsPyRecordMetadata` | `.station` | `.station_code` |
| `LongPeriodResult` | `.absolute_velocity_cm_s` | `.absolute_velocity_time_series_cm_s` |
| `LongPeriodUpdate` | `.class_so_far` | `.long_period_class_so_far` |

`station_code`は`JMARecordMetadata`側の名前に合わせたものです。どちらのリーダーから来た記録でも同じ属性名で観測点を読めるようになります。

`absolute_velocity_cm_s`は`(サンプル数, 周期数)`の時系列でしたが、同じデータクラスにある`sva_cm_s`(周期ごとのスカラー)と名前から区別できませんでした。`_time_series_`を含む名前は`ResponseSpectrumResult`・`SpectrumIntensityResult`の同種フィールドと同じ規則です。

`class_so_far`は同じサブパッケージ内の`long_period_class`と揃えました。

---

### B. 戻り値・既定値が変わったもの

**置換では見つかりません。該当する呼び出しを目視で確認してください。**

#### B-1. `realtime_intensity` の `reported` 既定値が `False` から `True` へ

`measured_intensity`(`reported=True`)と`realtime_intensity`(`reported=False`)が、同じ名前のキーワードで逆の既定値を持ち、**エラーを出さずに異なる量**を返していました。結果のデータクラスで`.intensity`が丸め後・`.intensity_raw`が丸め前であることに合わせ、両方とも`reported=True`に統一しています。

```python
# 0.2.2: 丸め前の連続値が返っていた
series = realtime_intensity(acceleration, 100.0)

# 0.3.0: 同じ結果を得るには明示する
series = realtime_intensity(acceleration, 100.0, reported=False)
```

プロットや後段の計算には`reported=False`(丸め前)が適しています。

#### B-2. PGAが常にgalを返すようになり、`unit=` を受け取るようになりました

`peak_ground_acceleration`・`component_peak_acceleration`は、0.2.2では`unit=`を持たず**入力単位のまま**の値を返していました。一方PGV・PGDは`unit=`を取り常にcm/s・cmを返します。同じ記録からPGAとPGVを計算すると、片方がm/s²・片方がcm/sという状態になり得ました。

```python
# 0.2.2: 入力がm/s^2なら戻り値もm/s^2
pga = peak_ground_acceleration(acceleration_mps2)

# 0.3.0: unitを明示する。戻り値は常にgal
pga = peak_ground_acceleration(acceleration_mps2, unit="m/s^2")
```

**gal単位の入力しか使っていない場合、既定が`unit="gal"`なので挙動は変わりません。** m/s²やgを渡していた箇所だけ確認してください。

#### B-3. `apply_jma_filter_fft` がタプルではなくデータクラスを返します

```python
# 0.2.2
filtered, frequency, response = apply_jma_filter_fft(acceleration_gal, 100.0)

# 0.3.0
result = apply_jma_filter_fft(acceleration, 100.0, unit="gal")
filtered = result.filtered_acceleration_gal
frequency = result.frequency_hz
response = result.response
```

第1引数名も`acceleration_gal`から`acceleration`に変わり、`unit=`を受け取るようになりました(キーワードで渡していた場合のみ影響します)。`sampling_rate_hz`に既定値100.0が付きました。

#### B-4. `scale_acceleration_to_intensity` の `target_intensity_raw` がキーワード専用に

0.2.2では第2引数が`target_intensity_raw`で、他のすべての加速度を取る関数(第2引数が`sampling_rate_hz`)と食い違っていました。`f(record, 200.0)`が「200 Hz」ではなく「生震度200へスケール」を意味し、エラーも出ませんでした。

```python
# 0.2.2
scaled, factor = scale_acceleration_to_intensity(acceleration, 5.0, 100.0)

# 0.3.0
scaled, factor = scale_acceleration_to_intensity(acceleration, 100.0, target_intensity_raw=5.0)
```

もともとキーワードで渡していた場合は、引数の順序だけ確認すれば動きます。

#### B-5. `SpectrumIntensityEstimator` が非100 Hzで警告しなくなりました

`warn_nonstandard_rate` 引数も削除しました。SI値には特定のサンプリング周波数に紐づく公表定数がなく(`calculate_spectrum_intensity`のdocstringに以前から明記)、バッチ側は最初から警告していませんでした。同じ記録に対して推定器だけが警告する状態を解消したものです。

`SpectrumIntensityEstimator(rate, warn_nonstandard_rate=...)`と書いていた場合は、その引数を削除してください。

#### B-6. `RealtimeIntensityEstimator` の設定属性が読み取り専用になりました

`sampling_rate_hz`・`input_unit`・`component_axis`・`allow_fewer_components`・`duration_samples`・`window_samples`・`filter_design`は、いずれも構築時にサンプリング周波数から導出される値と相互に依存しています。0.2.2では書き換え可能な属性で、例えば`estimator.sampling_rate_hz = 50.0`としても`window_samples`やフィルタ設計は追従せず、静かに不整合になりました。

読み取りは従来どおりです。設定を変えたい場合は新しいインスタンスを作ってください。

---

### C. 追加されたもの(後方互換)

#### C-1. 例外・警告クラスがトップレベルから使えます

`pyshindo.exceptions`経由の従来の書き方も引き続き動作します。

```python
from pyshindo import InvalidAccelerationError, NonstandardSamplingRateWarning
```

対象: `PyShindoError`、`InvalidAccelerationError`、`InsufficientDataError`、`UnstableFilterError`、`DataFormatError`、`PyShindoWarning`、`NonstandardSamplingRateWarning`、`NonstandardProcessingWarning`、`MissingComponentWarning`、`FractionalDurationWarning`。

#### C-2. トップレベルに追加されたもの

- `pyshindo.long_period` — `import pyshindo`だけで`pyshindo.long_period.calculate_long_period_class(...)`が使えます(従来は`from pyshindo.long_period import ...`が必須でした)。
- `MeasuredIntensityTiming`、`RealtimeChunkTiming` — 他の`*Timing`型と同様にトップレベルから型注釈に使えます。
- `report_intensity_array`、`intensity_series_from_threshold_acceleration` — それぞれ`report_intensity`・`intensity_from_threshold_acceleration`の配列版。
- `JMAFilterResult` — B-3の戻り値型。

#### C-3. `RealtimeIntensityEstimator.approximate_scale`

ストリーミング中の現時点での震度階級。入力前は`None`です。従来は`classify_intensity(estimator.approximate_intensity)`と自分で書く必要がありました。

なお`LongPeriodEstimator`・`SpectrumIntensityEstimator`にある`result()`は、`RealtimeIntensityEstimator`には意図的に用意していません。理由は[`docs/api.md`](api.md)の該当節を参照してください。

#### C-4. 警告の発生位置が呼び出し元を指すようになりました

`NonstandardSamplingRateWarning`・`MissingComponentWarning`・`FractionalDurationWarning`が、pyshindo内部のソース行ではなく利用者自身のコード行を指すようになりました。警告をモジュール単位でフィルタしている場合は、対象が変わる可能性があります。

---

### PGDの計算方法について(0.2.2で変更済み)

**0.2.2より前のバージョンから移行する場合のみ**該当します。

気象庁が`max.csv`で公表している最大変位は、加速度の二重積分ではなく、気象庁の機械式1倍強震計(固有周期6秒、減衰定数0.55)の特性を再現する専用フィルタで算出されています。公表値に合わせる目的であればこちらを使ってください。

```python
import numpy as np
from pyshindo import apply_strong_motion_displacement_filter, vector_resultant

displacement = apply_strong_motion_displacement_filter(acceleration, 100.0, unit="gal")
pgd = float(np.max(vector_resultant(displacement)))
```

`integrate_to_displacement` / `component_peak_displacement` / `peak_ground_displacement`(補正なしの二重台形積分)は引き続き利用でき、動作も変わっていません。ただし気象庁の公表値とは一致しません。詳細は[`docs/validation.md`](validation.md)を参照してください。
