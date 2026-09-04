# 観測データの入出力

## JMA強震記録テキスト形式

`pyshindo.io` は気象庁が公開する7行ヘッダのテキスト形式を解釈します。

1. 観測点コード
2. 緯度
3. 経度
4. サンプリング周波数
5. 単位
6. 開始時刻
7. 成分名

8行目以降が加速度データです。北・東・上方向を正とする形式で、`parse_jma_text` / `parse_jma_bytes` / `read_jma_record` は成分名と取得元をメタデータに保持します。

公式フォーマット: https://ds.data.jma.go.jp/eqev/data/kyoshin/jishin/format.html

## ダウンロード

観測記録はリポジトリに同梱しません。明示的に指定した1件のURLだけを取得します。

```python
from pathlib import Path
from pyshindo.io import download_jma_record, read_jma_record

url = "https://ds.data.jma.go.jp/eqev/data/kyoshin/jishin/001006_tottori-seibu/dat/AA06EA01.csv"
path = Path(".cache/pyshindo/AA06EA01.csv")
if not path.exists():
    download_jma_record(url, path)
record = read_jma_record(path)
```

サイズ上限を設け、一時ファイルに書き込んでから原子的に配置し、SHA-256を任意で検証します。公開前に提供元の利用規約(第三者への再配布禁止、出典明記など)を確認してください。

気象庁の計算解説ページで使われている2000年鳥取県西部地震(米子)の記録は、公表値として計測震度5.1・閾値加速度127.85 galを示しています。実際にこの記録をダウンロードして`calculate_measured_intensity`にかけると計測震度5.1が再現されます(サンプリング周波数・単位・加速度データの解釈は正しいということです)。ただし、この記録のヘッダは観測点コードや緯度・経度の行が資料どおりの単純な1行1項目形式になっておらず、`station_code` / `latitude_deg` / `longitude_deg` は信頼できません。古い記録には形式のばらつきがあるため、これらのメタデータ項目は必要に応じて元ファイルを直接確認してください。

## その他の観測網

K-NET/KiK-netなど他のアーカイブは独自の形式・認証・利用規約を持つため、本パッケージは提供元ごとのスクレイピングには対応していません。その代わりに、既にこれらの形式を読める [ObsPy](https://docs.obspy.org/) との変換ヘルパーを用意しています。

## ObsPyとの連携

```bash
python -m pip install "pyshindo[obspy]"
```

```python
import obspy
from pyshindo import calculate_measured_intensity
from pyshindo.obspy_interop import from_obspy_stream

stream = obspy.read("...")            # K-NET / KiK-net / miniSEED / SAC など
stream = stream.select(station="...")  # 1観測点の3成分に絞る

record = from_obspy_stream(stream, unit="gal")
result = calculate_measured_intensity(
    record.acceleration,
    record.metadata.sampling_rate_hz,
    unit=record.metadata.unit,
)
```

`from_obspy_stream` は変換だけを行う薄いアダプタです。リサンプリング・トリミング・マージ・回転・スケーリングは一切行いません。いずれも記録そのものを書き換える操作であり、ObsPy側に `Stream.resample()` / `Stream.trim()` / `Stream.merge()` / `Stream.remove_response()` として用意されているため、必要なら呼び出し側が先に明示的に適用してください。サンプリング周波数・サンプル数・開始時刻・観測点が食い違うトレースは、暗黙に整合させるのではなくエラーとして拒否します。

### 単位を引数で必須にしている理由

SEEDおよびその周辺の交換形式には、物理単位を確実に伝えるフィールドがありません。ObsPyが返すのはリーダーが生成した数値そのもの(応答除去済みなら `remove_response(output=...)` で指定した量)であり、パッケージ側から単位を判別する手段がありません。そのため `unit` は推測せず必須の引数とし、呼び出し側の申告としてメタデータに記録します。

### 成分の並びとラベル

列はSEEDの方位コード(チャンネル名の末尾1文字)から水平・水平・上下の順に並べ替えられます。`N`/`E`/`Z` はそれぞれ `NS`/`EW`/`UD` になります。`1`/`2` はSEEDでは「方位が特定されていない直交する水平2成分」を意味するため、北・東であるかのように偽らず `H1`/`H2` とラベルします。これは数値的には無害です。計測震度もPGVも成分をユークリッドノルムで合成するため、水平2成分が面内でどう回転していても結果は変わりません(`tests/test_velocity.py` に回転不変性の回帰テストがあります)。

方位コードから判別できない場合は `channel_order=("...", "...", "...")` で並び順を明示できます。
