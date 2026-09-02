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

K-NET/KiK-netなど他のアーカイブは独自の形式・認証・利用規約を持つため、本パッケージは提供元ごとのスクレイピングには対応しません。
