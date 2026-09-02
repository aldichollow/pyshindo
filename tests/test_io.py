from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from pyshindo.exceptions import DataFormatError
from pyshindo.io import parse_jma_bytes, parse_jma_text

_SAMPLE = """AA06EA01
35.43
133.34
100
 gal
2000 10 06 13 30 00.50
NS,EW,UD
1.0,2.0,3.0
4.0,5.0,6.0
"""


def test_parse_jma_text() -> None:
    record = parse_jma_text(_SAMPLE, source="memory")
    assert record.metadata.station_code == "AA06EA01"
    assert record.metadata.sampling_rate_hz == 100.0
    assert record.metadata.start_time == datetime(2000, 10, 6, 13, 30, 0, 500_000)
    assert record.metadata.component_names == ("NS", "EW", "UD")
    assert np.array_equal(record.acceleration, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.array_equal(record.time_s, [0.0, 0.01])


def test_parse_rejects_wrong_column_count() -> None:
    with pytest.raises(DataFormatError):
        parse_jma_text(_SAMPLE.replace("4.0,5.0,6.0", "4.0,5.0"))


@pytest.mark.parametrize("encoding", ["utf-8", "cp932"])
@pytest.mark.parametrize("station_name", ["東京", "女川町"])
def test_parse_jma_bytes_decodes_japanese_station_names_regardless_of_encoding(
    encoding: str,
    station_name: str,
) -> None:
    # CP932 rarely raises UnicodeDecodeError on arbitrary bytes, so trying it
    # before UTF-8 can silently mis-decode a genuine no-BOM UTF-8 record.
    text = _SAMPLE.replace("AA06EA01", f"AA06EA01 {station_name}")
    record = parse_jma_bytes(text.encode(encoding))
    assert record.metadata.station_code == station_name


def test_read_jma_record_can_preserve_upstream_source(tmp_path) -> None:
    path = tmp_path / "record.csv"
    path.write_text(_SAMPLE, encoding="utf-8")
    record = __import__("pyshindo.io", fromlist=["read_jma_record"]).read_jma_record(
        path, source="https://example.test/record.csv"
    )
    assert record.metadata.source == "https://example.test/record.csv"
