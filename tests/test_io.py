from __future__ import annotations

import hashlib
import io
from datetime import datetime
from email.message import Message
from unittest import mock

import numpy as np
import pytest

from pyshindo.exceptions import DataFormatError
from pyshindo.io import download_jma_record, parse_jma_bytes, parse_jma_text

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


class _FakeResponse:
    """Stands in for urllib.request.urlopen()'s context-manager result."""

    def __init__(self, data: bytes, *, content_length: int | None = None) -> None:
        self._buffer = io.BytesIO(data)
        self.headers = Message()
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)


def test_download_jma_record_rejects_non_http_url(tmp_path) -> None:
    with pytest.raises(ValueError):
        download_jma_record("ftp://example.test/a.csv", tmp_path / "a.csv")


def test_download_jma_record_rejects_malformed_checksum(tmp_path) -> None:
    with pytest.raises(ValueError):
        download_jma_record(
            "https://example.test/a.csv", tmp_path / "a.csv", expected_sha256="not-hex"
        )


def test_download_jma_record_refuses_to_overwrite_by_default(tmp_path) -> None:
    existing = tmp_path / "a.csv"
    existing.write_text("existing")
    with pytest.raises(FileExistsError):
        download_jma_record("https://example.test/a.csv", existing)


def test_download_jma_record_rejects_declared_size_over_limit(tmp_path) -> None:
    response = _FakeResponse(b"x" * 10, content_length=1000)
    with mock.patch("urllib.request.urlopen", return_value=response), pytest.raises(ValueError):
        download_jma_record("https://example.test/a.csv", tmp_path / "a.csv", max_bytes=100)
    assert not (tmp_path / "a.csv").exists()


def test_download_jma_record_writes_file_and_matches_checksum(tmp_path) -> None:
    payload = _SAMPLE.encode("utf-8")
    expected_hash = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "a.csv"
    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
        result = download_jma_record(
            "https://example.test/a.csv", destination, expected_sha256=expected_hash
        )
    assert destination.read_bytes() == payload
    assert result.sha256 == expected_hash
    assert result.byte_count == len(payload)


def test_download_jma_record_rejects_checksum_mismatch_and_cleans_up(tmp_path) -> None:
    destination = tmp_path / "a.csv"
    with (
        mock.patch("urllib.request.urlopen", return_value=_FakeResponse(b"some data")),
        pytest.raises(DataFormatError),
    ):
        download_jma_record("https://example.test/a.csv", destination, expected_sha256="0" * 64)
    # Neither the destination nor a leftover temp file should remain.
    assert list(tmp_path.iterdir()) == []


def test_download_jma_record_rejects_empty_response(tmp_path) -> None:
    with (
        mock.patch("urllib.request.urlopen", return_value=_FakeResponse(b"")),
        pytest.raises(DataFormatError),
    ):
        download_jma_record("https://example.test/a.csv", tmp_path / "a.csv")
