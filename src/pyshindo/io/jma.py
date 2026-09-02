"""Read and download JMA strong-motion acceleration text records.

The parser follows the seven-line header documented by the Japan
Meteorological Agency. Download helpers are deliberately explicit: this package
never bundles or mirrors provider records.

Reference
---------
Japan Meteorological Agency, *Strong-motion data format*.
https://ds.data.jma.go.jp/eqev/data/kyoshin/jishin/format.html
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from ..exceptions import DataFormatError
from ..models import DownloadedRecord, JMARecord, JMARecordMetadata

_TOKEN_SEPARATOR = re.compile(r"[\s,]+")
_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
_DEFAULT_ENCODINGS = ("utf-8-sig", "cp932")
"""UTF-8 first: CP932 rarely raises ``UnicodeDecodeError`` on arbitrary bytes,
so trying it first would silently mis-decode a genuine no-BOM UTF-8 record.
CP932 (the documented official format) is the fallback. Not airtight either
way -- some uncommon CP932 kanji happen to form valid UTF-8 too -- so pass
``encoding`` explicitly when it is known."""


def _split(line: str) -> list[str]:
    return [token for token in _TOKEN_SEPARATOR.split(line.strip()) if token]


def _numbers(line: str) -> list[float]:
    return [float(match.group(0)) for match in _FLOAT_PATTERN.finditer(line)]


def _first_float(line: str, field_name: str) -> float:
    numbers = _numbers(line)
    if not numbers:
        raise DataFormatError(f"Could not parse {field_name} from {line!r}.")
    return numbers[0]


def _parse_start_time(line: str) -> datetime:
    numbers = _numbers(line)
    if len(numbers) < 6:
        raise DataFormatError(f"Could not parse the record start time from {line!r}.")
    year, month, day, hour, minute = (int(value) for value in numbers[:5])
    second = numbers[5]
    whole_second = int(second)
    microsecond = int(round((second - whole_second) * 1_000_000))
    try:
        base = datetime(year, month, day, hour, minute, whole_second)
    except ValueError as exc:
        raise DataFormatError(f"The record start time is invalid: {line!r}.") from exc
    return base + timedelta(microseconds=microsecond)


def parse_jma_text(text: str, *, source: str | None = None) -> JMARecord:
    """Parse one JMA strong-motion text record.

    The official format contains seven header lines followed by one to three
    acceleration values per row. Comma-separated and whitespace-separated rows
    are both accepted because the format is distributed as MS-DOS text.
    """
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    if len(lines) < 8:
        raise DataFormatError("A JMA record must contain seven header lines and data rows.")

    station_tokens = _split(lines[0])
    if not station_tokens:
        raise DataFormatError("The station-code line is empty.")
    station_code = station_tokens[-1]
    latitude = _first_float(lines[1], "latitude")
    longitude = _first_float(lines[2], "longitude")
    sampling_rate = _first_float(lines[3], "sampling rate")
    if not np.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise DataFormatError("The latitude must be finite and within [-90, 90].")
    if not np.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise DataFormatError("The longitude must be finite and within [-180, 180].")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise DataFormatError("The sampling rate must be finite and greater than zero.")

    unit_tokens = _split(lines[4])
    if not unit_tokens:
        raise DataFormatError("The unit line is empty.")
    unit = unit_tokens[-1]
    start_time = _parse_start_time(lines[5])
    component_names = tuple(_split(lines[6]))
    if not 1 <= len(component_names) <= 3:
        raise DataFormatError(
            "The component-name line must define one, two, or three components."
        )

    rows: list[list[float]] = []
    for line_number, line in enumerate(lines[7:], start=8):
        if not line:
            continue
        tokens = _split(line)
        if len(tokens) != len(component_names):
            raise DataFormatError(
                f"Line {line_number} contains {len(tokens)} values, but "
                f"{len(component_names)} components were declared."
            )
        try:
            rows.append([float(token) for token in tokens])
        except ValueError as exc:
            raise DataFormatError(f"Line {line_number} contains a non-numeric value.") from exc
    if not rows:
        raise DataFormatError("The record contains no acceleration samples.")

    acceleration = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(acceleration)):
        raise DataFormatError("The record contains a non-finite acceleration value.")
    return JMARecord(
        metadata=JMARecordMetadata(
            station_code=station_code,
            latitude_deg=latitude,
            longitude_deg=longitude,
            sampling_rate_hz=sampling_rate,
            unit=unit,
            start_time=start_time,
            component_names=component_names,
            source=source,
        ),
        acceleration=np.ascontiguousarray(acceleration),
    )


def parse_jma_bytes(
    data: bytes,
    *,
    source: str | None = None,
    encoding: str | None = None,
) -> JMARecord:
    """Decode and parse a JMA record from bytes.

    UTF-8 (with an optional byte-order mark) is attempted first when no
    encoding is supplied, followed by CP932 -- the encoding the official
    format is documented in. See :data:`_DEFAULT_ENCODINGS` for why this
    order matters.
    """
    encodings = (encoding,) if encoding is not None else _DEFAULT_ENCODINGS
    last_error: UnicodeDecodeError | None = None
    for candidate in encodings:
        try:
            return parse_jma_text(data.decode(candidate), source=source)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        message = f"Could not decode the record using {', '.join(encodings)}."
        raise DataFormatError(message) from last_error
    raise RuntimeError("No text encoding was attempted.")


def read_jma_record(
    path: str | os.PathLike[str],
    *,
    encoding: str | None = None,
    source: str | None = None,
) -> JMARecord:
    """Read a local JMA strong-motion text record.

    ``source`` can preserve an upstream URL while ``path`` points to a local
    cache. When omitted, the local path is stored as provenance.
    """
    record_path = Path(path)
    return parse_jma_bytes(
        record_path.read_bytes(),
        source=str(record_path) if source is None else source,
        encoding=encoding,
    )


def download_jma_record(
    url: str,
    destination: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    timeout_s: float = 30.0,
    max_bytes: int = 64 * 1024 * 1024,
    expected_sha256: str | None = None,
    user_agent: str = "pyshindo/0.1 (+scientific-use)",
) -> DownloadedRecord:
    """Download one explicitly selected record to a local file atomically.

    Review the provider's current terms, retain provenance, and cite the
    source when publishing results. Downloads exactly one requested file; does
    not crawl, mirror, or poll. ``overwrite=False`` is checked once before the
    download starts, not held as a lock against a concurrent writer.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an absolute HTTP or HTTPS URL.")
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be finite and greater than zero.")
    if max_bytes < 1:
        raise ValueError("max_bytes must be at least one.")
    if expected_sha256 is not None:
        expected_sha256 = expected_sha256.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("expected_sha256 must contain 64 hexadecimal characters.")

    output_path = Path(destination)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass overwrite=True to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    temp_path: Path | None = None
    byte_count = 0
    digest = hashlib.sha256()
    headers: dict[str, str]
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            headers = dict(response.headers.items())
            declared_length = response.headers.get("Content-Length")
            if declared_length is not None and int(declared_length) > max_bytes:
                raise ValueError(
                    f"The declared response size exceeds max_bytes={max_bytes}."
                )
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{output_path.name}.",
                dir=output_path.parent,
                delete=False,
            ) as temporary:
                temp_path = Path(temporary.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    if byte_count > max_bytes:
                        raise ValueError(f"The response exceeds max_bytes={max_bytes}.")
                    digest.update(chunk)
                    temporary.write(chunk)
                temporary.flush()
                os.fsync(temporary.fileno())
        if byte_count == 0:
            raise DataFormatError("The download returned an empty file.")
        sha256 = digest.hexdigest()
        if expected_sha256 is not None and sha256 != expected_sha256:
            raise DataFormatError(
                f"SHA-256 mismatch: expected {expected_sha256}, received {sha256}."
            )
        if temp_path is None:
            raise RuntimeError("The temporary download file was not created.")
        temp_path.replace(output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return DownloadedRecord(
        path=output_path,
        url=url,
        byte_count=byte_count,
        sha256=sha256,
        headers=headers,
    )
