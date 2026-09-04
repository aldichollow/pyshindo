"""Long-period ground motion classes, thresholds, and period bands.

Mirrors the role of :mod:`pyshindo.scale` for instrumental intensity: the
class enum, the published intervals, and the classification functions, kept
separate from the numerical response calculation.

References
----------
Japan Meteorological Agency, "長周期地震動階級および長周期地震動階級関連解説表について".
https://www.jma.go.jp/jma/kishou/know/jishin/ltpgm_explain/about_level.html

Japan Meteorological Agency, "長周期地震動の観測結果ページの見方" (period bands,
and the published use of "階級0").
https://www.data.jma.go.jp/eew/data/ltpgm_explain/about_contents.pdf
"""

from __future__ import annotations

from enum import StrEnum
from math import isnan
from types import MappingProxyType
from typing import Final

import numpy as np

from ..units import FloatArray

# Integer tenths of a second, so neither the grid nor the band membership
# below depends on floating-point accumulation.
PERIOD_DECISECONDS: Final = tuple(range(16, 80, 2))
OFFICIAL_PERIODS_S: Final[FloatArray] = np.array(PERIOD_DECISECONDS, dtype=np.float64) / 10.0
"""The 32 periods JMA evaluates: 1.6, 1.8, ..., 7.8 seconds."""

OFFICIAL_DAMPING_RATIO: Final = 0.05
"""Damping ratio h for the published class definition (5 percent)."""


class LongPeriodClass(StrEnum):
    """JMA long-period ground motion classes.

    JMA publishes classes 1 through 4. ``ZERO`` is not an invention here:
    the observation pages label motion that did not reach class 1 as
    "階級0", and the committee comparison tables use the same category.
    """

    ZERO = "0"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"

    @property
    def japanese(self) -> str:
        """Return the conventional Japanese class label."""
        return self.value

    @property
    def english(self) -> str:
        """Return a readable English class label."""
        return self.value


LONG_PERIOD_CLASS_INTERVALS: Final = MappingProxyType(
    {
        LongPeriodClass.ZERO: (0.0, 5.0),
        LongPeriodClass.ONE: (5.0, 15.0),
        LongPeriodClass.TWO: (15.0, 50.0),
        LongPeriodClass.THREE: (50.0, 100.0),
        LongPeriodClass.FOUR: (100.0, np.inf),
    }
)
"""Absolute velocity response Sva intervals in cm/s, lower bound inclusive.

The lower-bound-inclusive reading is confirmed by JMA's own worked example
(8th committee, 資料3): a record with a maximum Sva of 14.67 cm/s is class 1
and 15.12 cm/s is class 2.
"""

_CLASS_LOWER_BOUNDS: Final[tuple[tuple[float, LongPeriodClass], ...]] = tuple(
    (lower, scale)
    for scale, (lower, _upper) in reversed(tuple(LONG_PERIOD_CLASS_INTERVALS.items()))
)

PERIOD_BANDS: Final = tuple(range(1, 8))
"""The seven period bands JMA reports alongside the overall class, keyed by
their integer second: 1秒台 covers 1.6-1.8 s, 2秒台 covers 2.0-2.8 s, and so
on to 7秒台."""


def band_of_period_deciseconds(deciseconds: int) -> int:
    """Return the JMA period band (1-7) containing a period in tenths of a second."""
    return deciseconds // 10


def band_period_range_s(band: int) -> tuple[float, float]:
    """Return the first and last official period of a band, in seconds."""
    members = [d for d in PERIOD_DECISECONDS if band_of_period_deciseconds(d) == band]
    if not members:
        raise ValueError(f"band must be one of {PERIOD_BANDS}; received {band}.")
    return members[0] / 10.0, members[-1] / 10.0


def classify_long_period(sva_cm_s: float) -> LongPeriodClass:
    """Map an absolute velocity response maximum in cm/s to a JMA class."""
    value = float(sva_cm_s)
    if isnan(value):
        raise ValueError("NaN cannot be classified as a long-period class.")
    if value < 0.0:
        raise ValueError("sva_cm_s must be non-negative.")
    for lower_bound, scale in _CLASS_LOWER_BOUNDS:
        if value >= lower_bound:
            return scale
    return LongPeriodClass.ZERO




def long_period_class_label(
    value: float,
    *,
    language: str = "ja",
) -> str:
    """Return a display label such as ``長周期地震動階級3``."""
    scale = classify_long_period(value)
    normalized = language.strip().lower()
    if normalized in {"ja", "jp", "japanese"}:
        return f"長周期地震動階級{scale.japanese}"
    if normalized in {"en", "english"}:
        return f"Long-period ground motion class {scale.english}"
    raise ValueError("language must be 'ja' or 'en'.")
