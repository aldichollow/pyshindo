"""JMA long-period ground motion class (長周期地震動階級).

A different quantity from instrumental seismic intensity and a different
calculation: it uses only the two horizontal components, drives a bank of
damped oscillators covering 1.6 to 7.8 seconds, and classifies the largest
absolute velocity response. It estimates how the upper floors of a tall
building would respond to the motion recorded at the ground, which is not the
same thing as a structural analysis of any particular building.

This is the calculation for an *observed* record. JMA's earthquake early
warning also predicts a long-period class before the shaking arrives; that is
a separate problem and is not implemented here.

See ``docs/long-period.md`` for the algorithm and its primary sources.
"""

from ._core import OscillatorSolver
from .calculation import apply_ground_motion_high_pass, calculate_long_period_class
from .models import (
    LongPeriodBandResult,
    LongPeriodResult,
    LongPeriodTiming,
    LongPeriodUpdate,
)
from .realtime import LongPeriodEstimator
from .scale import (
    LONG_PERIOD_CLASS_INTERVALS,
    OFFICIAL_DAMPING_RATIO,
    OFFICIAL_PERIODS_S,
    PERIOD_BANDS,
    LongPeriodClass,
    band_period_range_s,
    classify_long_period,
    long_period_class_label,
)

__all__ = [
    "LONG_PERIOD_CLASS_INTERVALS",
    "OFFICIAL_DAMPING_RATIO",
    "OFFICIAL_PERIODS_S",
    "PERIOD_BANDS",
    "LongPeriodBandResult",
    "LongPeriodClass",
    "LongPeriodEstimator",
    "LongPeriodResult",
    "LongPeriodTiming",
    "LongPeriodUpdate",
    "OscillatorSolver",
    "apply_ground_motion_high_pass",
    "band_period_range_s",
    "calculate_long_period_class",
    "classify_long_period",
    "long_period_class_label",
]
