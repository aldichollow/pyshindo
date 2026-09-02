"""Optional Plotly visualizations."""

from .figures import (
    acceleration_figure,
    add_intensity_bands,
    amplitude_duration_figure,
    filter_response_figure,
    intensity_comparison_figure,
    jma_filter_components_figure,
    measured_result_figure,
    realtime_result_figure,
)
from .theme import JMA_INTENSITY_COLORS, LINE_COLORS, TEXT_ON_INTENSITY, apply_theme

__all__ = [
    "JMA_INTENSITY_COLORS",
    "LINE_COLORS",
    "TEXT_ON_INTENSITY",
    "acceleration_figure",
    "add_intensity_bands",
    "amplitude_duration_figure",
    "apply_theme",
    "filter_response_figure",
    "intensity_comparison_figure",
    "jma_filter_components_figure",
    "measured_result_figure",
    "realtime_result_figure",
]
