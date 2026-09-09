"""Optional Plotly visualizations."""

from .figures import (
    acceleration_figure,
    add_intensity_bands,
    amplitude_duration_figure,
    filter_response_figure,
    filter_stages_figure,
    intensity_comparison_figure,
    jma_filter_components_figure,
    long_period_spectrum_figure,
    measured_result_figure,
    realtime_result_figure,
)
from .maps import (
    continuous_value_map_figure,
    intensity_map_figure,
    long_period_class_map_figure,
)
from .theme import (
    JMA_INTENSITY_COLORS,
    LINE_COLORS,
    LONG_PERIOD_CLASS_COLORS,
    STAGE_COLORS,
    TEXT_ON_INTENSITY,
    apply_theme,
)

__all__ = [
    "JMA_INTENSITY_COLORS",
    "LINE_COLORS",
    "LONG_PERIOD_CLASS_COLORS",
    "STAGE_COLORS",
    "TEXT_ON_INTENSITY",
    "acceleration_figure",
    "add_intensity_bands",
    "amplitude_duration_figure",
    "apply_theme",
    "continuous_value_map_figure",
    "filter_response_figure",
    "filter_stages_figure",
    "intensity_comparison_figure",
    "intensity_map_figure",
    "jma_filter_components_figure",
    "long_period_class_map_figure",
    "long_period_spectrum_figure",
    "measured_result_figure",
    "realtime_result_figure",
]
