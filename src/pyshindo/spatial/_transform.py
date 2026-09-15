"""Monotone value transforms applied before interpolation and undone after.

Internal to :mod:`pyshindo.spatial`; the public surface is
:class:`~pyshindo.spatial.ValueTransform` plus the ``transform=`` keyword
every plan's ``interpolate`` method accepts.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .models import ValueTransform

type FloatArray = npt.NDArray[np.float64]


def forward_transform(values: FloatArray, transform: ValueTransform) -> FloatArray:
    """Map validated station values into the space interpolation weights average in."""
    if transform is ValueTransform.IDENTITY:
        return values
    if np.any(values <= 0.0):
        raise ValueError(
            "transform='log' requires every station value to be strictly positive. "
            "PGA, PGV, SI, and Sva are non-negative by definition but a station "
            "reading of exactly zero has no logarithm; use transform='identity', or "
            "exclude that station before interpolating."
        )
    return np.log(values)


def inverse_transform(values: FloatArray, transform: ValueTransform) -> FloatArray:
    """Undo :func:`forward_transform` on interpolated values.

    ``values`` may contain NaN for unsupported cells; ``exp(nan)`` is NaN,
    so no separate masking is needed here.
    """
    if transform is ValueTransform.IDENTITY:
        return values
    return np.exp(values)
