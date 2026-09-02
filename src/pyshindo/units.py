"""Acceleration-unit conversion utilities."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type ArrayLike = npt.ArrayLike

STANDARD_GRAVITY_MPS2 = 9.80665
"""Conventional standard acceleration of gravity in m/s²."""


class AccelerationUnit(StrEnum):
    """Supported acceleration units."""

    GAL = "gal"
    MPS2 = "m/s^2"
    G = "g"

    @classmethod
    def parse(cls, value: str | AccelerationUnit) -> AccelerationUnit:
        """Return a canonical acceleration unit from common spellings."""
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower().replace("²", "2").replace(" ", "")
        aliases = {
            "gal": cls.GAL,
            "gal(cm/s/s)": cls.GAL,  # as written in official JMA strong-motion files
            "cm/s2": cls.GAL,
            "cm/s^2": cls.GAL,
            "cms-2": cls.GAL,
            "m/s2": cls.MPS2,
            "m/s^2": cls.MPS2,
            "ms-2": cls.MPS2,
            "g": cls.G,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            supported = ", ".join(unit.value for unit in cls)
            message = f"Unsupported acceleration unit {value!r}; choose {supported}."
            raise ValueError(message) from exc


def conversion_factor(
    from_unit: str | AccelerationUnit,
    to_unit: str | AccelerationUnit,
) -> float:
    """Return the multiplicative conversion factor between acceleration units."""
    source = AccelerationUnit.parse(from_unit)
    target = AccelerationUnit.parse(to_unit)
    to_mps2 = {
        AccelerationUnit.GAL: 0.01,
        AccelerationUnit.MPS2: 1.0,
        AccelerationUnit.G: STANDARD_GRAVITY_MPS2,
    }
    return to_mps2[source] / to_mps2[target]


def convert_acceleration(
    values: ArrayLike,
    from_unit: str | AccelerationUnit,
    to_unit: str | AccelerationUnit,
    *,
    copy: bool = True,
) -> FloatArray:
    """Convert acceleration values while preserving their array shape.

    Unit-changing conversions always allocate so the caller's array is never
    modified. ``copy=False`` only avoids a copy when the source and target units
    are identical and the input already has a compatible dtype.
    """
    factor = conversion_factor(from_unit, to_unit)
    if factor == 1.0:
        array = np.asarray(values, dtype=np.float64)
        return array.copy() if copy else array
    array = np.array(values, dtype=np.float64, copy=True)
    array *= factor
    return array


def to_gal(
    values: ArrayLike,
    unit: str | AccelerationUnit,
    *,
    copy: bool = True,
) -> FloatArray:
    """Convert acceleration values to gal (cm/s²)."""
    return convert_acceleration(values, unit, AccelerationUnit.GAL, copy=copy)
