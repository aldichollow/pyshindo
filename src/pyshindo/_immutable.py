"""Helpers for freezing array and mapping fields in ``__post_init__``.

A ``frozen=True`` dataclass refuses attribute *reassignment*, but a NumPy
array or ``dict`` held by one of its fields is a mutable object underneath:
``result.some_array[0] = 999.0`` or ``result.some_mapping["x"] = 1`` both
succeed silently, leaving the dataclass only half-frozen. Every result type
in this package that holds an array or a plain ``dict`` calls one of these
from its own ``__post_init__`` to close that gap, using
``object.__setattr__`` the same way a frozen dataclass's own ``__post_init__``
always must to set a field at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import numpy.typing as npt


def frozen_array(
    values: npt.ArrayLike, *, dtype: npt.DTypeLike | None = None
) -> npt.NDArray[np.generic]:
    """Return a read-only, contiguous copy of ``values``.

    Always copies rather than trusting the caller not to mutate the array
    they passed in after construction -- the copy is what this function
    makes read-only, not the caller's own array.
    """
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def frozen_mapping[K, V](values: Mapping[K, V]) -> MappingProxyType[K, V]:
    """Return a read-only view over a copy of ``values``.

    Copies first, the same reasoning as :func:`frozen_array`: a
    ``MappingProxyType`` over the caller's own dict would still change
    underneath the dataclass if the caller kept mutating that dict.
    """
    return MappingProxyType(dict(values))
