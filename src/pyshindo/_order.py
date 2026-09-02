"""Exact rolling order statistics with bounded-memory lazy heaps."""

from __future__ import annotations

import heapq
from collections import deque


class RollingKthLargest:
    """Track the k-th largest value in a fixed-size sliding window.

    Updates are amortized ``O(log window_size)``. The smaller ``top`` heap holds
    the current ``k`` largest values, so its minimum is the requested order
    statistic. Unique sample identifiers make duplicate values unambiguous.
    """

    __slots__ = (
        "window_size",
        "k",
        "_top",
        "_rest",
        "_queue",
        "_location",
        "_value",
        "_top_count",
        "_rest_count",
        "_next_id",
    )

    def __init__(self, window_size: int, k: int) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least one.")
        if k < 1:
            raise ValueError("k must be at least one.")
        if k > window_size:
            raise ValueError("k must not exceed window_size.")
        self.window_size = window_size
        self.k = k
        self._top: list[tuple[float, int]] = []
        self._rest: list[tuple[float, int]] = []
        self._queue: deque[int] = deque()
        self._location: dict[int, int] = {}
        self._value: dict[int, float] = {}
        self._top_count = 0
        self._rest_count = 0
        self._next_id = 0

    def clear(self) -> None:
        """Remove all values while preserving the configured window and rank."""
        self._top.clear()
        self._rest.clear()
        self._queue.clear()
        self._location.clear()
        self._value.clear()
        self._top_count = 0
        self._rest_count = 0
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._location)

    @property
    def ready(self) -> bool:
        """Return whether at least ``k`` values are available."""
        return len(self) >= self.k

    @property
    def threshold(self) -> float | None:
        """Return the current k-th largest value, or ``None`` before ready."""
        if not self.ready:
            return None
        self._clean_top()
        if not self._top:
            raise RuntimeError("Rolling order-statistic invariant was violated.")
        return self._top[0][0]

    def update(self, value: float) -> float | None:
        """Insert one value, evict the oldest if needed, and return the threshold."""
        sample_id = self._next_id
        self._next_id += 1
        self._queue.append(sample_id)
        self._value[sample_id] = value

        self._clean_top()
        if self._top_count < self.k:
            self._push_top(value, sample_id)
        elif self._top and value > self._top[0][0]:
            moved_value, moved_id = self._pop_top()
            self._push_rest(moved_value, moved_id)
            self._push_top(value, sample_id)
        else:
            self._push_rest(value, sample_id)

        if len(self._queue) > self.window_size:
            self._evict(self._queue.popleft())

        self._rebalance()
        self._maybe_compact()

        # self._top is already clean here, so read it directly instead of
        # going through the threshold property's own clean pass.
        if len(self._location) < self.k:
            return None
        top = self._top
        if not top:
            raise RuntimeError("Rolling order-statistic invariant was violated.")
        return top[0][0]

    def _push_top(self, value: float, sample_id: int) -> None:
        heapq.heappush(self._top, (value, sample_id))
        self._location[sample_id] = 1
        self._top_count += 1

    def _push_rest(self, value: float, sample_id: int) -> None:
        heapq.heappush(self._rest, (-value, sample_id))
        self._location[sample_id] = 0
        self._rest_count += 1

    def _clean_top(self) -> None:
        top = self._top
        location = self._location
        value_map = self._value
        while top:
            value, sample_id = top[0]
            if location.get(sample_id) == 1 and value_map.get(sample_id) == value:
                return
            heapq.heappop(top)

    def _clean_rest(self) -> None:
        rest = self._rest
        location = self._location
        value_map = self._value
        while rest:
            negative_value, sample_id = rest[0]
            value = -negative_value
            if location.get(sample_id) == 0 and value_map.get(sample_id) == value:
                return
            heapq.heappop(rest)

    def _pop_top(self) -> tuple[float, int]:
        self._clean_top()
        value, sample_id = heapq.heappop(self._top)
        if self._location.get(sample_id) != 1:
            raise RuntimeError("Invalid top-heap entry.")
        self._top_count -= 1
        return value, sample_id

    def _pop_rest(self) -> tuple[float, int]:
        self._clean_rest()
        negative_value, sample_id = heapq.heappop(self._rest)
        if self._location.get(sample_id) != 0:
            raise RuntimeError("Invalid rest-heap entry.")
        self._rest_count -= 1
        return -negative_value, sample_id

    def _evict(self, sample_id: int) -> None:
        location = self._location.pop(sample_id)
        self._value.pop(sample_id)
        if location == 1:
            self._top_count -= 1
        else:
            self._rest_count -= 1

    def _rebalance(self) -> None:
        target_top_count = min(self.k, len(self))
        self._clean_top()
        self._clean_rest()

        while self._top_count > target_top_count:
            value, sample_id = self._pop_top()
            self._push_rest(value, sample_id)
        while self._top_count < target_top_count:
            value, sample_id = self._pop_rest()
            self._push_top(value, sample_id)

        # Both heaps are already clean at the front: only eviction (already
        # handled above) can make an entry stale, and pushing never does.
        top = self._top
        rest = self._rest
        while top and rest and top[0][0] < -rest[0][0]:
            top_value, top_id = self._pop_top()
            rest_value, rest_id = self._pop_rest()
            self._push_top(rest_value, rest_id)
            self._push_rest(top_value, top_id)
            self._clean_top()
            self._clean_rest()

    def _maybe_compact(self) -> None:
        limit = 2 * self.window_size + 64
        if len(self._top) + len(self._rest) <= limit:
            return
        self._top = [
            (value, sample_id)
            for sample_id, value in self._value.items()
            if self._location[sample_id] == 1
        ]
        self._rest = [
            (-value, sample_id)
            for sample_id, value in self._value.items()
            if self._location[sample_id] == 0
        ]
        heapq.heapify(self._top)
        heapq.heapify(self._rest)
