from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from pyshindo._order import RollingKthLargest


def _check_against_brute_force(window_size: int, k: int, values: list[float]) -> None:
    rolling = RollingKthLargest(window_size, k)
    history: deque[float] = deque(maxlen=window_size)
    storage_limit = 2 * window_size + 64
    for value in values:
        history.append(value)
        actual = rolling.update(value)
        expected = None if len(history) < k else sorted(history, reverse=True)[k - 1]
        assert actual == expected
        # The lazy-deletion heaps must stay bounded, not grow with the stream.
        assert len(rolling._top) + len(rolling._rest) <= storage_limit


@pytest.mark.parametrize(
    ("window_size", "k"),
    [(50, 5), (40, 40), (40, 1), (1, 1), (30, 10)],
    ids=["typical", "k-equals-window", "k-equals-one", "window-equals-one", "alternating"],
)
def test_rolling_kth_largest_matches_brute_force_on_adversarial_patterns(
    window_size: int,
    k: int,
) -> None:
    values = [1e9 if i % 2 == 0 else -1e9 for i in range(2000)]
    _check_against_brute_force(window_size, k, values)


def test_rolling_kth_largest_matches_brute_force_on_large_random_stream() -> None:
    # Long enough, and with a small enough window, to trigger several
    # rounds of lazy-heap compaction while values are still churning.
    generator = np.random.default_rng(42)
    values = [float(generator.integers(0, 5)) for _ in range(20_000)]
    _check_against_brute_force(window_size=137, k=11, values=values)
