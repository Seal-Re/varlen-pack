"""Best-fit-decreasing (BFD) bin packing for variable-length sequences.

The algorithm is deliberately small and dependency-free.  Given a list of
sequence lengths and a fixed row width (``pack_len``), it returns a list of
rows, where each row is a list of indices into the input list.  A sequence is
never split across rows.

Properties:

* Every sample is placed exactly once.
* Every row's total length is ``<= pack_len``.
* The result is deterministic for a given input order: the sort key is
  ``(-length, original_index)``.
* Over-length and empty sequences are rejected rather than silently truncated.

Complexity is ``O(n log n + n log r)`` where ``r`` is the number of rows
(``bisect`` over the sorted remaining-capacity list).
"""

from __future__ import annotations

import bisect
from typing import List, Sequence


def bfd_pack(lengths: Sequence[int], pack_len: int) -> List[List[int]]:
    """Pack ``lengths`` into rows of at most ``pack_len`` tokens.

    Args:
        lengths: Non-empty, positive sequence lengths.
        pack_len: Row width.  Must be at least ``max(lengths)``.

    Returns:
        A list of rows.  Each row is a list of indices into ``lengths`` in
        placement order (which is decreasing length order for a fresh row).

    Raises:
        ValueError: If ``pack_len`` is not positive, a length is not positive,
            or a sequence is longer than ``pack_len``.
    """
    if pack_len <= 0:
        raise ValueError(f"pack_len must be positive, got {pack_len}")

    n = len(lengths)
    if n == 0:
        return []

    for i, length in enumerate(lengths):
        if length <= 0:
            raise ValueError(
                f"sequence {i} has non-positive length {length}; "
                "filter empty samples before packing"
            )
        if length > pack_len:
            raise ValueError(
                f"sequence {i} length {length} exceeds pack_len {pack_len}; "
                "increase pack_len or split/filter the sample"
            )

    # Decreasing length, then original index for determinism.
    order = sorted(range(n), key=lambda i: (-lengths[i], i))

    rows: List[List[int]] = []
    # Sorted (remaining_capacity, row_index).  bisect_left((length, -1)) finds
    # the tightest row that can still fit this sequence.
    remaining: List[tuple[int, int]] = []

    for i in order:
        length = lengths[i]
        k = bisect.bisect_left(remaining, (length, -1))
        if k == len(remaining):
            rows.append([i])
            bisect.insort(remaining, (pack_len - length, len(rows) - 1))
        else:
            capacity, row_index = remaining.pop(k)
            rows[row_index].append(i)
            bisect.insort(remaining, (capacity - length, row_index))

    return rows
