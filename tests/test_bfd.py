"""Unit tests for the dependency-free BFD packer."""

import random

import pytest

from varlen_pack.bfd import bfd_pack


def _naive_pack(lengths, pack_len):
    """Sequential greedy packing, used only as a baseline."""
    rows = []
    current = []
    used = 0
    for i, length in enumerate(lengths):
        if used + length > pack_len:
            rows.append(current)
            current = []
            used = 0
        current.append(i)
        used += length
    if current:
        rows.append(current)
    return rows


def test_capacity_and_conservation():
    rng = random.Random(0)
    lengths = [rng.randint(11, 608) for _ in range(500)]
    rows = bfd_pack(lengths, 2048)

    placed = sorted(i for row in rows for i in row)
    assert placed == list(range(len(lengths)))
    for row in rows:
        assert sum(lengths[i] for i in row) <= 2048


def test_determinism():
    lengths = [17, 200, 5, 1200, 42, 608, 11]
    assert bfd_pack(lengths, 2048) == bfd_pack(lengths, 2048)


def test_exact_fit_uses_one_row():
    assert bfd_pack([1024, 1024], 2048) == [[0, 1]]


def test_empty_input():
    assert bfd_pack([], 2048) == []


def test_overlength_raises():
    with pytest.raises(ValueError, match="exceeds pack_len"):
        bfd_pack([1, 2049], 2048)


def test_nonpositive_length_raises():
    with pytest.raises(ValueError, match="non-positive"):
        bfd_pack([10, 0], 2048)
    with pytest.raises(ValueError, match="non-positive"):
        bfd_pack([-1], 2048)


def test_invalid_pack_len_raises():
    with pytest.raises(ValueError, match="pack_len"):
        bfd_pack([1], 0)


def test_bfd_is_no_worse_than_naive():
    rng = random.Random(1)
    lengths = [rng.randint(11, 608) for _ in range(1000)]
    assert len(bfd_pack(lengths, 2048)) <= len(_naive_pack(lengths, 2048))
