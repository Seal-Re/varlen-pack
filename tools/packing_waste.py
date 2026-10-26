#!/usr/bin/env python3
"""Measure packing efficiency on a real length distribution.

This is a decision tool, not part of the training path.  Given a file with one
sequence length per line (or a JSON array), it compares:

* naive sequential greedy packing (no sorting),
* first-fit decreasing (FFD),
* best-fit decreasing (BFD, the algorithm used by the collator),

at a chosen micro-batch size and row width.  It reports rows per batch, waste
percentage, average gap per row, and CPU time per batch.

Waste matters: gap tokens are pad tokens, but they still go through the GEMMs
and the backward pass.  At a small micro-batch, the last row's structural tail
can dominate; at a large micro-batch, BFD should approach the theoretical
optimum.

Example:
    python tools/packing_waste.py --lengths lengths.txt --micro 100 --pack-len 2048
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from varlen_pack.bfd import bfd_pack  # noqa: E402


def naive_pack(lengths, pack_len):
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


def ffd_pack(lengths, pack_len):
    rows = []
    used = []
    for i in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        length = lengths[i]
        for row_index, capacity in enumerate(used):
            if capacity + length <= pack_len:
                rows[row_index].append(i)
                used[row_index] += length
                break
        else:
            rows.append([i])
            used.append(length)
    return rows


def simulate(lengths, micro, pack_len, packer):
    """Drop the last partial micro-batch, matching ``drop_last=True``."""
    started = time.perf_counter()
    batches = rows = real_tokens = 0
    for start in range(0, len(lengths) - micro + 1, micro):
        chunk = lengths[start : start + micro]
        packed = packer(chunk, pack_len)
        batches += 1
        rows += len(packed)
        real_tokens += sum(chunk)
    elapsed = time.perf_counter() - started
    capacity = rows * pack_len
    return {
        "batches": batches,
        "rows_per_batch": rows / batches if batches else 0.0,
        "waste_pct": 100.0 * (capacity - real_tokens) / real_tokens if real_tokens else 0.0,
        "avg_gap_per_row": (capacity - real_tokens) / rows if rows else 0.0,
        "ms_per_batch": 1000.0 * elapsed / batches if batches else 0.0,
    }


def load_lengths(path):
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return [int(x) for x in json.loads(text)]
    return [int(line) for line in text.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", required=True,
                        help="file with one integer length per line, or a JSON array")
    parser.add_argument("--micro", type=int, default=1024,
                        help="samples per micro-batch (default: 1024)")
    parser.add_argument("--pack-len", type=int, default=2048,
                        help="row width in tokens (default: 2048)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args()

    lengths = load_lengths(args.lengths)
    if not lengths:
        parser.error("no lengths found")
    random.Random(args.seed).shuffle(lengths)

    over = [length for length in lengths if length > args.pack_len]
    if over:
        parser.error(
            f"{len(over)} sample(s) exceed pack_len={args.pack_len}; "
            "filter or split them before packing"
        )

    algorithms = [
        ("naive", naive_pack),
        ("ffd", ffd_pack),
        ("bfd", bfd_pack),
    ]
    results = []
    for name, packer in algorithms:
        stats = simulate(lengths, args.micro, args.pack_len, packer)
        stats.update({"algorithm": name, "micro": args.micro,
                      "pack_len": args.pack_len})
        results.append(stats)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(
        f"samples={len(lengths)}  min/mean/max="
        f"{min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)}  "
        f"micro={args.micro}  pack_len={args.pack_len}"
    )
    print(f"{'algorithm':<10} {'rows/batch':>10} {'waste':>8} "
          f"{'gap/row':>9} {'ms/batch':>9}")
    for stats in results:
        print(
            f"{stats['algorithm']:<10} {stats['rows_per_batch']:>10.1f} "
            f"{stats['waste_pct']:>7.2f}% {stats['avg_gap_per_row']:>9.1f} "
            f"{stats['ms_per_batch']:>9.2f}"
        )


if __name__ == "__main__":
    main()
