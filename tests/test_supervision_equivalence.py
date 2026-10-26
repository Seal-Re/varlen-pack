"""Supervision-set equivalence tests.

These tests do not need a model.  They verify the property that makes packed
training equivalent to padded training at the loss level:

* every sample keeps exactly ``len - 1`` supervised next-token targets;
* the *first* label of every sample is masked, so the previous segment's last
  logit cannot be trained to predict it;
* no supervised target crosses a segment boundary.
"""

from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from varlen_pack.collator import PackedDataCollator  # noqa: E402

PAD = 0


def make_feature(length, seed):
    ids = list(range(seed * 10_000 + 1, seed * 10_000 + 1 + length))
    return {"input_ids": ids, "labels": list(ids)}


def _expected_targets(feature):
    """Padded-path targets: positions 1..L-1 (position 0 is masked)."""
    ids = feature["input_ids"]
    return [(j, ids[j]) for j in range(1, len(ids))]


def _actual_targets(batch, segment):
    start, end = segment.start, segment.end
    out = []
    for j in range(1, end - start):
        label = int(batch["labels"][segment.row, start + j])
        if label != -100:
            out.append((j, int(batch["input_ids"][segment.row, start + j])))
    return out


@pytest.mark.parametrize("mode", ["fa2_positions", "cu_seqlens"])
def test_per_sample_targets_match_padded_reference(mode):
    features = [make_feature(length, i) for i, length in enumerate([7, 5, 4, 11, 3])]
    pack_len = 64
    batch = PackedDataCollator(
        PAD, pack_len=pack_len, mode=mode, return_segment_map=True
    )(features)

    by_sample = {s.sample_index: s for s in batch["segment_map"]}
    assert set(by_sample) == set(range(len(features)))
    for i, feature in enumerate(features):
        assert _actual_targets(batch, by_sample[i]) == _expected_targets(feature)


@pytest.mark.parametrize("mode", ["fa2_positions", "cu_seqlens"])
def test_supervised_count_is_len_minus_one(mode):
    features = [make_feature(length, i) for i, length in enumerate([4, 9, 6])]
    batch = PackedDataCollator(
        PAD, pack_len=32, mode=mode, return_segment_map=True
    )(features)

    counts = Counter()
    for segment in batch["segment_map"]:
        start, end = segment.start, segment.end
        counts[segment.sample_index] += int(
            (batch["labels"][segment.row, start:end] != -100).sum()
        )
    for i, feature in enumerate(features):
        assert counts[i] == len(feature["input_ids"]) - 1


def _segment_ids_per_position(batch, width, segments):
    """Map each output position to its segment id; tail joins the last segment."""
    seg_ids = [-1] * width
    for sid, segment in enumerate(segments):
        seg_ids[segment.start:segment.end] = [sid] * (segment.end - segment.start)
    if segments:
        last = segments[-1]
        seg_ids[last.end:] = [len(segments) - 1] * (width - last.end)
    return seg_ids


def test_no_cross_segment_supervision_pair():
    features = [make_feature(length, i) for i, length in enumerate([7, 5, 4, 9, 6])]
    width = 64
    batch = PackedDataCollator(
        PAD, pack_len=width, mode="fa2_positions", return_segment_map=True
    )(features)

    for row in range(batch["labels"].shape[0]):
        row_segments = [s for s in batch["segment_map"] if s.row == row]
        seg_ids = _segment_ids_per_position(batch, width, row_segments)
        labels = batch["labels"][row]
        for j in range(1, width):
            if int(labels[j]) != -100:
                # The logit at j-1 predicts labels[j].  Both must belong to
                # the same sample segment.
                assert seg_ids[j] == seg_ids[j - 1]
