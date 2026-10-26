"""Unit tests for the packed collator (pure CPU; no model, no GPU)."""

import pytest

torch = pytest.importorskip("torch")

from varlen_pack.collator import PackedDataCollator, Segment  # noqa: E402

PAD = 0


def make_feature(length, seed, *, with_labels=True, group_id=None, channel_id=None):
    """Create a sample with unique token ids so segments are identifiable."""
    ids = list(range(seed * 10_000 + 1, seed * 10_000 + 1 + length))
    feature = {"input_ids": ids}
    if with_labels:
        feature["labels"] = list(ids)
    if group_id is not None:
        feature["group_id"] = group_id
    if channel_id is not None:
        feature["channel_id"] = channel_id
    return feature


def test_fa2_layout_and_boundaries():
    features = [make_feature(7, 0), make_feature(5, 1), make_feature(4, 2)]
    collator = PackedDataCollator(PAD, pack_len=32, mode="fa2_positions",
                                  return_segment_map=True)
    batch = collator(features)

    assert set(batch) == {"input_ids", "labels", "position_ids",
                          "attention_mask", "segment_map"}
    assert batch["input_ids"].shape == (1, 32)
    assert batch["labels"].shape == (1, 32)
    assert batch["position_ids"].shape == (1, 32)
    assert torch.equal(batch["attention_mask"], torch.ones((1, 32), dtype=torch.long))

    segments = batch["segment_map"]
    assert len(segments) == 3
    assert [s.sample_index for s in segments] == [0, 1, 2]
    assert [s.start for s in segments] == [0, 7, 12]

    for segment in segments:
        start, end = segment.start, segment.end
        length = end - start
        # First label is masked; the rest are the real next-token targets.
        assert int(batch["labels"][0, start]) == -100
        assert torch.equal(
            batch["labels"][0, start + 1:end],
            batch["input_ids"][0, start + 1:end],
        )
        # Position ids restart at zero for every sample.
        assert torch.equal(
            batch["position_ids"][0, start:end],
            torch.arange(length, dtype=torch.long),
        )

    # Tail is pad + ignore, and its position ids continue (no new segment).
    assert torch.equal(
        batch["input_ids"][0, 16:], torch.full((16,), PAD, dtype=torch.long)
    )
    assert torch.equal(
        batch["labels"][0, 16:], torch.full((16,), -100, dtype=torch.long)
    )
    assert torch.equal(
        batch["position_ids"][0, 16:], torch.arange(4, 20, dtype=torch.long)
    )


def test_source_features_are_not_mutated():
    features = [{"input_ids": [11, 12, 13, 14]}, {"input_ids": [21, 22, 23]}]
    before = [list(f["input_ids"]) for f in features]
    collator = PackedDataCollator(PAD, pack_len=16)
    first = collator(features)
    second = collator(features)

    assert [list(f["input_ids"]) for f in features] == before
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert torch.equal(first["labels"], second["labels"])


def test_missing_labels_default_to_input_ids():
    features = [{"input_ids": [11, 12, 13, 14]}]
    batch = PackedDataCollator(PAD, pack_len=8)(features)
    assert int(batch["labels"][0, 0]) == -100
    assert batch["labels"][0, 1:4].tolist() == [12, 13, 14]
    assert batch["labels"][0, 4:].tolist() == [-100] * 4


def test_overlength_raises():
    collator = PackedDataCollator(PAD, pack_len=8)
    with pytest.raises(ValueError, match="exceeds pack_len"):
        collator([make_feature(9, 0)])


def test_empty_batch_returns_empty_dict():
    assert PackedDataCollator(PAD, pack_len=8)([]) == {}


def test_exact_pack_len():
    batch = PackedDataCollator(PAD, pack_len=8)([make_feature(8, 0)])
    assert batch["input_ids"].shape == (1, 8)
    assert torch.equal(
        batch["position_ids"][0], torch.arange(8, dtype=torch.long)
    )
    assert int(batch["labels"][0, 7]) == 8  # last real token stays a target
    assert int(batch["labels"][0, 0]) == -100


def test_group_id_and_legacy_channel_id():
    features = [
        make_feature(3, 0, group_id=7),
        make_feature(3, 1, channel_id=9),
        make_feature(3, 2),
    ]
    batch = PackedDataCollator(
        PAD, pack_len=16, return_segment_map=True
    )(features)
    assert [s.group_id for s in batch["segment_map"]] == [7, 9, None]
    assert all(isinstance(s, Segment) for s in batch["segment_map"])


def test_segment_map_is_off_by_default():
    batch = PackedDataCollator(PAD, pack_len=16)([make_feature(3, 0)])
    assert "segment_map" not in batch


def test_attention_mask_can_be_omitted():
    batch = PackedDataCollator(
        PAD, pack_len=16, return_attention_mask=False
    )([make_feature(3, 0)])
    assert batch["attention_mask"] is None


def test_labels_length_mismatch_raises():
    collator = PackedDataCollator(PAD, pack_len=16)
    with pytest.raises(ValueError, match="len\\(labels\\)"):
        collator([{"input_ids": [1, 2, 3], "labels": [1, 2]}])


def test_empty_sample_raises():
    collator = PackedDataCollator(PAD, pack_len=16)
    with pytest.raises(ValueError, match="empty"):
        collator([{"input_ids": []}])


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        PackedDataCollator(PAD, pack_len=16, mode="dense_mask")


def test_tuple_and_tensor_inputs():
    labels = torch.tensor([5, 6, 7])
    features = [
        {"input_ids": (1, 2, 3), "labels": (1, 2, 3)},
        {"input_ids": torch.tensor([8, 9, 10]), "labels": labels},
    ]
    batch = PackedDataCollator(PAD, pack_len=8)(features)
    assert batch["input_ids"][0, :6].tolist() == [1, 2, 3, 8, 9, 10]
    # The source label tensor must not be modified by the boundary mask.
    assert labels.tolist() == [5, 6, 7]
    assert batch["labels"][0, 3].item() == -100


def test_tokenizer_duck_typing():
    class Tok:
        pad_token_id = PAD

    batch = PackedDataCollator(Tok(), pack_len=8)([make_feature(3, 0)])
    assert batch["input_ids"].shape == (1, 8)


# ----------------------------------------------------------------------
# cu_seqlens mode
# ----------------------------------------------------------------------
def test_cu_seqlens_single_row():
    features = [make_feature(7, 0), make_feature(5, 1), make_feature(4, 2)]
    batch = PackedDataCollator(PAD, pack_len=16, mode="cu_seqlens",
                               return_segment_map=True)(features)

    assert batch["input_ids"].shape == (1, 16)
    assert batch["labels"].shape == (1, 16)
    assert batch["position_ids"].shape == (1, 16)
    assert batch["attention_mask"] is None
    assert batch["cu_seq_lens_q"].tolist() == [0, 7, 12, 16]
    assert batch["cu_seq_lens_k"].tolist() == [0, 7, 12, 16]
    assert batch["max_length_q"] == 7
    assert batch["max_length_k"] == 7
    assert batch["seq_idx"].tolist() == [[0] * 7 + [1] * 5 + [2] * 4]

    assert [s.start for s in batch["segment_map"]] == [0, 7, 12]
    assert all(s.row == 0 for s in batch["segment_map"])
    assert [s.sample_index for s in batch["segment_map"]] == [0, 1, 2]


def test_cu_seqlens_tail_is_part_of_last_segment():
    features = [make_feature(7, 0), make_feature(5, 1), make_feature(4, 2)]
    batch = PackedDataCollator(PAD, pack_len=20, mode="cu_seqlens")(features)
    # Last segment absorbs the 4-token tail: [0,7) [7,12) [12,20).
    assert batch["cu_seq_lens_q"].tolist() == [0, 7, 12, 20]
    assert batch["max_length_q"] == 8
    assert batch["seq_idx"].tolist() == [[0] * 7 + [1] * 5 + [2] * 8]
    assert torch.equal(
        batch["labels"][0, 16:], torch.full((4,), -100, dtype=torch.long)
    )


def test_cu_seqlens_multiple_rows():
    # pack_len=12: BFD puts 7+5 in row 0 and 4 in row 1.
    features = [make_feature(7, 0), make_feature(5, 1), make_feature(4, 2)]
    batch = PackedDataCollator(PAD, pack_len=12, mode="cu_seqlens")(features)

    assert batch["input_ids"].shape == (1, 24)
    # Row 0: [0,7) [7,12); row 1: [12,24) (4 real tokens + 8-token tail).
    assert batch["cu_seq_lens_q"].tolist() == [0, 7, 12, 24]
    assert batch["max_length_q"] == 12
    assert batch["seq_idx"].tolist() == [[0] * 7 + [1] * 5 + [2] * 12]


def test_cu_seqlens_boundary_values_are_consistent():
    rng = torch.Generator().manual_seed(0)
    lengths = [int(x) for x in torch.randint(1, 31, (20,), generator=rng)]
    features = [make_feature(length, i) for i, length in enumerate(lengths)]
    batch = PackedDataCollator(PAD, pack_len=64, mode="cu_seqlens")(features)

    cu = batch["cu_seq_lens_q"]
    assert int(cu[0]) == 0
    assert int(cu[-1]) == batch["input_ids"].numel()
    assert torch.all(cu[1:] > cu[:-1])
    assert batch["max_length_q"] == int((cu[1:] - cu[:-1]).max())

    # seq_idx is constant within each cu segment and increments by one.
    seq_idx = batch["seq_idx"][0]
    for segment_id in range(len(cu) - 1):
        start, end = int(cu[segment_id]), int(cu[segment_id + 1])
        assert torch.equal(
            seq_idx[start:end],
            torch.full((end - start,), segment_id, dtype=torch.int32),
        )


def test_cu_seqlens_no_segment_map_by_default():
    batch = PackedDataCollator(PAD, pack_len=16, mode="cu_seqlens")(
        [make_feature(3, 0)]
    )
    assert "segment_map" not in batch
