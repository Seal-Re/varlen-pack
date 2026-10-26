"""Padding-free sequence packing for causal language-model training.

The collator takes a list of per-sample dicts (the same dicts a Hugging Face
``Dataset`` yields) and packs them into fixed-width rows with no padding
overhead beyond the last row's tail.

Two output layouts are supported:

``mode="fa2_positions"``
    2-D ``[rows, pack_len]`` tensors.  ``attention_mask`` is all ones (or
    ``None``), and ``position_ids`` reset to 0 at every sample boundary.
    With ``transformers`` + FlashAttention-2 this is the packed path:
    the model converts the all-ones mask to ``None`` and derives
    ``cu_seqlens`` from the position-id resets.

``mode="cu_seqlens"``
    Flattened ``[1, rows * pack_len]`` tensors plus explicit boundary kwargs
    in the shape used by Hugging Face's padding-free collator:
    ``cu_seq_lens_q``, ``cu_seq_lens_k``, ``max_length_q``, ``max_length_k``,
    and ``seq_idx``.  Models that need explicit boundaries (including hybrid /
    linear-attention models) consume this layout instead of inferring
    boundaries from ``position_ids``.

Loss correctness
----------------
Causal-LM cross entropy shifts by one: ``logits[j]`` predicts
``labels[j + 1]``.  When two samples share a row, the last logit of sample A
would predict the first label of sample B -- a cross-sample supervision pair,
even though attention is isolated.  The collator therefore sets the **first
label of every sample segment** to ``-100``.  Masking the *last* label would
delete a valid in-segment target while leaving the cross-sample pair intact.
See ``DESIGN.md`` for the full argument.

Assumptions
-----------
* The consumer computes shifted causal-LM loss (the Hugging Face convention).
* Attention isolation is provided by the model/kernel, not by a dense mask.
  Do not use this collator with an attention backend that ignores both
  position-id resets and explicit boundaries; samples would leak into each
  other.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import torch

from .bfd import bfd_pack

__all__ = ["Segment", "PackedDataCollator"]

IGNORE_INDEX = -100

_MODES = ("fa2_positions", "cu_seqlens")


class Segment(NamedTuple):
    """Location of one packed sample in the output batch.

    Attributes:
        row: Row index in the output tensors.  Always 0 in flattened
            ``cu_seqlens`` mode.
        start: Inclusive start offset in the output layout.
        end: Exclusive end offset in the output layout (the sample's real
            end, excluding any row tail).
        sample_index: Index of this sample in the original ``features`` list.
        group_id: Optional caller-supplied group/channel id (``group_id`` or
            the legacy ``channel_id`` key).
    """

    row: int
    start: int
    end: int
    sample_index: int
    group_id: Optional[int] = None


def _resolve_pad_token_id(tokenizer_or_pad_id: Union[int, Any]) -> int:
    if isinstance(tokenizer_or_pad_id, int):
        return tokenizer_or_pad_id
    pad_token_id = getattr(tokenizer_or_pad_id, "pad_token_id", None)
    if pad_token_id is None:
        raise ValueError(
            "tokenizer has no pad_token_id; set tokenizer.pad_token_id or pass "
            "an integer pad token id explicitly"
        )
    return int(pad_token_id)


def _to_list(values: Sequence[int]) -> List[int]:
    """Convert a list/tensor/array of token ids to a plain Python list."""
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


class PackedDataCollator:
    """Pack a micro-batch of tokenized samples into fixed-width rows.

    Args:
        tokenizer_or_pad_id: A tokenizer with ``pad_token_id``, or the pad
            token id itself.  Only ``pad_token_id`` is used.
        pack_len: Row width in tokens.  Must be at least the longest sample.
        mode: ``"fa2_positions"`` or ``"cu_seqlens"`` (see module docstring).
        return_attention_mask: In ``fa2_positions`` mode, return an all-ones
            ``[rows, pack_len]`` mask.  Set to ``False`` to return ``None``
            (Hugging Face's padding-free documentation recommends omitting the
            mask).  In ``cu_seqlens`` mode the key is always ``None``.
        return_segment_map: Include a ``segment_map`` list of :class:`Segment`
            entries.  This is metadata for per-segment loss accounting and is
            **not** a model input; pop it before calling the model, or leave
            this option off (the default) when using a stock Hugging Face
            trainer that forwards every batch key to the model.

    Raises:
        ValueError: On an unknown mode, a non-positive ``pack_len``, an empty
            sample, an over-length sample, or mismatched ``input_ids`` /
            ``labels`` lengths.
    """

    def __init__(
        self,
        tokenizer_or_pad_id: Union[int, Any],
        pack_len: int = 2048,
        mode: str = "fa2_positions",
        *,
        return_attention_mask: bool = True,
        return_segment_map: bool = False,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {_MODES}")
        if pack_len <= 0:
            raise ValueError(f"pack_len must be positive, got {pack_len}")

        self.pad_token_id = _resolve_pad_token_id(tokenizer_or_pad_id)
        self.pack_len = int(pack_len)
        self.mode = mode
        self.return_attention_mask = bool(return_attention_mask)
        self.return_segment_map = bool(return_segment_map)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Collate ``features`` into a packed batch dict.

        Each feature must contain ``input_ids`` (a sequence of token ids) and
        may contain ``labels`` (same length; defaults to ``input_ids``),
        ``group_id``, and the legacy alias ``channel_id``.
        """
        if not features:
            return {}

        input_ids_list, labels_list, lengths, group_ids = self._validate(features)
        rows = bfd_pack(lengths, self.pack_len)

        batch, segments = self._build_rows(
            input_ids_list, labels_list, group_ids, rows
        )

        if self.mode == "fa2_positions":
            batch["attention_mask"] = (
                torch.ones(batch["input_ids"].shape, dtype=torch.long)
                if self.return_attention_mask
                else None
            )
            if self.return_segment_map:
                batch["segment_map"] = segments
        else:  # cu_seqlens
            batch, flat_segments = self._flatten_with_boundaries(
                batch, rows, segments
            )
            if self.return_segment_map:
                batch["segment_map"] = flat_segments

        return batch

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(
        features: Sequence[Mapping[str, Any]],
    ) -> Tuple[List[Sequence[int]], List[Sequence[int]], List[int], List[Optional[int]]]:
        input_ids_list: List[Sequence[int]] = []
        labels_list: List[Sequence[int]] = []
        lengths: List[int] = []
        group_ids: List[Optional[int]] = []

        for i, feature in enumerate(features):
            if "input_ids" not in feature:
                raise ValueError(f"feature {i} has no 'input_ids'")
            ids = feature["input_ids"]
            length = len(ids)
            if length <= 0:
                raise ValueError(
                    f"feature {i} has an empty 'input_ids'; filter empty samples"
                )
            labels = feature.get("labels")
            if labels is None:
                labels = ids
            if len(labels) != length:
                raise ValueError(
                    f"feature {i}: len(labels)={len(labels)} != "
                    f"len(input_ids)={length}"
                )
            input_ids_list.append(ids)
            labels_list.append(labels)
            lengths.append(length)
            group_id = feature.get("group_id")
            if group_id is None:
                group_id = feature.get("channel_id")
            group_ids.append(group_id)

        return input_ids_list, labels_list, lengths, group_ids

    def _build_rows(
        self,
        input_ids_list: Sequence[Sequence[int]],
        labels_list: Sequence[Sequence[int]],
        group_ids: Sequence[Optional[int]],
        rows: Sequence[Sequence[int]],
    ) -> Tuple[Dict[str, Any], List[Segment]]:
        n_rows = len(rows)
        width = self.pack_len

        # Assemble in plain Python lists, then create one tensor per field.
        # Per-sample torch tensor creation and slice assignment is an order of
        # magnitude slower for micro-batches of hundreds/thousands of samples.
        flat_ids: List[int] = []
        flat_labels: List[int] = []
        flat_position_ids: List[int] = []
        segments: List[Segment] = []

        for row_index, sample_indices in enumerate(rows):
            pos = 0
            last_length = 0
            for sample_index in sample_indices:
                ids = _to_list(input_ids_list[sample_index])
                lab = _to_list(labels_list[sample_index])
                length = len(ids)

                # Boundary rule: kill the cross-sample supervision pair that
                # would otherwise be formed by the previous segment's last
                # logit predicting this segment's first token.  See DESIGN.md.
                lab[0] = IGNORE_INDEX

                flat_ids.extend(ids)
                flat_labels.extend(lab)
                flat_position_ids.extend(range(length))
                segments.append(
                    Segment(
                        row_index,
                        pos,
                        pos + length,
                        sample_index,
                        group_ids[sample_index],
                    )
                )
                pos += length
                last_length = length

            # The tail continues the last sample's local numbering: it must not
            # create a new segment, because its labels are all -100 and it is
            # never trained.  Causality means real tokens cannot see the tail.
            tail_length = width - pos
            if tail_length:
                flat_ids.extend([self.pad_token_id] * tail_length)
                flat_labels.extend([IGNORE_INDEX] * tail_length)
                flat_position_ids.extend(
                    range(last_length, last_length + tail_length)
                )

            if len(flat_ids) != (row_index + 1) * width:
                raise RuntimeError(
                    f"internal error: row {row_index} assembled "
                    f"{len(flat_ids)} tokens, expected {(row_index + 1) * width}"
                )

        input_ids = torch.tensor(flat_ids, dtype=torch.long).reshape(n_rows, width)
        labels = torch.tensor(flat_labels, dtype=torch.long).reshape(n_rows, width)
        position_ids = torch.tensor(
            flat_position_ids, dtype=torch.long
        ).reshape(n_rows, width)

        return (
            {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
            },
            segments,
        )

    def _flatten_with_boundaries(
        self,
        batch: Dict[str, Any],
        rows: Sequence[Sequence[int]],
        segments: Sequence[Segment],
    ) -> Tuple[Dict[str, Any], List[Segment]]:
        """Flatten to ``[1, rows * pack_len]`` and emit explicit boundaries.

        Each row's tail belongs to that row's last sample segment.  The next
        row starts a new segment, so rows cannot attend to each other.
        """
        n_rows = len(rows)
        width = self.pack_len
        total = n_rows * width

        boundaries: List[int] = [0]
        seq_idx = torch.empty((1, total), dtype=torch.int32)
        flat_segments: List[Segment] = []

        cursor = 0
        segment_id = 0
        for row_index, sample_indices in enumerate(rows):
            base = row_index * width
            row_segments = segments[cursor : cursor + len(sample_indices)]
            cursor += len(sample_indices)

            for j, seg in enumerate(row_segments):
                start = base + seg.start
                # The last sample of a row owns the row tail; earlier samples
                # end at their real end.  A boundary at ``row_end`` also
                # prevents the next row from attending to this row's tail.
                end = base + (width if j == len(row_segments) - 1 else seg.end)
                seq_idx[0, start:end] = segment_id
                boundaries.append(end)
                segment_id += 1
                flat_segments.append(
                    Segment(
                        0,
                        base + seg.start,
                        base + seg.end,
                        seg.sample_index,
                        seg.group_id,
                    )
                )

        if cursor != len(segments):
            raise RuntimeError(
                f"internal error: consumed {cursor} of {len(segments)} segments"
            )
        if boundaries[-1] != total:
            raise RuntimeError(
                f"internal error: boundary end {boundaries[-1]} != total {total}"
            )

        cu_seq_lens = torch.tensor(boundaries, dtype=torch.int32)
        max_length = max(
            int(cu_seq_lens[i + 1] - cu_seq_lens[i])
            for i in range(len(cu_seq_lens) - 1)
        )

        flat_batch = {
            "input_ids": batch["input_ids"].reshape(1, total),
            "labels": batch["labels"].reshape(1, total),
            "position_ids": batch["position_ids"].reshape(1, total),
            "attention_mask": None,
            "cu_seq_lens_q": cu_seq_lens,
            # Full causal attention: query and key boundaries are identical.
            "cu_seq_lens_k": cu_seq_lens.clone(),
            "max_length_q": max_length,
            "max_length_k": max_length,
            "seq_idx": seq_idx,
        }
        return flat_batch, flat_segments
