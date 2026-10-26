# Design and correctness notes

This document explains the invariants the collator maintains and why they are
sufficient for loss-equivalent packed training. It is intentionally
implementation-focused; the README covers usage.

## 1. Scope

The collator handles the **data layout** of packing:

* bin-pack variable-length samples into fixed-width rows;
* build `input_ids`, `labels`, `position_ids`, and either an all-ones
  `attention_mask` or explicit boundary kwargs;
* mask the one label per sample that would otherwise create a cross-sample
  supervision pair.

It does **not** implement attention isolation. Isolation is provided by the
model/kernel:

* `fa2_positions`: FlashAttention-2's packed-position-id path derives
  `cu_seqlens` from `position_ids` resets.
* `cu_seqlens`: the model consumes explicit `cu_seq_lens_q/k`,
  `max_length_q/k`, and `seq_idx`.

If the backend ignores both, samples leak. The probe in `tools/` exists to
catch that.

## 2. Output layouts

### 2.1 `fa2_positions`

```
input_ids      [rows, pack_len]  int64
labels         [rows, pack_len]  int64   (-100 for boundary/tail)
position_ids   [rows, pack_len]  int64   (0..L-1 per sample; tail continues)
attention_mask [rows, pack_len]  int64   (all ones by default)
segment_map    list[Segment]     optional
```

Each row is a concatenation of whole samples followed by a tail. The tail:

* is filled with `pad_token_id`;
* has `label = -100`;
* has `position_ids` continuing from the last sample (no new segment).

Why no new segment for the tail? The tail comes after every real token in the
row. Causal attention means real tokens cannot attend to it. Its labels are
ignored. Whether it forms a separate segment is therefore semantically
irrelevant; continuing the numbering avoids a spurious boundary.

### 2.2 `cu_seqlens`

```
input_ids      [1, rows * pack_len]  int64
labels         [1, rows * pack_len]  int64
position_ids   [1, rows * pack_len]  int64
attention_mask None
cu_seq_lens_q  int32 [num_segments + 1]
cu_seq_lens_k  int32 [num_segments + 1]
max_length_q   int
max_length_k   int
seq_idx        int32 [1, rows * pack_len]
segment_map    list[Segment]         optional
```

Rows are flattened. Each row's tail belongs to that row's last sample segment;
the next row starts a new segment. `cu_seq_lens_*` therefore has a boundary at
every sample start/end and at every row end, and `seq_idx[i]` identifies the
segment of position `i`. The next row cannot attend to the previous row's
tail because they are different segments.

The key names follow Hugging Face's flat collator (`DataCollatorWithFlattening`
with `return_flash_attn_kwargs=True` / `return_seq_idx=True`). Some models use
slightly different names internally; check the model's forward signature.

## 3. Loss correctness

### 3.1 Shift semantics

For a causal LM, Hugging Face computes:

```
loss = CE(logits[:, :-1], labels[:, 1:], ignore_index=-100)
```

So the logit at position `j` is trained to predict `labels[j + 1]`.

### 3.2 The boundary problem

Consider two samples in one row:

```
positions:  0    1    2   |  3    4
tokens:     A1   A2   A3  |  B1   B2
labels:     A1   A2   A3  |  B1   B2
```

The shifted pairs are:

| logit position | predicts label | kind |
|---|---|---|
| 0 (A1) | A2 | within A |
| 1 (A2) | A3 | within A |
| 2 (A3) | B1 | **cross-sample** |
| 3 (B1) | B2 | within B |

Attention isolation does not remove the pair at position 2: the *logit* comes
from A, but the *target* comes from B. This is a loss-level leak.

### 3.3 Why mask the first label

Mask `B1`'s label (`labels[3] = -100`):

```
labels:     A1   A2   A3  | -100  B2
```

* position 2 now predicts `-100` -> ignored;
* position 3 predicts `B2` -> within B, preserved;
* all within-A pairs are preserved.

Mask `A3`'s label instead (`labels[2] = -100`):

```
labels:     A1   A2  -100 |  B1   B2
```

* position 1 predicts `-100` -> the valid `A2 -> A3` pair is deleted;
* position 2 still predicts `B1` -> the cross-sample pair remains.

Therefore the correct rule is:

> **Set the first label of every sample segment to `-100`.**

### 3.4 Supervised-set equivalence

For a sample of length `L` with all labels valid, the packed path supervises
positions `1 .. L-1` within the segment: `L - 1` targets. A padded path that
masks the first real label supervises the same `L - 1` targets. The token ids
at those positions are identical. Hence the per-sample target multiset is
identical.

The first segment's first label is also masked. In a padded path, the first
real label would be predicted from the last padding position; masking it is the
standard padding-free convention (and the origin pipeline does the same). If
your padded baseline trains that token, the packed path has one fewer target
per sample; that is a deliberate, documented choice, not a silent change.

### 3.5 SFT labels

For SFT, `labels` may already contain `-100` in the prompt region. The
collator leaves those untouched and additionally masks the first label. Setting
an already-masked label to `-100` is idempotent. The same proof applies
segment-wise.

## 4. Attention isolation

### 4.1 `fa2_positions`

In Transformers 4.5x, the FA2 dispatch has two relevant branches:

* a 2D `attention_mask` containing `0` -> padding/unpad branch, which treats
  the row as **one** sequence;
* no mask (or an all-ones mask that the model converts to `None`) plus
  non-monotonic `position_ids` -> packed-position-id branch, which derives
  `cu_seqlens` from the resets.

An all-ones mask is therefore deliberate: it must not contain a `0`. A single
`0` silently destroys isolation while training continues with plausible-looking
loss. The probe includes a mask-with-zero negative test.

The all-ones mask also means there is no padding mask. The tail is pad, but it
is untrained and comes after all real tokens; causal attention keeps it out of
the real tokens' receptive field.

### 4.2 `cu_seqlens`

`position_ids` inference is treated as legacy in current Transformers
documentation. It is sufficient for standard attention, but:

* it can cause graph breaks / device-to-host syncs under `torch.compile`;
* it is **not** sufficient for hybrid/linear-attention models (Mamba/Gated
  DeltaNet-style layers), which need `cu_seq_lens_q`/`seq_idx` to reset their
  recurrent state;
* some models pass `position_ids` in a non-2D layout (e.g. MRoPE axes), which
  makes naive `cu_seqlens` inference wrong.

The `cu_seqlens` mode emits the explicit kwargs instead. This path does not
rely on position-id inference.

### 4.3 What about `sdpa`?

A dense 4D block-diagonal mask is possible on `sdpa`, but:

* attention cost becomes `O(rows * pack_len^2)` instead of `O(sum(L^2))`;
* some Transformers versions silently drop or mishandle 4D masks;
* FA2's unpad path can mis-index a 4D mask.

It is intentionally out of scope. Use FA2/FA3 or explicit boundary kwargs. If
you must support a GPU without FA2, use the padded path rather than a dense
packed mask.

## 5. Packing

### 5.1 BFD

Samples are sorted by `(-length, original_index)` and placed into the row with
the smallest remaining capacity that still fits (best-fit). If none fits, a new
row opens.

Properties:

* no sample is split;
* no row exceeds `pack_len`;
* deterministic for a given input order;
* `O(n log n + n log r)`.

### 5.2 Why BFD rather than sequential greedy

Sequential greedy (no sorting) can leave larger gaps: a short sample that
arrives after a nearly full row opens a new row even though it would fit
elsewhere. At `pack_len=1024` with samples up to ~608 tokens, two large
samples cannot share a row regardless of the algorithm.

BFD is not the only applicable algorithm; FFD performs similarly on many
distributions. This implementation uses BFD because it is deterministic and
supports a `bisect`-based best-fit search.

### 5.3 Waste vs micro-batch

The last row of each micro-batch carries a structural tail. Its share of the
row capacity varies with the micro-batch size and the length distribution.
`tools/packing_waste.py` reports the resulting waste for a given distribution
and micro-batch size.

### 5.4 Over-length samples

The collator raises. It never truncates silently. Truncation changes the
supervised target set and should be an explicit data-policy decision (split the
sample, filter it, or increase `pack_len`), not a collator side effect.

## 6. Collator-time vs dataset-level packing

| Property | Collator-time (this repo) | Dataset-level (TRL/torchtune) |
|---|---|---|
| Tokenized cache | unchanged | rebuilt/rewritten |
| Packing freshness | per micro-batch/epoch | fixed for the run |
| Sample counts/resume | unchanged | dataset rows change |
| Long-sample policy | raise | truncate/split at preprocessing |
| Global optimality | per-micro-batch BFD | can pack globally |
| Integration | any HF-style trainer | tied to the framework |

The per-micro-batch approach packs within each micro-batch rather than
globally; the amount of unused capacity depends on the length distribution and
batch size. It does not rewrite the dataset or change sample counts.

## 7. Validation

The repo uses four levels of checks:

1. **Unit tests** (`tests/`): bin invariants, tensor layouts, boundary masks,
   segment maps, `cu_seqlens` values.
2. **Supervision-set test**: every sample has exactly `L-1` supervised targets
   and no supervised target crosses a segment boundary. This is model-free.
3. **Loss equivalence test** (in the GPU probe): packed shifted CE vs a padded
   reference, within bf16 noise.
4. **Isolation test** (in the GPU probe): poison one segment; the other
   segments' logits must not move. Plus the mask-with-zero negative test for
   the FA2 branch trap.

For a new model/Transformers/GPU combination, run the GPU probe before a long
training run. Numeric differences from different tensor shapes are expected at
the bf16 level; isolation failures are not.

## 8. Limitations and pitfalls

* **Version drift.** Transformers' FA2 dispatch and flat-collator kwargs have
  changed across 4.5x/5.x. Pin versions and re-run the probe.
* **`position_ids` shape.** If a model uses a non-2D `position_ids` layout,
  deriving `cu_seqlens` from it can produce a multi-dimensional tensor that
  kernels silently ignore. Prefer explicit `cu_seq_lens_*`.
* **`seq_idx` for hybrid models.** Some linear-attention/conv layers need
  `seq_idx` (or an equivalent) to reset state at segment boundaries; without
  it, they silently carry state across samples. `cu_seqlens` mode emits it.
* **No bitwise comparability across `pack_len`/micro-batch.** Changing the
  grouping changes GEMM shapes and reduction order; runs are not bitwise
  comparable, just as with a batch-size change.
* **Extra batch keys.** `segment_map` and the boundary kwargs must be consumed
  by the model or popped by the trainer. A stock HF trainer forwards every key
  with `remove_unused_columns=False`; pop `segment_map` in `compute_loss`.
* **DeepSpeed ZeRO-3.** The collator is orthogonal to ZeRO-3: `position_ids`
  and boundary kwargs are ordinary tensors, and no parameter shapes change.
  The trainer must still pop non-model metadata.

## 9. References

* Hugging Face, *Padding-free training*:
  https://huggingface.co/docs/transformers/en/padding_free
* Transformers PR #36456 (flat collator `cu_seq_lens_*` / `seq_idx`):
  https://github.com/huggingface/transformers/pull/36456
* TRL PR #7533 / #7599 (boundary kwargs for padding-free batches):
  https://github.com/huggingface/trl/pull/7533 ,
  https://github.com/huggingface/trl/pull/7599
* TRL SFT packing (`packing_strategy`, dataset-level BFD):
  https://huggingface.co/docs/trl/sft_trainer
* torchtune `PackedDataset`:
  https://meta-pytorch.org/torchtune/stable/generated/torchtune.datasets.PackedDataset.html
* Axolotl issue #1375 (2-D `position_ids` / `cu_seqlens` pitfall):
  https://github.com/axolotl-ai-cloud/axolotl/issues/1375
* multipack sampler:
  https://github.com/imoneoi/multipack_sampler
* SeqPacker:
  https://github.com/AlphaKhaw/seqpacker
