# varlen-pack

A collator for padding-free sequence packing in causal language-model training,
with tools for checking packed-batch isolation and loss equivalence.

## Overview

`PackedDataCollator` takes a list of tokenized samples and packs them into
fixed-width rows using best-fit-decreasing (BFD) bin packing. It returns:

* `input_ids`, `labels`, and `position_ids`;
* an all-ones `attention_mask` with per-sample `position_ids` resets
  (`fa2_positions`), or flattened tensors with explicit boundary metadata
  (`cu_seqlens`);
* optional per-segment metadata.

Packing is performed in the collator. The tokenized dataset is not modified.

## Scope

The collator handles data layout. Attention isolation is provided by the model
or kernel:

* `fa2_positions` uses the FlashAttention-2 packed-position-id path;
* `cu_seqlens` passes explicit boundary metadata to the model.

The repository does not include a trainer, an offline dataset packer, or a
dense block-diagonal mask backend. Model quality is outside its scope.

## Installation

Python 3.9+ is required.

```bash
pip install -e .
# optional: tests and the GPU probe
pip install -e ".[test,probe]"
pytest
```

The BFD packer has no third-party dependency. The collator requires PyTorch.
The `cu_seqlens` mode does not import Transformers itself; the model consuming
the batch must accept the boundary metadata.

## Repository layout

```
src/varlen_pack/bfd.py          BFD packing (no third-party dependencies)
src/varlen_pack/collator.py     packed collator
tests/                          CPU unit tests
tools/packing_waste.py          packing-efficiency comparison
tools/probe_packing_contract.py GPU isolation/equivalence probe
```

## Usage

### `fa2_positions`

For Transformers 4.5x with FlashAttention-2:

```python
from varlen_pack import PackedDataCollator

collator = PackedDataCollator(
    tokenizer,              # only tokenizer.pad_token_id is used
    pack_len=2048,
    mode="fa2_positions",
)
batch = collator(features)
# input_ids, labels, position_ids: [rows, pack_len] int64
# attention_mask: [rows, pack_len] all ones
```

`position_ids` reset to 0 at every sample boundary. With FA2, the all-ones mask
is converted to `None` and the model derives `cu_seqlens` from those resets.

### `cu_seqlens`

For models that accept explicit boundary metadata:

```python
collator = PackedDataCollator(
    tokenizer,
    pack_len=2048,
    mode="cu_seqlens",
)
batch = collator(features)
# input_ids, labels, position_ids: [1, rows * pack_len]
# attention_mask: None
# cu_seq_lens_q, cu_seq_lens_k: int32 [segments + 1]
# max_length_q, max_length_k: int
# seq_idx: int32 [1, rows * pack_len]
```

This layout matches the boundary metadata used by Hugging Face's padding-free
collator. It is intended for models that do not infer boundaries from
`position_ids`, including hybrid and linear-attention models, and for
`torch.compile`. If the model does not accept these keys, the probe reports a
skip.

### Segment metadata

```python
collator = PackedDataCollator(
    tokenizer, pack_len=2048, return_segment_map=True
)
batch = collator(features)
segments = batch.pop("segment_map")   # metadata; pop before model(**batch)
for seg in segments:
    print(seg.row, seg.start, seg.end, seg.sample_index, seg.group_id)
```

`group_id` comes from `feature["group_id"]` or `feature["channel_id"]`. The
metadata can be used for per-channel loss accounting, per-sample loss
attribution, or debugging. Leave `return_segment_map=False` (the default) when
using a trainer that forwards every batch key to the model.

## Output fields

| Field | `fa2_positions` | `cu_seqlens` |
|---|---|---|
| `input_ids` | `[rows, pack_len]` int64 | `[1, rows * pack_len]` int64 |
| `labels` | `[rows, pack_len]` int64 | `[1, rows * pack_len]` int64 |
| `position_ids` | `[rows, pack_len]` int64 | `[1, rows * pack_len]` int64 |
| `attention_mask` | `[rows, pack_len]` int64 or `None` | `None` |
| `cu_seq_lens_q/k` | — | int32 `[segments + 1]` |
| `max_length_q/k` | — | int |
| `seq_idx` | — | int32 `[1, rows * pack_len]` |
| `segment_map` | optional | optional |

## Correctness notes

### Boundary labels

Causal-LM cross entropy shifts by one: `logits[j]` predicts `labels[j + 1]`.
For a row `[A1 A2 A3 | B1 B2]`:

* `A3`'s logit predicts `B1`, a cross-sample target;
* masking the last label of A (`A3`) deletes the valid `A2 -> A3` target and
  leaves `A3 -> B1`;
* masking the first label of B (`B1`) removes the cross-sample pair and keeps
  `A2 -> A3`.

The collator sets `labels[segment_start] = -100` for every sample. Each sample
keeps `len - 1` supervised targets, matching a padded path that masks the first
real label. `DESIGN.md` contains the full argument.

### Attention mask

In the Transformers 4.5x FA2 dispatch, a 2D `attention_mask` containing a `0`
takes the padding/unpad branch and treats the row as one sequence. The
`fa2_positions` layout therefore uses an all-ones mask and puts the boundaries
in `position_ids`. Set `return_attention_mask=False` to omit the mask.

### Tail tokens

The final row may contain a tail shorter than `pack_len`. The tail is filled
with `pad_token_id`, its labels are `-100`, and its position ids continue the
last sample. Real tokens are causal and precede the tail.

### Isolation

The collator does not build a dense block-diagonal mask. `fa2_positions`
relies on the model's packed-position-id path; `cu_seqlens` relies on explicit
boundary metadata. If the backend ignores both, samples are not isolated. Run
`tools/probe_packing_contract.py` on the target stack.

## Tools

### Packing efficiency

```bash
python tools/packing_waste.py --lengths lengths.txt --micro 100 --pack-len 2048
```

Compares naive sequential, FFD, and BFD packing on a file containing one
sequence length per line. The output includes rows per batch, waste percentage,
average gap per row, and CPU time per batch.

### Contract probe

```bash
python tools/probe_packing_contract.py \
    --model Qwen/Qwen3-0.6B --mode fa2_positions --pack-len 64
```

Builds a model from the config with a reduced number of layers (weights are
not loaded), packs three synthetic samples, and checks segment equivalence,
poison-sample isolation, the mask-with-zero behavior, and loss equivalence
against a padded reference. The probe requires CUDA, a FlashAttention build
compatible with the GPU, and Transformers.

## Compatibility

* `fa2_positions`: targets Transformers 4.5x with FlashAttention-2. Run the
  probe on the exact versions in use.
* `cu_seqlens`: layout and boundary values are covered by CPU unit tests;
  model support depends on the Transformers/model version.
* CPU unit tests do not require a GPU, Transformers, or FlashAttention.

Transformers documentation treats packed `position_ids` inference as a legacy
path: it works for standard attention, can cause graph breaks under
`torch.compile`, and is not sufficient for hybrid/linear-attention layers.
`cu_seqlens` is the explicit alternative.

## Testing

```bash
pytest
```

The tests cover:

* bin capacity, conservation, determinism, over-length and empty rejection;
* first-label boundary masking, tail handling, and position-id resets;
* segment map alignment, group/channel ids, and source aliasing;
* `cu_seq_lens_*`, `max_length_*`, and `seq_idx` values;
* per-sample target-set equivalence with the padded path;
* absence of cross-segment supervision pairs.

## Related work

| Project | Packing location | Notes |
|---|---|---|
| Hugging Face `DataCollatorWithFlattening` | collator (flatten only) | Padding-free layout, boundary labels, `cu_seq_lens_*`/`seq_idx`; no bin packing. |
| TRL `SFTConfig(packing=True, packing_strategy="bfd")` | dataset preparation (`pack_dataset`) | Dataset-level BFD; adds `seq_lengths`; used with TRL SFTTrainer. |
| torchtune `PackedDataset` | dataset preparation | Sequential greedy packing; block-causal mask. |
| `multipack_sampler` / Composer | batch sampler | Length-based packing with FFD/LPT; custom sampler and dataset-level lengths. |
| Axolotl sample packing | trainer-integrated | `position_ids` boundaries; see upstream issue #1375 for a 2-D `position_ids` case. |
| varlen-pack | collator | BFD per micro-batch; no dataset rewrite; optional explicit `cu_seqlens`/`seq_idx`; verification tools. |

If the dataset can be rewritten and TRL is already in use, TRL's dataset-level
packing is a direct option. This collator is intended for setups that need
per-batch packing without changing a tokenized dataset.

## License and provenance

No license has been selected. The code was extracted from an internal training
codebase. Confirm ownership and add a license before public distribution. The
repository contains the generic collator, tests, and tools; it does not contain
proprietary data, models, or training pipeline code.
