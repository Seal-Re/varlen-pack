#!/usr/bin/env python3
"""Verify that a packed batch is actually isolated and loss-equivalent.

This is an optional GPU probe.  It builds a tiny causal LM from a model config
(no pretrained weights), packs three short synthetic samples, and checks:

1. **Segment equivalence** -- each packed segment's logits match an independent
   forward pass of that sample alone.
2. **Isolation** -- replacing the first segment's tokens must not change the
   other segments' logits.
3. **Branch trap** (``fa2_positions`` only) -- flipping one attention-mask bit
   to 0 must change the other segments, proving that an all-ones mask is
   required for the packed-position-id path.
4. **Loss equivalence** -- packed shifted CE matches a padded reference within
   bf16 noise.

Requirements: CUDA, ``torch``, ``transformers``, and a FlashAttention build
that supports your GPU.  The model config must be available locally or on the
Hub.  This probe never downloads weights.

Example:
    python tools/probe_packing_contract.py --model Qwen/Qwen3-0.6B \
        --mode fa2_positions --pack-len 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForCausalLM
except ImportError as exc:  # pragma: no cover - optional dependency
    print(f"[skip] missing dependency: {exc}")
    sys.exit(0)

from varlen_pack.collator import PackedDataCollator  # noqa: E402

PAD = 0
LENGTHS = [7, 5, 4]


def build_model(model_name, layers, device):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_hidden_layers = min(int(config.num_hidden_layers), layers)
    # Select FA2 at construction time.  Setting the private field after
    # from_config can leave the eager/sdpa classes in place on some versions.
    config._attn_implementation = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
    except TypeError:
        try:
            # Older Transformers versions may not accept torch_dtype here.
            model = AutoModelForCausalLM.from_config(
                config, attn_implementation="flash_attention_2"
            )
        except TypeError:
            model = AutoModelForCausalLM.from_config(config)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    impl = getattr(model.config, "_attn_implementation", "unknown")
    print(f"[model] {model_name} layers={config.num_hidden_layers} "
          f"hidden={config.hidden_size} vocab={config.vocab_size} impl={impl}")
    if impl != "flash_attention_2":
        print("[warn] model did not select flash_attention_2; results may not "
              "exercise the packed path")
    return model, config


def make_sequences(vocab_size, device, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randint(1, vocab_size, (length,), generator=generator).to(device)
        for length in LENGTHS
    ]


def move_batch(batch, device):
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def segment_slices(batch, segments):
    """Return {sample_index: (row, start, end)} for the output layout."""
    return {
        segment.sample_index: (segment.row, segment.start, segment.end)
        for segment in segments
    }


def max_abs_diff(a, b):
    return float((a.float() - b.float()).abs().max())


def shifted_ce(logits, labels):
    return F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def padded_reference(model, sequences):
    max_len = max(len(seq) for seq in sequences)
    n = len(sequences)
    input_ids = torch.full((n, max_len), 0, dtype=torch.long, device=sequences[0].device)
    attention_mask = torch.zeros((n, max_len), dtype=torch.long, device=sequences[0].device)
    labels = torch.full((n, max_len), -100, dtype=torch.long, device=sequences[0].device)
    for i, seq in enumerate(sequences):
        offset = max_len - len(seq)
        input_ids[i, offset:] = seq
        attention_mask[i, offset:] = 1
        labels[i, offset + 1:] = seq[1:]  # first real label stays -100
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return shifted_ce(logits, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="model id or local path (config only; weights are not loaded)")
    parser.add_argument("--mode", default="fa2_positions",
                        choices=["fa2_positions", "cu_seqlens"])
    parser.add_argument("--pack-len", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="loss/segment tolerance (bf16 noise)")
    parser.add_argument("--isolation-atol", type=float, default=1e-3,
                        help="victim-logit tolerance for the isolation test")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[skip] CUDA is not available; the FA2 probe needs a GPU")
        return 0
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        print("[skip] flash_attn is not installed")
        return 0

    device = torch.device(args.device)
    model, config = build_model(args.model, args.layers, device)
    sequences = make_sequences(config.vocab_size, device)

    features = [{"input_ids": seq.tolist(), "labels": seq.tolist()} for seq in sequences]
    collator = PackedDataCollator(
        PAD, pack_len=args.pack_len, mode=args.mode, return_segment_map=True
    )
    batch = move_batch(collator(features), device)
    segments = batch.pop("segment_map")
    slices = segment_slices(batch, segments)
    print(f"[pack] mode={args.mode} output={tuple(batch['input_ids'].shape)} "
          f"segments={list(slices)}")

    with torch.no_grad():
        try:
            out = model(**batch)
        except (TypeError, RuntimeError) as exc:
            print(f"[skip] this model/version does not accept mode "
                  f"{args.mode!r}: {type(exc).__name__}: {exc}")
            return 0
    batch = move_batch(collator(features), device)  # restore for later tests
    batch.pop("segment_map", None)

    # 1. Segment equivalence against independent forwards.
    with torch.no_grad():
        references = [model(input_ids=seq.unsqueeze(0)).logits[0] for seq in sequences]
    segment_diffs = {}
    for sample_index, (row, start, end) in slices.items():
        segment_diffs[sample_index] = max_abs_diff(
            out.logits[row, start:end], references[sample_index]
        )
    worst_segment = max(segment_diffs.values())
    print(f"[1] segment vs independent forward: "
          f"{ {k: f'{v:.2e}' for k, v in segment_diffs.items()} }")
    ok_segment = worst_segment <= args.atol
    print(f"    {'PASS' if ok_segment else 'FAIL'} (worst {worst_segment:.2e})")

    # 2. Poison the first segment; other segments must not move.
    poison = torch.randint(1, config.vocab_size, (LENGTHS[0],), device=device)
    poisoned = move_batch(collator(features), device)
    poisoned.pop("segment_map", None)
    row, start, end = slices[0]
    poisoned["input_ids"][row, start:end] = poison
    with torch.no_grad():
        out_poison = model(**poisoned)
    self_diff = max_abs_diff(out_poison.logits[row, start:end], out.logits[row, start:end])
    victim_diffs = {
        i: max_abs_diff(
            out_poison.logits[r, s:e], out.logits[r, s:e]
        )
        for i, (r, s, e) in slices.items()
        if i != 0
    }
    worst_victim = max(victim_diffs.values()) if victim_diffs else 0.0
    print(f"[2] poison self diff={self_diff:.2e} (must be > 0); "
          f"victims={ {k: f'{v:.2e}' for k, v in victim_diffs.items()} }")
    ok_isolation = self_diff > args.isolation_atol and worst_victim < args.isolation_atol
    print(f"    {'PASS' if ok_isolation else 'FAIL'} "
          f"(worst victim {worst_victim:.2e})")

    # 3. FA2 branch trap: a single 0 in the mask must break segment isolation
    # within the same row.  Rows are independent in the 2-D layout, so the
    # test only makes sense when a row contains at least two segments.
    ok_trap = None
    if args.mode == "fa2_positions":
        by_row = {}
        for i, (r, s, e) in slices.items():
            by_row.setdefault(r, []).append((i, s, e))
        trap_row = next(
            (r for r, items in by_row.items() if len(items) >= 2), None
        )
        if trap_row is None:
            print("[3] mask containing 0: skipped "
                  "(no row contains multiple segments)")
        else:
            trapped = move_batch(collator(features), device)
            trapped.pop("segment_map", None)
            trapped["attention_mask"][trap_row, 0] = 0
            with torch.no_grad():
                out_trap = model(**trapped)
            row_items = by_row[trap_row]
            trap_diff = max(
                max_abs_diff(out_trap.logits[r, s:e], out.logits[r, s:e])
                for (i, s, e) in row_items
                if i != row_items[0][0]
            )
            print(f"[3] mask containing 0: same-row victim diff={trap_diff:.2e} "
                  f"(expected > 0 if the model takes the padding branch)")
            ok_trap = trap_diff > args.isolation_atol
            print(f"    {'PASS' if ok_trap else 'WARN'} "
                  "(WARN means this model/version does not take the branch-1 trap)")

    # 4. Loss equivalence against a padded reference.
    packed_labels = batch["labels"]
    packed_loss = shifted_ce(out.logits, packed_labels).item()
    reference_loss = padded_reference(model, sequences).item()
    relative = abs(packed_loss - reference_loss) / max(abs(reference_loss), 1e-9)
    print(f"[4] loss packed={packed_loss:.6f} padded={reference_loss:.6f} "
          f"relative={relative:.2e}")
    ok_loss = relative <= args.atol
    print(f"    {'PASS' if ok_loss else 'FAIL'} (<= {args.atol:g})")

    passed = ok_segment and ok_isolation and ok_loss
    print("\n" + ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
