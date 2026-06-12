"""
Quickstart addon: launches upstream scripts.chat_sft with two single-GPU fixes
applied at runtime, WITHOUT modifying any upstream file. Used by
scripts/quickstart.py for the SFT stage; safe to invoke manually too:

    python -m scripts.quickstart_sft_runner --model-tag=d4 --num-iterations=20 ...

Fix 1 — --num-iterations semantics (upstream PR #775):
    chat_sft's data generator compares its per-micro-batch counter directly
    against --num-iterations, but optimizer steps are grad_accum_steps
    micro-batches apart. With gradient accumulation > 1 (any single-GPU run),
    progress overshoots 100% inside the first optimizer step, the LR multiplier
    goes negative, and the loss becomes NaN. If (and only if) the upstream
    source still contains the buggy comparison, --num-iterations is pre-scaled
    by grad_accum_steps here so it means optimizer steps, as documented.

Fix 2 — fully-padded rows at short max_seq_len (upstream issue #486):
    render_conversation defaults to max_tokens=2048. When the inherited
    max_seq_len is shorter, conversations longer than row_capacity
    (max_seq_len + 1) can never fit a packing row, so the best-fit packer emits
    rows that are 100% padding with every target masked — and cross-entropy
    over zero valid targets is NaN, which poisons the weights on backward.
    We wrap RustBPETokenizer.render_conversation to cap at row_capacity.
"""

import json
import os
import re
import runpy
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUGGY_STOP_RE = re.compile(r"0\s*<\s*args\.num_iterations\s*<=\s*it")
BUGGY_PROGRESS_RE = re.compile(r"approx_progress\s*=\s*it\s*/\s*args\.num_iterations")


def log(msg):
    print(f"[quickstart-sft] {msg}", flush=True)


def arg_value(argv, name):
    """Read --name value from argv, supporting both '--name value' and '--name=value'."""
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def replace_arg(argv, name, value):
    """Replace --name in argv with --name=value (or append if absent)."""
    out = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a == name:
            skip_next = True
            continue
        if a.startswith(name + "="):
            continue
        out.append(a)
    out.append(f"{name}={value}")
    return out


def resolve_meta(argv):
    """Locate the base checkpoint meta json the same way chat_sft will."""
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    ckpts = os.path.join(base_dir, "base_checkpoints")
    tag = arg_value(argv, "--model-tag")
    if tag is None:
        # mirror checkpoint_manager.find_largest_model: biggest d<N>
        candidates = []
        for d in os.listdir(ckpts):
            m = re.match(r"d(\d+)", d)
            if m and os.path.isdir(os.path.join(ckpts, d)):
                candidates.append((int(m.group(1)), d))
        candidates.sort()
        tag = candidates[-1][1]
    ckpt_dir = os.path.join(ckpts, tag)
    step = arg_value(argv, "--model-step")
    if step is None:
        # mirror checkpoint_manager.find_last_step
        steps = [int(f.split("_")[-1].split(".")[0])
                 for f in os.listdir(ckpt_dir)
                 if f.startswith("model_") and f.endswith(".pt")]
        step = max(steps)
    meta_path = os.path.join(ckpt_dir, f"meta_{int(step):06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    argv = sys.argv[1:]

    # Resolve the effective SFT batch geometry (same inheritance rules as chat_sft)
    try:
        meta = resolve_meta(argv)
    except Exception as exc:
        log(f"WARNING: could not read base checkpoint meta ({exc}); running upstream chat_sft unmodified")
        meta = None

    if meta is not None:
        cfg = meta.get("model_config", {})
        max_seq_len = int(arg_value(argv, "--max-seq-len")
                          or meta.get("max_seq_len")
                          or cfg.get("sequence_len")
                          or 2048)
        dbs = int(arg_value(argv, "--device-batch-size")
                  or meta.get("device_batch_size")
                  or 32)
        total_bs = int(arg_value(argv, "--total-batch-size")
                       or meta.get("total_batch_size")
                       or 524288)
        grad_accum = max(1, total_bs // (dbs * max_seq_len))
        row_capacity = max_seq_len + 1

        # --- Fix 2: cap conversation rendering at row capacity ---
        import nanochat.tokenizer as tok_mod
        orig_render = tok_mod.RustBPETokenizer.render_conversation

        def render_capped(self, conversation, max_tokens=2048):
            return orig_render(self, conversation, max_tokens=min(max_tokens, row_capacity))

        tok_mod.RustBPETokenizer.render_conversation = render_capped
        log(f"conversations capped at row capacity {row_capacity} tokens (max_seq_len {max_seq_len})")

        # --- Fix 1: pre-scale --num-iterations if upstream still counts micro-batches ---
        n_iter = arg_value(argv, "--num-iterations")
        if n_iter is not None and int(n_iter) > 0 and grad_accum > 1:
            src_path = os.path.join(REPO_ROOT, "scripts", "chat_sft.py")
            with open(src_path, "r", encoding="utf-8") as f:
                src = f.read()
            if BUGGY_STOP_RE.search(src) and BUGGY_PROGRESS_RE.search(src):
                scaled = int(n_iter) * grad_accum
                argv = replace_arg(argv, "--num-iterations", scaled)
                log(f"upstream chat_sft counts micro-batches: scaled --num-iterations "
                    f"{n_iter} -> {scaled} (= {n_iter} optimizer steps x {grad_accum} grad accum)")
            else:
                log("upstream chat_sft looks fixed; --num-iterations passed through unscaled")

    sys.argv = ["scripts.chat_sft"] + argv
    runpy.run_module("scripts.chat_sft", run_name="__main__")


if __name__ == "__main__":
    main()
