# NanoChat CUDA Quickstart (Windows + NVIDIA)

A port of the [nanochat-mlx](https://github.com/scasella/nanochat-mlx) quickstart wizard UI
to the original PyTorch/CUDA [nanochat](https://github.com/karpathy/nanochat), so the full
"train your own ChatGPT" pipeline runs on an NVIDIA GPU (tested on RTX 5070 Ti 16 GB,
Windows 11, Python 3.10 venv via uv, torch 2.9.1+cu128).

**Zero upstream modifications.** The addon is purely additive — `git status` shows only
untracked new files, so `git pull` always works cleanly. The two upstream bugs the wizard
would otherwise hit are compensated at runtime by `scripts/quickstart_sft_runner.py`
(see below); both are also reported upstream
([PR #775](https://github.com/karpathy/nanochat/pull/775),
[issue #486](https://github.com/karpathy/nanochat/issues/486)).

## Setup

```
uv sync --extra gpu
uv pip install "triton-windows>=3.5,<3.6"   # torch.compile on native Windows (optional but ~faster)
```

The triton-windows minor version must match torch (torch 2.9 → triton 3.5.x). If it is
missing or mismatched the wizard detects this and automatically falls back to eager mode
for that run instead of crashing.

## Launch

Recommended from Windows PowerShell:

```powershell
uv run python -m scripts.quickstart --port 8000
```

Then open http://127.0.0.1:8000. If port 8000 is already in use:

```powershell
uv run python -m scripts.quickstart --port 8001
```

`start_quickstart.bat` is a convenience wrapper for the same command.

## Added files (everything else is untouched upstream nanochat)

| File | Purpose |
|---|---|
| `scripts/quickstart.py` | FastAPI backend: runs each pipeline stage as a subprocess, streams stdout + parsed metrics over SSE, serves the chat model in-process |
| `nanochat/quickstart_ui.html` | The wizard UI (adapted from nanochat-mlx: MLX/Apple knobs → CUDA knobs) |
| `scripts/quickstart_sft_runner.py` | Wraps upstream `chat_sft` with two runtime fixes (below) without editing it |
| `scripts/import_from_hf.py` | Downloads a pretrained nanochat checkpoint (e.g. `nanochat-students/base-d20`) straight into `~/.cache/nanochat/base_checkpoints` — no conversion needed since HF repos host the native torch format |
| `start_quickstart.bat` | Launcher |
| `QUICKSTART_README.md` | This file |

### The two runtime SFT fixes

1. **`--num-iterations` counts micro-batches upstream**, not optimizer steps; with
   gradient accumulation > 1 (any single-GPU run) the LR schedule overshoots and the
   loss goes NaN. The runner pre-scales the value by `grad_accum_steps` — and sniffs
   the upstream source first, so once upstream merges the fix the scaling switches
   itself off.
2. **Fully-padded packing rows at short `max_seq_len`** (conversations longer than a
   row never fit → all targets masked → NaN loss). The runner caps
   `render_conversation` at row capacity at runtime.

## The five steps

1. **Download Data** — ClimbMix shards (~100 MB each). 2 shards is enough to smoke-test;
   8 covers full 2B-char tokenizer training; ~170 for a GPT-2-grade run.
2. **Train Tokenizer** — rustbpe BPE, vocab 32768. Takes seconds-to-minutes.
3. **Get Model** — either train from scratch (depth 4–20) or import a pretrained
   checkpoint from HuggingFace. On a 16 GB card prefer small batch sizes for big
   depths — auto gradient accumulation keeps the effective batch size correct.
   Advanced CUDA controls can override total token batch, eval cadence, checkpoint
   save cadence, and `torch.compile`.
4. **Fine-Tune (SFT)** — teaches the chat special tokens. First run downloads the
   SmolTalk/MMLU/GSM8K datasets (a few GB) plus `identity_conversations.jsonl` (auto).
5. **Chat** — loads a checkpoint into the server process and chats over SSE.

## Notes / troubleshooting

- **CUDA out of memory** → lower **Batch Size** in the train/SFT panel (1 works everywhere).
- **Wizard ticks feel stale** → click **Reset UI**. This clears only the wizard progress
  marks; downloaded data, tokenizers, and checkpoints are kept.
- **Dark Mode** can be toggled from the sidebar and is remembered locally by the browser.
- **torch.compile problems** → tick **Disable torch.compile** (sets `TORCH_COMPILE_DISABLE=1`;
  slower but very compatible).
- The wizard sets `TORCHINDUCTOR_CACHE_DIR=%USERPROFILE%\.tic` and
  `TRITON_CACHE_DIR=%USERPROFILE%\.ttc` because fused-kernel filenames overflow the
  260-char Windows path limit under the default %TEMP% locations. Enabling long paths
  (`LongPathsEnabled=1` in the registry) is the system-wide alternative.
- **Realistic expectations**: depth 4 trains in minutes and babbles; depth 8–12 over
  hours starts to feel like a tiny LLM; karpathy's $100 speedrun model (d20, ~4h on
  8×H100) takes days on one consumer GPU. Importing `nanochat-students/base-d20`
  (~1.2 GB) gives you a real pretrained model to SFT/chat with immediately.
- A **base** (pretrained-only) model is not chat-tuned; run SFT on it for proper chat.
- Artifacts live in `~/.cache/nanochat` (override with `NANOCHAT_BASE_DIR`).
