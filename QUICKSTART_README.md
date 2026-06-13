# NanoChat CUDA Quickstart (Windows + NVIDIA)

A port of the [nanochat-mlx](https://github.com/scasella/nanochat-mlx) quickstart wizard UI
to the original PyTorch/CUDA [nanochat](https://github.com/karpathy/nanochat), so the full
"train your own ChatGPT" pipeline runs on an NVIDIA GPU (tested on RTX 5070 Ti 16 GB,
Windows 11, Python 3.10 venv via uv, torch 2.9.1+cu128).

This repo is now a UI/control layer over nanochat. It keeps the original nanochat
training and inference scripts intact, while exposing them through a single local
FastAPI server and a browser UI at `http://127.0.0.1:8000`.

**Zero upstream nanochat modifications.** Original nanochat training/chat scripts are not
edited. The addon lives in quickstart-specific files and adapts the original CLIs through
safe subprocess argv builders, runtime wrappers, and UI endpoints. The two upstream bugs
the wizard would otherwise hit are compensated at runtime by
`scripts/quickstart_sft_runner.py` (see below); both are also reported upstream
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

Recommended from Windows PowerShell. First enter the repo directory:

```powershell
cd C:\Users\berka\Desktop\Projects\nanochat-mlx\nanochat
```

If you cloned from GitHub instead:

```powershell
git clone https://github.com/walbis/nanochat-ui.git
cd nanochat-ui
```

Then run:

```powershell
uv run --extra gpu python -m scripts.quickstart --port 8000
```

Then open http://127.0.0.1:8000. If port 8000 is already in use:

```powershell
uv run --extra gpu python -m scripts.quickstart --port 8001
```

`start_quickstart.bat` is a convenience wrapper for the same command.

If you update the repo while the UI is open, restart this command. The HTML file can
reload without restarting Python, but new backend endpoints and CLI capabilities only
exist after the quickstart server process restarts.

## What we added

- **Pipeline wizard** for data, tokenizer, base model train/import, optional SFT, and chat.
- **Dashboard** for CUDA/VRAM, downloaded artifacts, checkpoint counts, active jobs,
  and chat worker status.
- **CLI capability registry** in `scripts/quickstart_commands.py`, covering nanochat's
  original scripts with validated argument schemas and safe command previews.
- **Generic job manager** in `scripts/quickstart_jobs.py`, with live SSE logs, parsed
  metrics, status, exit code, and Windows process-tree stop support.
- **Chat worker service** in `scripts/quickstart_chat_workers.py`, adapting
  `chat_web.py` worker-pool behavior into the existing quickstart server.
- **Advanced panels** for base training, SFT, tokenizer/base/chat evaluation, RL,
  report generation, one-shot chat CLI, checkpoint inspection, and run recipes.
- **Dark/light mode** plus per-page explanations for purpose, GPU/CPU usage,
  optionality, and produced artifacts.

## Added files (everything else is untouched upstream nanochat)

| File | Purpose |
|---|---|
| `scripts/quickstart.py` | FastAPI backend: runs each pipeline stage as a subprocess, streams stdout + parsed metrics over SSE, serves the chat model in-process |
| `nanochat/quickstart_ui.html` | The wizard UI (adapted from nanochat-mlx: MLX/Apple knobs → CUDA knobs) |
| `scripts/quickstart_commands.py` | Central CLI capability registry, argument validation, and safe argv command previews |
| `scripts/quickstart_jobs.py` | Generic job manager with SSE logs, parsed metrics, exit status, and process-tree stop support |
| `scripts/quickstart_chat_workers.py` | Additive chat_web-style worker pool for one or more CUDA GPUs |
| `scripts/quickstart_sft_runner.py` | Wraps upstream `chat_sft` with two runtime fixes (below) without editing it |
| `scripts/import_from_hf.py` | Downloads a pretrained nanochat checkpoint (e.g. `nanochat-students/base-d20`) straight into `~/.cache/nanochat/base_checkpoints` — no conversion needed since HF repos host the native torch format |
| `start_quickstart.bat` | Launcher |
| `QUICKSTART_README.md` | This file |

## GPU / CPU behavior

Not every button uses the GPU. The GPU is used when a model is loaded for training,
evaluation, generation, or chat. Dataset download, tokenizer training, checkpoint
inspection, report generation, and model import are mostly network/disk/CPU tasks.

| UI action | Uses GPU? | Notes |
|---|---:|---|
| Download Data | No | Downloads ClimbMix parquet shards; network/disk work. |
| Train Tokenizer | No | Trains the BPE tokenizer; CPU/disk work, not model training. |
| Import Model | No | Downloads/copies native nanochat checkpoint files from HuggingFace. |
| Train Base Model | Yes | CUDA if available; this is the main pretraining workload. |
| Fine-Tune (SFT) | Yes | CUDA if available; turns a base model into a chat-tuned model. |
| Chat | Yes | GPU if the model is loaded on CUDA; CPU fallback is possible but slow. |
| Advanced Train | Yes | Runs `base_train` and `chat_sft`; GPU-heavy. |
| Evaluation | Mixed | `tok_eval` is CPU; `base_eval` and `chat_eval` load models and can use GPU. |
| Chat Server | Yes | Integrated `chat_web.py`-style worker pool; multi-GPU requires CUDA. |
| RL | Yes | Runs `chat_rl` after SFT; GPU-heavy and optional. |
| Checkpoints | No | Reads checkpoint metadata from disk. |
| Recipes | Mixed | Depends on script; `speedrun` is GPU, `runcpu` is CPU/MPS. |

### The two runtime SFT fixes

1. **`--num-iterations` counts micro-batches upstream**, not optimizer steps; with
   gradient accumulation > 1 (any single-GPU run) the LR schedule overshoots and the
   loss goes NaN. The runner pre-scales the value by `grad_accum_steps` — and sniffs
   the upstream source first, so once upstream merges the fix the scaling switches
   itself off.
2. **Fully-padded packing rows at short `max_seq_len`** (conversations longer than a
   row never fit → all targets masked → NaN loss). The runner caps
   `render_conversation` at row capacity at runtime.

## UI sections

1. **Download Data** — ClimbMix shards (~100 MB each). 2 shards is enough to smoke-test;
   8 covers full 2B-char tokenizer training; ~170 for a GPT-2-grade run. Required
   only when training from scratch.
2. **Train Tokenizer** — rustbpe BPE, vocab 32768. Takes seconds-to-minutes.
   Required when training from scratch; optional when importing a compatible checkpoint.
3. **Get Model** — either train from scratch (depth 4–20) or import a pretrained
   checkpoint from HuggingFace. On a 16 GB card prefer small batch sizes for big
   depths — auto gradient accumulation keeps the effective batch size correct.
   Advanced CUDA controls can override total token batch, eval cadence, checkpoint
   save cadence, and `torch.compile`.
4. **Fine-Tune (SFT)** — teaches the chat special tokens. First run downloads the
   SmolTalk/MMLU/GSM8K datasets (a few GB) plus `identity_conversations.jsonl` (auto).
   Technically optional, but strongly recommended for proper chat behavior.
5. **Chat** — loads a checkpoint into the server process and chats over SSE.
6. **Dashboard** - CUDA/VRAM, artifacts, checkpoints, active jobs, and worker status.
7. **Advanced Train** - full `base_train` and `chat_sft` CLI forms with advanced flags.
8. **Evaluation** - tokenizer, base, chat, and report jobs.
9. **Chat Server** - chat_web-style worker pool controls, health, stats, and unload.
10. **RL** - `chat_rl` job launcher with original CLI flags. Optional and normally
    used after SFT.
11. **Checkpoints** - base/SFT/RL checkpoint discovery and metadata.
12. **Recipes** - `runs/*.sh` launchers with command preview. On native Windows these
    require bash or WSL.

## CLI coverage

| CLI / script | UI support |
|---|---|
| `nanochat.dataset` | Shards and workers |
| `scripts.tok_train` | `max-chars`, `doc-cap`, `vocab-size` |
| `scripts.tok_eval` | Generic eval job |
| `scripts.base_train` | Full advanced training form |
| `scripts.base_eval` | CORE/BPB/sample/HF eval options |
| `scripts.chat_sft` | SFT form through `quickstart_sft_runner` runtime wrapper |
| `scripts.chat_eval` | Task/source/sample/batch/max-problem controls |
| `scripts.chat_rl` | RL section |
| `scripts.chat_cli` | Prompt, source, step, temperature, top-k, max tokens |
| `scripts.chat_web` | Integrated worker-pool behavior, health/stats, multi-GPU chat |
| `nanochat.report` | Generate/reset report job |
| `runs/runcpu.sh`, `runs/speedrun.sh`, `runs/scaling_laws.sh`, `runs/miniseries.sh` | Recipe launchers |

## Backend API

- `GET /api/capabilities` returns the command registry used to render the UI.
- `POST /api/jobs` starts a validated subprocess job.
- `GET /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/events`, and
  `POST /api/jobs/{id}/stop` manage status, SSE logs, metrics, and process-tree stop.
- `GET /api/checkpoints` lists base/SFT/RL checkpoint metadata.
- `POST /api/chat/workers/load` and `POST /api/chat/workers/unload` manage the
  chat_web-style worker pool.
- `GET /api/chat/health` and `GET /api/chat/stats` expose worker readiness and usage.
- `POST /chat/completions` stays OpenAI-style SSE compatible. If the worker pool is
  loaded, chat is served through the pool; otherwise it uses the simple in-process model.

## Notes / troubleshooting

- **CUDA out of memory** → lower **Batch Size** in the train/SFT panel (1 works everywhere).
- **Wizard ticks feel stale** → click **Reset UI**. This clears only the wizard progress
  marks; downloaded data, tokenizers, and checkpoints are kept.
- **Dark Mode** can be toggled from the sidebar and is remembered locally by the browser.
- **"No commands available" / command registry warning** → restart the quickstart
  server. This means the browser loaded newer HTML than the Python backend currently
  running on port 8000.
- **GPU-heavy jobs and chat workers are mutually exclusive** to avoid VRAM conflicts.
  Unload the worker pool before training/SFT/RL/eval jobs, or unload the simple chat
  model before starting GPU-heavy jobs.
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
