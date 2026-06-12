"""
CUDA Quickstart GUI — step-by-step wizard for the full nanochat training pipeline.

A port of the nanochat-mlx quickstart wizard (https://github.com/scasella/nanochat-mlx)
to the original PyTorch/CUDA nanochat. Serves a web UI that walks through:
data download → tokenizer → training → SFT → chat.
Each stage runs as a subprocess with live SSE streaming of stdout/stderr.

Usage:
    python -m scripts.quickstart
    python -m scripts.quickstart --port 8080
"""

import argparse
import asyncio
import gc
import json
import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IDENTITY_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"

MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096


class SetupError(Exception):
    """Raised when a stage's prerequisites are not met."""


def build_parser():
    parser = argparse.ArgumentParser(description="NanoChat CUDA Quickstart")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    return parser


args = argparse.Namespace(port=8000, host="127.0.0.1")

# --- Globals ---

running_process: Optional[asyncio.subprocess.Process] = None
loaded_engine = None
loaded_tokenizer = None
loaded_model = None  # keep ref for explicit cleanup
loaded_depth = None
loaded_step = None
loaded_source = None

# base_train.py:  step 00010/00100 (10.00%) | loss: 4.123456 | ... | tok/sec: 12,345 | ...
METRIC_VALUE_RE = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"
BASE_METRIC_RE = re.compile(
    rf"step\s+(\d+)/(\d+).*?loss:\s*{METRIC_VALUE_RE}.*?tok/sec:\s*([\d,]+)",
    re.IGNORECASE,
)
# chat_sft.py:  step 00010 (3.21%) | loss: 4.123456 | ... | tok/sec: 12,345 | ...
SFT_METRIC_RE = re.compile(
    rf"step\s+(\d+)\s+\({METRIC_VALUE_RE}%\).*?loss:\s*{METRIC_VALUE_RE}.*?tok/sec:\s*([\d,]+)",
    re.IGNORECASE,
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def get_base_dir():
    base = os.environ.get("NANOCHAT_BASE_DIR")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    return base


def count_downloaded_shards():
    data_dir = os.path.join(get_base_dir(), "base_data_climbmix")
    if not os.path.isdir(data_dir):
        return 0
    return len([f for f in os.listdir(data_dir) if f.endswith(".parquet")])


def tokenizer_ready():
    tok_dir = os.path.join(get_base_dir(), "tokenizer")
    return (os.path.isfile(os.path.join(tok_dir, "tokenizer.pkl"))
            and os.path.isfile(os.path.join(tok_dir, "token_bytes.pt")))


def scan_checkpoints():
    """Scan base/sft checkpoint dirs, return a list of checkpoint dicts."""
    base = get_base_dir()
    results = []
    for source, dirname in [("base", "base_checkpoints"), ("sft", "chatsft_checkpoints")]:
        ckpt_base = os.path.join(base, dirname)
        if not os.path.isdir(ckpt_base):
            continue
        for d in sorted(os.listdir(ckpt_base)):
            if not (d.startswith("d") and d[1:].isdigit()):
                continue
            depth = int(d[1:])
            dpath = os.path.join(ckpt_base, d)
            for f in sorted(os.listdir(dpath)):
                m = re.match(r"meta_(\d+)\.json$", f)
                if not m:
                    continue
                step = int(m.group(1))
                model_path = os.path.join(dpath, f"model_{step:06d}.pt")
                if not os.path.isfile(model_path):
                    continue
                meta_path = os.path.join(dpath, f)
                try:
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    cfg = meta.get("model_config", {})
                    results.append({
                        "depth": depth,
                        "step": step,
                        "n_embd": cfg.get("n_embd", 0),
                        "n_head": cfg.get("n_head", 0),
                        "sequence_len": cfg.get("sequence_len", 0),
                        "window_pattern": cfg.get("window_pattern", "L"),
                        "source": source,
                        "date": os.path.getmtime(meta_path),
                    })
                except Exception:
                    pass
    results.sort(
        key=lambda c: (
            0 if c["source"] == "sft" else 1,
            -c["date"],
            -c["depth"],
            -c["step"],
        )
    )
    return results


# --- GPU info (via nvidia-smi so the server never holds a CUDA context) ---

_gpu_static = None  # cached (name, total_mib)

def get_gpu_info():
    global _gpu_static
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        line = out.stdout.strip().splitlines()[0]
        name, total, free = [p.strip() for p in line.split(",")]
        _gpu_static = (name, int(total))
        return {"name": name, "vram_total_mib": int(total), "vram_free_mib": int(free), "backend": "cuda"}
    except Exception:
        if _gpu_static:
            return {"name": _gpu_static[0], "vram_total_mib": _gpu_static[1], "vram_free_mib": None, "backend": "cuda"}
        return {"name": "CPU (no CUDA detected)", "vram_total_mib": None, "vram_free_mib": None, "backend": "cpu"}


# torch minor version -> compatible triton minor version (torch.compile on Windows
# only works when the community triton-windows build matches what torch expects)
TORCH_TRITON_COMPAT = {"2.9": "3.5", "2.8": "3.4", "2.7": "3.3"}


def check_triton_for_compile():
    """Return (ok, message) for torch.compile usability on native Windows."""
    from importlib.metadata import version, PackageNotFoundError
    triton_version = None
    for pkg in ("triton-windows", "triton"):
        try:
            triton_version = version(pkg)
            break
        except PackageNotFoundError:
            pass
    if triton_version is None:
        return False, ('triton is not installed -> torch.compile disabled for this run. '
                       'For faster training: uv pip install "triton-windows>=3.5,<3.6"')
    try:
        torch_minor = ".".join(version("torch").split(".")[:2])
        triton_minor = ".".join(triton_version.split(".")[:2])
    except Exception:
        return True, None
    expected = TORCH_TRITON_COMPAT.get(torch_minor)
    if expected is not None and triton_minor != expected:
        major, minor = expected.split(".")
        upper = f"{major}.{int(minor) + 1}"
        return False, (f"triton-windows {triton_version} does not match torch {torch_minor} "
                       f"(needs {expected}.x) -> torch.compile disabled for this run. "
                       f'Fix with: uv pip install "triton-windows>={expected},<{upper}"')
    return True, None


def check_status():
    """Check which pipeline stages are complete by inspecting the filesystem."""
    shard_count = count_downloaded_shards()
    checkpoints = scan_checkpoints()
    trained = {}
    sft_trained = {}
    for c in checkpoints:
        target = sft_trained if c["source"] == "sft" else trained
        target[c["depth"]] = target.get(c["depth"], 0) + 1

    chat_ready = loaded_engine is not None

    return {
        "data": shard_count >= 2,
        "data_shards": shard_count,
        "tokenizer": tokenizer_ready(),
        "train": trained,
        "sft": sft_trained,
        "chat": chat_ready,
        "chat_model": {"depth": loaded_depth, "step": loaded_step, "source": loaded_source} if chat_ready else None,
        "running": running_process is not None and running_process.returncode is None,
        "device": get_gpu_info(),
    }


# --- FastAPI ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse_error_response(message, code=400):
    """Return a one-shot SSE error response that the UI can render."""
    async def stream():
        yield f"data: {json.dumps({'type': 'error', 'text': message, 'code': code})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/")
async def root():
    ui_path = os.path.join(REPO_ROOT, "nanochat", "quickstart_ui.html")
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/status")
async def status():
    return check_status()


def preflight_stage(stage: str, depth: int, step: int):
    """Validate stage prerequisites before launching subprocesses."""
    if stage == "tokenizer":
        if count_downloaded_shards() < 2:
            raise SetupError("Training data is required first (at least 2 shards). Complete step 1.")
    elif stage == "train":
        if count_downloaded_shards() < 2:
            raise SetupError("Training data is required first (at least 2 shards). Complete step 1.")
        if not tokenizer_ready():
            raise SetupError("A tokenizer is required first. Complete step 2.")
    elif stage == "sft":
        if not tokenizer_ready():
            raise SetupError("A tokenizer is required first (train one or import from HuggingFace).")
        ckpt_dir = os.path.join(get_base_dir(), "base_checkpoints", f"d{depth}")
        if step > 0:
            ok = os.path.isfile(os.path.join(ckpt_dir, f"model_{step:06d}.pt"))
        else:
            ok = os.path.isdir(ckpt_dir) and any(
                f.startswith("model_") and f.endswith(".pt") for f in os.listdir(ckpt_dir)
            )
        if not ok:
            raise SetupError(f"No d{depth} base model found. Train or import a model in step 3 first.")
    elif stage == "import":
        try:
            import huggingface_hub  # noqa: F401
        except ModuleNotFoundError:
            raise SetupError("The huggingface_hub package is missing. Install it with: uv pip install huggingface_hub")


def _download_identity_file(dest_path):
    import urllib.request
    tmp_path = dest_path + ".tmp"
    with urllib.request.urlopen(IDENTITY_URL, timeout=60) as resp:
        content = resp.read()
    with open(tmp_path, "wb") as f:
        f.write(content)
    os.replace(tmp_path, dest_path)


def _kill_process_tree(process):
    """Terminate a subprocess and its children (Windows needs taskkill for the tree)."""
    if process is None or process.returncode is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
            )
        else:
            process.terminate()
    except Exception:
        pass


@app.get("/run/{stage}")
async def run_stage(stage: str, n_shards: int = 4, depth: int = 4,
                    step: int = -1,
                    num_iterations: int = -1,
                    window_pattern: str = "L", max_seq_len: int = 1024,
                    device_batch_size: int = 8, total_batch_size: int = -1,
                    save_every: int = -1,
                    eval_every: int = 100,
                    max_chars: int = 2_000_000_000,
                    repo: str = "nanochat-students/base-d20",
                    force_tokenizer: bool = False,
                    disable_compile: bool = False):
    """Run a pipeline stage as a subprocess, streaming output via SSE."""
    global running_process

    if running_process is not None and running_process.returncode is None:
        raise HTTPException(status_code=409, detail="A process is already running")

    python = sys.executable

    try:
        if stage == "data":
            cmd = [python, "-m", "nanochat.dataset", "-n", str(n_shards)]
        elif stage == "tokenizer":
            preflight_stage(stage, depth, step)
            cmd = [python, "-m", "scripts.tok_train", f"--max-chars={max_chars}"]
        elif stage == "train":
            preflight_stage(stage, depth, step)
            # keep evals light so the wizard stays responsive on a single consumer GPU
            eval_tokens = 24 * device_batch_size * max_seq_len
            cmd = [python, "-m", "scripts.base_train",
                   f"--depth={depth}",
                   f"--max-seq-len={max_seq_len}",
                   f"--window-pattern={window_pattern}",
                   f"--device-batch-size={device_batch_size}",
                   f"--eval-every={eval_every}",
                   f"--eval-tokens={eval_tokens}",
                   "--core-metric-every=-1",
                   "--sample-every=-1"]
            if num_iterations > 0:
                cmd.append(f"--num-iterations={num_iterations}")
            if total_batch_size > 0:
                cmd.append(f"--total-batch-size={total_batch_size}")
            effective_save_every = save_every if save_every > 0 else 500
            cmd.append(f"--save-every={effective_save_every}")
        elif stage == "sft":
            preflight_stage(stage, depth, step)
            eval_tokens = 8 * device_batch_size * 2048
            # quickstart_sft_runner wraps upstream chat_sft with two single-GPU
            # fixes applied at runtime (no upstream files are modified)
            cmd = [python, "-m", "scripts.quickstart_sft_runner",
                   f"--model-tag=d{depth}",
                   f"--device-batch-size={device_batch_size}",
                   f"--eval-every={eval_every}",
                   f"--eval-tokens={eval_tokens}",
                   "--chatcore-every=-1"]
            if step > 0:
                cmd.append(f"--model-step={step}")
            if num_iterations > 0:
                cmd.append(f"--num-iterations={num_iterations}")
            if total_batch_size > 0:
                cmd.append(f"--total-batch-size={total_batch_size}")
        elif stage == "import":
            preflight_stage(stage, depth, step)
            cmd = [python, "-m", "scripts.import_from_hf", f"--repo={repo}"]
            if force_tokenizer:
                cmd.append("--force")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown stage: {stage}")
    except SetupError as exc:
        return sse_error_response(str(exc))

    # graceful degradation: a missing/mismatched triton would crash torch.compile
    # mid-run on Windows, so fall back to eager mode for this run instead
    compile_notice = None
    if os.name == "nt" and not disable_compile and stage in ("train", "sft"):
        triton_ok, triton_msg = check_triton_for_compile()
        if not triton_ok:
            disable_compile = True
            compile_notice = triton_msg

    child_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    if os.name == "nt":
        # inductor/triton fused-kernel filenames are huge; under the default %TEMP%
        # cache dirs they overflow the 260-char Windows path limit (LongPathsEnabled=0).
        # Both vars must be set: torch re-derives TRITON_CACHE_DIR from the inductor
        # dir in some code paths and from its own default in others.
        home = os.path.expanduser("~")
        child_env.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(home, ".tic"))
        child_env.setdefault("TRITON_CACHE_DIR", os.path.join(home, ".ttc"))
    if disable_compile:
        child_env["TORCH_COMPILE_DISABLE"] = "1"
        child_env["TORCHDYNAMO_DISABLE"] = "1"

    unload_notice = None
    if stage in ("train", "sft") and loaded_model is not None:
        _unload_model()
        unload_notice = "Unloaded the chat model first to free GPU memory for training."

    # don't starve the rest of the system (and this server) while training
    popen_kwargs = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        def _low_priority():
            try:
                os.nice(10)
            except OSError:
                pass
        popen_kwargs["preexec_fn"] = _low_priority

    def parse_metric(line):
        m = BASE_METRIC_RE.search(line)
        if m:
            return {
                "type": "metric",
                "step": int(m.group(1)),
                "total": int(m.group(2)),
                "loss": float(m.group(3)),
                "tok_per_sec": int(m.group(4).replace(",", "")),
            }
        m = SFT_METRIC_RE.search(line)
        if m:
            step_i = int(m.group(1))
            pct = float(m.group(2))
            total = round(step_i / pct * 100) if pct > 0 else 0
            return {
                "type": "metric",
                "step": step_i,
                "total": total,
                "loss": float(m.group(3)),
                "tok_per_sec": int(m.group(4).replace(",", "")),
            }
        return None

    async def stream():
        global running_process
        try:
            if compile_notice:
                yield f"data: {json.dumps({'type': 'output', 'text': compile_notice})}\n\n"
            if unload_notice:
                yield f"data: {json.dumps({'type': 'output', 'text': unload_notice})}\n\n"
            if stage == "sft":
                ident_path = os.path.join(get_base_dir(), "identity_conversations.jsonl")
                if not os.path.isfile(ident_path):
                    yield f"data: {json.dumps({'type': 'output', 'text': 'Downloading identity_conversations.jsonl (~2.3 MB)...'})}\n\n"
                    try:
                        await asyncio.to_thread(_download_identity_file, ident_path)
                        yield f"data: {json.dumps({'type': 'output', 'text': 'Downloaded identity_conversations.jsonl.'})}\n\n"
                    except Exception as exc:
                        yield f"data: {json.dumps({'type': 'error', 'text': f'Failed to download identity_conversations.jsonl: {exc}'})}\n\n"
                        return

            running_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=child_env,
                cwd=REPO_ROOT,
                limit=2**20,
                **popen_kwargs,
            )

            # manual line reader: handles \n and bare \r (tqdm progress bars)
            buf = b""
            while True:
                chunk = await running_process.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                while True:
                    nl = -1
                    for sep in (b"\r\n", b"\n", b"\r"):
                        idx = buf.find(sep)
                        if idx != -1 and (nl == -1 or idx < nl):
                            nl = idx
                            sep_len = len(sep)
                    if nl == -1:
                        break
                    raw, buf = buf[:nl], buf[nl + sep_len:]
                    line = ANSI_RE.sub("", raw.decode("utf-8", errors="replace")).rstrip()
                    if not line:
                        continue
                    metric = parse_metric(line)
                    if metric:
                        yield f"data: {json.dumps(metric)}\n\n"
                    yield f"data: {json.dumps({'type': 'output', 'text': line})}\n\n"
            if buf:
                line = ANSI_RE.sub("", buf.decode("utf-8", errors="replace")).rstrip()
                if line:
                    yield f"data: {json.dumps({'type': 'output', 'text': line})}\n\n"

            await running_process.wait()
            code = running_process.returncode
            if code == 0:
                yield f"data: {json.dumps({'type': 'done', 'code': 0})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'text': f'Process exited with code {code}', 'code': code})}\n\n"

        except asyncio.CancelledError:
            _kill_process_tree(running_process)
            raise
        finally:
            running_process = None

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/stop")
async def stop():
    global running_process
    if running_process is None or running_process.returncode is not None:
        return {"status": "no_process"}
    _kill_process_tree(running_process)
    try:
        await asyncio.wait_for(running_process.wait(), timeout=5.0)
    except (asyncio.TimeoutError, ProcessLookupError):
        try:
            running_process.kill()
        except Exception:
            pass
    running_process = None
    return {"status": "stopped"}


@app.get("/checkpoints")
async def list_checkpoints():
    return scan_checkpoints()


class LoadRequest(BaseModel):
    depth: int = 12
    step: Optional[int] = None
    source: str = "base"


def _unload_model():
    """Free the currently loaded chat model and reclaim memory."""
    global loaded_engine, loaded_tokenizer, loaded_model, loaded_depth, loaded_step, loaded_source
    loaded_engine = None
    loaded_tokenizer = None
    loaded_model = None
    loaded_depth = None
    loaded_step = None
    loaded_source = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_model_sync(depth, step, source):
    import torch
    from nanochat.checkpoint_manager import load_model as ncm_load_model
    from nanochat.engine import Engine
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, tokenizer, _meta = ncm_load_model(
        source, device, phase="eval",
        model_tag=f"d{depth}",
        step=step if (step is not None and step > 0) else None,
    )
    engine = Engine(model, tokenizer)
    return engine, tokenizer, model


@app.post("/chat/load")
async def chat_load(req: LoadRequest):
    global loaded_engine, loaded_tokenizer, loaded_model, loaded_depth, loaded_step, loaded_source

    if running_process is not None and running_process.returncode is None:
        raise HTTPException(status_code=409, detail="A training/import process is running. Stop it before loading a chat model.")

    # Free previous model first to avoid double memory usage
    if loaded_model is not None:
        _unload_model()

    try:
        engine, tokenizer, model = await asyncio.to_thread(
            _load_model_sync, req.depth, req.step, req.source
        )
        loaded_engine = engine
        loaded_tokenizer = tokenizer
        loaded_model = model
        loaded_depth = req.depth
        loaded_step = req.step
        loaded_source = req.source
        return {"status": "loaded", "depth": req.depth, "source": req.source}
    except (FileNotFoundError, AssertionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/chat/unload")
async def chat_unload():
    """Unload the chat model to free memory."""
    if loaded_model is None:
        return {"status": "no_model"}
    _unload_model()
    return {"status": "unloaded"}


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.8
    max_tokens: int = 256
    top_k: int = 50


def validate_chat_request(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="At least one message is required.")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Too many messages. Maximum is {MAX_MESSAGES_PER_REQUEST}.")

    total_length = 0
    for i, msg in enumerate(request.messages):
        if msg.role not in {"user", "assistant"}:
            raise HTTPException(status_code=400, detail=f"Message {i} role must be 'user' or 'assistant'.")
        if not msg.content:
            raise HTTPException(status_code=400, detail=f"Message {i} is empty.")
        if len(msg.content) > MAX_MESSAGE_LENGTH:
            raise HTTPException(status_code=400, detail=f"Message {i} is too long. Maximum is {MAX_MESSAGE_LENGTH} characters.")
        total_length += len(msg.content)

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(status_code=400, detail=f"Conversation is too long. Maximum is {MAX_TOTAL_CONVERSATION_LENGTH} characters.")
    if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
        raise HTTPException(status_code=400, detail=f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}.")
    if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
        raise HTTPException(status_code=400, detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}.")
    if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
        raise HTTPException(status_code=400, detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}.")


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    if loaded_engine is None or loaded_tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded. POST /chat/load first.")
    validate_chat_request(request)

    tokenizer = loaded_tokenizer
    engine = loaded_engine
    bos_id = tokenizer.get_bos_token_id()

    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    tokens = [bos_id]
    for msg in request.messages:
        if msg.role == "user":
            tokens.append(user_start)
            tokens.extend(tokenizer.encode(msg.content))
            tokens.append(user_end)
        elif msg.role == "assistant":
            tokens.append(assistant_start)
            tokens.extend(tokenizer.encode(msg.content))
            tokens.append(assistant_end)
    tokens.append(assistant_start)

    async def stream():
        import random
        accumulated = []
        last_clean = ""
        for token_column, token_masks in engine.generate(
            tokens, num_samples=1,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k if request.top_k > 0 else None,
            seed=random.randint(0, 2**31 - 1),
        ):
            tok = token_column[0]
            if tok == assistant_end or tok == bos_id:
                break
            accumulated.append(tok)
            text = tokenizer.decode(accumulated)
            if not text.endswith("�"):
                new = text[len(last_clean):]
                if new:
                    yield f"data: {json.dumps({'token': new}, ensure_ascii=False)}\n\n"
                    last_clean = text
            await asyncio.sleep(0)  # let other requests breathe between tokens
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def main(argv=None):
    global args
    args = build_parser().parse_args(argv)
    if sys.platform == "win32":
        # Proactor loop is required for asyncio subprocess support on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    import uvicorn
    print(f"NanoChat CUDA Quickstart -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
