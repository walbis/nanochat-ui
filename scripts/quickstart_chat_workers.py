"""Internal chat worker pool for the CUDA quickstart addon.

This adapts the behavior of upstream scripts.chat_web without importing that
module, because chat_web parses CLI args at import time. Upstream files stay
untouched.
"""

from __future__ import annotations

import asyncio
import gc
import json
import random
import threading
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional


@dataclass
class ChatWorker:
    gpu_id: int
    device: Any
    engine: Any
    tokenizer: Any
    model: Any


class ChatWorkerPool:
    def __init__(self):
        self.workers: List[ChatWorker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()
        self.settings: Dict[str, Any] = {}
        self.loading = False
        self.inflight = 0  # number of in-progress /chat/completions streams

    @property
    def loaded(self) -> bool:
        return bool(self.workers)

    async def load(self, source: str = "sft", model_tag: Optional[str] = None,
                   step: Optional[int] = None, num_gpus: int = 1,
                   device_type: str = "", temperature: float = 0.8,
                   top_k: int = 50, max_tokens: int = 512) -> Dict[str, Any]:
        if self.loaded or self.loading:
            raise RuntimeError("Chat worker pool is already loaded. Unload it first.")
        self.loading = True
        try:
            await asyncio.to_thread(
                self._load_sync, source, model_tag or None,
                step if step and step > 0 else None,
                max(1, int(num_gpus)), device_type,
                float(temperature), int(top_k), int(max_tokens),
            )
        finally:
            self.loading = False
        return self.health()

    def _load_sync(self, source: str, model_tag: Optional[str], step: Optional[int],
                   num_gpus: int, device_type: str, temperature: float,
                   top_k: int, max_tokens: int):
        import torch
        from nanochat.checkpoint_manager import load_model
        from nanochat.common import autodetect_device_type
        from nanochat.engine import Engine

        resolved_device_type = autodetect_device_type() if device_type == "" else device_type
        if num_gpus > 1 and resolved_device_type != "cuda":
            raise RuntimeError("Only CUDA supports multiple chat workers.")
        if resolved_device_type == "cuda":
            available = torch.cuda.device_count()
            if available <= 0:
                raise RuntimeError("CUDA was requested but no CUDA GPUs are visible.")
            if num_gpus > available:
                raise RuntimeError(f"Requested {num_gpus} GPUs, but only {available} are visible.")
        else:
            num_gpus = 1

        loaded_workers: List[ChatWorker] = []
        try:
            for gpu_id in range(num_gpus):
                if resolved_device_type == "cuda":
                    device = torch.device(f"cuda:{gpu_id}")
                else:
                    device = torch.device(resolved_device_type)
                model, tokenizer, _meta = load_model(
                    source, device, phase="eval", model_tag=model_tag, step=step,
                )
                loaded_workers.append(ChatWorker(
                    gpu_id=gpu_id,
                    device=device,
                    engine=Engine(model, tokenizer),
                    tokenizer=tokenizer,
                    model=model,
                ))
        except Exception:
            self._clear_workers(loaded_workers)
            raise

        self.workers = loaded_workers
        self.available_workers = asyncio.Queue()
        for worker in self.workers:
            self.available_workers.put_nowait(worker)
        self.settings = {
            "source": source,
            "model_tag": model_tag,
            "step": step,
            "num_gpus": num_gpus,
            "device_type": resolved_device_type,
            "temperature": temperature,
            "top_k": top_k,
            "max_tokens": max_tokens,
        }

    async def unload(self) -> Dict[str, Any]:
        if self.inflight > 0:
            raise RuntimeError(
                f"{self.inflight} chat completion(s) still streaming. Stop them before unloading."
            )
        self._clear_workers(self.workers)
        self.workers = []
        self.available_workers = asyncio.Queue()
        self.settings = {}
        return self.health()

    def _clear_workers(self, workers: List[ChatWorker]):
        workers.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "ready": self.loaded,
            "loading": self.loading,
            "num_gpus": self.settings.get("num_gpus", 0),
            "available_workers": self.available_workers.qsize() if self.loaded else 0,
            "inflight": self.inflight,
            "settings": self.settings,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "total_workers": len(self.workers),
            "available_workers": self.available_workers.qsize() if self.loaded else 0,
            "busy_workers": len(self.workers) - (self.available_workers.qsize() if self.loaded else 0),
            "settings": self.settings,
            "workers": [
                {"gpu_id": worker.gpu_id, "device": str(worker.device)}
                for worker in self.workers
            ],
        }

    async def complete(self, messages: List[Any], temperature: Optional[float],
                       top_k: Optional[int], max_tokens: Optional[int]) -> AsyncGenerator[str, None]:
        if not self.loaded:
            raise RuntimeError("Chat worker pool is not loaded.")
        worker = await self.available_workers.get()
        self.inflight += 1
        try:
            conversation_tokens = build_conversation_tokens(worker.tokenizer, messages)
            async for chunk in generate_stream(
                worker,
                conversation_tokens,
                temperature if temperature is not None else self.settings.get("temperature", 0.8),
                top_k if top_k is not None else self.settings.get("top_k", 50),
                max_tokens if max_tokens is not None else self.settings.get("max_tokens", 512),
            ):
                yield chunk
        finally:
            self.inflight -= 1
            # only return the worker to the pool if it still belongs to this pool
            # (an unload() during streaming would have replaced the queue/workers)
            if worker in self.workers:
                await self.available_workers.put(worker)


def build_conversation_tokens(tokenizer: Any, messages: List[Any]) -> List[int]:
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    tokens = [bos]
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        if role == "user":
            tokens.append(user_start)
            tokens.extend(tokenizer.encode(content))
            tokens.append(user_end)
        elif role == "assistant":
            tokens.append(assistant_start)
            tokens.extend(tokenizer.encode(content))
            tokens.append(assistant_end)
    tokens.append(assistant_start)
    return tokens


async def aiter_engine_tokens(engine: Any, tokens: List[int], temperature: float,
                              top_k: int, max_tokens: int):
    """Drive the synchronous engine.generate loop on a worker thread and yield
    token ids to the event loop, so the blocking CUDA forward passes don't freeze
    the server. A stop flag lets us abandon generation early without running the
    model all the way to max_tokens. Shared by the worker pool and the simple
    single-model chat path."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    SENTINEL = object()

    def produce():
        try:
            for token_column, _masks in engine.generate(
                tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                seed=random.randint(0, 2**31 - 1),
            ):
                if stop.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, int(token_column[0]))
        except Exception as exc:  # surface generation errors to the consumer
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    fut = loop.run_in_executor(None, produce)
    try:
        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()  # tell the producer to stop on early break / client disconnect
        try:
            await fut
        except Exception:
            pass


async def generate_stream(worker: ChatWorker, tokens: List[int], temperature: float,
                          top_k: int, max_tokens: int) -> AsyncGenerator[str, None]:
    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()
    accumulated: List[int] = []
    last_clean = ""

    async for tok in aiter_engine_tokens(worker.engine, tokens, temperature, top_k, max_tokens):
        if tok == assistant_end or tok == bos:
            break
        accumulated.append(tok)
        text = worker.tokenizer.decode(accumulated)
        if not text.endswith("\ufffd"):
            new = text[len(last_clean):]
            if new:
                yield f"data: {json.dumps({'token': new, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean = text
    yield f"data: {json.dumps({'done': True})}\n\n"
