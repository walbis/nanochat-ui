"""Async job runner for the CUDA quickstart addon."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from scripts.quickstart_commands import build_argv, command_preview, get_command


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
METRIC_VALUE_RE = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"
BASE_METRIC_RE = re.compile(
    rf"step\s+(\d+)/(\d+).*?loss:\s*{METRIC_VALUE_RE}.*?tok/sec:\s*([\d,]+)",
    re.IGNORECASE,
)
SFT_METRIC_RE = re.compile(
    rf"step\s+(\d+)\s+\({METRIC_VALUE_RE}%\).*?loss:\s*{METRIC_VALUE_RE}.*?tok/sec:\s*([\d,]+)",
    re.IGNORECASE,
)


class JobError(RuntimeError):
    pass


@dataclass
class Job:
    id: int
    command_id: str
    args: Dict[str, Any]
    argv: List[str]
    preview: str
    status: str = "queued"
    returncode: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    subscribers: List[asyncio.Queue] = field(default_factory=list)
    process: Optional[asyncio.subprocess.Process] = None
    task: Optional[asyncio.Task] = None
    stop_requested: bool = False

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "command_id": self.command_id,
            "args": self.args,
            "argv": self.argv,
            "preview": self.preview,
            "status": self.status,
            "returncode": self.returncode,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


BusyCheck = Callable[[str], Awaitable[Optional[str]]]


class JobManager:
    def __init__(self, repo_root: str, busy_check: Optional[BusyCheck] = None):
        self.repo_root = repo_root
        self.busy_check = busy_check
        self._next_id = 1
        self._jobs: Dict[int, Job] = {}
        self._active_id: Optional[int] = None
        self._lock = asyncio.Lock()

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [job.public() for job in sorted(self._jobs.values(), key=lambda j: j.id, reverse=True)]

    def get_job(self, job_id: int) -> Job:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobError(f"Unknown job: {job_id}") from exc

    async def start(self, command_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        spec = get_command(command_id)
        async with self._lock:
            if self._active_id is not None:
                active = self._jobs.get(self._active_id)
                if active and active.status == "running":
                    raise JobError(f"Job {active.id} is already running")
            if self.busy_check:
                message = await self.busy_check(command_id)
                if message:
                    raise JobError(message)
            argv = build_argv(command_id, args or {})
            preview = command_preview(command_id, args or {})
            job = Job(
                id=self._next_id,
                command_id=command_id,
                args=args or {},
                argv=argv,
                preview=preview,
            )
            self._next_id += 1
            self._jobs[job.id] = job
            self._active_id = job.id
            job.task = asyncio.create_task(self._run(job, spec.gpu_heavy))
            return job.public()

    async def stop(self, job_id: int) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if job.process is None or job.process.returncode is not None:
            return {"status": "no_process", "job": job.public()}
        job.stop_requested = True
        _kill_process_tree(job.process)
        try:
            await asyncio.wait_for(job.process.wait(), timeout=5.0)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                job.process.kill()
            except Exception:
                pass
        job.returncode = job.process.returncode
        job.status = "stopped"
        job.finished_at = time.time()
        await self._emit(job, {"type": "error", "text": "Job stopped by user", "code": -9})
        if self._active_id == job.id:
            self._active_id = None
        return {"status": "stopped", "job": job.public()}

    async def event_stream(self, job_id: int):
        job = self.get_job(job_id)
        queue: asyncio.Queue = asyncio.Queue()
        for event in job.events:
            yield _sse(event)
        if job.status in {"done", "error", "stopped"}:
            return
        job.subscribers.append(queue)
        try:
            while True:
                event = await queue.get()
                yield _sse(event)
                if event.get("type") in {"done", "error"}:
                    break
        finally:
            try:
                job.subscribers.remove(queue)
            except ValueError:
                pass

    async def _run(self, job: Job, gpu_heavy: bool):
        job.status = "running"
        job.started_at = time.time()
        await self._emit(job, {"type": "output", "text": "$ " + job.preview})

        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        if os.name == "nt":
            home = os.path.expanduser("~")
            env.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(home, ".tic"))
            env.setdefault("TRITON_CACHE_DIR", os.path.join(home, ".ttc"))

        popen_kwargs: Dict[str, Any] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
        else:
            def _low_priority():
                try:
                    os.nice(10)
                except OSError:
                    pass
            if gpu_heavy:
                popen_kwargs["preexec_fn"] = _low_priority

        try:
            job.process = await asyncio.create_subprocess_exec(
                *job.argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                cwd=self.repo_root,
                limit=2**20,
                **popen_kwargs,
            )
            assert job.process.stdout is not None
            await self._read_stdout(job)
            await job.process.wait()
            job.returncode = job.process.returncode
            job.finished_at = time.time()
            if job.stop_requested:
                job.status = "stopped"
                if not any(event.get("text") == "Job stopped by user" for event in job.events):
                    await self._emit(job, {"type": "error", "text": "Job stopped by user", "code": job.returncode or -9})
            elif job.returncode == 0:
                job.status = "done"
                await self._emit(job, {"type": "done", "code": 0})
            else:
                job.status = "error"
                await self._emit(job, {
                    "type": "error",
                    "text": f"Process exited with code {job.returncode}",
                    "code": job.returncode,
                })
        except asyncio.CancelledError:
            _kill_process_tree(job.process)
            job.status = "stopped"
            job.finished_at = time.time()
            raise
        except Exception as exc:
            job.status = "error"
            job.finished_at = time.time()
            await self._emit(job, {"type": "error", "text": str(exc), "code": 1})
        finally:
            if self._active_id == job.id:
                self._active_id = None

    async def _read_stdout(self, job: Job):
        assert job.process is not None and job.process.stdout is not None
        buf = b""
        while True:
            chunk = await job.process.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                nl = -1
                sep_len = 0
                for sep in (b"\r\n", b"\n", b"\r"):
                    idx = buf.find(sep)
                    if idx != -1 and (nl == -1 or idx < nl):
                        nl = idx
                        sep_len = len(sep)
                if nl == -1:
                    break
                raw, buf = buf[:nl], buf[nl + sep_len:]
                await self._emit_line(job, raw)
        if buf:
            await self._emit_line(job, buf)

    async def _emit_line(self, job: Job, raw: bytes):
        line = ANSI_RE.sub("", raw.decode("utf-8", errors="replace")).rstrip()
        if not line:
            return
        metric = parse_metric(line)
        if metric:
            await self._emit(job, metric)
        await self._emit(job, {"type": "output", "text": line})

    async def _emit(self, job: Job, event: Dict[str, Any]):
        job.events.append(event)
        if len(job.events) > 2000:
            job.events = job.events[-2000:]
        for queue in list(job.subscribers):
            await queue.put(event)


def parse_metric(line: str) -> Optional[Dict[str, Any]]:
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


def _kill_process_tree(process: Optional[asyncio.subprocess.Process]):
    if process is None or process.returncode is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
        else:
            process.terminate()
    except Exception:
        pass


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
