"""Command registry for the CUDA quickstart addon.

This module describes nanochat CLI entrypoints without modifying upstream
scripts. The quickstart backend uses these specs to validate UI payloads and to
build safe argv lists for subprocess execution.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CommandValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ArgSpec:
    name: str
    label: str
    kind: str = "str"  # str|int|float|bool|select
    default: Any = None
    flag: Optional[str] = None
    help: str = ""
    choices: Optional[List[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    required: bool = False
    advanced: bool = False
    positional: bool = False
    include_if_default: bool = False

    def public(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "flag": self.flag,
            "help": self.help,
            "choices": self.choices,
            "min": self.min,
            "max": self.max,
            "required": self.required,
            "advanced": self.advanced,
            "positional": self.positional,
        }


@dataclass(frozen=True)
class CommandSpec:
    id: str
    label: str
    category: str
    description: str
    module: Optional[str] = None
    script: Optional[str] = None
    args: List[ArgSpec] = field(default_factory=list)
    requires_cuda: bool = False
    gpu_heavy: bool = False
    recipe: bool = False

    def public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "description": self.description,
            "module": self.module,
            "script": self.script,
            "requires_cuda": self.requires_cuda,
            "gpu_heavy": self.gpu_heavy,
            "recipe": self.recipe,
            "args": [arg.public() for arg in self.args],
        }


def _a(name: str, label: str, **kwargs) -> ArgSpec:
    return ArgSpec(name=name, label=label, flag=kwargs.pop("flag", f"--{name.replace('_', '-')}"), **kwargs)


COMMANDS: Dict[str, CommandSpec] = {}


def _register(spec: CommandSpec) -> CommandSpec:
    COMMANDS[spec.id] = spec
    return spec


def _train_common() -> List[ArgSpec]:
    return [
        _a("run", "W&B Run", default="dummy", help="'dummy' disables wandb logging"),
        _a("device_type", "Device Type", kind="select", default="", choices=["", "cuda", "cpu", "mps"]),
        _a("fp8", "Enable FP8", kind="bool", default=False, help="Requires H100+ and torchao"),
        _a("fp8_recipe", "FP8 Recipe", kind="select", default="tensorwise", choices=["tensorwise", "rowwise"], advanced=True),
        _a("depth", "Depth", kind="int", default=20, min=1),
        _a("aspect_ratio", "Aspect Ratio", kind="int", default=64, min=1, advanced=True),
        _a("head_dim", "Head Dim", kind="int", default=128, min=1, advanced=True),
        _a("max_seq_len", "Max Seq Len", kind="int", default=2048, min=128),
        _a("window_pattern", "Window Pattern", default="SSSL"),
        _a("num_iterations", "Num Iterations", kind="int", default=-1, min=-1),
        _a("target_flops", "Target FLOPs", kind="float", default=-1.0, advanced=True),
        _a("target_param_data_ratio", "Target Param/Data Ratio", kind="float", default=12, advanced=True),
        _a("device_batch_size", "Device Batch Size", kind="int", default=32, min=1),
        _a("total_batch_size", "Total Batch Size", kind="int", default=-1, min=-1),
        _a("embedding_lr", "Embedding LR", kind="float", default=0.3, advanced=True),
        _a("unembedding_lr", "Unembedding LR", kind="float", default=0.008, advanced=True),
        _a("weight_decay", "Weight Decay", kind="float", default=0.28, advanced=True),
        _a("matrix_lr", "Matrix LR", kind="float", default=0.02, advanced=True),
        _a("scalar_lr", "Scalar LR", kind="float", default=0.5, advanced=True),
        _a("warmup_steps", "Warmup Steps", kind="int", default=40, min=0, advanced=True),
        _a("warmdown_ratio", "Warmdown Ratio", kind="float", default=0.65, min=0, max=1, advanced=True),
        _a("final_lr_frac", "Final LR Fraction", kind="float", default=0.05, min=0, advanced=True),
        _a("resume_from_step", "Resume From Step", kind="int", default=-1, min=-1),
        _a("eval_every", "Eval Every", kind="int", default=250, min=-1),
        _a("eval_tokens", "Eval Tokens", kind="int", default=80 * 524288, min=1, advanced=True),
        _a("core_metric_every", "CORE Metric Every", kind="int", default=2000, min=-1),
        _a("core_metric_max_per_task", "CORE Max Per Task", kind="int", default=500, min=-1, advanced=True),
        _a("sample_every", "Sample Every", kind="int", default=2000, min=-1),
        _a("save_every", "Save Every", kind="int", default=-1, min=-1),
        _a("model_tag", "Model Tag", default="", advanced=True),
    ]


_register(CommandSpec(
    id="dataset",
    label="Download Dataset",
    category="pipeline",
    description="Download NVIDIA ClimbMix pretraining shards.",
    module="nanochat.dataset",
    args=[
        _a("num_files", "Train Shards", kind="int", default=-1, min=-1, flag="-n"),
        _a("num_workers", "Download Workers", kind="int", default=4, min=1, flag="-w"),
    ],
))

_register(CommandSpec(
    id="tok_train",
    label="Train Tokenizer",
    category="pipeline",
    description="Train the rustbpe tokenizer.",
    module="scripts.tok_train",
    args=[
        _a("max_chars", "Max Chars", kind="int", default=2_000_000_000, min=1),
        _a("doc_cap", "Doc Cap", kind="int", default=10_000, min=1),
        _a("vocab_size", "Vocab Size", kind="int", default=32768, min=256),
    ],
))

_register(CommandSpec(
    id="tok_eval",
    label="Evaluate Tokenizer",
    category="evaluation",
    description="Run tokenizer evaluation.",
    module="scripts.tok_eval",
))

_register(CommandSpec(
    id="base_train",
    label="Train Base Model",
    category="training",
    description="Run scripts.base_train with full CLI controls.",
    module="scripts.base_train",
    args=_train_common(),
    requires_cuda=True,
    gpu_heavy=True,
))

_register(CommandSpec(
    id="base_eval",
    label="Evaluate Base Model",
    category="evaluation",
    description="Run CORE/BPB/sample evaluation for a base model or HuggingFace model.",
    module="scripts.base_eval",
    args=[
        _a("eval", "Evaluations", default="core,bpb,sample", help="Comma-separated: core,bpb,sample"),
        _a("hf_path", "HuggingFace Path", default=""),
        _a("model_tag", "Model Tag", default=""),
        _a("step", "Step", kind="int", default=-1, min=-1),
        _a("max_per_task", "Max Per CORE Task", kind="int", default=-1, min=-1),
        _a("device_batch_size", "Device Batch Size", kind="int", default=32, min=1),
        _a("split_tokens", "Split Tokens", kind="int", default=40 * 524288, min=1),
        _a("device_type", "Device Type", kind="select", default="", choices=["", "cuda", "cpu", "mps"]),
    ],
    gpu_heavy=True,
))

_register(CommandSpec(
    id="chat_sft",
    label="Supervised Fine-Tune",
    category="training",
    description="Run upstream SFT with full CLI controls.",
    module="scripts.quickstart_sft_runner",
    args=[
        _a("run", "W&B Run", default="dummy"),
        _a("device_type", "Device Type", kind="select", default="", choices=["", "cuda", "cpu", "mps"]),
        _a("model_tag", "Model Tag", default="", required=True),
        _a("model_step", "Model Step", kind="int", default=-1, min=-1),
        _a("load_optimizer", "Load Optimizer", kind="int", default=1, min=0, max=1, advanced=True),
        _a("num_iterations", "Num Iterations", kind="int", default=-1, min=-1),
        _a("max_seq_len", "Max Seq Len", kind="int", default=-1, min=-1),
        _a("device_batch_size", "Device Batch Size", kind="int", default=-1, min=-1),
        _a("total_batch_size", "Total Batch Size", kind="int", default=-1, min=-1),
        _a("embedding_lr", "Embedding LR", kind="float", default=-1, advanced=True),
        _a("unembedding_lr", "Unembedding LR", kind="float", default=-1, advanced=True),
        _a("matrix_lr", "Matrix LR", kind="float", default=-1, advanced=True),
        _a("init_lr_frac", "Init LR Fraction", kind="float", default=0.8, advanced=True),
        _a("warmup_ratio", "Warmup Ratio", kind="float", default=0.0, min=0, advanced=True),
        _a("warmdown_ratio", "Warmdown Ratio", kind="float", default=0.5, min=0, advanced=True),
        _a("final_lr_frac", "Final LR Fraction", kind="float", default=0.0, min=0, advanced=True),
        _a("eval_every", "Eval Every", kind="int", default=200, min=-1),
        _a("eval_tokens", "Eval Tokens", kind="int", default=40 * 524288, min=1, advanced=True),
        _a("chatcore_every", "ChatCORE Every", kind="int", default=200, min=-1),
        _a("chatcore_max_cat", "ChatCORE Max Cat", kind="int", default=-1, min=-1, advanced=True),
        _a("chatcore_max_sample", "ChatCORE Max Sample", kind="int", default=24, min=-1, advanced=True),
        _a("mmlu_epochs", "MMLU Epochs", kind="int", default=3, min=0, advanced=True),
        _a("gsm8k_epochs", "GSM8K Epochs", kind="int", default=4, min=0, advanced=True),
    ],
    requires_cuda=True,
    gpu_heavy=True,
))

_register(CommandSpec(
    id="chat_eval",
    label="Evaluate Chat Model",
    category="evaluation",
    description="Run chat benchmark evaluation.",
    module="scripts.chat_eval",
    args=[
        _a("source", "Source", kind="select", default="sft", choices=["sft", "rl"], required=True, flag="-i"),
        _a("task_name", "Task Name", default="", flag="-a"),
        _a("temperature", "Temperature", kind="float", default=0.0, min=0, max=2, flag="-t"),
        _a("max_new_tokens", "Max New Tokens", kind="int", default=512, min=1, flag="-m"),
        _a("num_samples", "Num Samples", kind="int", default=1, min=1, flag="-n"),
        _a("top_k", "Top-K", kind="int", default=50, min=0, max=200, flag="-k"),
        _a("batch_size", "Batch Size", kind="int", default=8, min=1, flag="-b"),
        _a("model_tag", "Model Tag", default="", flag="-g"),
        _a("step", "Step", kind="int", default=-1, min=-1, flag="-s"),
        _a("max_problems", "Max Problems", kind="int", default=-1, min=-1, flag="-x"),
        _a("device_type", "Device Type", kind="select", default="", choices=["", "cuda", "cpu", "mps"]),
    ],
    gpu_heavy=True,
))

_register(CommandSpec(
    id="chat_rl",
    label="Reinforcement Learning",
    category="training",
    description="Run GSM8K reinforcement learning.",
    module="scripts.chat_rl",
    args=[
        _a("run", "W&B Run", default="dummy"),
        _a("device_type", "Device Type", kind="select", default="", choices=["", "cuda", "cpu", "mps"]),
        _a("model_tag", "Model Tag", default=""),
        _a("model_step", "Model Step", kind="int", default=-1, min=-1),
        _a("num_epochs", "Num Epochs", kind="int", default=1, min=1),
        _a("device_batch_size", "Device Batch Size", kind="int", default=8, min=1),
        _a("examples_per_step", "Examples Per Step", kind="int", default=16, min=1),
        _a("num_samples", "Num Samples", kind="int", default=16, min=1),
        _a("max_new_tokens", "Max New Tokens", kind="int", default=256, min=1),
        _a("temperature", "Temperature", kind="float", default=1.0, min=0, max=2),
        _a("top_k", "Top-K", kind="int", default=50, min=0, max=200),
        _a("embedding_lr", "Embedding LR", kind="float", default=0.2, advanced=True),
        _a("unembedding_lr", "Unembedding LR", kind="float", default=0.004, advanced=True),
        _a("matrix_lr", "Matrix LR", kind="float", default=0.02, advanced=True),
        _a("weight_decay", "Weight Decay", kind="float", default=0.0, advanced=True),
        _a("init_lr_frac", "Init LR Fraction", kind="float", default=0.05, advanced=True),
        _a("eval_every", "Eval Every", kind="int", default=60, min=-1),
        _a("eval_examples", "Eval Examples", kind="int", default=400, min=1, advanced=True),
        _a("save_every", "Save Every", kind="int", default=60, min=-1),
    ],
    requires_cuda=True,
    gpu_heavy=True,
))

_register(CommandSpec(
    id="chat_cli",
    label="Chat CLI Prompt",
    category="chat",
    description="Run a one-shot scripts.chat_cli prompt.",
    module="scripts.chat_cli",
    args=[
        _a("source", "Source", kind="select", default="sft", choices=["sft", "rl"], flag="-i"),
        _a("model_tag", "Model Tag", default="", flag="-g"),
        _a("step", "Step", kind="int", default=-1, min=-1, flag="-s"),
        _a("prompt", "Prompt", default="", flag="-p", required=True),
        _a("temperature", "Temperature", kind="float", default=0.6, min=0, max=2, flag="-t"),
        _a("top_k", "Top-K", kind="int", default=50, min=0, max=200, flag="-k"),
        _a("device_type", "Device Type", kind="select", default="", choices=["", "cuda", "cpu", "mps"]),
    ],
    gpu_heavy=True,
))

_register(CommandSpec(
    id="report",
    label="Training Report",
    category="reports",
    description="Generate or reset nanochat training reports.",
    module="nanochat.report",
    args=[
        ArgSpec("command", "Command", kind="select", default="generate", choices=["generate", "reset"], positional=True),
    ],
))

for recipe_id, script, label in [
    ("recipe_runcpu", "runs/runcpu.sh", "CPU Recipe"),
    ("recipe_speedrun", "runs/speedrun.sh", "Speedrun Recipe"),
    ("recipe_scaling_laws", "runs/scaling_laws.sh", "Scaling Laws Recipe"),
    ("recipe_miniseries", "runs/miniseries.sh", "Miniseries Recipe"),
]:
    _register(CommandSpec(
        id=recipe_id,
        label=label,
        category="recipes",
        description=f"Launch {script}. On native Windows this requires bash/WSL.",
        script=script,
        recipe=True,
        gpu_heavy=recipe_id != "recipe_runcpu",
    ))


def capabilities() -> List[Dict[str, Any]]:
    return [COMMANDS[key].public() for key in sorted(COMMANDS)]


def get_command(command_id: str) -> CommandSpec:
    try:
        return COMMANDS[command_id]
    except KeyError as exc:
        raise CommandValidationError(f"Unknown command_id: {command_id}") from exc


def _clean_empty(value: Any) -> Any:
    return None if value == "" else value


def coerce_arg(spec: ArgSpec, raw: Any) -> Any:
    raw = spec.default if raw is None else raw
    raw = _clean_empty(raw)
    if raw is None:
        if spec.required:
            raise CommandValidationError(f"{spec.name} is required")
        return None
    if spec.kind == "bool":
        value = raw if isinstance(raw, bool) else str(raw).lower() in {"1", "true", "yes", "on"}
    elif spec.kind == "int":
        value = int(raw)
    elif spec.kind == "float":
        value = float(raw)
    elif spec.kind in {"str", "select"}:
        value = str(raw)
    else:
        raise CommandValidationError(f"Unsupported argument kind for {spec.name}: {spec.kind}")

    if spec.choices is not None and value not in spec.choices:
        raise CommandValidationError(f"{spec.name} must be one of {spec.choices}")
    if spec.min is not None and isinstance(value, (int, float)) and value < spec.min:
        raise CommandValidationError(f"{spec.name} must be >= {spec.min}")
    if spec.max is not None and isinstance(value, (int, float)) and value > spec.max:
        raise CommandValidationError(f"{spec.name} must be <= {spec.max}")
    return value


def normalize_args(spec: CommandSpec, args: Dict[str, Any]) -> Dict[str, Any]:
    return {arg.name: coerce_arg(arg, args.get(arg.name)) for arg in spec.args}


def build_argv(command_id: str, args: Dict[str, Any]) -> List[str]:
    spec = get_command(command_id)
    values = normalize_args(spec, args or {})
    if spec.module:
        argv = [sys.executable, "-m", spec.module]
    elif spec.script:
        argv = ["bash", spec.script]
    else:
        raise CommandValidationError(f"{command_id} has no executable target")

    for arg in spec.args:
        value = values[arg.name]
        if value is None:
            continue
        if arg.kind == "bool":
            if value:
                argv.append(arg.flag or f"--{arg.name.replace('_', '-')}")
            continue
        if not arg.include_if_default and value == arg.default and not arg.required:
            continue
        # -1 is the "inherit/auto/disable" sentinel ONLY for args whose own default is -1
        # (e.g. model_step, num_iterations). For args like eval_every (default 200/250) a
        # user-supplied -1 means "disable" and MUST be passed through to upstream.
        if (arg.kind in {"int", "float"} and value == -1 and arg.default == -1
                and not arg.include_if_default):
            continue
        if arg.positional:
            argv.append(str(value))
        else:
            argv.append(arg.flag or f"--{arg.name.replace('_', '-')}")
            argv.append(str(value))
    return argv


def command_preview(command_id: str, args: Dict[str, Any]) -> str:
    argv = build_argv(command_id, args)
    return " ".join(_quote(part) for part in argv)


def _quote(part: str) -> str:
    if not part or any(ch.isspace() for ch in part):
        return '"' + part.replace('"', '\\"') + '"'
    return part


def command_groups() -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for command in COMMANDS.values():
        groups.setdefault(command.category, []).append(command.id)
    return {key: sorted(value) for key, value in sorted(groups.items())}


def gpu_heavy_commands() -> Iterable[str]:
    return (cmd.id for cmd in COMMANDS.values() if cmd.gpu_heavy)
