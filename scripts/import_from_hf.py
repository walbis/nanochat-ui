"""
Download a pretrained nanochat (PyTorch) checkpoint from HuggingFace.

The HF repos (e.g. nanochat-students/base-d20) host checkpoints in nanochat's
native format (model_*.pt + meta_*.json + tokenizer.pkl + token_bytes.pt),
so no conversion is needed — files are placed straight into the local
checkpoint/tokenizer directories that the rest of nanochat expects.

Usage:
    python -m scripts.import_from_hf                                  # default: nanochat-students/base-d20
    python -m scripts.import_from_hf --repo nanochat-students/base-d20
    python -m scripts.import_from_hf --force                          # overwrite existing tokenizer
"""

import argparse
import hashlib
import json
import os
import re
import shutil

from huggingface_hub import hf_hub_download, list_repo_files

from nanochat.common import get_base_dir


def resolve_files(repo):
    """Find model, meta, and tokenizer files in the HF repo (largest step wins)."""
    files = list_repo_files(repo)
    candidates = []
    for filename in files:
        m = re.match(r"model_(\d+)\.pt$", filename)
        if m:
            candidates.append((int(m.group(1)), filename))
    if not candidates:
        raise SystemExit(f"No model_*.pt found in {repo}. Files: {files}")
    candidates.sort()
    step, model_file = candidates[-1]
    meta_file = f"meta_{step:06d}.json"
    if meta_file not in files:
        # pick the meta whose step matches the chosen model, else the largest-step
        # meta (numeric, not list order, so a multi-meta repo can't mismatch)
        metas = []
        for f in files:
            mm = re.match(r"meta_(\d+)\.json$", f)
            if mm:
                metas.append((int(mm.group(1)), f))
        if not metas:
            raise SystemExit(f"No meta_*.json found in {repo}. Files: {files}")
        metas.sort()
        match = [f for s, f in metas if s == step]
        meta_file = match[0] if match else metas[-1][1]
    has_tokenizer = "tokenizer.pkl" in files and "token_bytes.pt" in files
    return model_file, meta_file, step, has_tokenizer


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install_tokenizer(repo, force=False):
    base_dir = get_base_dir()
    tok_dir = os.path.join(base_dir, "tokenizer")
    pkl_dst = os.path.join(tok_dir, "tokenizer.pkl")
    tb_dst = os.path.join(tok_dir, "token_bytes.pt")

    print("Downloading tokenizer from HuggingFace...")
    pkl_src = hf_hub_download(repo, "tokenizer.pkl")
    tb_src = hf_hub_download(repo, "token_bytes.pt")

    if os.path.exists(pkl_dst) and not force:
        # NEVER silently keep or overwrite a tokenizer that doesn't match the
        # imported model. A mismatch means correct-looking loads but garbage output.
        # If the local tokenizer is identical, keep it; otherwise require the user
        # to opt in via --force / "Overwrite tokenizer".
        same_pkl = _sha256(pkl_src) == _sha256(pkl_dst)
        same_token_bytes = os.path.exists(tb_dst) and _sha256(tb_src) == _sha256(tb_dst)
        if same_pkl and same_token_bytes:
            print(f"Local tokenizer already matches the imported model; keeping {tok_dir}")
            return
        raise SystemExit(
            "A different local tokenizer already exists. Import stopped to avoid "
            "silently breaking locally trained checkpoints. Re-run with --force "
            "or tick 'Overwrite tokenizer' in the UI if you want to replace the "
            "local tokenizer with the imported model's tokenizer."
        )

    os.makedirs(tok_dir, exist_ok=True)
    shutil.copy2(pkl_src, pkl_dst)
    shutil.copy2(tb_src, tb_dst)
    print(f"Installed tokenizer to {tok_dir}")


def main():
    parser = argparse.ArgumentParser(description="Import a pretrained nanochat checkpoint from HuggingFace")
    parser.add_argument("--repo", type=str, default="nanochat-students/base-d20")
    parser.add_argument("--force", action="store_true", help="overwrite existing tokenizer")
    args = parser.parse_args()

    print(f"Inspecting {args.repo}...")
    model_file, meta_file, step, has_tokenizer = resolve_files(args.repo)

    print(f"Downloading {meta_file}...")
    meta_src = hf_hub_download(args.repo, meta_file)
    with open(meta_src, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model_config = meta.get("model_config", {})
    depth = model_config.get("n_layer")
    if depth is None:
        raise SystemExit(f"meta file has no model_config.n_layer: {meta_src}")
    print(f"Checkpoint: d{depth} | step {step} | n_embd {model_config.get('n_embd')} "
          f"| seq len {model_config.get('sequence_len')}")

    base_dir = get_base_dir()
    ckpt_dir = os.path.join(base_dir, "base_checkpoints", f"d{depth}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_dst = os.path.join(ckpt_dir, f"model_{step:06d}.pt")
    meta_dst = os.path.join(ckpt_dir, f"meta_{step:06d}.json")

    if os.path.exists(model_dst):
        print(f"Model already exists at {model_dst}, skipping download")
    else:
        print(f"Downloading {model_file} (this can be 1+ GB, may take a few minutes)...")
        model_src = hf_hub_download(args.repo, model_file)
        size_gb = os.path.getsize(model_src) / 1e9
        print(f"Downloaded ({size_gb:.2f} GB), installing to {ckpt_dir}...")
        shutil.copy2(model_src, model_dst)

    shutil.copy2(meta_src, meta_dst)

    if has_tokenizer:
        install_tokenizer(args.repo, force=args.force)
    else:
        print("WARNING: repo has no tokenizer.pkl/token_bytes.pt — the local tokenizer must match this model!")

    print(f"Import complete: d{depth} step {step} -> {ckpt_dir}")
    print("Note: a base (pretrained-only) model is not chat-tuned; responses will be rough until you run SFT.")


if __name__ == "__main__":
    main()
