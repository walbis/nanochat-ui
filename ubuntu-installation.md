# Ubuntu Installation (NVIDIA GPU)

Step-by-step commands to run nanochat-ui on a fresh Ubuntu machine with an NVIDIA GPU.

> The main `QUICKSTART_README.md` targets Windows. This guide is the Linux/Ubuntu
> equivalent. The codebase is fully Linux-compatible: all OS-specific code is guarded,
> and on Linux `torch` already bundles `triton`, so `torch.compile` works out of the box
> (the `triton-windows` step from the Windows guide is **not** needed here).

## Requirements

- Ubuntu (x86_64 or aarch64)
- An NVIDIA GPU, **Turing (RTX 20-series / sm_75) or newer** recommended
  (the `cu128` PyTorch wheels do not ship kernels for very old Pascal/GTX 10-series cards)
- NVIDIA driver capable of CUDA 12.8 — driver **≥ 525**, ideally **≥ 570**

## 1. Update the system and install base tools

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl
```

## 2. Verify the NVIDIA driver

```bash
nvidia-smi
```

- **Works** (shows GPU name + CUDA version) → go to step 3.
- **`command not found` / error** → install the driver and reboot:

```bash
sudo ubuntu-drivers devices      # shows the recommended driver
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot, run `nvidia-smi` again to confirm.

## 3. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env      # activate uv in the current shell
uv --version                     # verify
```

## 4. Clone the repository

```bash
git clone https://github.com/walbis/nanochat-ui.git
cd nanochat-ui
```

## 5. Install dependencies (GPU)

```bash
uv sync --extra gpu
```

> ⚠️ Do **not** run the `uv pip install "triton-windows..."` line from the Windows guide —
> it is Windows-only. On Linux `torch` already pulls in `triton`.

## 6. Verify that PyTorch sees the GPU

```bash
uv run --extra gpu python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see `True` and your GPU name, e.g. `2.9.1+cu128 True NVIDIA GeForce RTX ...`.

## 7. Start the quickstart server

```bash
uv run --extra gpu python -m scripts.quickstart --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.
If port 8000 is taken, use `--port 8001`.

## Remote access (optional)

If the Ubuntu box is headless, open an SSH tunnel from your own machine:

```bash
ssh -L 8000:127.0.0.1:8000 user@ubuntu-pc-ip
```

Then browse to `http://127.0.0.1:8000` locally.

## Notes

- **CUDA out of memory** → lower the **Batch Size** in the train/SFT panel (1 works everywhere).
- **Job stop on Linux**: stopping a job sends `SIGTERM` to the direct child process. The
  single-GPU quickstart jobs run as one process, so this is fine; multi-GPU `torchrun`
  jobs may leave orphan processes.
- **Recipes** (`runs/*.sh`) run natively on Linux (no WSL needed).
- Artifacts live in `~/.cache/nanochat` (override with `NANOCHAT_BASE_DIR`).
