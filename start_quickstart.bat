@echo off
rem NanoChat CUDA Quickstart launcher
cd /d "%~dp0"
uv run python -m scripts.quickstart %*
