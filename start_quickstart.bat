@echo off
rem NanoChat CUDA Quickstart launcher
cd /d "%~dp0"
uv run --extra gpu python -m scripts.quickstart %*
