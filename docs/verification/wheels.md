# Stack wheel availability and GPU-platform Python

Purpose: confirm before building anything that PyTorch and ONNX Runtime ship wheels for every
interpreter and platform the project targets, and that `requires-python` does not exceed the Python
of the free GPU platforms (ADR-0002).

## 2026-09-07 — lock resolution on uv 0.11.13

Method: `uv lock` with `tool.uv.required-environments` = Linux aarch64, Linux x86_64, macOS arm64;
`requires-python = ">=3.12"`; torch from the CPU-only index. Wheel tags read from `uv.lock`.

| Package | Version | cp312 | cp314 | linux aarch64 | linux x86_64 | macOS arm64 | win amd64 |
|---------|---------|:-----:|:-----:|:-------------:|:------------:|:-----------:|:---------:|
| torch (`+cpu`, download.pytorch.org/whl/cpu) | 2.14.0 | yes | yes | manylinux_2_28 | manylinux_2_28 | macosx_14_0 (no `+cpu` suffix) | yes |
| onnxruntime (PyPI) | 1.29.0 | yes | yes | manylinux_2_28 | manylinux_2_28 | macosx_14_0 | yes |
| numpy | 2.5.3 | yes | yes | yes | yes | yes | yes |
| pydantic-core | 2.46.5 | yes | yes | yes | yes | yes | yes |

Result: **pass**. Lock resolves in one pass; `uv sync --all-extras` on Windows / CPython 3.14.5
installs torch 2.14.0+cpu and onnxruntime 1.29.0; `tests/ml` (hard imports, CPU tensor op,
`CPUExecutionProvider`) pass locally.

Constraints observed:

- torch macOS wheels require macOS 14 (`macosx_14_0_arm64`). The M1 development machine must run
  macOS 14 or newer.
- CUDA builds for cp314 are not verified here and are not needed: GPU platforms use their own
  preinstalled torch.

## 2026-09-07 — GPU platform Python versions

| Platform | Python | Source |
|----------|--------|--------|
| Kaggle notebooks | 3.12 | `Kaggle/docker-python` `Dockerfile.tmpl` on `main`: `PACKAGE_PATH=/usr/local/lib/python3.12/dist-packages`, `sitecustomize.py` installed under `/usr/lib/python3.12` |
| Google Colab | 3.12 | the runtime's own `python --version`, state as of 2026-09 |

Result: **floor `>=3.12` holds** for both platforms. Both remain to be confirmed from inside a live
notebook.
