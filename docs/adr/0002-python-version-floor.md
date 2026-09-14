# ADR-0002: Python lower bound set by the GPU platform, not the local machine

- Status: accepted
- Date: 2026-09-07

## Context

Training runs on free GPU platforms (Kaggle, Colab) whose Python version we do not control:
as of 2026-09 both ship 3.12 (Kaggle: `Kaggle/docker-python` image paths under `python3.12`;
Colab: see `docs/verification/wheels.md`). The package is installed into
that interpreter from the repository,
so a `requires-python` bound above the platform's version would make the package uninstallable
exactly where training happens. Locally, development uses CPython 3.14. Stable PyTorch wheels for
cp314 exist for CPU builds; CUDA builds for cp314 are not on the official index.

## Decision

- `requires-python = ">=3.12"` with **no upper bound**. The floor equals the GPU platform's Python.
- Local development pins 3.14 (`.python-version`), the lockfile is resolved on 3.14.
- Code imported by training notebooks must not use syntax or standard-library features newer
  than 3.12. Enforced by `ruff target-version = "py312"`, `mypy python_version = "3.12"`,
  `ty python-version = "3.12"` and a CI matrix on **3.12 and 3.14**.
- CI installs PyTorch from the CPU-only index (`tool.uv.index` in `pyproject.toml`); GPU platforms
  keep their preinstalled CUDA build because `pip` ignores uv-specific index configuration.
- `tool.uv.required-environments` forces the lock to contain wheels for Linux aarch64, Linux
  x86_64 and macOS arm64, so a missing wheel fails at lock time, not at deploy time.

## Consequences

- Two Python versions are exercised on every change; a 3.13/3.14-only feature fails CI on 3.12.
- Type checkers run with 3.12 semantics locally even on 3.14; typeshed differences may surface as
  false positives only for features unavailable on 3.12, which is the intended signal.
- The Kaggle smoke test is a manual step outside CI.

## Fallback path

If the GPU platform's Python drops below 3.12, or a required wheel disappears for 3.14:

1. Lower `requires-python` to the platform version and add it to the CI matrix; audit `ruff`/`mypy`
   targets accordingly. Nothing in the architecture depends on a specific minor version.
2. Move the local pin down (3.13 or 3.12) and re-lock. The lockfile is the only artefact that
   changes.
3. If CUDA wheels are needed for a version the official index does not serve, the training notebook
   uses the platform's preinstalled `torch` and the package declares `torch` as an optional extra
   (already the case: `emblema[ml]`), so a version mismatch is a warning, not an install failure.

## Alternatives considered

- **Pin to the local version (3.14) as the floor** — rejected: uninstallable on Colab/Kaggle.
- **Develop on 3.12 everywhere** — rejected in plan review: the modern stack is a deliberate
  choice, and the risk is mitigated by the CI matrix rather than by lowering the local version.
- **Upper bound (`<3.15`)** — rejected: blocks installation on a future platform interpreter for no
  gain; incompatibilities are caught by the matrix, not prevented by a bound.
