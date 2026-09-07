# Emblema

[![CI](https://github.com/lstasiak/emblema/actions/workflows/ci.yml/badge.svg)](https://github.com/lstasiak/emblema/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lstasiak/emblema/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/lstasiak/emblema/blob/python-coverage-comment-action-data/htmlcov/index.html)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Architecture: import-linter](https://img.shields.io/badge/architecture-import--linter-blue)](https://import-linter.readthedocs.io/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **Work in progress.** The project is under active development. This README will be completed once the implementation is finished.

Emblema is a research platform for label-efficient representation learning on heterogeneous sensor streams. A single permutation-invariant, variable-channel transformer encoder is pretrained without labels on measurement corpora that differ in channel count and sampling regime, then transferred to new, small labelled tasks. The platform measures, with confidence intervals and strong classical baselines, whether and when that pretraining actually pays off, and serves the winning candidate through an API.

## Development

Requires [uv](https://docs.astral.sh/uv/). One command builds the environment, a second runs the tests:

```sh
uv sync --all-extras
uv run pytest
```

Supported Python: 3.12 and newer. The floor is set by the free GPU platforms the training code runs on.

Architecture rules (framework-free core, inward-pointing layers, bounded contexts that share only published contracts) are [import-linter](https://import-linter.readthedocs.io/) contracts in `pyproject.toml`. `uv run lint-imports` checks them; CI enforces them alongside lint, types and coverage.
