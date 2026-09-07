# Emblema

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
