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

### Local environment

Postgres, an S3-compatible artifact store ([Garage](https://garagehq.deuxfleurs.fr/)) and MLflow run in Docker. One command brings the stack up, a second checks it, a third creates the application's tables, a fourth runs the adapter contracts against it:

```sh
cp env.example .env
docker compose up -d --wait
bash scripts/smoke.sh
uv run alembic upgrade head
uv run pytest -m integration
```

The metadata database holds one schema per bounded context and one [Alembic](https://alembic.sqlalchemy.org/) migration tree for all of them (`migrations/`); `uv run alembic check` reports any table the model has and the migrations do not.

The artifact store speaks S3 to Garage locally and to a Cloudflare R2 bucket that GPU platforms can reach. The same contract tests run against the remote bucket by configuration alone: `uv run --env-file .env.r2 pytest -m integration` (variables in `env.example`). `docker compose down -v` removes the stack and its data.

### Data

The raw corpora come from their sources of record into `data/raw/` (not tracked) and are priced before any tokeniser exists: units, windows and tokens are counted from the files, and the GPU-hour budget per compute tier is derived from `scripts/corpus_budget.toml` and the tier profiles in `src/emblema/config/compute_tiers.toml`. The satellite telemetry ships as pandas pickles, so reading it needs the `corpora` extra, which `--all-extras` installs.

```sh
uv run scripts/fetch_corpora.py            # about 12 GB; a re-run skips what is already there
uv run scripts/corpus_facts.py             # counts, printed as TOML to paste into the budget file
uv run scripts/corpus_budget_report.py     # the budget tables
```

A corpus is published once, as a block of token windows beside a manifest in the artifact store;
the manifest's reference is what a training run is pointed at:

```sh
uv run python -m emblema.entrypoints.cli.publish_corpus --corpus cmapss --window 50 --stride 5
```

### Pretraining

A run is made in three steps that may happen on two machines. The first registers the backbone
and places an order in the artifact store; the second fulfils the order wherever there is a GPU,
here or in a notebook that installed the package at the commit the order names; the third takes
the result back, holds it to the order, records the run in MLflow and registers the weights.

```sh
uv run python -m emblema.entrypoints.cli.pretrain order \
    --experiment experiments/control-a-s.toml --corpus <manifest key> <manifest checksum> --run first
uv run python -m emblema.entrypoints.cli.pretrain run --order <order key> <order checksum>
uv run python -m emblema.entrypoints.cli.pretrain accept \
    --result <result key> <result checksum> --track http://127.0.0.1:5000
```

Every parameter of a run lives in its experiment file under `experiments/`; the shape of the model
comes from the compute tier the file declares. The commit the order and the result carry is read
from the installed package or the working tree: an order is placed only from a committed tree
unless `--commit` states the revision, a run on other code than ordered stops before it trains,
and a result made with other code, over other data or under another configuration is refused.
Accepting records the replayed run against the MLflow server `--track` names, so the flag is
required there.
