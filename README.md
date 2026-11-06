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

Emblema is a research platform for label-efficient representation learning on heterogeneous
sensor streams. One permutation-invariant, variable-channel transformer is pretrained without
labels on corpora that differ in channel count and sampling regime. It is then adapted to small
labelled tasks. The aim is one backbone for irregular, multichannel sensor streams, judged on
intensive-care stays and satellite telemetry against tuned classical baselines and a published
network. The platform measures, with paired confidence intervals, whether and when that
pretraining pays off. It can serve whichever candidate a finished comparison measured.

## The name

In a Roman mosaic the *emblema* is the central panel. It was laid in a workshop from the finest
tesserae, then carried to the site and set into a floor of coarser tiles. The pretrained core
here is made once, over many corpora, and carried into new arrangements of sensors. Each
measurement is a *tessera*: a self-contained token of channel, time and value.

![A mosaic emblema with Pegasus inside a guilloche border, second century AD, Archaeological Museum of Córdoba](docs/images/emblema.jpg)

*Emblema with Pegasus, second century AD, Archaeological Museum of Córdoba. Photograph by
[Carole Raddato](https://commons.wikimedia.org/wiki/File:Mosaic_emblema_with_Pegasus,_the_immortal_winged_horse_which_sprang_forth_from_the_neck_of_Medusa_when_she_was_beheaded_by_the_hero_Perseus,_2nd_century_AD,_Archaeological_Museum_of_C%C3%B3rdoba,_Spain_(24783223879).jpg),
[CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/), via Wikimedia Commons; scaled down.*

## How it works

**The problem.** Industrial and scientific sensor data rarely looks like a neat table. A jet
engine reports 21 sensors every cycle; an intensive-care patient has a heart rate every few
minutes, a blood test twice a day and some values never measured; a satellite sends telemetry in
bursts between passes. Most time-series models expect a fixed set of channels on a regular
clock, so each new dataset is resampled, filled in and given a model of its own, trained from
its own labels. Labels are the expensive part.

**The idea.** Train one model on many unlabelled sensor datasets first, then adapt it to a new
task with only a few labels. This is what pretrained language models do for text. For it to work
across datasets, the model must not care how many sensors a dataset has or how often they are
read.

**Measurements as a bag of tokens.** Each measurement becomes one token: which sensor, its value
scaled per sensor, when it was taken within the window, and how long since that sensor's
previous reading. A window is the set of its tokens. Three sensors or thirty, read every second
or twice a day, it is the same kind of input, with nothing interpolated. Fixed facts, such as a
patient's age, become tokens without a time.

![Sensor readings of an intensive-care stay and a turbofan engine, and the same windows as sets of tokens](docs/images/token-view.png)

*Real data, tokenised by this repository's code. Top: 48 hours of one intensive-care stay, where
heart rate is read 189 times and glucose 5. Bottom: the last 50 cycles of one turbofan engine,
every sensor every cycle. On the right, the same kind of input for each.*

**One encoder for any sensor set.** A transformer reads the whole set at once, in any order.
Each sensor has a learned identity vector, so a sensor it has never seen gets a new vector
learned from a little data while everything else carries over.

**Learning without labels.** During pretraining, parts of each window are hidden: whole sensors,
stretches of one sensor's time, or single readings. The model predicts the hidden values from
what remains, and counts as having learnt something only where it beats a simple method for the
same gap, such as interpolation or a regression on the other sensors.

**Adapting to a task.** The encoder is then used on a small labelled task: frozen under a small
output layer, lightly adjusted through a few extra weights (LoRA), fully fine-tuned, or trained
from scratch as the control. Each runs at several label budgets, tuned by the same declared
procedure as the classical baselines.

**Keeping the comparison honest.**

- **Rules first.** The comparisons, the size of an improvement that counts and what is reported
  if the answer is no are committed before the runs they judge
  ([`docs/preregistration.md`](docs/preregistration.md)).
- **Engines, not windows.** Intervals are computed over engines or patients, the independent
  units, not over overlapping windows.
- **Strong baselines.** Gradient-boosted trees on window statistics, frequency features and
  MiniRocket compete in the same grid, tuned by the same declared procedure as the networks.
- **A positive control.** A generated dataset with known shared structure checks that the
  pipeline finds structure when it is there, so a negative result on real data means something.

## Status

Preliminary, on validation data; each task's test data is used once, at the end.

**The research question: does pretraining without labels pay off at few labels?**

- Not shown. On turbofan remaining life (NASA C-MAPSS) and on death in hospital (PhysioNet
  2012) no pretrained arm beats the same network trained from scratch beyond the noise of the
  comparison; a gain of 0.016 in area at 50 labelled stays awaits fresh pretrainings. A linear
  probe reads 0.06 to 0.09 above the untrained encoder at 500 stays, still below the trees.
- Ruled out: more pretraining data (up to about 10⁹ values), a five times larger model,
  continuing on the task's own domain, the adaptation recipe and six variants of the pretext.

**The product: one backbone for irregular, multichannel sensor streams.**

- On intensive-care stays a small network over channel-by-time cells is level with
  gradient-boosted trees at 200 stays and ahead at 50, 0.008 below the tuned trees on the full
  set; a published network (STraTS) reads 0.827 on the same stays, ours from scratch 0.818.
- A new sensor set joins without labels: fitting only the channel rows brings the pretext to
  within 0.006 of a backbone that read the corpus, and carries part of a task (+0.031 on sepsis
  at 4,000 patients).
- On satellite telemetry the features find no events without labels. With every label the best
  readout is a linear direction over the frozen features, 0.26 in event-weighted PR-AUC against
  0.15 for trees, with or without pretraining; windows of 6 to 72 hours instead of one read
  lower, not higher. Each channel's departure from its own recent hours, with no model and no
  label, reads level with the best classical bar on the first mission and 0.885 against 0.627
  on the second, finding as many events as the taught direction on both; taught by trees, those
  departures find 50 to 54 of 61 events at 1 % false alarms on the first mission.

**Next:** the shape of a pretext for telemetry, now that the missed events are read per
channel against its own history; confirmation at few labels on fresh pretrainings.

Numbers, intervals and limitations: [`docs/findings.md`](docs/findings.md).

![Validation RMSE of the network arms and classical baselines by number of labelled windows, and each pretrained arm's reduction against training from scratch](docs/verification/figures/label-efficiency-curve.png)

*Error by number of labelled windows on C-MAPSS FD001, lower is better. Top: the ways of using the
encoder. Middle: classical baselines. Bottom: each pretrained arm's reduction against training
from scratch, with its 95 % interval; the grey band is the practical floor. Validation, not test.*

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+, the Python of the free GPU platforms.

```sh
uv sync --all-extras
uv run pytest                    # unit, domain and application tests
uv run lint-imports              # architecture contracts
```

### Local stack

Postgres, Garage, MLflow, RabbitMQ and the API run in Docker:

```sh
cp env.example .env
docker compose up -d --wait      # builds the images and migrates the database
bash scripts/smoke.sh
uv run pytest -m integration     # adapter contracts against the stack
```

The API listens on `http://127.0.0.1:8000` (`/docs`, `/openapi.json`, `/health`, `/ready`,
`/metrics`).

The merge gate is the same suite inside the image:

```sh
docker compose run --rm tests pytest -o addopts="-ra --strict-markers" --cov
```

On macOS XGBoost's OpenMP runtime cannot share a process with torch's, so its fits run apart:
`uv run pytest --without-torch tests/evaluation/adapters/xgboost`.

### Workflow

**Data.** Download from the sources of record, count, and publish a corpus as a block of windows
beside a manifest:

```sh
uv run scripts/fetch_corpora.py
uv run python -m emblema.entrypoints.cli.publish_corpus --corpus cmapss --window 50 --stride 5
```

**Pretraining.** Order a run here, run it on any machine with a GPU, accept the result against
the order ([ADR-0024](docs/adr/0024-handing-a-run-to-another-machine.md)); every parameter is in
the experiment file:

```sh
uv run python -m emblema.entrypoints.cli.pretrain order \
    --experiment experiments/<file>.toml --corpus <manifest key> <checksum> --run <name>
uv run python -m emblema.entrypoints.cli.pretrain run --order <key> <checksum>
uv run python -m emblema.entrypoints.cli.pretrain accept --result <key> <checksum> \
    --track http://127.0.0.1:5000
```

**Evaluation.** A campaign is declared from a file under `campaigns/`; its cells run on the
workers or, where no broker reaches, from an order in the artifact store:

```sh
uv run python -m emblema.entrypoints.cli.campaign define-task --corpus <key> <checksum> --task turbofan-fd001
uv run python -m emblema.entrypoints.cli.campaign define --file campaigns/baselines-fd001.toml --task <task id>

# either through the queue
docker compose --profile workers up -d worker-ml worker-general
uv run python -m emblema.entrypoints.cli.campaign advance --campaign <id>

# or through an order, run anywhere and accepted back
uv run python -m emblema.entrypoints.cli.campaign order --campaign <id> --pool ml [--budget 200]
uv run python -m emblema.entrypoints.cli.campaign_run --order <key> <checksum>
uv run python -m emblema.entrypoints.cli.campaign accept --result <key> <checksum>
```

**Serving.** Only an artifact a finished campaign kept can be promoted, by checksum
([ADR-0037](docs/adr/0037-promoting-what-a-campaign-kept.md)). A promoted model answers over HTTP
with raw readings by channel name
([ADR-0042](docs/adr/0042-the-prediction-service.md)):

```sh
uv run python -m emblema.entrypoints.cli.serving promote --checksum <checksum>
curl -s -X POST http://127.0.0.1:8000/served-models/<model id>/predictions \
  -H 'Content-Type: application/json' \
  -d '{"windows": [{"start": 1, "length": 50,
                    "observations": [{"channel": "T24", "time": 1, "value": 641.8}]}]}'
```

The API's own page at `/docs` covers the rest: promotion and withdrawal under a token, embeddings,
campaigns, and the refusals when a request cannot start in time.

## Documentation

| Where | What |
| --- | --- |
| [`docs/findings.md`](docs/findings.md) | Results, setup and limitations on one page |
| [`docs/preregistration.md`](docs/preregistration.md) | Rules and configuration in force, with a register of every change |
| [`docs/adr/`](docs/adr/README.md) | Architecture decision records |
| [`docs/verification/`](docs/verification/README.md) | Dated measurement notes: the lab notebook |
