"""Can each channel's summaries of the hidden part of a window be answered without a model?

The audit a pretext target passes before any accelerator trains on it. The target is, per window
and per channel with a hidden token, the summaries of that channel's hidden values: their mean
(level), their minimum and maximum (extremes) and their slope over time (trend). Each is answered
on the validation windows by answers that learn nothing from them:

- ``visible``: the same summary of the channel's visible tokens, where it keeps any;
- ``interpolation``: the summary of a line through the channel's visible tokens at the hidden
  instants;
- ``combined``: the summary of the regression on the channel's own line and the other channels;
- ``schedule``: a ridge map from when and how often the channel was hidden and seen, no value;
- ``stacked``: a ridge map from all of the above, the strongest linear answer without a model.

The two ridge maps are fitted per channel and summary on one half of the training windows, the
regression the others summarise on the other half, so neither is scored where it was fitted. A
backbone's own reconstruction is summarised the same way (``model``), and each token is scored as
well (``token``), the objective as it stands. The masks are the backbone's own, or those of an
experiment file's ``[masking]`` (``--masks``), drawn from fixed seeds.

Scores are kept as sums per corpus, masks, channel, summary, part and answer, so the R² of any
grouping is computed from the CSV without the corpora at hand; the tables are rendered from it.

    uv run --env-file .env.r2 scripts/summary_target_audit.py --backbone <id>
        [--masks experiments/<file>.toml] [--no-model]
    uv run scripts/summary_target_audit.py --report-only data/report/summary-target/<backbone>
"""

import argparse
import csv
import sys
import time
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.config.settings import Settings
from emblema.entrypoints.configured import configured_store
from emblema.entrypoints.source_revision import SourceRevision
from emblema.pretraining.adapters.diagnostics.linear_interpolation_baseline import (
    LinearInterpolationBaseline,
)
from emblema.pretraining.adapters.diagnostics.own_and_cross_channel_ridge_baseline import (
    OwnAndCrossChannelRidgeBaseline,
)
from emblema.pretraining.adapters.diagnostics.window_arrays import WindowArrays
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.adapters.training.devices import available_device
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.shared.adapters.loaders.window_loader import WindowLoader
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.tokens import TokenWindow
from scripts.corpus_saturation_report import BLOCKS
from scripts.masked_reconstruction_report import DEVICES, device_available
from scripts.pretext_triviality_report import (
    BATCH_SIZE,
    FIT_SEED,
    TRAINING_WINDOWS,
    VALIDATION_SEED,
    VALIDATION_WINDOWS,
    CorpusWindows,
    _backbone_id,
    _registry,
    corpora_of,
)
from scripts.reporting import machine, table

WORKSPACE = REPO_ROOT / "data" / "report" / "summary-target"
SUMMARIES = ("mean", "minimum", "maximum", "slope")
FAMILY = {"mean": "level", "minimum": "extremes", "maximum": "extremes", "slope": "trend"}
TOKEN = "token"
PARTIAL, WHOLE = "partial", "whole"
VISIBLE, INTERPOLATION, COMBINED = "visible", "interpolation", "combined"
SCHEDULE, STACKED, MODEL = "schedule", "stacked", "model"
# The answers summarised from values at the hidden instants, and the ones fitted on summaries.
SUMMARISED = (INTERPOLATION, COMBINED, MODEL)
ANSWERS = (VISIBLE, INTERPOLATION, COMBINED, SCHEDULE, STACKED, MODEL)
# The ridge maps' penalty on standardised features, and the samples a channel needs before one is
# fitted; below that it answers the training mean.
PENALTY = 1.0
LEAST_SAMPLES = 30
# A channel enters a median when it has this many scored samples of a summary and part.
LEAST_SCORED = 30
# Channels the stays measure at most this many times a window on average are read apart: the
# laboratory values a loss over tokens weighs least.
SPARSE_TOKENS = 3.0
SETTINGS, SCORES = "settings.csv", "scores.csv"
SCORE_FIELDS = (
    "corpus",
    "masks",
    "channel_id",
    "summary",
    "part",
    "answer",
    "count",
    "total",
    "squares",
    "errors",
    "tokens_per_window",
)


# ------------------------------------------------------------------------------------------------
# What is answered
# ------------------------------------------------------------------------------------------------


def summarised(times: NDArray[np.float64], values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mean, minimum, maximum and least-squares slope over time; no slope from one instant."""
    slope = np.nan
    if np.unique(times).size >= 2:
        centred = times - times.mean()
        slope = float(centred @ (values - values.mean()) / (centred @ centred))
    return np.array([values.mean(), values.min(), values.max(), slope])


def schedule_of(
    hidden_times: NDArray[np.float64], visible_count: int, length: float
) -> NDArray[np.float64]:
    """When and how often a channel was hidden and seen in a window, without a value."""
    scale = length if length > 0.0 else 1.0
    return np.array(
        [
            np.log1p(hidden_times.size),
            np.log1p(visible_count),
            (hidden_times.max() - hidden_times.min()) / scale,
            hidden_times.mean() / scale,
            float(visible_count == 0),
        ]
    )


@dataclass(frozen=True)
class Sample:
    """One channel of one window with a hidden token, and every answer summarised.

    Attributes:
        channel_id: The channel.
        part: ``whole`` where the channel kept no visible token, ``partial`` otherwise.
        target: The hidden values' summaries, in ``SUMMARIES`` order; NaN where undefined.
        answers: Each summarised answer's summaries, NaN where it has none.
        schedule: ``schedule_of`` the channel in the window.
        tokens: Tokens the channel holds in the window, hidden and visible.
    """

    channel_id: int
    part: str
    target: NDArray[np.float64]
    answers: Mapping[str, NDArray[np.float64]]
    schedule: NDArray[np.float64]
    tokens: int


@dataclass(frozen=True)
class TokenScore:
    """One channel's hidden tokens of one window, answered token by token."""

    channel_id: int
    part: str
    values: NDArray[np.float64]
    answers: Mapping[str, NDArray[np.float64]]


def samples_of(
    batch: TokenTensors,
    masks: TokenMasks,
    predictions: Mapping[str, NDArray[np.float64]],
    channels_apart: frozenset[int],
) -> tuple[list[Sample], list[TokenScore]]:
    """Every channel of every window that lost a token, with its summaries and its tokens."""
    arrays = WindowArrays.of(batch)
    timeless = batch.timeless.cpu().numpy()
    hidden = masks.hidden.cpu().numpy() & arrays.observed & ~timeless
    seen = arrays.observed & ~hidden & ~timeless
    samples, tokens = [], []
    for row in range(arrays.rows):
        length = float(arrays.times[row, arrays.observed[row]].max(initial=0.0))
        for channel in np.unique(arrays.channel_ids[row, hidden[row]]):
            if int(channel) in channels_apart:
                continue
            own = arrays.channel_ids[row] == channel
            lost, kept = hidden[row] & own, seen[row] & own
            times = arrays.times[row, lost]
            part = PARTIAL if kept.any() else WHOLE
            nan = np.full(len(SUMMARIES), np.nan)
            answers = {
                name: summarised(times, predicted[row, lost])
                for name, predicted in predictions.items()
            }
            answers[VISIBLE] = (
                summarised(arrays.times[row, kept], arrays.values[row, kept])
                if part == PARTIAL
                else nan
            )
            samples.append(
                Sample(
                    channel_id=int(channel),
                    part=part,
                    target=summarised(times, arrays.values[row, lost]),
                    answers=answers,
                    schedule=schedule_of(times, int(kept.sum()), length),
                    tokens=int((lost | kept).sum()),
                )
            )
            tokens.append(
                TokenScore(
                    channel_id=int(channel),
                    part=part,
                    values=arrays.values[row, lost],
                    answers={name: predicted[row, lost] for name, predicted in predictions.items()},
                )
            )
    return samples, tokens


# ------------------------------------------------------------------------------------------------
# The ridge maps
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RidgeMap:
    """A ridge map on standardised features, or the training mean where too few samples were."""

    centre: NDArray[np.float64]
    scale: NDArray[np.float64]
    weights: NDArray[np.float64]
    intercept: float

    @classmethod
    def fitted(cls, features: NDArray[np.float64], target: NDArray[np.float64]) -> "RidgeMap":
        width = features.shape[1]
        if target.size < LEAST_SAMPLES:
            mean = float(target.mean()) if target.size else 0.0
            return cls(np.zeros(width), np.ones(width), np.zeros(width), mean)
        centre = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale == 0.0] = 1.0
        standard = (features - centre) / scale
        intercept = float(target.mean())
        weights = np.linalg.solve(
            standard.T @ standard + PENALTY * np.eye(width), standard.T @ (target - intercept)
        )
        return cls(centre, scale, weights, intercept)

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return ((features - self.centre) / self.scale) @ self.weights + self.intercept


def stacked_features(samples: Sequence[Sample], summary: int) -> NDArray[np.float64]:
    """The schedule beside each trivial answer's summary, a missing one as zero and a flag."""
    columns = [np.stack([sample.schedule for sample in samples])]
    for name in (VISIBLE, INTERPOLATION, COMBINED):
        answer = np.array([sample.answers[name][summary] for sample in samples])
        missing = np.isnan(answer)
        columns += [np.where(missing, 0.0, answer)[:, None], missing[:, None].astype(np.float64)]
    return np.hstack(columns)


Key = tuple[int, str]


def grouped(samples: Iterable[Sample]) -> dict[Key, list[Sample]]:
    groups: dict[Key, list[Sample]] = {}
    for sample in samples:
        groups.setdefault((sample.channel_id, sample.part), []).append(sample)
    return groups


@dataclass(frozen=True)
class FittedMaps:
    """Per channel, part and summary: the schedule's map, the stacked map, the training mean."""

    schedule: Mapping[tuple[int, str, int], RidgeMap]
    stacked: Mapping[tuple[int, str, int], RidgeMap]
    means: Mapping[tuple[int, str, int], float]

    @classmethod
    def fitted(cls, samples: Iterable[Sample]) -> "FittedMaps":
        schedule, stacked, means = {}, {}, {}
        for (channel, part), group in grouped(samples).items():
            for summary in range(len(SUMMARIES)):
                target = np.array([sample.target[summary] for sample in group])
                kept = ~np.isnan(target)
                rows = [sample for sample, keep in zip(group, kept, strict=True) if keep]
                key = (channel, part, summary)
                means[key] = float(target[kept].mean()) if kept.any() else 0.0
                if not rows:
                    continue
                schedule[key] = RidgeMap.fitted(
                    np.stack([sample.schedule for sample in rows]), target[kept]
                )
                stacked[key] = RidgeMap.fitted(stacked_features(rows, summary), target[kept])
        return cls(schedule, stacked, means)


# ------------------------------------------------------------------------------------------------
# The scores
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Score:
    """Sums from which an R² over any set of samples is recovered."""

    count: int
    total: float
    squares: float
    errors: float

    def __add__(self, other: "Score") -> "Score":
        return Score(
            self.count + other.count,
            self.total + other.total,
            self.squares + other.squares,
            self.errors + other.errors,
        )

    @classmethod
    def of(cls, target: NDArray[np.float64], answer: NDArray[np.float64]) -> "Score":
        return cls(
            int(target.size),
            float(target.sum()),
            float(target @ target),
            float((target - answer) @ (target - answer)),
        )

    @property
    def r2(self) -> float:
        """One less the errors over the spread about the samples' own mean; NaN without spread."""
        spread = self.squares - self.total**2 / self.count if self.count else 0.0
        return 1.0 - self.errors / spread if spread > 1e-12 else np.nan


@dataclass(frozen=True)
class ScoreRow:
    corpus: str
    masks: str
    channel_id: int
    summary: str
    part: str
    answer: str
    score: Score
    tokens_per_window: float

    def record(self) -> dict[str, str]:
        return {
            "corpus": self.corpus,
            "masks": self.masks,
            "channel_id": str(self.channel_id),
            "summary": self.summary,
            "part": self.part,
            "answer": self.answer,
            "count": str(self.score.count),
            "total": repr(self.score.total),
            "squares": repr(self.score.squares),
            "errors": repr(self.score.errors),
            "tokens_per_window": repr(self.tokens_per_window),
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> "ScoreRow":
        return cls(
            corpus=record["corpus"],
            masks=record["masks"],
            channel_id=int(record["channel_id"]),
            summary=record["summary"],
            part=record["part"],
            answer=record["answer"],
            score=Score(
                int(record["count"]),
                float(record["total"]),
                float(record["squares"]),
                float(record["errors"]),
            ),
            tokens_per_window=float(record["tokens_per_window"]),
        )


def scored(
    corpus: str,
    masks: str,
    samples: Sequence[Sample],
    tokens: Sequence[TokenScore],
    maps: FittedMaps,
) -> list[ScoreRow]:
    """Every answer to every summary, and every token answer, scored per channel and part.

    An answer that has none for a sample, a slope from one visible instant, answers the training
    mean of its channel and part, so every answer is scored on the same samples.
    """
    rows = []
    for (channel, part), group in grouped(samples).items():
        mean_tokens = float(np.mean([sample.tokens for sample in group]))
        for summary, name in enumerate(SUMMARIES):
            target = np.array([sample.target[summary] for sample in group])
            kept = ~np.isnan(target)
            if not kept.any():
                continue
            members = [sample for sample, keep in zip(group, kept, strict=True) if keep]
            key = (channel, part, summary)
            fallback = maps.means.get(key, 0.0)
            answers: dict[str, NDArray[np.float64]] = {}
            for answer in (VISIBLE, *SUMMARISED):
                if answer not in members[0].answers:
                    continue
                given = np.array([sample.answers[answer][summary] for sample in members])
                answers[answer] = np.where(np.isnan(given), fallback, given)
            for answer, ridge in ((SCHEDULE, maps.schedule), (STACKED, maps.stacked)):
                features = (
                    np.stack([sample.schedule for sample in members])
                    if answer == SCHEDULE
                    else stacked_features(members, summary)
                )
                answers[answer] = (
                    ridge[key].predict(features)
                    if key in ridge
                    else np.full(len(members), fallback)
                )
            rows += [
                ScoreRow(
                    corpus,
                    masks,
                    channel,
                    name,
                    part,
                    answer,
                    Score.of(target[kept], given),
                    mean_tokens,
                )
                for answer, given in answers.items()
            ]
    per_token: dict[tuple[int, str, str], Score] = {}
    for token in tokens:
        for answer, given in token.answers.items():
            key3 = (token.channel_id, token.part, answer)
            score = Score.of(token.values, given)
            per_token[key3] = per_token[key3] + score if key3 in per_token else score
    mean_of = {
        key: float(np.mean([sample.tokens for sample in group]))
        for key, group in grouped(samples).items()
    }
    rows += [
        ScoreRow(corpus, masks, channel, TOKEN, part, answer, score, mean_of[(channel, part)])
        for (channel, part, answer), score in per_token.items()
    ]
    return rows


# ------------------------------------------------------------------------------------------------
# One corpus under one set of masks
# ------------------------------------------------------------------------------------------------


def drawn(
    windows: Sequence[TokenWindow], strategy: MaskingStrategy, *, batch_size: int, seed: int
) -> list[tuple[TokenTensors, TokenMasks]]:
    """The windows in a fixed order, each batch under masks drawn from its own seed."""
    loader = WindowLoader(windows, batch_size=batch_size, seed=seed, shuffle=False)
    masking = TokenMasking(strategy)
    return [
        (batch, masking.draw(batch, torch.Generator().manual_seed(seed * 1_000 + index)))
        for index, batch in enumerate(loader.batches_of(0))
    ]


def collected(
    batches: Iterable[tuple[TokenTensors, TokenMasks]],
    combined: OwnAndCrossChannelRidgeBaseline,
    channels_apart: frozenset[int],
    model: MaskedReconstruction | None = None,
    device: str = "cpu",
) -> tuple[list[Sample], list[TokenScore]]:
    interpolation = LinearInterpolationBaseline()
    samples, tokens = [], []
    for batch, masks in batches:
        predictions = {
            INTERPOLATION: interpolation.predict(batch, masks).double().numpy(),
            COMBINED: combined.predict(batch, masks).double().numpy(),
        }
        if model is not None:
            with torch.no_grad():
                answered = model(batch.to(device), masks.to(device)).detach().to("cpu")
            # Off the accelerator first, widened after: one call that does both reads zeros.
            predictions[MODEL] = answered.to(torch.float64).numpy()
        found, answered_tokens = samples_of(batch, masks, predictions, channels_apart)
        samples += found
        tokens += answered_tokens
    return samples, tokens


def audited(
    corpus: CorpusWindows,
    strategy: MaskingStrategy,
    masks: str,
    *,
    model: MaskedReconstruction | None,
    device: str,
    batch_size: int = BATCH_SIZE,
) -> list[ScoreRow]:
    """One corpus's scores under one set of masks.

    The regression is fitted on the first half of the training windows, the ridge maps on the
    second, each under masks of the fitting seed; the validation windows are scored under masks
    of the validation seed.
    """
    half = len(corpus.training) // 2
    first = drawn(corpus.training[:half], strategy, batch_size=batch_size, seed=FIT_SEED)
    combined = OwnAndCrossChannelRidgeBaseline.fitted(first, vocabulary_size=corpus.vocabulary_size)
    second = drawn(corpus.training[half:], strategy, batch_size=batch_size, seed=FIT_SEED + 1)
    fitting, _ = collected(second, combined, corpus.channels_apart)
    maps = FittedMaps.fitted(fitting)
    if model is not None:
        model = model.to(device).eval()
    validation = drawn(corpus.validation, strategy, batch_size=batch_size, seed=VALIDATION_SEED)
    samples, tokens = collected(validation, combined, corpus.channels_apart, model, device)
    return scored(corpus.corpus, masks, samples, tokens, maps)


# ------------------------------------------------------------------------------------------------
# Files and the report
# ------------------------------------------------------------------------------------------------


def write_scores(directory: Path, rows: Iterable[ScoreRow]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / SCORES).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(row.record() for row in rows)


def read_scores(directory: Path) -> list[ScoreRow]:
    with (directory / SCORES).open(newline="") as handle:
        return [ScoreRow.parse(record) for record in csv.DictReader(handle)]


def write_settings(directory: Path, settings: Mapping[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / SETTINGS).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("key", "value"))
        writer.writerows(settings.items())


def read_settings(directory: Path) -> dict[str, str]:
    with (directory / SETTINGS).open(newline="") as handle:
        return {record["key"]: record["value"] for record in csv.DictReader(handle)}


def median_r2(rows: Iterable[ScoreRow], *, sparse: bool | None = None) -> tuple[float, int]:
    """The median over channels of R², and how many channels had enough samples to enter it."""
    values = [
        row.score.r2
        for row in rows
        if row.score.count >= LEAST_SCORED
        and (sparse is None or (row.tokens_per_window <= SPARSE_TOKENS) == sparse)
    ]
    values = [value for value in values if not np.isnan(value)]
    return (float(np.median(values)), len(values)) if values else (np.nan, 0)


def _cell(value: tuple[float, int]) -> str:
    median, channels = value
    return "—" if channels == 0 else f"{median:.2f} ({channels})"


def render(directory: Path) -> str:
    """Per masks, corpus and part: the median R² over channels of each answer to each summary."""
    settings = read_settings(directory)
    rows = read_scores(directory)
    header = ", ".join(
        f"{key} {settings[key]}" for key in ("backbone", "revision") if key in settings
    )
    sections = [f"{header}; device {settings.get('device', '?')}.", ""]
    answers = [answer for answer in ANSWERS if any(row.answer == answer for row in rows)]
    for masks in dict.fromkeys(row.masks for row in rows):
        for corpus in dict.fromkeys(row.corpus for row in rows if row.masks == masks):
            for part in (PARTIAL, WHOLE):
                chosen = [r for r in rows if (r.masks, r.corpus, r.part) == (masks, corpus, part)]
                if not chosen:
                    continue
                sections.append(
                    f"### {corpus}, masks {masks}, channel {part}ly hidden — median R² over "
                    "channels (channels)"
                )
                body = []
                for summary in (*SUMMARIES, TOKEN):
                    for sparse, label in ((None, "all"), (True, "sparse"), (False, "dense")):
                        cells = [
                            _cell(
                                median_r2(
                                    (r for r in chosen if (r.summary, r.answer) == (summary, a)),
                                    sparse=sparse,
                                )
                            )
                            for a in answers
                        ]
                        if all(cell == "—" for cell in cells):
                            continue
                        body.append([summary, label, *cells])
                sections += [table(["summary", "channels", *answers], body), ""]
    return "\n".join(sections)


# ------------------------------------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------------------------------------


def masks_of(path: Path) -> MaskingStrategy:
    """The ``[masking]`` section of an experiment file, as the run would draw it."""
    with path.open("rb") as handle:
        section = tomllib.load(handle)["masking"]
    return MaskingStrategy(**{key: float(value) for key, value in section.items()})


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", metavar="ID", help="a backbone the registry holds ready")
    parser.add_argument(
        "--masks",
        type=Path,
        action="append",
        default=[],
        metavar="FILE",
        help="an experiment file whose masks are audited as well as the backbone's own",
    )
    parser.add_argument(
        "--no-model", action="store_true", help="leave the backbone's reconstruction unscored"
    )
    parser.add_argument("--report-only", type=Path, metavar="DIR", help="render a stored audit")
    parser.add_argument(
        "--out", type=Path, help="data/report/summary-target/<backbone> unless given"
    )
    parser.add_argument("--device", choices=DEVICES, default=available_device())
    parser.add_argument("--validation-windows", type=int, default=VALIDATION_WINDOWS)
    parser.add_argument("--training-windows", type=int, default=TRAINING_WINDOWS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--blocks", type=Path, default=BLOCKS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - needs the registry
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(render(arguments.report_only))
        return
    if arguments.backbone is None:
        raise SystemExit("--backbone is needed to audit anything")
    if not device_available(arguments.device):
        raise SystemExit(f"device {arguments.device} is not available on this machine")
    backbone = _registry().get(_backbone_id(arguments.backbone, "--backbone"))
    if backbone.artifact is None:
        raise SystemExit(f"backbone {backbone.id} is not ready: it has no weights")
    store = configured_store(Settings())
    model: MaskedReconstruction | None = None
    if not arguments.no_model:
        built = TrainedModel.read(store.get(backbone.artifact)).build()
        if not isinstance(built, MaskedReconstruction):
            raise SystemExit(f"backbone {backbone.id} does not reconstruct values")
        model = built
    strategies = {"own": backbone.configuration.masking}
    for path in arguments.masks:
        strategies[path.stem] = masks_of(path)
    directory = arguments.out or WORKSPACE / str(backbone.id)
    write_settings(
        directory,
        {
            "backbone": str(backbone.id),
            "weights": backbone.artifact.checksum.digest,
            "experiment": backbone.configuration.name,
            "masks": ", ".join(["own", *map(str, arguments.masks)]),
            "validation_windows": str(arguments.validation_windows),
            "training_windows": str(arguments.training_windows),
            "batch_size": str(arguments.batch_size),
            "seeds": f"validation {VALIDATION_SEED}, fit {FIT_SEED} and {FIT_SEED + 1}",
            "device": arguments.device,
            "machine": machine(),
            "torch": version("torch"),
            "revision": SourceRevision().current(),
            "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        },
    )
    rows: list[ScoreRow] = []
    for corpus in corpora_of(
        backbone,
        store,
        arguments.blocks,
        training_windows=arguments.training_windows,
        validation_windows=arguments.validation_windows,
    ):
        for name, strategy in strategies.items():
            started = time.perf_counter()
            # The model is scored under the masks it was trained under alone.
            rows += audited(
                corpus,
                strategy,
                name,
                model=model if name == "own" else None,
                device=arguments.device,
                batch_size=arguments.batch_size,
            )
            write_scores(directory, rows)
            print(
                f"{corpus.corpus}, {name}: {time.perf_counter() - started:.0f} s", file=sys.stderr
            )
    print(render(directory))
    print(f"\nstored under {directory}")


if __name__ == "__main__":
    main()
