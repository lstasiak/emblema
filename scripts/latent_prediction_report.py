"""Does a backbone that predicts states learn each kind of mask beyond the trivial answers?

The question the triviality report asks of the objective that predicts values, asked of the one
that predicts the teacher's states: for every kind of mask, the predictor's error on the
validation windows against the state interpolated along the channel between the hidden token's
visible neighbours — the line, carried forward past the last visible token — and against the
nearest visible neighbour's state copied over; a channel hidden whole, having no neighbours, is
read against the mean state of the window's visible tokens, which is also the trivial predictor
everywhere. Each comparison carries a bootstrap interval over validation units, and a kind is
learnt where the whole interval of the line's error less the predictor's lies above zero.

Beside the tallies, what the teacher's targets are like: their variance over tokens and the
effective rank of their covariance, which is where a collapse shows — targets drawn onto one
point or into a few directions make every predictor look good at nothing.

Measuring and reporting are two steps: the tallies per unit, the summaries per kind and the
targets' spread are written as CSV, and the tables are rendered from those files.

    uv run --env-file .env.r2 scripts/latent_prediction_report.py --backbone <id>
    uv run scripts/latent_prediction_report.py --report-only data/report/pretext-latent/<backbone>
"""

import argparse
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Self
from uuid import UUID

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
from emblema.pretraining.adapters.diagnostics.latent_interpolation_baseline import (
    LatentInterpolationBaseline,
)
from emblema.pretraining.adapters.diagnostics.latent_nearest_baseline import LatentNearestBaseline
from emblema.pretraining.adapters.diagnostics.triviality_diagnostic import TrivialityDiagnostic
from emblema.pretraining.adapters.objective.latent_prediction import LatentPrediction
from emblema.pretraining.adapters.objective.latent_prediction_loss import LatentPredictionLoss
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.adapters.training.devices import available_device
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.pretraining.domain.assessment.interval import CONFIDENCE, Interval, Verdict
from emblema.pretraining.domain.assessment.kind_summary import BEYOND_LINEAR, KindSummary
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.objective_loss import ObjectiveLoss
from emblema.shared.adapters.loaders.window_loader import WindowLoader
from scripts.corpus_saturation_report import BLOCKS
from scripts.masked_reconstruction_report import DEVICES, device_available
from scripts.pretext_triviality_report import (
    BATCH_SIZE,
    RESAMPLES,
    VALIDATION_SEED,
    VALIDATION_WINDOWS,
    CorpusRow,
    CorpusWindows,
    _number,
    _registry,
    corpora_of,
    corpora_table,
    read_corpora,
    read_rows,
    read_settings,
    summarise,
    write_corpora,
    write_rows,
    write_settings,
    write_tallies,
)
from scripts.reporting import machine, table

WORKSPACE = REPO_ROOT / "data" / "report" / "pretext-latent"
KINDS, TARGETS = "kinds.csv", "targets.csv"
# Observed tokens whose targets the spread is read from, per corpus: enough for a covariance of a
# few hundred coordinates, few enough to hold.
SPREAD_TOKENS = 4_096


# ------------------------------------------------------------------------------------------------
# What is diagnosed
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSpread:
    """How the teacher's targets are spread over the observed tokens of one corpus.

    Attributes:
        corpus: Name the corpus is registered under.
        tokens: Observed tokens the spread was read from.
        variance: Mean over the coordinates of the variance across tokens.
        effective_rank: Exponential of the entropy of the covariance's normalised eigenvalues:
            how many directions the targets fill, one where they collapse onto a line.
    """

    corpus: str
    tokens: int
    variance: float
    effective_rank: float

    FIELDS = ("corpus", "tokens", "variance", "effective_rank")

    @classmethod
    def of(cls, corpus: str, targets: NDArray[np.float64]) -> Self:
        """The spread of these targets, ``[tokens, width]``, over their tokens."""
        centred = targets - targets.mean(axis=0, keepdims=True)
        variance = float(centred.var(axis=0, ddof=0).mean()) if len(targets) > 1 else 0.0
        covariance = centred.T @ centred / max(len(targets) - 1, 1)
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        total = float(eigenvalues.sum())
        if total <= 0.0:
            return cls(corpus, len(targets), variance, 1.0)
        shares = eigenvalues[eigenvalues > 0.0] / total
        return cls(corpus, len(targets), variance, float(np.exp(-(shares * np.log(shares)).sum())))

    def record(self) -> dict[str, str]:
        return {
            "corpus": self.corpus,
            "tokens": str(self.tokens),
            "variance": _number(self.variance),
            "effective_rank": _number(self.effective_rank),
        }

    @classmethod
    def parse(cls, record: dict[str, str]) -> Self:
        return cls(
            record["corpus"],
            int(record["tokens"]),
            float(record["variance"]),
            float(record["effective_rank"]),
        )


def diagnose(
    model: LatentPrediction,
    strategy: MaskingStrategy,
    loss: ObjectiveLoss,
    corpus: CorpusWindows,
    *,
    device: str,
    batch_size: int = BATCH_SIZE,
    validation_seed: int = VALIDATION_SEED,
    spread_tokens: int = SPREAD_TOKENS,
) -> tuple[tuple[MaskKindTally, ...], TargetSpread]:
    """Tally the predictor against the trivial answers on the validation windows, per unit.

    The masks are the strategy's, drawn on the host once per batch from a seed, as the run drew
    its own; the targets and the prediction are brought back to the host before they are widened
    and scored. The baselines need no fitting: each reads the teacher's targets of the window.
    """
    diagnostic = TrivialityDiagnostic(loss, channels_apart=corpus.channels_apart)
    scorer = LatentPredictionLoss(loss)
    interpolation, nearest = LatentInterpolationBaseline(), LatentNearestBaseline()
    masking = TokenMasking(strategy)
    loader = WindowLoader(
        corpus.validation, batch_size=batch_size, seed=validation_seed, shuffle=False
    )
    model = model.to(device).eval()
    offset = 0
    sampled: list[NDArray[np.float64]] = []
    kept = 0
    with torch.no_grad():
        for index, batch in enumerate(loader.batches_of(0)):
            units = corpus.validation_units[offset : offset + batch.batch_size]
            offset += batch.batch_size
            masks = masking.draw(
                batch, torch.Generator().manual_seed(validation_seed * 1_000 + index)
            )
            on_device = batch.to(device)
            targets = model.targets(on_device).to("cpu").to(torch.float64)
            predicted = model(on_device, masks.to(device)).to("cpu").to(torch.float64)
            observed = ~batch.padding_mask
            errors = {
                name: scorer.of_tokens(prediction, targets)
                for name, prediction in (
                    ("model", predicted),
                    ("interpolation", interpolation.predict(targets, batch, masks)),
                    ("nearest", nearest.predict(targets, batch, masks)),
                    ("mean", scorer.visible_mean(targets, batch, masks)),
                )
            }
            diagnostic = diagnostic.observe_errors(
                batch,
                masks,
                units,
                model=errors["model"],
                matched=dict.fromkeys(MaskKind, errors["interpolation"]),
                linear=dict.fromkeys(MaskKind, errors["nearest"]),
                mean=errors["mean"],
            )
            if kept < spread_tokens:
                taken = targets[observed].numpy()[: spread_tokens - kept]
                sampled.append(taken)
                kept += len(taken)
    stacked = (
        np.concatenate(sampled) if sampled else np.zeros((0, model.encoder.architecture.width))
    )
    return diagnostic.tallies(), TargetSpread.of(corpus.corpus, stacked)


# ------------------------------------------------------------------------------------------------
# Files
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentKindRow:
    """One kind's summary on one corpus, under the names the latent baselines go by.

    The summary is the same value the triviality report writes; here the matched baseline is the
    line between the neighbours' states and the one beyond it the nearest neighbour's state, and
    the file says so.
    """

    corpus: str
    summary: KindSummary

    FIELDS = (
        "corpus",
        "kind",
        "apart",
        "tokens",
        "units",
        "model",
        "interpolation",
        "nearest",
        "mean",
        "interpolation_low",
        "interpolation_high",
        "verdict",
        "nearest_low",
        "nearest_high",
        "nearest_verdict",
    )

    def record(self) -> dict[str, str]:
        summary = self.summary
        return {
            "corpus": self.corpus,
            "kind": summary.kind.value,
            "apart": str(summary.apart).lower(),
            "tokens": str(summary.tokens),
            "units": str(summary.units),
            "model": _number(summary.model_error),
            "interpolation": _number(summary.matched_error),
            "nearest": _number(summary.linear_error),
            "mean": _number(summary.mean_error),
            "interpolation_low": _number(summary.matched_excess.low),
            "interpolation_high": _number(summary.matched_excess.high),
            "verdict": summary.verdict.value,
            "nearest_low": _number(summary.linear_excess.low),
            "nearest_high": _number(summary.linear_excess.high),
            "nearest_verdict": summary.linear_excess.verdict.value,
        }

    @classmethod
    def parse(cls, record: dict[str, str]) -> Self:
        summary = KindSummary(
            kind=MaskKind(record["kind"]),
            apart=record["apart"] == "true",
            tokens=int(record["tokens"]),
            units=int(record["units"]),
            model_error=float(record["model"]),
            matched_error=float(record["interpolation"]),
            linear_error=float(record["nearest"]),
            mean_error=float(record["mean"]),
            noise_floor=None,
            matched_excess=Interval(
                float(record["interpolation_low"]), float(record["interpolation_high"])
            ),
            linear_excess=Interval(float(record["nearest_low"]), float(record["nearest_high"])),
        )
        return cls(record["corpus"], summary)


def kinds_of(
    tallies: Sequence[tuple[str, MaskKindTally]], *, resamples: int = RESAMPLES
) -> list[LatentKindRow]:
    """The summaries of every corpus's tallies, each corpus bootstrapped on its own units."""
    corpora = list(dict.fromkeys(corpus for corpus, _ in tallies))
    return [
        LatentKindRow(corpus, summary)
        for corpus in corpora
        for summary in summarise(
            (tally for named, tally in tallies if named == corpus), resamples=resamples
        )
    ]


def write_kinds(directory: Path, rows: Iterable[LatentKindRow]) -> None:
    write_rows(directory / KINDS, LatentKindRow.FIELDS, (row.record() for row in rows))


def read_kinds(directory: Path) -> list[LatentKindRow]:
    return [LatentKindRow.parse(record) for record in read_rows(directory / KINDS)]


def write_targets(directory: Path, rows: Iterable[TargetSpread]) -> None:
    write_rows(directory / TARGETS, TargetSpread.FIELDS, (row.record() for row in rows))


def read_targets(directory: Path) -> list[TargetSpread]:
    return [TargetSpread.parse(record) for record in read_rows(directory / TARGETS)]


# ------------------------------------------------------------------------------------------------
# The report
# ------------------------------------------------------------------------------------------------


def kinds_table(rows: Sequence[LatentKindRow]) -> str:
    lines = [
        (
            row.corpus,
            row.summary.kind.value + (" (apart)" if row.summary.apart else ""),
            f"{row.summary.tokens:,}",
            str(row.summary.units),
            f"{row.summary.model_error:.4f}",
            f"{row.summary.matched_error:.4f}",
            str(row.summary.matched_excess),
            row.summary.verdict.value,
            f"{row.summary.linear_error:.4f}",
            f"{row.summary.linear_excess} {row.summary.linear_excess.verdict.value}"
            if row.summary.kind in BEYOND_LINEAR
            else "—",
            f"{row.summary.mean_error:.4f}",
        )
        for row in rows
    ]
    header = (
        "Corpus",
        "Kind",
        "Tokens",
        "Units",
        "Model",
        "Line",
        f"Excess [{CONFIDENCE:.0%}]",
        "Verdict",
        "Nearest",
        f"Beyond nearest [{CONFIDENCE:.0%}]",
        "Visible mean",
    )
    return table(header, lines)


def targets_table(rows: Sequence[TargetSpread]) -> str:
    return table(
        ("Corpus", "Tokens", "Variance", "Effective rank"),
        (
            (row.corpus, f"{row.tokens:,}", f"{row.variance:.4f}", f"{row.effective_rank:.1f}")
            for row in rows
        ),
    )


def render(directory: Path) -> str:
    settings = read_settings(directory)
    learnt = [
        f"{row.corpus}/{row.summary.kind.value}"
        for row in read_kinds(directory)
        if not row.summary.apart and row.summary.verdict is Verdict.LEARNT
    ]
    return "\n\n".join(
        (
            f"### {settings['started'][:10]} — {settings['experiment']}, backbone "
            f"`{settings['backbone'][:8]}-…`, weights `sha256:{settings['weights'][:8]}…`",
            f"Target: the mean of the teacher's top {settings['target_layers']} blocks, each "
            f"normalised without parameters; validation masks from seed "
            f"{settings['validation_seed']}; {settings['resamples']} resamples over units; "
            f"loss {settings['loss']} averaged over the coordinates of a state; "
            f"{settings['device']}, {settings['machine']}, torch {settings['torch']}, commit "
            f"{settings['revision']}.",
            corpora_table(read_corpora(directory)),
            kinds_table(read_kinds(directory)),
            "Errors are the run's own reading of the loss on hidden tokens, averaged over the "
            "coordinates of the state. Each interval is a bootstrap over validation units under "
            "one draw of the masks; a kind is learnt only when the whole interval of the line's "
            "error less the model's lies above zero. A channel hidden whole has no line and no "
            "nearest neighbour: both read as the visible mean, so its verdict is against that.",
            targets_table(read_targets(directory)),
            "The variance is over the observed tokens' targets, the effective rank the number of "
            "directions they fill; targets that collapse show as a variance near zero or a rank "
            "near one.",
            "Learnt: " + (", ".join(learnt) if learnt else "none") + ".",
        )
    )


# ------------------------------------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------------------------------------


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", metavar="ID", help="a backbone the registry holds ready")
    parser.add_argument(
        "--report-only", type=Path, metavar="DIR", help="render a stored diagnostic"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="where the files go; data/report/pretext-latent/<backbone> unless given",
    )
    parser.add_argument("--device", choices=DEVICES, default=available_device())
    parser.add_argument("--validation-windows", type=int, default=VALIDATION_WINDOWS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--blocks", type=Path, default=BLOCKS, help="where blocks are fetched to and mapped from"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(render(arguments.report_only))
        return
    if arguments.backbone is None:
        raise SystemExit("--backbone is needed to diagnose anything")
    if not device_available(arguments.device):
        raise SystemExit(f"device {arguments.device} is not available on this machine")
    try:
        backbone_id = BackboneId(UUID(arguments.backbone))
    except ValueError as error:
        raise SystemExit(f"--backbone must be a backbone's id: {error}") from error
    backbone = _registry().get(backbone_id)
    if backbone.artifact is None:
        raise SystemExit(f"backbone {backbone.id} is not ready: it has no weights")
    store = configured_store(Settings())
    configuration = backbone.configuration
    model = TrainedModel.read(store.get(backbone.artifact)).build()
    if not isinstance(model, LatentPrediction):
        raise SystemExit(
            f"backbone {backbone.id} was pretrained to predict values; its predictions are "
            "scored against the baselines of a value by scripts/pretext_triviality_report.py"
        )
    directory = arguments.out or WORKSPACE / str(backbone.id)
    write_settings(
        directory,
        {
            "backbone": str(backbone.id),
            "weights": backbone.artifact.checksum.digest,
            "experiment": configuration.name,
            "loss": str(configuration.loss.kind),
            "target_layers": str(configuration.target.layers),
            "momentum": _number(configuration.target.momentum),
            "final_momentum": _number(configuration.target.final_momentum),
            **{
                key: _number(value)
                for key, value in configuration.parameters().items()
                if key.startswith(("channel_rate", "block_", "token_rate", "horizon_"))
                and isinstance(value, float)
            },
            "validation_windows": str(arguments.validation_windows),
            "batch_size": str(arguments.batch_size),
            "validation_seed": str(VALIDATION_SEED),
            "resamples": str(arguments.resamples),
            "device": arguments.device,
            "machine": machine(),
            "torch": version("torch"),
            "revision": SourceRevision().current(),
            "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        },
    )
    tallies: list[tuple[str, MaskKindTally]] = []
    rows, spreads = [], []
    for corpus in corpora_of(
        backbone,
        store,
        arguments.blocks,
        training_windows=1,
        validation_windows=arguments.validation_windows,
    ):
        started = time.perf_counter()
        found, spread = diagnose(
            model,
            configuration.masking,
            configuration.loss,
            corpus,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        tallies += [(corpus.corpus, tally) for tally in found]
        rows.append(CorpusRow.of(corpus, time.perf_counter() - started))
        spreads.append(spread)
        print(f"{corpus.corpus}: {len(found)} tallies in {rows[-1].seconds:.0f} s", file=sys.stderr)
        write_tallies(directory, tallies)
        write_corpora(directory, rows)
        write_targets(directory, spreads)
    write_kinds(directory, kinds_of(tallies, resamples=arguments.resamples))
    print(render(directory))


if __name__ == "__main__":
    main()
