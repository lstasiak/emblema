"""Does a backbone learn each kind of mask beyond its trivial baseline, on the corpora it read?

The same question the synthetic control answers at the published tier, asked of an accepted
backbone on its real corpora: for every kind of mask, the model's error on the validation
windows against the baseline matched to the kind — a line through the channel's own visible
tokens for a block, a single token or the window's tail, where the line carries the last value
forward; the cross-channel regression for a channel hidden whole — and against the strongest
linear answer on the same sources. Each comparison carries a bootstrap interval over validation
units. The registry's record of the backbone names everything the diagnostic needs: the
configuration whose masks are drawn, the corpora as read, and the weights kept.

Measuring and reporting are two steps: the tallies per unit and the summaries per kind are
written as CSV, and the tables are rendered from those files, so a note is redrawn without the
backbone or the corpora in hand.

    uv run --env-file .env.r2 scripts/pretext_triviality_report.py --backbone <id>
    uv run scripts/pretext_triviality_report.py --report-only data/report/pretext/<backbone>
"""

import argparse
import csv
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

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.config.settings import Settings
from emblema.entrypoints.configured import configured_store
from emblema.entrypoints.source_revision import SourceRevision
from emblema.pretraining.adapters.diagnostics.cross_channel_ridge_baseline import (
    CrossChannelRidgeBaseline,
)
from emblema.pretraining.adapters.diagnostics.linear_interpolation_baseline import (
    LinearInterpolationBaseline,
)
from emblema.pretraining.adapters.diagnostics.own_and_cross_channel_ridge_baseline import (
    OwnAndCrossChannelRidgeBaseline,
)
from emblema.pretraining.adapters.diagnostics.triviality_diagnostic import TrivialityDiagnostic
from emblema.pretraining.adapters.diagnostics.unit_bootstrap import UnitBootstrap
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.adapters.training.devices import available_device
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.pretraining.domain.assessment.interval import CONFIDENCE, Interval, Verdict
from emblema.pretraining.domain.assessment.kind_summary import (
    BEYOND_LINEAR,
    MATCHED_BASELINE,
    KindSummary,
)
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.backbone.backbone import Backbone
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.objective_loss import ObjectiveLoss
from emblema.pretraining.ports.backbone_repository import BackboneRepository
from emblema.shared.adapters.loaders.window_loader import WindowLoader
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore
from scripts.corpus_saturation_report import BLOCKS
from scripts.excursion_report import fetched_block, sides_of
from scripts.masked_reconstruction_report import DEVICES, device_available
from scripts.reporting import machine, table

WORKSPACE = REPO_ROOT / "data" / "report" / "pretext"
# Windows a corpus contributes to each side, spaced evenly over the side so every unit is reached;
# a side with fewer is read whole.
VALIDATION_WINDOWS = 2_000
TRAINING_WINDOWS = 2_000
# Windows a forward pass takes at once; the accelerator's graph refuses a batch of a thousand
# tokens a window past a few hundred windows.
BATCH_SIZE = 64
RESAMPLES = 2_000
# The masks the model is scored under are drawn once per validation batch from the first seed;
# the masks the linear baselines are fitted under come from the second, so the baselines learn
# from windows as incomplete as the ones they are asked about, and not the same ones.
VALIDATION_SEED = 1
FIT_SEED = 2
SETTINGS, CORPORA, TALLIES, KINDS = "settings.csv", "corpora.csv", "tallies.csv", "kinds.csv"


# ------------------------------------------------------------------------------------------------
# What is diagnosed
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusWindows:
    """One corpus of the backbone's mixture, cut down to the windows the diagnostic reads.

    Attributes:
        corpus: Name the corpus is registered under.
        vocabulary_size: Entries of the channel vocabulary the backbone was pretrained over.
        channels_apart: Channels the verdict leaves aside: timeless ones and ones that never vary.
        training: Windows the linear baselines are fitted on.
        validation: Windows the model and the baselines are scored on.
        validation_units: The unit each validation window was cut from, in that order.
        block_checksum: Of the block the windows were read from.
    """

    corpus: str
    vocabulary_size: int
    channels_apart: frozenset[int]
    training: Sequence[TokenWindow]
    validation: Sequence[TokenWindow]
    validation_units: tuple[str, ...]
    block_checksum: str

    @classmethod
    def of(
        cls,
        corpus: str,
        vocabulary_size: int,
        manifest: PublishedCorpusManifest,
        block: WindowBlock,
        *,
        training_windows: int,
        validation_windows: int,
    ) -> Self:
        sides = sides_of(manifest, block)
        training, validation = sides["training"], sides["validation"]
        kept = evenly_spaced(len(validation.windows), validation_windows)
        return cls(
            corpus=corpus,
            vocabulary_size=vocabulary_size,
            channels_apart=channels_apart_of(manifest),
            training=[
                training.windows[position]
                for position in evenly_spaced(len(training.windows), training_windows)
            ],
            validation=[validation.windows[position] for position in kept],
            validation_units=tuple(validation.units[position] for position in kept),
            block_checksum=str(manifest.block.checksum),
        )


def evenly_spaced(count: int, wanted: int) -> list[int]:
    """At most ``wanted`` positions out of ``count``, spread evenly from the first to the last."""
    if count <= wanted:
        return list(range(count))
    return [int(position) for position in np.unique(np.linspace(0, count - 1, wanted).round())]


def channels_apart_of(manifest: PublishedCorpusManifest) -> frozenset[int]:
    """Channels the verdict leaves aside: timeless ones, and ones whose published std is zero."""
    return frozenset(
        channel.channel_id
        for channel in manifest.channels
        if channel.timeless or (channel.statistics is not None and channel.statistics.std == 0.0)
    )


# ------------------------------------------------------------------------------------------------
# The diagnostic
# ------------------------------------------------------------------------------------------------


def fitted_batches(
    windows: Sequence[TokenWindow], strategy: MaskingStrategy, *, batch_size: int, seed: int
) -> list[tuple[TokenTensors, TokenMasks]]:
    """The training windows in a fixed order under masks drawn once, to fit the baselines on."""
    loader = WindowLoader(windows, batch_size=batch_size, seed=seed, shuffle=False)
    masking = TokenMasking(strategy)
    draws = torch.Generator().manual_seed(seed)
    return [(batch, masking.draw(batch, draws)) for batch in loader.batches_of(0)]


def diagnose(
    model: MaskedReconstruction,
    strategy: MaskingStrategy,
    loss: ObjectiveLoss,
    corpus: CorpusWindows,
    *,
    device: str,
    batch_size: int = BATCH_SIZE,
    validation_seed: int = VALIDATION_SEED,
    fit_seed: int = FIT_SEED,
) -> tuple[MaskKindTally, ...]:
    """Tally the model against every baseline on the corpus's validation windows, per unit.

    The masks are the strategy's, drawn on the host once per batch from a seed, so a backbone is
    scored under the same masks on any device; the model's prediction is brought back to the host
    before it is widened. The baselines are fitted on the training windows under masks of another
    seed.
    """
    fit = fitted_batches(corpus.training, strategy, batch_size=batch_size, seed=fit_seed)
    interpolation = LinearInterpolationBaseline()
    ridge = CrossChannelRidgeBaseline.fitted(fit, vocabulary_size=corpus.vocabulary_size)
    combined = OwnAndCrossChannelRidgeBaseline.fitted(fit, vocabulary_size=corpus.vocabulary_size)
    diagnostic = TrivialityDiagnostic(loss, channels_apart=corpus.channels_apart)
    masking = TokenMasking(strategy)
    loader = WindowLoader(
        corpus.validation, batch_size=batch_size, seed=validation_seed, shuffle=False
    )
    model = model.to(device).eval()
    offset = 0
    with torch.no_grad():
        for index, batch in enumerate(loader.batches_of(0)):
            units = corpus.validation_units[offset : offset + batch.batch_size]
            offset += batch.batch_size
            masks = masking.draw(
                batch, torch.Generator().manual_seed(validation_seed * 1_000 + index)
            )
            predicted = model(batch.to(device), masks.to(device)).detach().to("cpu")
            diagnostic = diagnostic.observe(
                batch,
                masks,
                units,
                model=predicted,
                interpolation=interpolation.predict(batch, masks),
                ridge=ridge.predict(batch, masks),
                combined=combined.predict(batch, masks),
            )
    return diagnostic.tallies()


def summarise(
    tallies: Iterable[MaskKindTally], *, resamples: int = RESAMPLES
) -> tuple[KindSummary, ...]:
    return UnitBootstrap(resamples=resamples).summarise(tallies)


# ------------------------------------------------------------------------------------------------
# Files
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusRow:
    """What one corpus contributed, and what reading it cost."""

    corpus: str
    vocabulary_size: int
    training_windows: int
    validation_windows: int
    validation_units: int
    channels_apart: int
    block_checksum: str
    seconds: float

    FIELDS = (
        "corpus",
        "vocabulary_size",
        "training_windows",
        "validation_windows",
        "validation_units",
        "channels_apart",
        "block_checksum",
        "seconds",
    )

    @classmethod
    def of(cls, corpus: CorpusWindows, seconds: float) -> Self:
        return cls(
            corpus.corpus,
            corpus.vocabulary_size,
            len(corpus.training),
            len(corpus.validation),
            len(set(corpus.validation_units)),
            len(corpus.channels_apart),
            corpus.block_checksum,
            seconds,
        )

    def record(self) -> dict[str, str]:
        return {
            "corpus": self.corpus,
            "vocabulary_size": str(self.vocabulary_size),
            "training_windows": str(self.training_windows),
            "validation_windows": str(self.validation_windows),
            "validation_units": str(self.validation_units),
            "channels_apart": str(self.channels_apart),
            "block_checksum": self.block_checksum,
            "seconds": f"{self.seconds:.1f}",
        }

    @classmethod
    def parse(cls, record: dict[str, str]) -> Self:
        return cls(
            record["corpus"],
            int(record["vocabulary_size"]),
            int(record["training_windows"]),
            int(record["validation_windows"]),
            int(record["validation_units"]),
            int(record["channels_apart"]),
            record["block_checksum"],
            float(record["seconds"]),
        )


@dataclass(frozen=True)
class KindRow:
    """One kind's summary on one corpus, as the file holds it and the table reads it."""

    corpus: str
    summary: KindSummary

    FIELDS = (
        "corpus",
        "kind",
        "apart",
        "tokens",
        "units",
        "model",
        "matched",
        "linear",
        "mean",
        "matched_low",
        "matched_high",
        "verdict",
        "linear_low",
        "linear_high",
        "linear_verdict",
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
            "matched": _number(summary.matched_error),
            "linear": _number(summary.linear_error),
            "mean": _number(summary.mean_error),
            "matched_low": _number(summary.matched_excess.low),
            "matched_high": _number(summary.matched_excess.high),
            "verdict": summary.verdict.value,
            "linear_low": _number(summary.linear_excess.low),
            "linear_high": _number(summary.linear_excess.high),
            "linear_verdict": summary.linear_excess.verdict.value,
        }

    @classmethod
    def parse(cls, record: dict[str, str]) -> Self:
        summary = KindSummary(
            kind=MaskKind(record["kind"]),
            apart=record["apart"] == "true",
            tokens=int(record["tokens"]),
            units=int(record["units"]),
            model_error=float(record["model"]),
            matched_error=float(record["matched"]),
            linear_error=float(record["linear"]),
            mean_error=float(record["mean"]),
            noise_floor=None,
            matched_excess=Interval(float(record["matched_low"]), float(record["matched_high"])),
            linear_excess=Interval(float(record["linear_low"]), float(record["linear_high"])),
        )
        return cls(record["corpus"], summary)


TALLY_FIELDS = (
    "corpus",
    "kind",
    "apart",
    "unit",
    "tokens",
    "model_sse",
    "matched_sse",
    "linear_sse",
    "mean_sse",
)


def tally_record(corpus: str, tally: MaskKindTally) -> dict[str, str]:
    return {
        "corpus": corpus,
        "kind": tally.kind.value,
        "apart": str(tally.apart).lower(),
        "unit": tally.group,
        "tokens": str(tally.tokens),
        "model_sse": _number(tally.model),
        "matched_sse": _number(tally.matched),
        "linear_sse": _number(tally.linear),
        "mean_sse": _number(tally.mean),
    }


def parse_tally(record: dict[str, str]) -> tuple[str, MaskKindTally]:
    return record["corpus"], MaskKindTally(
        kind=MaskKind(record["kind"]),
        apart=record["apart"] == "true",
        group=record["unit"],
        tokens=int(record["tokens"]),
        model=float(record["model_sse"]),
        matched=float(record["matched_sse"]),
        linear=float(record["linear_sse"]),
        mean=float(record["mean_sse"]),
        floor=None,
    )


def _number(value: float) -> str:
    """A number as text that reads back to the same float."""
    return repr(float(value))


def write_rows(path: Path, fields: Sequence[str], records: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_settings(directory: Path, settings: dict[str, str]) -> None:
    write_rows(
        directory / SETTINGS,
        ("key", "value"),
        ({"key": key, "value": value} for key, value in settings.items()),
    )


def read_settings(directory: Path) -> dict[str, str]:
    return {row["key"]: row["value"] for row in read_rows(directory / SETTINGS)}


def write_tallies(directory: Path, tallies: Iterable[tuple[str, MaskKindTally]]) -> None:
    write_rows(directory / TALLIES, TALLY_FIELDS, (tally_record(c, t) for c, t in tallies))


def read_tallies(directory: Path) -> list[tuple[str, MaskKindTally]]:
    return [parse_tally(record) for record in read_rows(directory / TALLIES)]


def write_kinds(directory: Path, rows: Iterable[KindRow]) -> None:
    write_rows(directory / KINDS, KindRow.FIELDS, (row.record() for row in rows))


def read_kinds(directory: Path) -> list[KindRow]:
    return [KindRow.parse(record) for record in read_rows(directory / KINDS)]


def write_corpora(directory: Path, rows: Iterable[CorpusRow]) -> None:
    write_rows(directory / CORPORA, CorpusRow.FIELDS, (row.record() for row in rows))


def read_corpora(directory: Path) -> list[CorpusRow]:
    return [CorpusRow.parse(record) for record in read_rows(directory / CORPORA)]


def kinds_of(
    tallies: Sequence[tuple[str, MaskKindTally]], *, resamples: int = RESAMPLES
) -> list[KindRow]:
    """The summaries of every corpus's tallies, each corpus bootstrapped on its own units."""
    corpora = list(dict.fromkeys(corpus for corpus, _ in tallies))
    return [
        KindRow(corpus, summary)
        for corpus in corpora
        for summary in summarise(
            (tally for named, tally in tallies if named == corpus), resamples=resamples
        )
    ]


# ------------------------------------------------------------------------------------------------
# The report
# ------------------------------------------------------------------------------------------------


def kinds_table(rows: Sequence[KindRow]) -> str:
    lines = [
        (
            row.corpus,
            row.summary.kind.value + (" (apart)" if row.summary.apart else ""),
            f"{row.summary.tokens:,}",
            str(row.summary.units),
            f"{row.summary.model_error:.4f}",
            f"{MATCHED_BASELINE[row.summary.kind]} {row.summary.matched_error:.4f}",
            f"{row.summary.excess:+.4f} {row.summary.matched_excess}",
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
        "Matched baseline",
        f"Excess [{CONFIDENCE:.0%}]",
        "Verdict",
        "Linear",
        f"Beyond linear [{CONFIDENCE:.0%}]",
        "Channel mean",
    )
    return table(header, lines)


def corpora_table(rows: Sequence[CorpusRow]) -> str:
    return table(
        ("Corpus", "Vocabulary", "Fitted on", "Scored on", "Units", "Apart", "Seconds"),
        (
            (
                row.corpus,
                str(row.vocabulary_size),
                f"{row.training_windows:,}",
                f"{row.validation_windows:,}",
                str(row.validation_units),
                str(row.channels_apart),
                f"{row.seconds:.0f}",
            )
            for row in rows
        ),
    )


def render(directory: Path) -> str:
    settings = read_settings(directory)
    strategy = ", ".join(
        f"{key} {settings[key]}"
        for key in (
            "channel_rate",
            "block_rate",
            "block_span",
            "token_rate",
            "horizon_rate",
            "horizon_min_span",
            "horizon_max_span",
        )
        if key in settings
    )
    learnt = [
        f"{row.corpus}/{row.summary.kind.value}"
        for row in read_kinds(directory)
        if not row.summary.apart and row.summary.verdict is Verdict.LEARNT
    ]
    return "\n\n".join(
        (
            f"### {settings['started'][:10]} — {settings['experiment']}, backbone "
            f"`{settings['backbone'][:8]}-…`, weights `sha256:{settings['weights'][:8]}…`",
            f"Masks: {strategy}; validation masks from seed {settings['validation_seed']}, the "
            f"baselines fitted under seed {settings['fit_seed']}; {settings['resamples']} "
            f"resamples over units; loss {settings['loss']}; {settings['device']}, "
            f"{settings['machine']}, torch {settings['torch']}, commit {settings['revision']}.",
            corpora_table(read_corpora(directory)),
            kinds_table(read_kinds(directory)),
            "Errors are the run's own reading of the loss on hidden tokens, in normalised units. "
            "Each interval is a bootstrap over validation units under one draw of the masks; a "
            "kind is learnt only when the whole interval of the matched baseline's error less the "
            "model's lies above zero. The matched baseline for the tail is the channel's own line "
            "carried past its last visible token.",
            "Learnt: " + (", ".join(learnt) if learnt else "none") + ".",
        )
    )


# ------------------------------------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------------------------------------


def corpora_of(
    backbone: Backbone,
    store: ArtifactStore,
    blocks: Path,
    *,
    training_windows: int,
    validation_windows: int,
) -> list[CorpusWindows]:
    """Every corpus the backbone read, cut to the diagnostic's windows, from the blocks it names.

    The registry records each input as the manifest the run was pointed at and the checksum of
    the block it read; the block each manifest points at now is held against that checksum.

    Raises:
        ValueError: If a manifest's block is not the block the backbone was pretrained over.
    """
    corpora = []
    codec = PublishedCorpusManifestJson()
    for stated in backbone.inputs:
        manifest = codec.decode(store.get(stated.manifest))
        if manifest.block.checksum != stated.block_checksum:
            raise ValueError(
                f"{stated.corpus}: the manifest's block is {manifest.block.checksum}, the backbone "
                f"read {stated.block_checksum}"
            )
        corpora.append(
            CorpusWindows.of(
                stated.corpus,
                stated.vocabulary_size,
                manifest,
                fetched_block(store, manifest.block, blocks),
                training_windows=training_windows,
                validation_windows=validation_windows,
            )
        )
    return corpora


def _registry() -> BackboneRepository:  # pragma: no cover - environment
    from emblema.entrypoints.configured import configured_engine
    from emblema.pretraining.adapters.persistence.backbone_repository import (
        SqlAlchemyBackboneRepository,
    )

    return SqlAlchemyBackboneRepository(configured_engine(Settings()))


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", metavar="ID", help="a backbone the registry holds ready")
    parser.add_argument(
        "--report-only", type=Path, metavar="DIR", help="render a stored diagnostic"
    )
    parser.add_argument(
        "--out", type=Path, help="where the files go; data/report/pretext/<backbone> unless given"
    )
    parser.add_argument("--device", choices=DEVICES, default=available_device())
    parser.add_argument("--validation-windows", type=int, default=VALIDATION_WINDOWS)
    parser.add_argument("--training-windows", type=int, default=TRAINING_WINDOWS)
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
    directory = arguments.out or WORKSPACE / str(backbone.id)
    strategy = configuration.masking
    write_settings(
        directory,
        {
            "backbone": str(backbone.id),
            "weights": backbone.artifact.checksum.digest,
            "experiment": configuration.name,
            "loss": str(configuration.loss.kind),
            **{
                key: _number(value)
                for key, value in configuration.parameters().items()
                if key.startswith(("channel_rate", "block_", "token_rate", "horizon_"))
                and isinstance(value, float)
            },
            "validation_windows": str(arguments.validation_windows),
            "training_windows": str(arguments.training_windows),
            "batch_size": str(arguments.batch_size),
            "validation_seed": str(VALIDATION_SEED),
            "fit_seed": str(FIT_SEED),
            "resamples": str(arguments.resamples),
            "device": arguments.device,
            "machine": machine(),
            "torch": version("torch"),
            "revision": SourceRevision().current(),
            "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        },
    )
    tallies: list[tuple[str, MaskKindTally]] = []
    rows = []
    for corpus in corpora_of(
        backbone,
        store,
        arguments.blocks,
        training_windows=arguments.training_windows,
        validation_windows=arguments.validation_windows,
    ):
        started = time.perf_counter()
        found = diagnose(
            model,
            strategy,
            configuration.loss,
            corpus,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        tallies += [(corpus.corpus, tally) for tally in found]
        rows.append(CorpusRow.of(corpus, time.perf_counter() - started))
        print(f"{corpus.corpus}: {len(found)} tallies in {rows[-1].seconds:.0f} s", file=sys.stderr)
        write_tallies(directory, tallies)
        write_corpora(directory, rows)
    write_kinds(directory, kinds_of(tallies, resamples=arguments.resamples))
    print(render(directory))
    print(f"\nstored under {directory}")


if __name__ == "__main__":
    main()
