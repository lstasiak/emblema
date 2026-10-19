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

A corpus the backbone never read is read through another backbone's windows (``--corpus``
with ``--windows-of``): the question is then what the pretext carries to a corpus it has not
seen, against baselines fitted on that corpus alone, its channels passing through rows of the
channel table no pretraining set — drawn, as the evaluation gives them, or set to the mean of
the learnt rows (``--unset-rows mean``), which tells a row the model never saw from a row
unlike any it saw. The encoder at its initialisation (``--at-initialisation``) is the floor
of either reading. The rows alone can be fitted on the corpus's unlabelled training windows
under the pretext, every other weight frozen (``--fit-rows STEPS``): what the encoder's
machinery carries once the corpus's channels have an identity it chose, read on the same
validation windows; the losses of the fit go to ``fit.csv``. Or a row can be copied from the
row the backbone learnt for the same quantity in a corpus it read (``--unset-rows map`` with
``--map FILE``, the counterparts one per line; ``map-mean`` sets the rows the map does not
name to the mean, ``map`` leaves them drawn): whether a channel's identity is the quantity's,
shared across corpora, or the corpus's own. Under a map the tokens of the channels it names are
tallied apart as well (``tallies-mapped.csv``, ``kinds-mapped.csv``), since the map reaches
those channels alone; ``--map`` with any other rows copies nothing and tallies those channels
apart all the same, which reads a backbone that set the rows itself on the same tokens.

    uv run --env-file .env.r2 scripts/pretext_triviality_report.py --backbone <id>
    uv run --env-file .env.r2 scripts/pretext_triviality_report.py --backbone <id>
        --corpus <name> --windows-of <id>
    uv run scripts/pretext_triviality_report.py --report-only data/report/pretext/<backbone>
"""

import argparse
import csv
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Self
from uuid import UUID

import numpy as np
import torch
from torch import Tensor

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.contracts.published_channel import PublishedChannel
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
from emblema.pretraining.adapters.encoder.grown_channel_embedding import (
    GrownChannelEmbedding,
)
from emblema.pretraining.adapters.encoder.learned_channel_embedding import (
    LearnedChannelEmbedding,
)
from emblema.pretraining.adapters.objective.latent_prediction import LatentPrediction
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape
from emblema.pretraining.adapters.objective.pretraining_objective import PretrainingObjective
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
from emblema.pretraining.domain.backbone.pretraining_input import PretrainingInput
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.objective_loss import ObjectiveLoss
from emblema.pretraining.ports.backbone_repository import BackboneRepository
from emblema.shared.adapters.loaders.window_loader import WindowLoader
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.kernel.tokens import PADDING_CHANNEL_ID, TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore
from scripts.channel_identity_report import Counterpart, read_map, swapped
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
TALLIES_MAPPED, KINDS_MAPPED = "tallies-mapped.csv", "kinds-mapped.csv"
FIT = "fit.csv"
# The rate the rows alone are fitted at: a row starts where torch drew it, at unit scale, and
# the learnt rows lie within a few units of each other, so a hundredth a step reaches them in
# a few hundred steps without leaving them.
FIT_RATE = 1e-2


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
        channel_ids: The corpus's own channels, which a map's tally stands apart from.
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
    channel_ids: frozenset[int]

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
            channel_ids=frozenset(own_channels(manifest.channels, corpus)),
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


def own_channels(channels: Iterable[PublishedChannel], corpus: str) -> list[int]:
    """The identifiers of the channels ``corpus`` itself has.

    A published manifest carries the whole vocabulary its chain had reached, every earlier
    corpus's channels included; only the rows of this corpus's own channels are the ones no
    pretraining without it set.
    """
    return [channel.channel_id for channel in channels if channel.corpus == corpus]


def unset_rows_to_mean(model: PretrainingObjective, channel_ids: Iterable[int]) -> None:
    """Set the rows of ``channel_ids``, which no pretraining set, to the mean of the learnt rows.

    A row torch drew is unlike any row the backbone learnt; the mean of the learnt rows is a
    channel the backbone could have seen, without saying which. Rows past the learnt table are
    the grown ones; rows within it that the backbone never read keep their drawn values until
    they are set here. Padding's zero row is left out of the mean.
    """
    table = model.encoder.channel_embedding
    learnt = table.learnt if isinstance(table, GrownChannelEmbedding) else table
    if not isinstance(learnt, LearnedChannelEmbedding):
        raise ValueError(f"only a learnt channel table has unset rows, not {type(learnt).__name__}")
    with torch.no_grad():
        mean = learnt.table.weight[PADDING_CHANNEL_ID + 1 :].mean(dim=0)
        for channel_id in channel_ids:
            if channel_id <= learnt.vocabulary_size:
                learnt.table.weight[channel_id] = mean
            elif isinstance(table, GrownChannelEmbedding):
                table.grown.weight[channel_id - learnt.vocabulary_size - 1] = mean
            else:
                raise ValueError(f"channel {channel_id} lies past a table that was not grown")


def named_channels(
    channels: Iterable[PublishedChannel], corpus: str, pairs: Iterable[Counterpart]
) -> list[int]:
    """The channels of ``corpus`` a map asks about, for a tally that copies nothing.

    Raises:
        ValueError: If a pair asks about a channel of another corpus or one the manifest lacks.
    """
    ids = {(channel.corpus, channel.channel): channel.channel_id for channel in channels}
    named = []
    for pair in pairs:
        if pair.corpus != corpus:
            raise ValueError(f"{pair.corpus}/{pair.channel} is not a channel of {corpus}")
        if (pair.corpus, pair.channel) not in ids:
            raise ValueError(f"the manifest names no row for {pair.corpus}/{pair.channel}")
        named.append(ids[(pair.corpus, pair.channel)])
    return named


def vocabulary_of(backbone: Backbone, store: ArtifactStore) -> list[PublishedChannel]:
    """Every channel of the backbone's own chain, from the manifest that reached furthest.

    A corpus from the middle of the chain is published under a shorter vocabulary than the
    backbone's; its counterparts in a later corpus are named by the backbone's widest manifest.
    """
    widest = max(backbone.inputs, key=lambda stated: stated.vocabulary_size)
    return list(PublishedCorpusManifestJson().decode(store.get(widest.manifest)).channels)


def corpora_channels_of(
    backbone: Backbone, store: ArtifactStore, corpus: str
) -> list[PublishedChannel]:
    """The channels the manifest of ``corpus``, as the backbone read it, names."""
    for stated in backbone.inputs:
        if stated.corpus == corpus:
            return list(PublishedCorpusManifestJson().decode(store.get(stated.manifest)).channels)
    raise ValueError(f"backbone {backbone.id} did not read {corpus}")


def unset_rows_from_counterparts(
    model: PretrainingObjective,
    channels: Iterable[PublishedChannel],
    own: Iterable[int],
    pairs: Iterable[Counterpart],
) -> list[int]:
    """Set each mapped row of the corpus's own channels to the learnt row of its counterpart.

    The query of every pair must be one of ``own``, the channels no pretraining set, and its
    counterpart a channel of another corpus whose row the backbone learnt: a row within the
    learnt table and outside ``own``. Rows the map does not name are left as they are.

    Returns:
        The identifiers of the rows set, in the map's order.

    Raises:
        ValueError: If a pair names a channel the manifest lacks, a query that is not one of the
            corpus's own channels, or a counterpart whose row no pretraining set.
    """
    table = model.encoder.channel_embedding
    learnt = table.learnt if isinstance(table, GrownChannelEmbedding) else table
    if not isinstance(learnt, LearnedChannelEmbedding):
        raise ValueError(f"only a learnt channel table has unset rows, not {type(learnt).__name__}")
    ids = {(channel.corpus, channel.channel): channel.channel_id for channel in channels}
    unset = set(own)
    mapped = []
    with torch.no_grad():
        for pair in pairs:
            query = ids.get((pair.corpus, pair.channel))
            counterpart = ids.get((pair.counterpart_corpus, pair.counterpart_channel))
            if query is None or counterpart is None:
                raise ValueError(f"the manifest names no row for a channel of the pair {pair}")
            if query not in unset:
                raise ValueError(
                    f"{pair.corpus}/{pair.channel} is not a channel of the corpus never read"
                )
            if counterpart in unset or counterpart > learnt.vocabulary_size:
                raise ValueError(
                    f"no pretraining set the row of {pair.counterpart_corpus}/"
                    f"{pair.counterpart_channel}, so it is no counterpart"
                )
            row = learnt.table.weight[counterpart].detach().clone()
            if query <= learnt.vocabulary_size:
                learnt.table.weight[query] = row
            elif isinstance(table, GrownChannelEmbedding):
                table.grown.weight[query - learnt.vocabulary_size - 1] = row
            else:
                raise ValueError(f"channel {query} lies past a table that was not grown")
            mapped.append(query)
    return mapped


def supplied_by(source: Backbone, backbone: Backbone, corpus: str) -> list[PretrainingInput]:
    """The inputs of ``source`` that carry ``corpus``, for a backbone that never read it.

    Raises:
        ValueError: If ``backbone`` read the corpus itself, so the reading would not be of a
            corpus it never saw, or if ``source`` did not read it.
    """
    if any(stated.corpus == corpus for stated in backbone.inputs):
        raise ValueError(f"backbone {backbone.id} read {corpus} itself; read it on its own windows")
    supplied = [stated for stated in source.inputs if stated.corpus == corpus]
    if not supplied:
        raise ValueError(f"backbone {source.id} did not read {corpus}")
    return supplied


def narrowed[T: TokenTensors | TokenMasks](batch: T, start: int, stop: int) -> T:
    """``batch`` cut to its windows ``start`` to ``stop``, every tensor sliced alike."""
    return replace(batch, **{f.name: getattr(batch, f.name)[start:stop] for f in fields(batch)})


def unset_rows_fitted(
    model: MaskedReconstruction,
    channel_ids: Iterable[int],
    windows: Sequence[TokenWindow],
    strategy: MaskingStrategy,
    loss: ObjectiveLoss,
    *,
    device: str,
    steps: int,
    learning_rate: float = FIT_RATE,
    batch_size: int = BATCH_SIZE,
    seed: int = FIT_SEED,
    chunk: int | None = None,
) -> list[float]:
    """Fit the rows of ``channel_ids`` on ``windows`` under the pretext, everything else frozen.

    The loop is the pretraining's in miniature — the strategy's masks drawn per batch from a
    seed, the loss on the hidden tokens, Adam — over the two tables' weights alone, with the
    gradient of every row outside ``channel_ids`` zeroed before each step, so no learnt row
    moves and no weight of the encoder or the decoder does. Adam's moments of a row that never
    receives a gradient stay zero, so zeroing the gradient holds the row still.

    A batch whose activations do not fit the device at once is scored ``chunk`` windows at a
    time under the masks drawn for the whole batch, each chunk's summed error divided by the
    batch's count of hidden tokens, so the step's gradient and loss are the whole batch's up to
    the order of a floating-point sum. The model is scored in evaluation mode, the forward pass
    the rows are read with afterwards, so that equality does not depend on a dropout the
    configuration may set. The model's mode and which of its parameters take gradients are as
    they were when this returns, whether the fit ends or fails.

    Returns:
        The mean loss on hidden tokens of each step's batch, in the order taken.
    """
    table = model.encoder.channel_embedding
    learnt = table.learnt if isinstance(table, GrownChannelEmbedding) else table
    if not isinstance(learnt, LearnedChannelEmbedding):
        raise ValueError(f"only a learnt channel table has unset rows, not {type(learnt).__name__}")
    chosen = set(channel_ids)
    within = torch.zeros(learnt.vocabulary_size + 1, 1)
    for channel_id in chosen:
        if channel_id <= learnt.vocabulary_size:
            within[channel_id] = 1.0
        elif not isinstance(table, GrownChannelEmbedding):
            raise ValueError(f"channel {channel_id} lies past a table that was not grown")
    beyond: Tensor | None = None
    if isinstance(table, GrownChannelEmbedding):
        beyond = torch.zeros(table.grown.num_embeddings, 1)
        for channel_id in chosen:
            if channel_id > learnt.vocabulary_size:
                beyond[channel_id - learnt.vocabulary_size - 1] = 1.0
    fitted = [learnt.table.weight]
    if isinstance(table, GrownChannelEmbedding):
        fitted.append(table.grown.weight)
    took_gradients = [parameter.requires_grad for parameter in model.parameters()]
    was_training = model.training
    for parameter in model.parameters():
        parameter.requires_grad_(any(parameter is weight for weight in fitted))
    model = model.to(device).eval()
    within = within.to(device)
    beyond = beyond.to(device) if beyond is not None else None
    optimiser = torch.optim.Adam(fitted, lr=learning_rate)
    masking = TokenMasking(strategy)
    loader = WindowLoader(windows, batch_size=batch_size, seed=seed, shuffle=True)
    losses: list[float] = []
    epoch = 0
    try:
        while len(losses) < steps:
            for batch in loader.batches_of(epoch):
                if len(losses) == steps:
                    break
                generator = torch.Generator().manual_seed(seed * 1_000 + len(losses))
                masks = masking.draw(batch, generator)
                batch, masks = batch.to(device), masks.to(device)
                tokens = (masks.hidden & ~batch.padding_mask).sum().clamp(min=1)
                optimiser.zero_grad()
                mean = torch.zeros((), device=device)
                stride = chunk or batch.batch_size
                for start in range(0, batch.batch_size, stride):
                    stop = start + stride
                    score = model.score(
                        narrowed(batch, start, stop), narrowed(masks, start, stop), loss
                    )
                    part = score.error / tokens
                    torch.autograd.backward(part)
                    mean += part.detach()
                if learnt.table.weight.grad is not None:
                    learnt.table.weight.grad.mul_(within)
                if (
                    beyond is not None
                    and isinstance(table, GrownChannelEmbedding)
                    and table.grown.weight.grad is not None
                ):
                    table.grown.weight.grad.mul_(beyond)
                optimiser.step()
                losses.append(float(mean.detach().to("cpu")))
            epoch += 1
    finally:
        for parameter, took in zip(model.parameters(), took_gradients, strict=True):
            parameter.requires_grad_(took)
        model.train(was_training)
    return losses


def grown(model: PretrainingObjective, vocabulary_size: int) -> PretrainingObjective:
    """The model over a vocabulary that continues its own, for a corpus it never read.

    The rows its table learnt stay as they are; a channel past them gets a row drawn from
    torch's generator here, which no pretraining set, as the evaluation gives it. An objective
    that predicts states is refused: its teacher reads the same table and would not be grown.

    Raises:
        ValueError: If the objective predicts states.
    """
    if isinstance(model, LatentPrediction):
        raise ValueError("an objective that predicts states is read on its own corpora only")
    model.encoder.grown_to(vocabulary_size)
    return model


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
    [tallies] = diagnose_each(
        model,
        strategy,
        loss,
        corpus,
        [corpus.channels_apart],
        device=device,
        batch_size=batch_size,
        validation_seed=validation_seed,
        fit_seed=fit_seed,
    )
    return tallies


def diagnose_each(
    model: MaskedReconstruction,
    strategy: MaskingStrategy,
    loss: ObjectiveLoss,
    corpus: CorpusWindows,
    apart: Sequence[frozenset[int]],
    *,
    device: str,
    batch_size: int = BATCH_SIZE,
    validation_seed: int = VALIDATION_SEED,
    fit_seed: int = FIT_SEED,
) -> list[tuple[MaskKindTally, ...]]:
    """``diagnose`` once per set of channels tallied apart, over one pass of predictions.

    A second set that holds every channel but the ones a map names tallies those alone on the
    other side, from the same predictions and under the same masks.
    """
    fit = fitted_batches(corpus.training, strategy, batch_size=batch_size, seed=fit_seed)
    interpolation = LinearInterpolationBaseline()
    ridge = CrossChannelRidgeBaseline.fitted(fit, vocabulary_size=corpus.vocabulary_size)
    combined = OwnAndCrossChannelRidgeBaseline.fitted(fit, vocabulary_size=corpus.vocabulary_size)
    diagnostics = [TrivialityDiagnostic(loss, channels_apart=channels) for channels in apart]
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
            predictions = {
                "interpolation": interpolation.predict(batch, masks),
                "ridge": ridge.predict(batch, masks),
                "combined": combined.predict(batch, masks),
            }
            diagnostics = [
                diagnostic.observe(batch, masks, units, model=predicted, **predictions)
                for diagnostic in diagnostics
            ]
    return [diagnostic.tallies() for diagnostic in diagnostics]


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


def write_fit(directory: Path, losses: Sequence[float]) -> None:
    write_rows(
        directory / FIT,
        ["step", "loss"],
        ({"step": str(step), "loss": _number(value)} for step, value in enumerate(losses, 1)),
    )


def read_fit(directory: Path) -> list[float]:
    return [float(record["loss"]) for record in read_rows(directory / FIT)]


def read_settings(directory: Path) -> dict[str, str]:
    return {row["key"]: row["value"] for row in read_rows(directory / SETTINGS)}


def write_tallies(
    directory: Path, tallies: Iterable[tuple[str, MaskKindTally]], name: str = TALLIES
) -> None:
    write_rows(directory / name, TALLY_FIELDS, (tally_record(c, t) for c, t in tallies))


def read_tallies(directory: Path) -> list[tuple[str, MaskKindTally]]:
    return [parse_tally(record) for record in read_rows(directory / TALLIES)]


def write_kinds(directory: Path, rows: Iterable[KindRow], name: str = KINDS) -> None:
    write_rows(directory / name, KindRow.FIELDS, (row.record() for row in rows))


def read_kinds(directory: Path, name: str = KINDS) -> list[KindRow]:
    return [KindRow.parse(record) for record in read_rows(directory / name)]


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
    fit = read_fit(directory) if (directory / FIT).exists() else []
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
            f"{settings['machine']}, torch {settings['torch']}, commit {settings['revision']}."
            + (
                f" The windows of `{settings['corpus']}` are the ones backbone "
                f"`{settings['windows_of'][:8]}-…` read; this backbone never read it, the "
                "baselines are fitted on that corpus alone, and its channels pass through rows "
                "no pretraining set, " + _rows_text(settings)
                if "windows_of" in settings
                else ""
            )
            + (
                " The encoder is read at its initialisation, before a step of pretraining."
                if settings.get("at_initialisation") == "true"
                else ""
            )
            + (
                f" The rows of the corpus's own channels were then fitted for "
                f"{settings['fit_rows']} steps at {settings['fit_rate']} on the training "
                f"windows under the pretext, every other weight frozen; the loss of a step's "
                f"batch went from {fit[0]:.3f} to {fit[-1]:.3f}."
                if fit
                else ""
            )
            + (
                f" Each batch of the fit was scored {settings['fit_chunk']} windows at a time "
                f"under the batch's masks, its error divided once by the batch's hidden tokens."
                if fit and "fit_chunk" in settings
                else ""
            ),
            corpora_table(read_corpora(directory)),
            kinds_table(read_kinds(directory)),
            *(
                (
                    "On the channels the map names alone (the rows marked apart are every "
                    "other channel of the corpus):",
                    kinds_table(read_kinds(directory, KINDS_MAPPED)),
                )
                if (directory / KINDS_MAPPED).exists()
                else ()
            ),
            "Errors are the run's own reading of the loss on hidden tokens, in normalised units. "
            "Each interval is a bootstrap over validation units under one draw of the masks; a "
            "kind is learnt only when the whole interval of the matched baseline's error less the "
            "model's lies above zero. The matched baseline for the tail is the channel's own line "
            "carried past its last visible token.",
            "Learnt: " + (", ".join(learnt) if learnt else "none") + ".",
        )
    )


def _rows_text(settings: dict[str, str]) -> str:
    rows = settings.get("unset_rows", "drawn")
    if rows == "mean":
        return "set to the mean of the learnt rows."
    if rows.startswith("map"):
        rest = "set to the mean of the learnt rows" if rows == "map-mean" else "drawn"
        return (
            f"the {settings.get('mapped', '?')} the map names copied from the rows learnt for "
            f"their counterparts, the rest {rest}."
        )
    return "drawn as the evaluation draws them."


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
    corpus: str | None = None,
) -> list[CorpusWindows]:
    """Every corpus the backbone read, cut to the diagnostic's windows, from the blocks it names.

    The registry records each input as the manifest the run was pointed at and the checksum of
    the block it read; the block each manifest points at now is held against that checksum.
    Named, ``corpus`` is the one input read.

    Raises:
        ValueError: If a manifest's block is not the block the backbone was pretrained over, or
            if the backbone did not read the corpus named.
    """
    inputs = [stated for stated in backbone.inputs if corpus in (None, stated.corpus)]
    if not inputs:
        raise ValueError(f"backbone {backbone.id} did not read {corpus}")
    corpora = []
    codec = PublishedCorpusManifestJson()
    for stated in inputs:
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


def _backbone_id(text: str, option: str) -> BackboneId:
    try:
        return BackboneId(UUID(text))
    except ValueError as error:
        raise SystemExit(f"{option} must be a backbone's id: {error}") from error


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", metavar="ID", help="a backbone the registry holds ready")
    parser.add_argument(
        "--corpus", metavar="NAME", help="one corpus to read, the backbone's own or another's"
    )
    parser.add_argument(
        "--windows-of",
        metavar="ID",
        help="the backbone whose windows of --corpus are read, where this one never read it",
    )
    parser.add_argument(
        "--unset-rows",
        choices=("drawn", "mean", "map", "map-mean"),
        default="drawn",
        help=(
            "rows of channels the backbone never read: drawn, the mean of the learnt rows, or "
            "copied from their counterparts' learnt rows (--map), the rest drawn or at the mean"
        ),
    )
    parser.add_argument(
        "--map", type=Path, metavar="FILE", help="the counterparts, one per line, for map modes"
    )
    parser.add_argument("--swap", action="store_true", help="read the map the other way round")
    parser.add_argument(
        "--at-initialisation",
        action="store_true",
        help="read the encoder as the run drew it, before a step, instead of the weights kept",
    )
    parser.add_argument(
        "--fit-rows",
        type=int,
        default=0,
        metavar="STEPS",
        help="fit the rows of the corpus's own channels for this many steps, the rest frozen",
    )
    parser.add_argument(
        "--fit-rate",
        type=float,
        default=FIT_RATE,
        metavar="RATE",
        help="the rate Adam fits the rows at",
    )
    parser.add_argument(
        "--fit-chunk",
        type=int,
        default=0,
        metavar="WINDOWS",
        help="score each batch of the fit this many windows at a time, zero for the whole batch",
    )
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
    registry = _registry()
    backbone = registry.get(_backbone_id(arguments.backbone, "--backbone"))
    if backbone.artifact is None:
        raise SystemExit(f"backbone {backbone.id} is not ready: it has no weights")
    # The windows come from this backbone's own inputs unless another's are asked for.
    source = backbone
    mapped: list[int] = []
    if arguments.windows_of is not None:
        if arguments.corpus is None:
            raise SystemExit("--windows-of needs --corpus, the corpus that backbone supplies")
        source = registry.get(_backbone_id(arguments.windows_of, "--windows-of"))
    store = configured_store(Settings())
    configuration = backbone.configuration
    model: PretrainingObjective
    if arguments.at_initialisation and configuration.start is not None:
        raise SystemExit(
            f"backbone {backbone.id} continued the weights of another backbone: its "
            "initialisation is that backbone's weights, read with --backbone and its id"
        )
    if arguments.at_initialisation:
        # The objective drawn as the run drew it, before a step, over the backbone's vocabulary.
        torch.manual_seed(configuration.budget.seed)
        model = ObjectiveShape.of(
            configuration, max(stated.vocabulary_size for stated in backbone.inputs)
        ).build()
    else:
        model = TrainedModel.read(store.get(backbone.artifact)).build()
    if not isinstance(model, MaskedReconstruction):
        raise SystemExit(
            f"backbone {backbone.id} was pretrained to predict states, not values; its "
            "predictions are not scored against the baselines of a value — see "
            "scripts/latent_prediction_report.py"
        )
    if source is not backbone:
        try:
            supplied = supplied_by(source, backbone, arguments.corpus)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        # The rows no pretraining set are drawn from one seed, so a reading repeats.
        torch.manual_seed(VALIDATION_SEED)
        grown(model, max(stated.vocabulary_size for stated in supplied))
        manifest = PublishedCorpusManifestJson().decode(store.get(supplied[0].manifest))
        own = own_channels(manifest.channels, arguments.corpus)
        if arguments.unset_rows in ("mean", "map-mean"):
            unset_rows_to_mean(model, own)
        if arguments.unset_rows.startswith("map"):
            if arguments.map is None:
                raise SystemExit(f"--unset-rows {arguments.unset_rows} needs --map")
            pairs = read_map(arguments.map)
            # The counterparts are rows of the backbone's own chain, which may reach past the
            # corpus's manifest; the corpus's channels come from its own.
            known = {channel.channel_id: channel for channel in vocabulary_of(backbone, store)}
            known.update({channel.channel_id: channel for channel in manifest.channels})
            try:
                mapped = unset_rows_from_counterparts(
                    model, known.values(), own, swapped(pairs) if arguments.swap else pairs
                )
            except ValueError as error:
                raise SystemExit(str(error)) from error
    elif arguments.unset_rows != "drawn":
        raise SystemExit(
            f"--unset-rows {arguments.unset_rows} needs --windows-of: a backbone's own rows are "
            "all set"
        )
    tallied_apart: list[int] = list(mapped)
    if arguments.map is not None and not mapped:
        if arguments.corpus is None:
            raise SystemExit("--map needs --corpus, the corpus whose channels it names")
        pairs = read_map(arguments.map)
        tallied_apart = named_channels(
            corpora_channels_of(source, store, arguments.corpus),
            arguments.corpus,
            swapped(pairs) if arguments.swap else pairs,
        )
    if arguments.fit_rows and source is backbone:
        raise SystemExit("--fit-rows needs --windows-of: a backbone's own rows are all set")
    if arguments.fit_rows < 0:
        raise SystemExit("--fit-rows takes a number of steps, zero for none")
    if arguments.fit_chunk < 0:
        raise SystemExit("--fit-chunk takes a number of windows, zero for the whole batch")
    directory = arguments.out or WORKSPACE / str(backbone.id)
    strategy = configuration.masking
    settings = {
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
    }
    if arguments.corpus is not None:
        settings["corpus"] = arguments.corpus
    if source is not backbone:
        settings["windows_of"] = str(source.id)
        settings["unset_rows"] = arguments.unset_rows
        settings["rows_seed"] = str(VALIDATION_SEED)
        if mapped:
            settings["mapped"] = str(len(mapped))
    if arguments.map is not None:
        settings["map"] = str(arguments.map)
        settings["swap"] = "true" if arguments.swap else "false"
        settings["tallied_apart"] = str(len(tallied_apart))
    if arguments.at_initialisation:
        settings["at_initialisation"] = "true"
    if arguments.fit_rows:
        settings["fit_rows"] = str(arguments.fit_rows)
        settings["fit_rate"] = _number(arguments.fit_rate)
        settings["fit_seed"] = str(FIT_SEED)
        if arguments.fit_chunk:
            settings["fit_chunk"] = str(arguments.fit_chunk)
    write_settings(directory, settings)
    tallies: list[tuple[str, MaskKindTally]] = []
    tallies_mapped: list[tuple[str, MaskKindTally]] = []
    rows = []
    for corpus in corpora_of(
        source,
        store,
        arguments.blocks,
        training_windows=arguments.training_windows,
        validation_windows=arguments.validation_windows,
        corpus=arguments.corpus,
    ):
        started = time.perf_counter()
        if arguments.fit_rows:
            fitted = unset_rows_fitted(
                model,
                own,
                corpus.training,
                strategy,
                configuration.loss,
                device=arguments.device,
                steps=arguments.fit_rows,
                learning_rate=arguments.fit_rate,
                batch_size=arguments.batch_size,
                chunk=arguments.fit_chunk or None,
            )
            write_fit(directory, fitted)
            print(
                f"{corpus.corpus}: rows fitted {len(fitted)} steps, loss {fitted[0]:.3f} to "
                f"{fitted[-1]:.3f}",
                file=sys.stderr,
            )
        apart = [corpus.channels_apart]
        if tallied_apart:
            # Every channel of the corpus but the named ones on the other side: the named
            # channels' tokens then stand alone as the side not apart.
            named = frozenset(tallied_apart)
            others = frozenset(
                channel_id for channel_id in corpus.channel_ids if channel_id not in named
            )
            apart.append(corpus.channels_apart | others)
        found, *rest = diagnose_each(
            model,
            strategy,
            configuration.loss,
            corpus,
            apart,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        tallies += [(corpus.corpus, tally) for tally in found]
        if rest:
            tallies_mapped += [(corpus.corpus, tally) for tally in rest[0]]
        rows.append(CorpusRow.of(corpus, time.perf_counter() - started))
        print(f"{corpus.corpus}: {len(found)} tallies in {rows[-1].seconds:.0f} s", file=sys.stderr)
        write_tallies(directory, tallies)
        write_corpora(directory, rows)
    write_kinds(directory, kinds_of(tallies, resamples=arguments.resamples))
    if tallies_mapped:
        write_tallies(directory, tallies_mapped, TALLIES_MAPPED)
        write_kinds(
            directory, kinds_of(tallies_mapped, resamples=arguments.resamples), KINDS_MAPPED
        )
    print(render(directory))
    print(f"\nstored under {directory}")


if __name__ == "__main__":
    main()
