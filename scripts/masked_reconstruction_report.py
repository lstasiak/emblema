"""Train the objective on the positive control and the spectral probe, and show what it learnt.

The self-supervised objective is judged on three things before any real corpus sees it: that its
loss falls, that its reconstructions look like the signal, and that what it learnt is not what a
trivial baseline already knew. This script does all three on the control corpora, which exist
for exactly this — a corpus whose structure was put there on purpose, so that a model finding
nothing is at fault. It publishes the corpus through the same use cases the command line runs,
trains the encoder with the masked-reconstruction objective on the training units, and measures
on the validation units against the baseline matched to each kind of mask: interpolation within
the channel for blocks and single tokens, a ridge regression from the other channels for a
channel hidden whole. The regressions are fitted on training windows under the masks the strategy
draws, so that they meet their regressors missing as they will be. Beside the matched baselines
stands the strongest linear one on the same inputs, and a least-squares spectrum of the channels
hidden whole says which frequencies the model gives back. The control's signal is too slow for a
window to show more than one of them, so the spectrum is read on ``spectral_probe`` — the dense
control watching faster factors — which the report trains on too.

    uv sync --all-extras
    uv run scripts/masked_reconstruction_report.py
    uv run scripts/masked_reconstruction_report.py --corpus control-b --epochs 10

The training loop here is the smallest that answers the question — no checkpoints, no tracker,
no precision policy; those belong to the run that trains a backbone for real. It warms the
learning rate up and decays it, because a verdict read off a model still circling its minimum is
read off whichever point of the circle the last epoch landed on. The output is markdown, meant to
be pasted under a dated heading in the verification note; the figures are written next to it.
What the run measured is stored as CSV, one directory per run with a line in an index of every
run, and ``AssessReconstructionRun`` turns it into the decision printed under the verdict —
comparing the run with a stored run of the same configuration and half the epochs, which is how it
tells a model that stopped learning from a schedule that stopped it.

Transitional. The training loop, the CSV store and the code digest here stand in for what T-2.3
and T-2.4 build — the training runtime, the experiment tracker and the provenance of a run — and
go when those exist; the diagnostics and the rules they feed already live in the package. What
the runs showed is recorded in ``docs/verification/masked-reconstruction.md``.
"""

import argparse
import hashlib
import io
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

# A report writes files and never opens a window; the backend has to be chosen before pyplot is.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.adapters.synthetic.latent_factor_process import LatentFactorProcess
from emblema.catalog.adapters.synthetic.layouts import CONTROL_PROCESS, LAYOUTS
from emblema.catalog.adapters.synthetic.sensor_layout import SensorLayout
from emblema.catalog.adapters.synthetic.synthetic_corpus_reader import SyntheticCorpusReader
from emblema.catalog.application.use_cases.publish_corpus import PublishCorpusCommand
from emblema.catalog.domain.tokenisation.tokenisation_manifest import TokenisationManifest
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.entrypoints.cli.composition_root import CompositionRoot
from emblema.entrypoints.cli.known_corpora import (
    GENERATED_LICENCE,
    GENERATED_SOURCE,
    KnownCorpora,
    KnownCorpus,
)
from emblema.pretraining.adapters.diagnostics.cross_channel_ridge_baseline import (
    CrossChannelRidgeBaseline,
)
from emblema.pretraining.adapters.diagnostics.linear_interpolation_baseline import (
    LinearInterpolationBaseline,
)
from emblema.pretraining.adapters.diagnostics.own_and_cross_channel_ridge_baseline import (
    OwnAndCrossChannelRidgeBaseline,
)
from emblema.pretraining.adapters.diagnostics.spectral_recovery import SpectralRecovery
from emblema.pretraining.adapters.diagnostics.triviality_diagnostic import TrivialityDiagnostic
from emblema.pretraining.adapters.diagnostics.unit_bootstrap import UnitBootstrap
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.reconstruction_loss import ReconstructionLoss
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.application.use_cases.assess_reconstruction_run import (
    AssessReconstructionRun,
)
from emblema.pretraining.domain.assessment.assessment import Assessment
from emblema.pretraining.domain.assessment.curve import Curve
from emblema.pretraining.domain.assessment.kind_summary import BEYOND_LINEAR, MATCHED_BASELINE
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.assessment.results import Results
from emblema.pretraining.domain.assessment.spectrum import Spectrum
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.exceptions import InvalidLearningRateScheduleError
from emblema.pretraining.domain.learning_rate_schedule import LearningRateSchedule
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.loaders.window_loader import WindowLoader
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.tokens import TokenWindow
from scripts.budget_file import architecture_of, tier_named
from scripts.masked_reconstruction_assessment import (
    assessment_section,
    find_shorter,
    kinds_table,
    store,
)
from scripts.reporting import dated_heading, machine, table
from scripts.spectral_probe import SPECTRAL_PROBE, SPECTRAL_PROCESS
from tests.support.settings import unreachable_store

FIGURES = REPO_ROOT / "docs" / "verification" / "figures"
# What the report trains on when the command line names nothing: both controls, and the probe the
# spectrum is read on.
DEFAULT_CORPORA = ("control-a", "control-b", SPECTRAL_PROBE.name)
RESULTS = REPO_ROOT / "data" / "report" / "results"

# The mixture the plan asks for: blocks and whole channels carry most of the hiding, single tokens
# are the minority ingredient, and together they take close to half of a window.
MIXTURE = MaskingStrategy(channel_rate=0.15, block_rate=0.6, block_span=0.5, token_rate=0.1)
DECODER_LAYERS = 1
# Cycles per window the spectrum is fitted up to, at most. On the control corpora that is more than
# their signal holds: the fastest harmonic of the control process completes about one cycle in a
# window of this length, so every frequency above the first holds noise alone, and the verdict and
# the assessment call the spectrum uninformative there rather than read recovery into it. A sparse
# layout holds fewer tokens per channel than that many coefficients need, so each corpus fits as
# many as it can.
CYCLES_AT_MOST = 6
WINDOW = WindowSpec(length=32.0, stride=12.0)
EXAMPLE_WINDOWS = 3
# One epoch of warmup is a hundred-odd steps on the control corpora — enough for Adam's moment
# estimates to settle before the peak rate — and a floor of one per cent of the peak is where the
# steps are too small to move the verdicts.
WARMUP_EPOCHS = 1
FINAL_LR_FRACTION = 0.01
# Where the model trains. Tier S is declared on MPS in the budget file; everything the diagnostics
# compute stays on the host, because the baselines are numpy and MPS holds no double precision.
DEVICES = ("cpu", "mps", "cuda")
# Epochs per tier and corpus when the command line names none, with how they were measured.
EPOCHS_FILE = Path(__file__).resolve().parent / "masked_reconstruction_epochs.toml"


def default_device() -> str:
    """The accelerator this machine has, the CPU where it has none."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_available(device: str) -> bool:
    if device == "mps":
        return torch.backends.mps.is_available()
    if device == "cuda":
        return torch.cuda.is_available()
    return device == "cpu"


def measured_epochs(tier: str, corpus: str, *, path: Path = EPOCHS_FILE) -> int | None:
    """The epochs measured for ``corpus`` at ``tier``; ``None`` where nothing was measured.

    Raises:
        ValueError: If the file states something other than a positive whole number of epochs.
    """
    with path.open("rb") as handle:
        epochs = tomllib.load(handle).get(tier, {}).get(corpus)
    if epochs is not None and (
        isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
    ):
        raise ValueError(
            f"{path.name}: epochs of {corpus} at tier {tier} must be positive, got {epochs!r}"
        )
    return epochs


@dataclass(frozen=True)
class Run:
    """Everything that decides what the training does, stated once and printed with the result."""

    corpus: str
    tier: str
    shape: EncoderArchitecture | None
    units: int | None
    epochs: int
    batch_size: int
    learning_rate: float
    warmup_epochs: int
    final_lr_fraction: float
    seed: int
    device: str

    @property
    def architecture(self) -> EncoderArchitecture:
        """The tier's shape, unless the run states a smaller one a slow machine can afford."""
        return self.shape if self.shape is not None else architecture_of(tier_named(self.tier))

    @property
    def shape_label(self) -> str:
        if self.shape is None:
            return f"tier {self.tier}"
        return f"tier {self.tier} cut down to {self.shape.width} wide, {self.shape.layers} deep"

    def schedule(self, steps_per_epoch: int) -> LearningRateSchedule:
        """The learning rate over the run, in optimiser steps."""
        return LearningRateSchedule(
            warmup_steps=self.warmup_epochs * steps_per_epoch,
            total_steps=self.epochs * steps_per_epoch,
            final_fraction=self.final_lr_fraction,
        )


@dataclass(frozen=True)
class Published:
    """The control corpus after the pipeline: its manifest and the windows of each split.

    Attributes:
        manifest: What the pipeline recorded about the corpus.
        training: Windows of the training units.
        validation: Windows of the validation units.
        validation_units: The unit each validation window was cut from, in the order of
            ``validation``: a verdict's uncertainty is resampled by unit, not by window.
        noise: Standard deviation of the layout's measurement noise in raw units, where the corpus
            states one.
    """

    manifest: TokenisationManifest
    training: tuple[TokenWindow, ...]
    validation: tuple[TokenWindow, ...]
    validation_units: tuple[str, ...]
    noise: float | None

    @property
    def vocabulary_size(self) -> int:
        return len(self.manifest.scheme.vocabulary)

    def spectral_cycles(self) -> int:
        """Cycles per window the spectrum can be fitted up to on a typical channel of a window."""
        tokens_per_channel = [
            count
            for window in self.validation[:64]
            for count in Counter(
                channel
                for channel, timeless in zip(window.channel_ids, window.timeless, strict=True)
                if not timeless
            ).values()
        ]
        typical = int(np.median(tokens_per_channel)) if tokens_per_channel else 0
        return max(1, min(CYCLES_AT_MOST, SpectralRecovery.cycles_fitting(typical)))

    def noise_variance(self) -> tuple[float, ...] | None:
        """Per vocabulary entry, the variance of the measurement noise in normalised units.

        Indexed by channel identifier, padding at zero. What no predictor can go below on a token of
        that channel. Zero for a channel reported apart, which carries no measurement noise;
        ``None`` for a corpus that states no noise.
        """
        if self.noise is None:
            return None
        scheme = self.manifest.scheme
        apart = self.channels_apart()
        variance = [0.0] * (self.vocabulary_size + 1)
        for entry in scheme.vocabulary.entries_of(self.manifest.corpus):
            statistics = scheme.statistics[entry.channel_id - 1]
            if entry.channel_id not in apart and statistics is not None:
                variance[entry.channel_id] = (self.noise / statistics.std) ** 2
        return tuple(variance)

    def channels_apart(self) -> frozenset[int]:
        """Channels the verdict leaves aside: timeless ones and ones that never vary."""
        scheme = self.manifest.scheme
        apart = set()
        for entry in scheme.vocabulary.entries_of(self.manifest.corpus):
            statistics = scheme.statistics[entry.channel_id - 1]
            if entry.timeless or (statistics is not None and statistics.std == 0.0):
                apart.add(entry.channel_id)
        return frozenset(apart)

    def channel_name(self, channel_id: int) -> str:
        return self.manifest.scheme.vocabulary.entry(channel_id).channel


@dataclass(frozen=True)
class Epoch:
    training_loss: float
    validation_loss: float
    seconds: float


@dataclass(frozen=True)
class Trained:
    model: MaskedReconstruction
    epochs: tuple[Epoch, ...]
    hidden_ratio: float


@dataclass(frozen=True)
class Example:
    """One channel of one validation window, as the figure draws it.

    Attributes:
        window: Position of the window among the validation windows.
        channel: Name of the channel.
        times: Instants of the channel's observed tokens.
        truth: Their values.
        visible: Which of them the model saw.
        of_kind: Which of them were hidden by the kind of mask the example shows — a token hidden by
            another kind is neither visible nor drawn as a prediction here.
        kind: The kind of mask the example shows.
        model: The model's prediction at every token.
        baseline: The matched baseline's.
    """

    window: int
    channel: str
    times: np.ndarray
    truth: np.ndarray
    visible: np.ndarray
    of_kind: np.ndarray
    kind: MaskKind
    model: np.ndarray
    baseline: np.ndarray


@dataclass(frozen=True)
class Diagnosis:
    interpolation_loss: float
    ridge_loss: float
    spectrum_of_model: SpectralRecovery
    spectrum_of_ridge: SpectralRecovery
    examples: tuple[Example, ...]
    tallies: tuple[MaskKindTally, ...]


def corpora() -> tuple[str, ...]:
    """Every corpus the report can train on: the control registry's and the spectral probe."""
    return (*sorted(LAYOUTS), SPECTRAL_PROBE.name)


def generated(corpus: str) -> tuple[LatentFactorProcess, SensorLayout, KnownCorpus]:
    """The process, the layout and the terms a corpus the report trains on is generated from."""
    if corpus == SPECTRAL_PROBE.name:
        return (
            SPECTRAL_PROCESS,
            SPECTRAL_PROBE,
            KnownCorpus(corpus, GENERATED_SOURCE, GENERATED_LICENCE),
        )
    return CONTROL_PROCESS, LAYOUTS[corpus], KnownCorpora.default().named(corpus)


def publish(run: Run, workspace: Path) -> Published:
    process, layout, known = generated(run.corpus)
    if run.units is not None:
        layout = layout.with_dials(units=run.units)
    root = CompositionRoot(
        unreachable_store(),
        corpus_root=workspace / "raw",
        workspace=workspace,
        corpora=InMemoryCorpusRepository(),
        reader=SyntheticCorpusReader(process, layout),
        store=InMemoryArtifactStore(),
    )
    ref = root.services.publish_corpus(
        PublishCorpusCommand(
            name=known.name,
            source=known.source,
            licence=known.licence,
            window=WINDOW,
            validation_fraction=0.25,
            seed=run.seed,
        )
    )
    archive = root.adapters.archive
    manifest = archive.read_manifest(ref)
    validation: list[TokenWindow] = []
    units: list[str] = []
    for unit in manifest.archived.units:
        if unit in manifest.split.validation:
            windows = archive.read_windows(manifest.archived, {unit})
            validation += windows
            units += [unit.value] * len(windows)
    return Published(
        manifest,
        tuple(archive.read_windows(manifest.archived, manifest.split.training)),
        tuple(validation),
        tuple(units),
        noise=layout.noise,
    )


def validation_batches(
    published: Published, run: Run
) -> Iterator[tuple[TokenTensors, TokenMasks, tuple[str, ...]]]:
    """The validation windows in a fixed order under masks drawn once per batch, every time.

    Batches and masks stay on the host, where the masks are drawn — so a run on any device is scored
    under the same masks — and whoever runs the model moves them. Each batch comes with the units
    its windows were cut from, counted off the windows actually delivered rather than computed from
    the batch size.
    """
    loader = WindowLoader(
        published.validation, batch_size=run.batch_size, seed=run.seed, shuffle=False
    )
    masking = TokenMasking(MIXTURE)
    offset = 0
    for index, batch in enumerate(loader.batches_of(0)):
        units = published.validation_units[offset : offset + batch.batch_size]
        offset += batch.batch_size
        yield (
            batch,
            masking.draw(batch, torch.Generator().manual_seed(run.seed * 1_000 + index)),
            units,
        )


def masked_training_batches(
    published: Published, run: Run
) -> list[tuple[TokenTensors, TokenMasks]]:
    """The training windows in a fixed order under masks drawn once, to fit the linear baselines.

    Drawn by the strategy the model trains under, from a generator of the run's seed, so that the
    baselines learn from windows as incomplete as the ones they will be asked about.
    """
    loader = WindowLoader(
        published.training, batch_size=run.batch_size, seed=run.seed, shuffle=False
    )
    masking = TokenMasking(MIXTURE)
    draws = torch.Generator().manual_seed(run.seed)
    return [(batch, masking.draw(batch, draws)) for batch in loader.batches_of(0)]


def train(published: Published, run: Run) -> Trained:
    torch.manual_seed(run.seed)
    # Built on the host, so that the seed gives the same initial weights whatever the device.
    model = MaskedReconstruction(
        SetEncoder.for_vocabulary(run.architecture, published.vocabulary_size),
        decoder_layers=DECODER_LAYERS,
    ).to(run.device)
    loss = ReconstructionLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=run.learning_rate)
    loader = WindowLoader(published.training, batch_size=run.batch_size, seed=run.seed)
    schedule = run.schedule(steps_per_epoch=len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule.factor)
    masking = TokenMasking(MIXTURE)
    draws = torch.Generator().manual_seed(run.seed)
    epochs = []
    for epoch in range(run.epochs):
        started = time.perf_counter()
        model.train()
        seen = []
        for on_host in loader.batches_of(epoch):
            masks = masking.draw(on_host, draws).to(run.device)
            batch = on_host.to(run.device)
            optimiser.zero_grad()
            step = loss(model(batch, masks), batch, masks)
            step.backward()
            optimiser.step()
            scheduler.step()
            seen.append(step.item())
        validation, _ = evaluate(model, published, run)
        epochs.append(Epoch(float(np.mean(seen)), validation, time.perf_counter() - started))
    _, ratio = evaluate(model, published, run)
    return Trained(model.eval(), tuple(epochs), ratio)


def evaluate(model: MaskedReconstruction, published: Published, run: Run) -> tuple[float, float]:
    """The validation loss under the fixed masks, and the share of observed tokens they hide."""
    model.eval()
    loss = ReconstructionLoss()
    total, hidden, observed = 0.0, 0, 0
    with torch.no_grad():
        for on_host, masks_on_host, _ in validation_batches(published, run):
            batch, masks = on_host.to(run.device), masks_on_host.to(run.device)
            scored = int(masks.hidden.sum())
            total += float(loss(model(batch, masks), batch, masks)) * scored
            hidden += scored
            observed += int((~batch.padding_mask).sum())
    return total / max(hidden, 1), hidden / max(observed, 1)


def diagnose(published: Published, trained: Trained, run: Run) -> Diagnosis:
    fit = masked_training_batches(published, run)
    interpolation = LinearInterpolationBaseline()
    ridge = CrossChannelRidgeBaseline.fitted(fit, vocabulary_size=published.vocabulary_size)
    combined = OwnAndCrossChannelRidgeBaseline.fitted(
        fit, vocabulary_size=published.vocabulary_size
    )
    apart = published.channels_apart()
    triviality = TrivialityDiagnostic(
        channels_apart=apart, noise_variance=published.noise_variance()
    )
    cycles = published.spectral_cycles()
    spectrum_of_model = SpectralRecovery.up_to(cycles)
    spectrum_of_ridge = SpectralRecovery.up_to(cycles)
    loss = ReconstructionLoss()
    interpolation_total, ridge_total, hidden, windows = 0.0, 0.0, 0, 0
    examples: list[Example] = []
    with torch.no_grad():
        for batch, masks, units in validation_batches(published, run):
            predicted = trained.model(batch.to(run.device), masks.to(run.device)).cpu()
            interpolated = interpolation.predict(batch, masks)
            regressed = ridge.predict(batch, masks)
            triviality = triviality.observe(
                batch,
                masks,
                units,
                model=predicted,
                interpolation=interpolated,
                ridge=regressed,
                combined=combined.predict(batch, masks),
            )
            spectrum_of_model = spectrum_of_model.observe(batch, masks, predicted)
            spectrum_of_ridge = spectrum_of_ridge.observe(batch, masks, regressed)
            scored = int(masks.hidden.sum())
            interpolation_total += float(loss(interpolated, batch, masks)) * scored
            ridge_total += float(loss(regressed, batch, masks)) * scored
            hidden += scored
            if len(examples) < EXAMPLE_WINDOWS * len(MaskKind):
                examples += pick_examples(
                    batch,
                    masks,
                    predicted,
                    interpolated,
                    regressed,
                    first_window=windows,
                    channel_name=published.channel_name,
                    apart=apart,
                )
            windows += batch.batch_size
    return Diagnosis(
        interpolation_loss=interpolation_total / max(hidden, 1),
        ridge_loss=ridge_total / max(hidden, 1),
        spectrum_of_model=spectrum_of_model,
        spectrum_of_ridge=spectrum_of_ridge,
        examples=tuple(examples[: EXAMPLE_WINDOWS * len(MaskKind)]),
        tallies=triviality.tallies(),
    )


def pick_examples(
    batch: TokenTensors,
    masks: TokenMasks,
    predicted: torch.Tensor,
    interpolated: torch.Tensor,
    regressed: torch.Tensor,
    *,
    first_window: int,
    channel_name: Callable[[int], str],
    apart: frozenset[int],
) -> list[Example]:
    """One channel per kind of mask from the first windows of the batch, for the figure.

    Args:
        batch: The windows.
        masks: What was hidden in them.
        predicted: The model's prediction.
        interpolated: The interpolation baseline's.
        regressed: The cross-channel regression's.
        first_window: Position of the batch's first window among the validation windows.
        channel_name: The name of a channel identifier.
        apart: Channels the verdict leaves aside, which the figure leaves aside too.
    """
    examples: list[Example] = []
    for row in range(min(batch.batch_size, EXAMPLE_WINDOWS)):
        observed = ~batch.padding_mask[row]
        visible = observed & ~masks.hidden[row]
        for kind in MaskKind:
            of_kind = masks.of_kind(kind)[row] & observed
            channels = [
                int(channel)
                for channel in torch.unique(batch.channel_ids[row][of_kind])
                if int(channel) not in apart
            ]
            if not channels:
                continue
            tokens = observed & (batch.channel_ids[row] == channels[0])
            baseline = regressed if kind is MaskKind.CHANNEL else interpolated
            examples.append(
                Example(
                    window=first_window + row,
                    channel=channel_name(channels[0]),
                    times=batch.timestamps[row][tokens].numpy(),
                    truth=batch.features[row, :, 0][tokens].numpy(),
                    visible=visible[tokens].numpy(),
                    of_kind=of_kind[tokens].numpy(),
                    kind=kind,
                    model=predicted[row][tokens].numpy(),
                    baseline=baseline[row][tokens].numpy(),
                )
            )
    return examples


def code_digest() -> str:
    """A digest of the repository code this process has loaded: the package, scripts and support.

    Two runs are compared only when their code agrees, uncommitted changes included, so that a
    shorter run from before a change to a baseline is never taken for a run of this configuration.
    Read off the modules actually loaded rather than off every file, so that editing a script the
    run never imports does not set its stored runs apart; by the time a run is stored, everything
    that computed its numbers has been imported.
    """
    roots = [REPO_ROOT / "src", REPO_ROOT / "scripts", REPO_ROOT / "tests"]
    files = set()
    for module in list(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        if path.suffix == ".py" and any(path.is_relative_to(root) for root in roots):
            files.add(path)
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def results_of(run: Run, published: Published, trained: Trained, diagnosis: Diagnosis) -> Results:
    """What the run measured, in the form the assessment reads and the CSV files store."""
    architecture = run.architecture
    settings = {
        "corpus": run.corpus,
        "date": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "code": code_digest(),
        "machine": machine(),
        "python": sys.version.split()[0],
        "torch": version("torch"),
        "tier": run.tier,
        "shape": (
            f"{architecture.width},{architecture.heads},{architecture.layers},"
            f"{architecture.feedforward_width}"
        ),
        "decoder_layers": str(DECODER_LAYERS),
        "units": "" if run.units is None else str(run.units),
        "training_windows": str(len(published.training)),
        "validation_windows": str(len(published.validation)),
        "validation_units": str(len(set(published.validation_units))),
        "window_length": f"{WINDOW.length:g}",
        "window_stride": f"{WINDOW.stride:g}",
        "batch_size": str(run.batch_size),
        "learning_rate": f"{run.learning_rate:g}",
        "warmup_epochs": str(run.warmup_epochs),
        "final_lr_fraction": f"{run.final_lr_fraction:g}",
        "seed": str(run.seed),
        "device": run.device,
        "noise": "" if published.noise is None else f"{published.noise:g}",
    }
    model, ridge = diagnosis.spectrum_of_model, diagnosis.spectrum_of_ridge
    return Results(
        settings=settings,
        strategy=MIXTURE,
        realised_ratio=trained.hidden_ratio,
        curve=Curve(
            training=tuple(epoch.training_loss for epoch in trained.epochs),
            validation=tuple(epoch.validation_loss for epoch in trained.epochs),
            seconds=tuple(epoch.seconds for epoch in trained.epochs),
        ),
        tallies=diagnosis.tallies,
        spectrum=Spectrum(
            truth=model.truth_energy,
            model_residual=model.residual_energy,
            ridge_residual=ridge.residual_energy,
            fitted=model.fitted,
            skipped=model.skipped,
        ),
    )


def heading(run: Run, published: Published, trained: Trained) -> str:
    architecture = run.architecture
    corpus = (
        f"{run.corpus}, {len(published.training)} training and {len(published.validation)} "
        f"validation windows of {WINDOW.length:g} steps, stride {WINDOW.stride:g}"
    )
    if run.units:
        corpus += f", cut to {run.units} units"
    encoder = (
        f"{run.shape_label}: {architecture.width} wide, {architecture.heads} heads, "
        f"{architecture.layers} blocks, "
        f"{architecture.parameter_count(published.vocabulary_size):,} parameters; "
        f"decoder of {DECODER_LAYERS} block"
    )
    masking = (
        f"channel {MIXTURE.channel_rate:g}, block {MIXTURE.block_rate:g} over "
        f"{MIXTURE.block_span:g} of the window, token {MIXTURE.token_rate:g}; "
        f"expected {MIXTURE.expected_ratio:.1%}, realised {trained.hidden_ratio:.1%} "
        "of observed tokens"
    )
    training = (
        f"{run.epochs} epochs, batch {run.batch_size}, Adam at a peak of {run.learning_rate:g}, "
        f"{run.warmup_epochs} epoch(s) of warmup then cosine decay to "
        f"{run.final_lr_fraction:g} of the peak, seed {run.seed}, {run.device.upper()}"
    )
    rows = [
        ("Machine", machine()),
        ("Python", sys.version.split()[0]),
        ("torch", version("torch")),
        ("Corpus", corpus),
        (
            "Vocabulary",
            f"{published.vocabulary_size} channels, {len(published.channels_apart())} apart",
        ),
        ("Encoder", encoder),
        ("Masking", masking),
        ("Training", training),
    ]
    return f"{dated_heading()} — {run.corpus}\n\n" + table(("", ""), rows)


def loss_section(trained: Trained, diagnosis: Diagnosis) -> str:
    rows = [
        (
            str(index + 1),
            f"{epoch.training_loss:.4f}",
            f"{epoch.validation_loss:.4f}",
            f"{epoch.seconds:.0f}",
        )
        for index, epoch in enumerate(trained.epochs)
    ]
    first, last = trained.epochs[0], trained.epochs[-1]
    factor = first.validation_loss / max(last.validation_loss, 1e-12)
    summary = (
        f"Validation loss went from {first.validation_loss:.4f} to {last.validation_loss:.4f} "
        f"(÷{factor:.1f}); over the same hidden tokens the interpolation baseline scores "
        f"{diagnosis.interpolation_loss:.4f} and the ridge baseline, fitted under the same "
        f"masking, {diagnosis.ridge_loss:.4f}."
    )
    return (
        "### Loss\n\n"
        + table(("Epoch", "Training", "Validation", "Seconds"), rows)
        + "\n\n"
        + summary
    )


def triviality_section(assessment: Assessment) -> str:
    return "### Triviality per kind of mask\n\n" + kinds_table(assessment.summaries)


def spectral_section(diagnosis: Diagnosis) -> str:
    model, ridge = diagnosis.spectrum_of_model, diagnosis.spectrum_of_ridge
    total = sum(model.truth_energy) or 1.0
    rows = [
        (
            str(k + 1),
            f"{truth:.3g}",
            f"{truth / total:.1%}",
            f"{recovered:.2f}",
            f"{by_ridge:.2f}",
        )
        for k, (truth, recovered, by_ridge) in enumerate(
            zip(model.truth_energy, model.recovered(), ridge.recovered(), strict=True)
        )
    ]
    return (
        "### Spectral recovery of channels hidden whole\n\n"
        + table(
            ("Cycles / window", "Truth energy", "Share", "Model recovers", "Ridge recovers"), rows
        )
        + f"\n\n{model.fitted} channel-windows fitted, {model.skipped} too short to fit. A share "
        "below zero means the residual holds more energy at that frequency than the truth does — "
        "where the truth holds next to none, that is the noise the model cannot recover."
    )


def verdict_section(run: Run, assessment: Assessment) -> str:
    """What the run shows, in one place and one voice with the assessment beneath it."""
    results = assessment.results
    first, last = results.curve.validation[0], results.curve.validation[-1]
    lines = [
        f"- Loss {'fell' if last < first else 'did not fall'}: "
        f"{first:.4f} → {last:.4f} on validation."
    ]
    for summary in assessment.summaries:
        if summary.apart:
            continue
        line = (
            f"- `{summary.kind.value}` masks: model {summary.model_error:.4f} against "
            f"{MATCHED_BASELINE[summary.kind]} {summary.matched_error:.4f}, excess "
            f"{summary.matched_excess} — {summary.verdict.value}"
        )
        if summary.kind in BEYOND_LINEAR:
            line += (
                f"; against the linear baseline {summary.linear_error:.4f}, "
                f"{summary.linear_excess} — {summary.linear_excess.verdict.value}"
            )
        lines.append(line + ".")
    lines.append(
        "- Triviality diagnostic "
        + (
            "**negative**: every kind of mask the strategy draws beats its matched baseline "
            "beyond the bootstrap's doubt."
            if assessment.triviality_negative
            else "**positive** for at least one kind: see the table and the assessment."
        )
    )
    lines.append("- " + spectrum_verdict(results.spectrum))
    lines.append(
        f"- Figures: `masked-reconstruction-{run.corpus}-loss.png`, `…-windows.png`, "
        "`…-diagnostics.png`."
    )
    return "### Verdict\n\n" + "\n".join(lines)


def spectrum_verdict(spectrum: Spectrum) -> str:
    if spectrum.fitted == 0:
        return "Spectrum: no channel hidden whole had tokens enough to fit."
    informative = spectrum.informative()
    if len(informative) <= 1:
        return (
            f"Spectrum: uninformative on this corpus — the truth holds {spectrum.shares()[0]:.1%} "
            "of its energy at one cycle per window and noise alone above it, so what the model "
            "recovers cannot tell structure from smoothness."
        )
    recovered = spectrum.recovered(spectrum.model_residual)
    lost = [k for k in informative if recovered[k - 1] < 0.5]
    return (
        f"Spectrum: the truth holds its energy at {', '.join(map(str, informative))} cycle(s) per "
        f"window of {len(spectrum.truth)} fitted; the model recovers "
        + ", ".join(f"{recovered[k - 1]:.0%} at {k}" for k in informative)
        + (
            f" — it loses the signal at {', '.join(map(str, lost))}."
            if lost
            else "; every frequency the signal has is recovered."
        )
    )


def render(
    run: Run, published: Published, trained: Trained, diagnosis: Diagnosis, assessment: Assessment
) -> str:
    return "\n\n".join(
        [
            heading(run, published, trained),
            loss_section(trained, diagnosis),
            triviality_section(assessment),
            spectral_section(diagnosis),
            verdict_section(run, assessment),
            assessment_section(assessment),
        ]
    )


def draw_figures(
    run: Run, trained: Trained, diagnosis: Diagnosis, assessment: Assessment, directory: Path
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"masked-reconstruction-{run.corpus}"

    epochs = np.arange(1, len(trained.epochs) + 1)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(epochs, [e.training_loss for e in trained.epochs], marker="o", label="training")
    axis.plot(epochs, [e.validation_loss for e in trained.epochs], marker="s", label="validation")
    axis.axhline(diagnosis.interpolation_loss, linestyle="--", color="grey", label="interpolation")
    axis.axhline(diagnosis.ridge_loss, linestyle=":", color="grey", label="ridge")
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("mean squared error on hidden tokens")
    axis.set_title(f"{run.corpus}: masked reconstruction, {run.shape_label}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / f"{stem}-loss.png", dpi=120)
    plt.close(figure)

    examples = diagnosis.examples
    if examples:
        figure, axes = plt.subplots(len(examples), 1, figsize=(8, 2.2 * len(examples)), sharex=True)
        for axis, example in zip(np.atleast_1d(axes), examples, strict=False):
            axis.plot(example.times, example.truth, color="black", linewidth=1, label="truth")
            axis.scatter(
                example.times[example.visible],
                example.truth[example.visible],
                s=12,
                color="black",
                label="visible",
            )
            axis.scatter(
                example.times[example.of_kind],
                example.model[example.of_kind],
                marker="x",
                color="tab:red",
                label="model",
            )
            axis.scatter(
                example.times[example.of_kind],
                example.baseline[example.of_kind],
                marker="^",
                s=14,
                color="tab:blue",
                label="matched baseline",
            )
            axis.set_ylabel(
                f"w{example.window} {example.channel}\n{example.kind.value}", fontsize=8
            )
        np.atleast_1d(axes)[0].legend(fontsize=7, ncol=4, loc="upper right")
        np.atleast_1d(axes)[-1].set_xlabel("position in window")
        figure.suptitle(
            f"{run.corpus}: tokens hidden by each kind of mask, model against the matched baseline"
        )
        figure.tight_layout()
        figure.savefig(directory / f"{stem}-windows.png", dpi=120)
        plt.close(figure)

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    judged = [summary for summary in assessment.summaries if not summary.apart]
    positions = np.arange(len(judged))
    left.bar(positions - 0.27, [s.model_error for s in judged], width=0.27, label="model")
    left.bar(positions, [s.matched_error for s in judged], width=0.27, label="matched baseline")
    left.bar(positions + 0.27, [s.linear_error for s in judged], width=0.27, label="linear")
    left.set_xticks(positions, [s.kind.value for s in judged])
    left.set_ylabel("mean squared error")
    left.set_title("triviality per kind of mask")
    left.legend()
    spectrum = diagnosis.spectrum_of_model
    frequencies = np.arange(1, spectrum.cycles + 1)
    right.plot(frequencies, spectrum.truth_energy, marker="o", color="black", label="truth")
    right.plot(
        frequencies, spectrum.residual_energy, marker="x", color="tab:red", label="model residual"
    )
    right.plot(
        frequencies,
        diagnosis.spectrum_of_ridge.residual_energy,
        marker="^",
        color="tab:blue",
        label="ridge residual",
    )
    right.set_yscale("log")
    right.set_xlabel("cycles per window")
    right.set_ylabel("energy over channel-windows hidden whole")
    right.set_title("spectrum of the truth and of what each method leaves")
    right.legend()
    figure.tight_layout()
    figure.savefig(directory / f"{stem}-diagnostics.png", dpi=120)
    plt.close(figure)


def shape_of(spec: str) -> EncoderArchitecture:
    """``WIDTH,HEADS,LAYERS,FEEDFORWARD`` as an architecture, with the tier's time frequencies."""
    width, heads, layers, feedforward = (int(part) for part in spec.split(","))
    return EncoderArchitecture(
        width=width,
        heads=heads,
        layers=layers,
        feedforward_width=feedforward,
        time_frequencies=architecture_of(tier_named("S")).time_frequencies,
    )


def shown(path: Path) -> Path:
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", choices=corpora())
    parser.add_argument("--tier", default="S", choices=["S", "M"])
    parser.add_argument(
        "--shape",
        type=shape_of,
        default=None,
        help="WIDTH,HEADS,LAYERS,FEEDFORWARD to build a smaller encoder than the tier's, where "
        "the tier's is too slow for the machine; the time frequencies stay the tier's",
    )
    parser.add_argument("--units", type=int, default=None, help="cut the layout to this many units")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="epochs for every corpus named; by default each corpus's measured budget",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="the peak rate")
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument(
        "--final-lr-fraction",
        type=float,
        default=FINAL_LR_FRACTION,
        help="the rate the decay ends at, as a fraction of the peak; 1 with no warmup is constant",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default=default_device(),
        help="where the model trains; runs on different devices are never compared",
    )
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT / "data" / "report")
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help="directory each run is stored under as CSV, beside an index of every run",
    )
    arguments = parser.parse_args(argv)
    if not device_available(arguments.device):
        parser.error(f"device {arguments.device} is not available on this machine")
    runs = []
    for corpus in arguments.corpus or DEFAULT_CORPORA:
        epochs = (
            arguments.epochs
            if arguments.epochs is not None
            else measured_epochs(arguments.tier, corpus)
        )
        if epochs is None:
            parser.error(
                f"no measured epochs for {corpus} at tier {arguments.tier}: name them with --epochs"
            )
        run = Run(
            corpus=corpus,
            tier=arguments.tier,
            shape=arguments.shape,
            units=arguments.units,
            epochs=epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            warmup_epochs=arguments.warmup_epochs,
            final_lr_fraction=arguments.final_lr_fraction,
            seed=arguments.seed,
            device=arguments.device,
        )
        try:
            # Whether the schedule holds does not depend on how many steps an epoch has.
            run.schedule(steps_per_epoch=1)
        except InvalidLearningRateScheduleError as error:
            parser.error(str(error))
        runs.append(run)
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    assess = AssessReconstructionRun(UnitBootstrap())
    sections = []
    for run in runs:
        published = publish(run, arguments.workspace / run.corpus)
        trained = train(published, run)
        diagnosis = diagnose(published, trained, run)
        results = results_of(run, published, trained, diagnosis)
        assessment = assess(results, find_shorter(results, arguments.results))
        draw_figures(run, trained, diagnosis, assessment, arguments.figures)
        stored = store(assessment, arguments.results)
        sections.append(
            render(run, published, trained, diagnosis, assessment)
            + f"\n\nFigures in `{shown(arguments.figures)}`; the run is stored in "
            f"`{shown(stored)}`."
        )
    if isinstance(sys.stdout, io.TextIOWrapper):
        # The tables use ×, → and —; the note this output is pasted into is UTF-8 and LF-only.
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
