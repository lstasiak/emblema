"""Train the objective on the control corpora and the spectral probe, and assess what it learnt.

Each experiment in ``experiments/`` names a corpus and everything a run of it does. The corpus is
published through the command line's use cases, trained by the training runtime on its training
units and scored on its validation units against the baselines of each kind of mask and the
spectrum of its channels hidden whole. Each run is stored as CSV, assessed against its stored run
with half the epochs, printed as markdown, and its figures are drawn from the files it stored —
never from what is still in memory, so a change of style costs a redraw and not a run.

    uv sync --all-extras
    uv run scripts/masked_reconstruction_report.py
    uv run scripts/masked_reconstruction_report.py --experiment control-b-s
    uv run scripts/masked_reconstruction_report.py --figures-only data/report/results/<run>

A run's figures are drawn into its own stored directory; the ones a note shows are drawn there on
purpose, with ``--figures docs/verification/figures``. What the runs showed is in
``docs/verification/masked-reconstruction.md``.
"""

import argparse
import hashlib
import io
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
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

from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.adapters.synthetic.latent_factor_process import LatentFactorProcess
from emblema.catalog.adapters.synthetic.layouts import CONTROL_PROCESS, LAYOUTS
from emblema.catalog.adapters.synthetic.sensor_layout import SensorLayout
from emblema.catalog.adapters.synthetic.synthetic_corpus_reader import SyntheticCorpusReader
from emblema.catalog.application.use_cases.publish_corpus import PublishCorpusCommand
from emblema.catalog.domain.tokenisation.split_policy import SeededSplit
from emblema.catalog.domain.tokenisation.tokenisation_manifest import TokenisationManifest
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.config.compute_tiers import ComputeTiers
from emblema.entrypoints.cli.publish_corpus.composition_root import CompositionRoot
from emblema.entrypoints.cli.publish_corpus.known_corpora import (
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
from emblema.pretraining.adapters.encoder.tier_architecture import architecture_of
from emblema.pretraining.adapters.experiments.experiment_file import ExperimentFile
from emblema.pretraining.adapters.in_memory.experiment_tracker import InMemoryExperimentTracker
from emblema.pretraining.adapters.mlflow.mlflow_experiment_tracker import MlflowExperimentTracker
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.reconstruction_loss import ReconstructionLoss
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.adapters.training.devices import available_device
from emblema.pretraining.adapters.training.torch_training_runtime import TorchTrainingRuntime
from emblema.pretraining.application.use_cases.assess_reconstruction_run import (
    AssessReconstructionRun,
)
from emblema.pretraining.application.use_cases.pretrain_backbone import (
    PretrainBackbone,
    PretrainBackboneCommand,
)
from emblema.pretraining.domain.assessment.assessment import Assessment
from emblema.pretraining.domain.assessment.curve import Curve
from emblema.pretraining.domain.assessment.kind_summary import BEYOND_LINEAR, MATCHED_BASELINE
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.assessment.results import Results
from emblema.pretraining.domain.assessment.spectrum import Spectrum
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.exceptions import InvalidTrainingBudgetError
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from emblema.pretraining.ports.experiment_tracker import ExperimentTracker
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.loaders.window_loader import WindowLoader
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow
from scripts.masked_reconstruction_assessment import (
    Example,
    RunFigures,
    assessment_section,
    find_shorter,
    kinds_table,
    run_name,
    store,
)
from scripts.masked_reconstruction_figures import drawn_from, figures_of
from scripts.reporting import dated_heading, machine, table
from scripts.spectral_probe import SPECTRAL_PROBE, SPECTRAL_PROCESS
from scripts.vocabulary import channel_names

# Where the experiments are written down: one file per experiment, and a run is reproduced by
# naming one.
EXPERIMENTS = REPO_ROOT / "experiments"
RESULTS = REPO_ROOT / "data" / "report" / "results"

# Cycles per window the spectrum is fitted up to, at most. On the control corpora that is more than
# their signal holds: the fastest harmonic of the control process completes about one cycle in a
# window of this length, so every frequency above the first holds noise alone, and the verdict and
# the assessment call the spectrum uninformative there rather than read recovery into it. A sparse
# layout holds fewer tokens per channel than that many coefficients need, so each corpus fits as
# many as it can.
CYCLES_AT_MOST = 6
WINDOW = WindowSpec(length=32.0, stride=12.0)
EXAMPLE_WINDOWS = 3
# Where the model trains. Everything the diagnostics compute stays on the host, because the
# baselines are numpy and MPS holds no double precision.
DEVICES = ("cpu", "mps", "cuda")


def device_available(device: str) -> bool:
    if device == "mps":
        return torch.backends.mps.is_available()
    if device == "cuda":
        return torch.cuda.is_available()
    return device == "cpu"


def experiments(directory: Path = EXPERIMENTS) -> dict[str, ExperimentFile]:
    """Every experiment stated in ``directory`` over one corpus this report can generate."""
    stated = (ExperimentFile.load(path) for path in sorted(directory.glob("*.toml")))
    return {
        file.name: file
        for file in stated
        if len(file.corpora) == 1 and file.corpora[0] in corpora()
    }


@dataclass(frozen=True)
class Run:
    """One run of the report: the experiment it follows and where it is run.

    The experiment says what is done and is reproduced by running the same file; what is left here
    is what belongs to the machine and to this report — the device, and a corpus cut short for a
    smoke run.

    Attributes:
        configuration: What the experiment states.
        corpus: The corpus the experiment reads, which this report generates.
        units: Units the layout is cut to, where a run is meant to be quick.
        device: Where the arithmetic happens.
    """

    configuration: ExperimentConfiguration
    corpus: str
    units: int | None
    device: str

    @property
    def epochs(self) -> int:
        return self.configuration.budget.epochs

    @property
    def seed(self) -> int:
        return self.configuration.budget.seed

    @property
    def batch_size(self) -> int:
        return self.configuration.budget.batch_size

    @property
    def architecture(self) -> EncoderArchitecture:
        return self.configuration.architecture

    @property
    def shape_label(self) -> str:
        """The tier the run declares, and whether its experiment changed the shape that tier states.

        Compared by value rather than by whether the file has a shape section: a section that
        repeats the tier's own numbers still trains the tier.
        """
        tier = self.configuration.tier
        if self.architecture == architecture_of(ComputeTiers.load().profile(tier)):
            return f"tier {tier}"
        return f"tier {tier} with its shape overridden"


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
class Trained:
    """What a run of the experiment produced: the model it stored, read back, and its epochs.

    Attributes:
        model: The model the run stored, read back from the store.
        epochs: What each epoch measured.
        hidden_ratio: Share of observed validation tokens the masks hid.
        name: What the tracker calls the run.
        started: When the run started, which its name and its stored date are read off.
        backbone: The weights the run stored; the tracker keeps their checksum as a tag.
    """

    model: MaskedReconstruction
    epochs: tuple[EpochOutcome, ...]
    hidden_ratio: float
    name: str
    started: datetime
    backbone: ArtifactRef


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
    root = CompositionRoot.over(
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
            split=SeededSplit(0.25, run.seed),
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
    # The share the experiment states, over the training units alone: the run is scored on the
    # whole validation side whatever it trained on.
    chosen = set(
        run.configuration.corpus_share.select(
            tuple(sorted(unit.value for unit in manifest.split.training))
        )
    )
    training = {unit for unit in manifest.split.training if unit.value in chosen}
    return Published(
        manifest,
        tuple(archive.read_windows(manifest.archived, training)),
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
    masking = TokenMasking(run.configuration.masking)
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
    masking = TokenMasking(run.configuration.masking)
    draws = torch.Generator().manual_seed(run.seed)
    return [(batch, masking.draw(batch, draws)) for batch in loader.batches_of(0)]


def train(
    published: Published,
    run: Run,
    workspace: Path,
    tracker: ExperimentTracker,
    started: datetime,
) -> Trained:
    """Train the experiment through the use case, and read back the model it stored.

    The run writes its checkpoints and its weights to a store under the workspace, and the model
    the diagnostics score is the one read back from it: what is diagnosed is then what was kept,
    not an object that happened to survive in memory. It is tracked under the name its stored
    directory takes, both read off ``started``, so the two are found from each other.
    """
    store = LocalDirectoryArtifactStore(workspace / "artifacts")
    runtime = TorchTrainingRuntime(store, device=run.device)
    name = run_name(run.corpus, started)
    outcome = PretrainBackbone(runtime, tracker)(
        PretrainBackboneCommand(
            configuration=run.configuration,
            mixture=TrainingMixture.of(
                TrainingCorpus(
                    name=run.corpus,
                    checksum=published.manifest.archived.block.checksum,
                    training=published.training,
                    validation=published.validation,
                    channels=channel_names(published.manifest.scheme.vocabulary),
                )
            ),
            run=name,
        )
    )
    return Trained(
        model=runtime.restore(outcome.backbone).to(run.device),
        epochs=outcome.epochs,
        hidden_ratio=outcome.hidden_ratio,
        name=name,
        started=started,
        backbone=outcome.backbone,
    )


def diagnose(published: Published, trained: Trained, run: Run) -> Diagnosis:
    fit = masked_training_batches(published, run)
    interpolation = LinearInterpolationBaseline()
    ridge = CrossChannelRidgeBaseline.fitted(fit, vocabulary_size=published.vocabulary_size)
    combined = OwnAndCrossChannelRidgeBaseline.fitted(
        fit, vocabulary_size=published.vocabulary_size
    )
    apart = published.channels_apart()
    triviality = TrivialityDiagnostic(
        run.configuration.loss, channels_apart=apart, noise_variance=published.noise_variance()
    )
    cycles = published.spectral_cycles()
    spectrum_of_model = SpectralRecovery.up_to(cycles)
    spectrum_of_ridge = SpectralRecovery.up_to(cycles)
    loss = ReconstructionLoss(run.configuration.loss)
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
    """A digest of the repository code this process has loaded: the package and the scripts.

    Two runs are compared only when their code agrees, uncommitted changes included, so that a
    shorter run from before a change to a baseline is never taken for a run of this configuration.
    Read off the modules actually loaded rather than off every file, so that editing a script the
    run never imports does not set its stored runs apart; by the time a run is stored, everything
    that computed its numbers has been imported.
    """
    roots = [REPO_ROOT / "src", REPO_ROOT / "scripts"]
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
    configuration = run.configuration
    architecture = configuration.architecture
    settings = {
        "corpus": run.corpus,
        "experiment": configuration.name,
        "date": f"{trained.started:%Y-%m-%d %H:%M:%S}",
        # Which tracked run this is, by name and by what it left behind: a tracker's names repeat,
        # the checksum of a run's weights does not.
        "tracked_run": trained.name,
        "backbone_checksum": str(trained.backbone.checksum),
        "code": code_digest(),
        "machine": machine(),
        "python": sys.version.split()[0],
        "torch": version("torch"),
        "tier": str(configuration.tier),
        "corpus_fraction": f"{configuration.corpus_fraction:g}",
        "shape": (
            f"{architecture.width},{architecture.heads},{architecture.layers},"
            f"{architecture.feedforward_width}"
        ),
        # Read from the experiment, which the code digest does not cover, so stated here for two
        # runs of different time encodings never to count as one configuration.
        "time_frequencies": str(architecture.time_frequencies),
        "decoder_layers": str(configuration.decoder_layers),
        "dropout": f"{configuration.dropout:g}",
        "precision": str(configuration.precision),
        "units": "" if run.units is None else str(run.units),
        "training_windows": str(len(published.training)),
        "validation_windows": str(len(published.validation)),
        "validation_units": str(len(set(published.validation_units))),
        "window_length": f"{WINDOW.length:g}",
        "window_stride": f"{WINDOW.stride:g}",
        "batch_size": str(configuration.budget.batch_size),
        "accumulation_steps": str(configuration.budget.accumulation_steps),
        "learning_rate": f"{configuration.budget.learning_rate:g}",
        "warmup_epochs": str(configuration.budget.warmup_epochs),
        "final_lr_fraction": f"{configuration.budget.final_lr_fraction:g}",
        "seed": str(configuration.budget.seed),
        "device": run.device,
        "noise": "" if published.noise is None else f"{published.noise:g}",
    }
    model, ridge = diagnosis.spectrum_of_model, diagnosis.spectrum_of_ridge
    return Results(
        settings=settings,
        strategy=configuration.masking,
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
    configuration = run.configuration
    architecture = configuration.architecture
    budget = configuration.budget
    corpus = (
        f"{run.corpus}, {len(published.training)} training and {len(published.validation)} "
        f"validation windows of {WINDOW.length:g} steps, stride {WINDOW.stride:g}"
    )
    if run.units:
        corpus += f", cut to {run.units} units"
    if not run.configuration.corpus_share.is_whole:
        corpus += f", {run.configuration.corpus_fraction:g} of the training units"
    encoder = (
        f"{run.shape_label}: {architecture.width} wide, {architecture.heads} heads, "
        f"{architecture.layers} blocks, "
        f"{architecture.parameter_count(published.vocabulary_size):,} parameters; "
        f"decoder of {configuration.decoder_layers} block(s)"
    )
    strategy = configuration.masking
    masking = (
        f"channel {strategy.channel_rate:g}, block {strategy.block_rate:g} over "
        f"{strategy.block_span:g} of the window, token {strategy.token_rate:g}; "
        f"expected {strategy.expected_ratio:.1%}, realised {trained.hidden_ratio:.1%} "
        "of observed tokens"
    )
    training = (
        f"{budget.epochs} epochs, batch {budget.batch_size}"
        + (f" × {budget.accumulation_steps} accumulated" if budget.accumulation_steps > 1 else "")
        + f", Adam at a peak of {budget.learning_rate:g}, "
        f"{budget.warmup_epochs} epoch(s) of warmup then cosine decay to "
        f"{budget.final_lr_fraction:g} of the peak, seed {budget.seed}, "
        f"{configuration.precision} on {run.device.upper()}"
    )
    rows = [
        ("Machine", machine()),
        ("Python", sys.version.split()[0]),
        ("torch", version("torch")),
        ("Experiment", f"`experiments/{configuration.name}.toml`"),
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


def shown(path: Path) -> Path:
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stated = experiments()
    parser.add_argument(
        "--experiment",
        action="append",
        choices=sorted(stated),
        help="experiment to run; every one this report can generate a corpus for unless given",
    )
    parser.add_argument(
        "--figures-only",
        type=Path,
        action="append",
        metavar="RUN",
        help="draw the figures of a stored run and train nothing; repeatable",
    )
    parser.add_argument("--units", type=int, default=None, help="cut the layout to this many units")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override the epochs every experiment states; recorded with the run",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default=available_device(),
        help="where the model trains; runs on different devices are never compared",
    )
    parser.add_argument(
        "--track",
        default=None,
        metavar="URI",
        help="MLflow tracking server or database to record the runs in; kept in memory unless "
        "given (the local stack serves one at http://127.0.0.1:5000)",
    )
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT / "data" / "report")
    parser.add_argument(
        "--figures",
        type=Path,
        default=None,
        help="directory to draw into; the stored run's own figures directory unless given",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help="directory each run is stored under as CSV, beside an index of every run",
    )
    arguments = parser.parse_args(argv)
    if arguments.figures_only:
        for directory in arguments.figures_only:
            for path in drawn_from(directory, arguments.figures or figures_of(directory)):
                print(shown(path))
        return
    if not device_available(arguments.device):
        parser.error(f"device {arguments.device} is not available on this machine")
    runs = []
    for name in arguments.experiment or sorted(stated):
        file = stated[name]
        configuration = file.configuration()
        if arguments.epochs is not None:
            try:
                configuration = replace(
                    configuration, budget=replace(configuration.budget, epochs=arguments.epochs)
                )
            except InvalidTrainingBudgetError as error:
                parser.error(f"{name}: {error}")
        (corpus_name,) = file.corpora
        runs.append(
            Run(
                configuration=configuration,
                corpus=corpus_name,
                units=arguments.units,
                device=arguments.device,
            )
        )
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    assess = AssessReconstructionRun(UnitBootstrap())
    sections = []
    for run in runs:
        published = publish(run, arguments.workspace / run.corpus)
        # One tracker per run, because a tracker records exactly one.
        tracker = (
            InMemoryExperimentTracker()
            if arguments.track is None
            else MlflowExperimentTracker(arguments.track)
        )
        trained = train(
            published,
            run,
            arguments.workspace / run.corpus,
            tracker,
            datetime.now(),
        )
        diagnosis = diagnose(published, trained, run)
        results = results_of(run, published, trained, diagnosis)
        assessment = assess(results, find_shorter(results, arguments.results))
        stored = store(
            assessment,
            RunFigures(
                interpolation_loss=diagnosis.interpolation_loss,
                ridge_loss=diagnosis.ridge_loss,
                examples=diagnosis.examples,
            ),
            arguments.results,
        )
        # Drawn from the files the run just wrote, never from what is still in memory: a figure
        # this report can draw is a figure anyone can redraw from the stored run.
        figures = arguments.figures or figures_of(stored)
        drawn_from(stored, figures)
        sections.append(
            render(run, published, trained, diagnosis, assessment)
            + f"\n\nFigures in `{shown(figures)}`; the run is stored in "
            f"`{shown(stored)}`."
        )
    if isinstance(sys.stdout, io.TextIOWrapper):
        # The tables use ×, → and —; the note this output is pasted into is UTF-8 and LF-only.
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
