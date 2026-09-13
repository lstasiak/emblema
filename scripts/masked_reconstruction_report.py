"""Train the objective on the positive control for a few epochs and show what it learnt.

The self-supervised objective is judged on three things before any real corpus sees it: that its
loss falls, that its reconstructions look like the signal, and that what it learnt is not what a
trivial baseline already knew. This script does all three on the control corpora, which exist
for exactly this — a corpus whose structure was put there on purpose, so that a model finding
nothing is at fault. It publishes the corpus through the same use cases the command line runs,
trains the encoder with the masked-reconstruction objective on the training units, and measures
on the validation units against the baseline matched to each kind of mask: interpolation within
the channel for blocks and single tokens, a ridge regression from the other channels for a
channel hidden whole. A least-squares spectrum of the channels hidden whole says which frequencies
the model gives back.

    uv sync --all-extras
    uv run scripts/masked_reconstruction_report.py --corpus control-a --corpus control-b

The training loop here is the smallest that answers the question — no checkpoints, no tracker,
no precision policy; those belong to the run that trains a backbone for real. The output is
markdown, meant to be pasted under a dated heading in the verification note; the figures are
written next to it.
"""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
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
from emblema.catalog.adapters.synthetic.layouts import CONTROL_PROCESS, LAYOUTS
from emblema.catalog.adapters.synthetic.synthetic_corpus_reader import SyntheticCorpusReader
from emblema.catalog.application.use_cases.publish_corpus import PublishCorpusCommand
from emblema.catalog.domain.tokenisation.tokenisation_manifest import TokenisationManifest
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.entrypoints.cli.composition_root import CompositionRoot
from emblema.entrypoints.cli.known_corpora import KnownCorpora
from emblema.pretraining.adapters.diagnostics.cross_channel_ridge_baseline import (
    CrossChannelRidgeBaseline,
)
from emblema.pretraining.adapters.diagnostics.linear_interpolation_baseline import (
    LinearInterpolationBaseline,
)
from emblema.pretraining.adapters.diagnostics.mask_kind_verdict import MaskKindVerdict
from emblema.pretraining.adapters.diagnostics.spectral_recovery import SpectralRecovery
from emblema.pretraining.adapters.diagnostics.triviality_diagnostic import TrivialityDiagnostic
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.masked_reconstruction import MaskedReconstruction
from emblema.pretraining.adapters.objective.reconstruction_loss import ReconstructionLoss
from emblema.pretraining.adapters.objective.token_masking import TokenMasking
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.loaders.window_loader import WindowLoader
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from emblema.shared.kernel.tokens import TokenWindow
from scripts.budget_file import architecture_of, tier_named
from scripts.reporting import dated_heading, machine, table
from tests.support.settings import unreachable_store

FIGURES = REPO_ROOT / "docs" / "verification" / "figures"

# The mixture the plan asks for: blocks and whole channels carry most of the hiding, single tokens
# are the minority ingredient, and together they take close to half of a window.
MIXTURE = MaskingStrategy(channel_rate=0.15, block_rate=0.6, block_span=0.5, token_rate=0.1)
DECODER_LAYERS = 1
# Cycles per window the spectrum is fitted up to, at most: the fastest harmonic of the control
# process completes about four cycles in a window of this length. A sparse layout holds fewer
# tokens per channel than that many coefficients need, so each corpus fits as many as it can.
CYCLES_AT_MOST = 6
WINDOW = WindowSpec(length=32.0, stride=12.0)
EXAMPLE_WINDOWS = 3


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
    seed: int

    @property
    def architecture(self) -> EncoderArchitecture:
        """The tier's shape, unless the run states a smaller one a slow machine can afford."""
        return self.shape if self.shape is not None else architecture_of(tier_named(self.tier))

    @property
    def shape_label(self) -> str:
        if self.shape is None:
            return f"tier {self.tier}"
        return f"tier {self.tier} cut down to {self.shape.width} wide, {self.shape.layers} deep"


@dataclass(frozen=True)
class Published:
    """The control corpus after the pipeline: its manifest and the windows of each split."""

    manifest: TokenisationManifest
    training: tuple[TokenWindow, ...]
    validation: tuple[TokenWindow, ...]

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

    def channels_apart(self) -> frozenset[int]:
        """Channels the verdict leaves aside: timeless ones and ones that never vary."""
        scheme = self.manifest.scheme
        apart = set()
        for entry in scheme.vocabulary.entries_of(self.manifest.corpus):
            statistics = scheme.statistics[entry.channel_id - 1]
            if entry.timeless or (statistics is not None and statistics.std == 0.0):
                apart.add(entry.channel_id)
        return frozenset(apart)


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
    """One channel of one validation window, as the figure draws it."""

    window: int
    channel: str
    times: np.ndarray
    truth: np.ndarray
    hidden: np.ndarray
    kind: MaskKind
    model: np.ndarray
    baseline: np.ndarray


@dataclass(frozen=True)
class Diagnosis:
    verdicts: tuple[MaskKindVerdict, ...]
    interpolation_loss: float
    ridge_loss: float
    spectrum_of_model: SpectralRecovery
    spectrum_of_ridge: SpectralRecovery
    examples: tuple[Example, ...]

    @property
    def negative(self) -> bool:
        """Whether every kind of mask, channels apart aside, taught more than its baseline knew."""
        return all(not verdict.trivial for verdict in self.verdicts if not verdict.apart)


def publish(run: Run, workspace: Path) -> Published:
    layout = LAYOUTS[run.corpus]
    if run.units is not None:
        layout = layout.with_dials(units=run.units)
    known = KnownCorpora.default().named(run.corpus)
    root = CompositionRoot(
        unreachable_store(),
        corpus_root=workspace / "raw",
        workspace=workspace,
        corpora=InMemoryCorpusRepository(),
        reader=SyntheticCorpusReader(CONTROL_PROCESS, layout),
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
    return Published(
        manifest,
        tuple(archive.read_windows(manifest.archived, manifest.split.training)),
        tuple(archive.read_windows(manifest.archived, manifest.split.validation)),
    )


def validation_batches(published: Published, run: Run) -> Iterator[tuple[TokenTensors, TokenMasks]]:
    """The validation windows in a fixed order under masks drawn once per batch, every time."""
    loader = WindowLoader(
        published.validation, batch_size=run.batch_size, seed=run.seed, shuffle=False
    )
    masking = TokenMasking(MIXTURE)
    for index, batch in enumerate(loader.batches_of(0)):
        yield batch, masking.draw(batch, torch.Generator().manual_seed(run.seed * 1_000 + index))


def train(published: Published, run: Run) -> Trained:
    torch.manual_seed(run.seed)
    model = MaskedReconstruction(
        SetEncoder.for_vocabulary(run.architecture, published.vocabulary_size),
        decoder_layers=DECODER_LAYERS,
    )
    loss = ReconstructionLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=run.learning_rate)
    loader = WindowLoader(published.training, batch_size=run.batch_size, seed=run.seed)
    masking = TokenMasking(MIXTURE)
    draws = torch.Generator().manual_seed(run.seed)
    epochs = []
    for epoch in range(run.epochs):
        started = time.perf_counter()
        model.train()
        seen = []
        for batch in loader.batches_of(epoch):
            masks = masking.draw(batch, draws)
            optimiser.zero_grad()
            step = loss(model(batch, masks), batch, masks)
            step.backward()
            optimiser.step()
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
        for batch, masks in validation_batches(published, run):
            scored = int(masks.hidden.sum())
            total += float(loss(model(batch, masks), batch, masks)) * scored
            hidden += scored
            observed += int((~batch.padding_mask).sum())
    return total / max(hidden, 1), hidden / max(observed, 1)


def diagnose(published: Published, trained: Trained, run: Run) -> Diagnosis:
    interpolation = LinearInterpolationBaseline()
    ridge = CrossChannelRidgeBaseline.fitted(
        WindowLoader(
            published.training, batch_size=run.batch_size, seed=run.seed, shuffle=False
        ).batches_of(0),
        vocabulary_size=published.vocabulary_size,
    )
    triviality = TrivialityDiagnostic(channels_apart=published.channels_apart())
    cycles = published.spectral_cycles()
    spectrum_of_model = SpectralRecovery.up_to(cycles)
    spectrum_of_ridge = SpectralRecovery.up_to(cycles)
    loss = ReconstructionLoss()
    interpolation_total, ridge_total, hidden = 0.0, 0.0, 0
    examples: list[Example] = []
    with torch.no_grad():
        for batch, masks in validation_batches(published, run):
            predicted = trained.model(batch, masks)
            interpolated = interpolation.predict(batch, masks)
            regressed = ridge.predict(batch, masks)
            triviality = triviality.observe(batch, masks, predicted, interpolated, regressed)
            spectrum_of_model = spectrum_of_model.observe(batch, masks, predicted)
            spectrum_of_ridge = spectrum_of_ridge.observe(batch, masks, regressed)
            scored = int(masks.hidden.sum())
            interpolation_total += float(loss(interpolated, batch, masks)) * scored
            ridge_total += float(loss(regressed, batch, masks)) * scored
            hidden += scored
            if len(examples) < EXAMPLE_WINDOWS * len(MaskKind):
                examples += pick_examples(
                    published, batch, masks, predicted, interpolated, regressed, len(examples)
                )
    return Diagnosis(
        verdicts=triviality.verdicts(),
        interpolation_loss=interpolation_total / max(hidden, 1),
        ridge_loss=ridge_total / max(hidden, 1),
        spectrum_of_model=spectrum_of_model,
        spectrum_of_ridge=spectrum_of_ridge,
        examples=tuple(examples[: EXAMPLE_WINDOWS * len(MaskKind)]),
    )


def pick_examples(
    published: Published,
    batch: TokenTensors,
    masks: TokenMasks,
    predicted: torch.Tensor,
    interpolated: torch.Tensor,
    regressed: torch.Tensor,
    offset: int,
) -> list[Example]:
    """One channel per kind of mask from the first windows of the batch, for the figure."""
    vocabulary = published.manifest.scheme.vocabulary
    apart = published.channels_apart()
    examples: list[Example] = []
    for row in range(min(batch.batch_size, EXAMPLE_WINDOWS)):
        observed = ~batch.padding_mask[row]
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
                    window=offset + row,
                    channel=vocabulary.entry(channels[0]).channel,
                    times=batch.timestamps[row][tokens].numpy(),
                    truth=batch.features[row, :, 0][tokens].numpy(),
                    hidden=masks.hidden[row][tokens].numpy(),
                    kind=kind,
                    model=predicted[row][tokens].numpy(),
                    baseline=baseline[row][tokens].numpy(),
                )
            )
    return examples


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
        f"{run.epochs} epochs, batch {run.batch_size}, Adam at {run.learning_rate:g}, "
        f"seed {run.seed}, CPU"
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
        f"{diagnosis.interpolation_loss:.4f} and the ridge baseline {diagnosis.ridge_loss:.4f}."
    )
    return (
        "### Loss\n\n"
        + table(("Epoch", "Training", "Validation", "Seconds"), rows)
        + "\n\n"
        + summary
    )


def triviality_section(diagnosis: Diagnosis) -> str:
    rows = [
        (
            verdict.kind.value + (" (apart)" if verdict.apart else ""),
            "ridge" if verdict.kind is MaskKind.CHANNEL else "interpolation",
            f"{verdict.tokens:,}",
            f"{verdict.model_error:.4f}",
            f"{verdict.baseline_error:.4f}",
            f"{verdict.excess:+.4f}",
            "trivial" if verdict.trivial else "learnt",
        )
        for verdict in diagnosis.verdicts
    ]
    return "### Triviality per kind of mask\n\n" + table(
        ("Kind", "Baseline", "Tokens", "Model", "Baseline error", "Excess", "Verdict"), rows
    )


# A frequency holding less than this share of the truth's energy is noise, not signal: nothing
# recovers noise, so a share of it "recovered" says nothing about the model.
INFORMATIVE_SHARE = 0.01


def informative_frequencies(spectrum: SpectralRecovery) -> list[int]:
    """Cycles per window at which the truth holds a share of its energy worth recovering."""
    total = sum(spectrum.truth_energy)
    return [
        k + 1
        for k, energy in enumerate(spectrum.truth_energy)
        if total > 0.0 and energy / total >= INFORMATIVE_SHARE
    ]


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


def verdict_section(run: Run, trained: Trained, diagnosis: Diagnosis) -> str:
    first, last = trained.epochs[0], trained.epochs[-1]
    lines = [
        f"- Loss {'fell' if last.validation_loss < first.validation_loss else 'did not fall'}: "
        f"{first.validation_loss:.4f} → {last.validation_loss:.4f} on validation."
    ]
    for verdict in diagnosis.verdicts:
        if verdict.apart:
            continue
        outcome = "trivial: nothing its baseline did not know" if verdict.trivial else "learnt"
        lines.append(
            f"- `{verdict.kind.value}` masks: model {verdict.model_error:.4f} against baseline "
            f"{verdict.baseline_error:.4f} — {outcome}."
        )
    lines.append(
        "- Triviality diagnostic "
        + (
            "**negative**: every kind of mask beats its baseline."
            if diagnosis.negative
            else "**positive** for at least one kind: see the table."
        )
    )
    spectrum = diagnosis.spectrum_of_model
    informative = informative_frequencies(spectrum)
    recovered = spectrum.recovered()
    lost = [k for k in informative if recovered[k - 1] < 0.5]
    lines.append(
        f"- Spectrum: the truth holds its energy at {', '.join(map(str, informative)) or 'no'} "
        f"cycle(s) per window of {spectrum.cycles} fitted; the model recovers "
        + ", ".join(f"{recovered[k - 1]:.0%} at {k}" for k in informative)
        + (
            f" — it loses the signal at {', '.join(map(str, lost))}."
            if lost
            else "; every frequency the signal has is recovered."
        )
    )
    lines.append(
        f"- Figures: `docs/verification/figures/masked-reconstruction-{run.corpus}-loss.png`, "
        f"`…-windows.png`, `…-diagnostics.png`."
    )
    return "### Verdict\n\n" + "\n".join(lines)


def render(run: Run, published: Published, trained: Trained, diagnosis: Diagnosis) -> str:
    return "\n\n".join(
        [
            heading(run, published, trained),
            loss_section(trained, diagnosis),
            triviality_section(diagnosis),
            spectral_section(diagnosis),
            verdict_section(run, trained, diagnosis),
        ]
    )


def draw_figures(run: Run, trained: Trained, diagnosis: Diagnosis, directory: Path) -> None:
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
                example.times[~example.hidden],
                example.truth[~example.hidden],
                s=12,
                color="black",
                label="visible",
            )
            hidden = example.hidden
            axis.scatter(
                example.times[hidden],
                example.model[hidden],
                marker="x",
                color="tab:red",
                label="model",
            )
            axis.scatter(
                example.times[hidden],
                example.baseline[hidden],
                marker="^",
                s=14,
                color="tab:blue",
                label="baseline",
            )
            axis.set_ylabel(
                f"w{example.window} {example.channel}\n{example.kind.value}", fontsize=8
            )
        np.atleast_1d(axes)[0].legend(fontsize=7, ncol=4, loc="upper right")
        np.atleast_1d(axes)[-1].set_xlabel("position in window")
        figure.suptitle(f"{run.corpus}: hidden tokens, model against the matched baseline")
        figure.tight_layout()
        figure.savefig(directory / f"{stem}-windows.png", dpi=120)
        plt.close(figure)

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    verdicts = [verdict for verdict in diagnosis.verdicts if not verdict.apart]
    positions = np.arange(len(verdicts))
    left.bar(positions - 0.2, [v.model_error for v in verdicts], width=0.4, label="model")
    left.bar(positions + 0.2, [v.baseline_error for v in verdicts], width=0.4, label="baseline")
    left.set_xticks(positions, [v.kind.value for v in verdicts])
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", choices=sorted(LAYOUTS))
    parser.add_argument("--tier", default="S", choices=["S", "M"])
    parser.add_argument(
        "--shape",
        type=shape_of,
        default=None,
        help="WIDTH,HEADS,LAYERS,FEEDFORWARD to build a smaller encoder than the tier's, where "
        "the tier's is too slow for the machine; the time frequencies stay the tier's",
    )
    parser.add_argument("--units", type=int, default=None, help="cut the layout to this many units")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT / "data" / "report")
    parser.add_argument("--figures", type=Path, default=FIGURES)
    arguments = parser.parse_args(argv)
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    sections = []
    for corpus in arguments.corpus or ["control-a", "control-b"]:
        run = Run(
            corpus=corpus,
            tier=arguments.tier,
            shape=arguments.shape,
            units=arguments.units,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            seed=arguments.seed,
        )
        published = publish(run, arguments.workspace / corpus)
        trained = train(published, run)
        diagnosis = diagnose(published, trained, run)
        draw_figures(run, trained, diagnosis, arguments.figures)
        sections.append(render(run, published, trained, diagnosis))
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")  # type: ignore[union-attr]
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
