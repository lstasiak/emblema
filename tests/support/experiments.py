"""Experiments and corpora small enough to run inside a test.

The overrides are typed ``Any`` because each names a field of the value object it builds and
carries that field's type; the value object refuses anything else on the way in.

The configuration here is the shape of a real one — every field stated, nothing defaulted away —
at a size where a whole run is a fraction of a second. The windows are invented rather than
published: what is being exercised is the loop around them, and a test that needs windows a
corpus really produced asks ``tests.support.control_corpus`` instead.
"""

import random
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.corpus_share import CorpusShare
from emblema.pretraining.domain.training.corpus_validation import CorpusValidation
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.pretraining.domain.training.precision import Precision
from emblema.pretraining.domain.training.training_budget import TrainingBudget
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.compute import ComputeTier
from emblema.shared.kernel.tokens import Token, TokenWindow

TINY = EncoderArchitecture(width=16, heads=2, layers=1, feedforward_width=32, time_frequencies=4)
MIXTURE = MaskingStrategy(channel_rate=0.15, block_rate=0.6, block_span=0.5, token_rate=0.1)
CHANNELS = 3
CHANNEL_NAMES = ("invented/a", "invented/b", "invented/c")


def configuration(**overrides: Any) -> ExperimentConfiguration:
    """A whole experiment at a test's scale; anything named is replaced."""
    stated = ExperimentConfiguration(
        name="test-experiment",
        tier=ComputeTier.S,
        corpus_fraction=1.0,
        architecture=TINY,
        dropout=0.0,
        decoder_layers=1,
        masking=MIXTURE,
        loss=ObjectiveLoss(kind=LossKind.MSE),
        budget=budget(),
        precision=Precision.FP32,
        checkpoint=CheckpointPolicy(every_steps=2),
    )
    return replace(stated, **overrides)


def share(fraction: float = 1.0, seed: int = 1) -> CorpusShare:
    """The share a run reads; the whole corpus, ranked by the test seed, unless said otherwise."""
    return CorpusShare(fraction=fraction, seed=seed)


def budget(**overrides: Any) -> TrainingBudget:
    stated = TrainingBudget(
        epochs=2,
        batch_size=2,
        accumulation_steps=1,
        learning_rate=1e-3,
        warmup_epochs=0,
        final_lr_fraction=0.5,
        seed=1,
    )
    return replace(stated, **overrides)


def corpus(
    *,
    training: int = 8,
    validation: int = 4,
    name: str = "invented",
    seed: int = 1,
    channels: tuple[str, ...] = CHANNEL_NAMES,
) -> TrainingCorpus:
    """A corpus of invented windows, each of three channels observed at four instants.

    The checksum stands for the artifact real windows are read out of: here it digests the windows
    themselves, so two corpora that hold different data are told apart as they would be in a run.
    """
    training_windows = windows(training, seed=seed)
    validation_windows = windows(validation, seed=seed + 1)
    return TrainingCorpus(
        name=name,
        checksum=checksum_of(training_windows + validation_windows),
        training=training_windows,
        validation=validation_windows,
        channels=channels,
    )


def mixture(*corpora: TrainingCorpus) -> TrainingMixture:
    """The mixture of these corpora; the test corpus alone unless others are given."""
    return TrainingMixture.of(*(corpora or (corpus(),)))


def continued(
    *, name: str, seed: int, training: int = 8, validation: int = 4, channels: int = 2
) -> TrainingCorpus:
    """A corpus published after the test corpus, continuing its vocabulary by that many channels.

    Its windows still carry the first three channels, so what the mixture checks — the chain of
    names — is exercised without windows of a wider vocabulary being invented.
    """
    return corpus(
        training=training,
        validation=validation,
        name=name,
        seed=seed,
        channels=CHANNEL_NAMES + tuple(f"{name}/{index}" for index in range(channels)),
    )


def checksum_of(read: Sequence[TokenWindow]) -> Checksum:
    """A digest of what the windows hold, as the block a run reads them from would be digested."""
    return Checksum.of_bytes(repr([window.values for window in read]).encode())


def windows(count: int, *, seed: int, steps: int = 4) -> list[TokenWindow]:
    """``count`` windows of the same channels at the same instants, with drawn values."""
    draws = random.Random(seed)
    times = [step / (steps - 1) for step in range(steps)]
    return [
        TokenWindow.of(
            Token(
                channel_id=channel,
                value=draws.gauss(0.0, 1.0),
                time=time,
                gap=time if index == 0 else times[index] - times[index - 1],
            )
            for channel in range(1, CHANNELS + 1)
            for index, time in enumerate(times)
        )
        for _ in range(count)
    ]


WEIGHTS = ArtifactRef("durable/weights", Checksum.of_bytes(b"weights"))


def epoch_outcome(number: int, **overrides: Any) -> EpochOutcome:
    """What an epoch of a run over one corpus reports; anything named is replaced."""
    stated: dict[str, Any] = {
        "epoch": number,
        "training_loss": 1.0 / (number + 1),
        "validation": (validated(loss=1.2 / (number + 1)),),
        "hidden_ratio": 0.46,
        "seconds": 0.25,
    }
    return EpochOutcome(**(stated | overrides))


def validated(**overrides: Any) -> CorpusValidation:
    """What one corpus of the held-out side cost an epoch; anything named is replaced."""
    stated = CorpusValidation(corpus="invented", tokens=40, loss=1.2, trivial=1.0)
    return replace(stated, **overrides)
