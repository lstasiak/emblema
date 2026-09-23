"""What the runtime that trains owes beyond the port: the arithmetic it claims to do.

The port's contract covers the shape of a run. Here are the claims only a runtime that really
trains can be held to — that accumulation is the larger batch, that a resumed run is the run it
resumed, and that the weights it stored are the weights it trained.
"""

import math
from typing import Any

import pytest

from emblema.pretraining.domain.exceptions import DivergedRunError, UnsupportedPrecisionError
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.pretraining.domain.training.precision import Precision
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.retention import Retention
from emblema.shared.kernel.tokens import Token, TokenWindow
from tests.support.experiments import (
    CHANNEL_NAMES,
    budget,
    checksum_of,
    configuration,
    continued,
    corpus,
    mixture,
)

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.objective.reconstruction_loss import (  # noqa: E402
    ReconstructionLoss,
)
from emblema.pretraining.adapters.training.torch_training_runtime import (  # noqa: E402
    TorchTrainingRuntime,
    _duration,
)
from emblema.pretraining.adapters.training.trained_model import TrainedModel  # noqa: E402
from emblema.pretraining.adapters.training.training_checkpoint import (  # noqa: E402
    TrainingCheckpoint,
)
from emblema.pretraining.domain.training.run_signature import RunSignature  # noqa: E402

pytestmark = pytest.mark.ml

CORPUS = corpus(training=8, validation=4)
MIXTURE = mixture(CORPUS)


class RecordingStore(InMemoryArtifactStore):
    """A store that remembers each thing it was asked to keep."""

    def __init__(self) -> None:
        super().__init__()
        self.kept: list[Retention] = []

    def put(self, content: bytes, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        self.kept.append(retention)
        return super().put(content, retention)


def weights_of(store: InMemoryArtifactStore, ref: ArtifactRef | None):
    """The weights an outcome points at, refusing an outcome that pointed at nothing."""
    assert ref is not None
    return TrainedModel.read(store.get(ref)).weights


def checkpoint_at(
    store: InMemoryArtifactStore, ref: ArtifactRef | None, signature: RunSignature
) -> TrainingCheckpoint:
    assert ref is not None
    return TrainingCheckpoint.read(store.get(ref), signature=signature)


def test_the_same_configuration_twice_trains_the_same_weights() -> None:
    stated = configuration(budget=budget(epochs=1, batch_size=2))
    first, second = InMemoryArtifactStore(), InMemoryArtifactStore()

    one = list(TorchTrainingRuntime(first, device="cpu").train(stated, MIXTURE))
    two = list(TorchTrainingRuntime(second, device="cpu").train(stated, MIXTURE))

    assert one[-1].training_loss == two[-1].training_loss
    assert one[-1].backbone == two[-1].backbone


def test_accumulated_micro_batches_make_one_step_between_them() -> None:
    """Four micro-batches accumulated in pairs are two steps, and the position says so.

    That the two are the gradient of the larger batch follows from the error being summed over
    tokens rather than averaged per batch, which is where that is tested: the masks a batch is
    drawn under depend on how the windows were grouped, so two groupings cannot train identically
    however the gradient is combined.
    """
    accumulated = configuration(
        budget=budget(epochs=1, batch_size=2, accumulation_steps=2),
        checkpoint=CheckpointPolicy(every_steps=1),
    )
    store = InMemoryArtifactStore()

    outcomes = list(TorchTrainingRuntime(store, device="cpu").train(accumulated, MIXTURE))

    written = checkpoint_at(store, outcomes[-1].checkpoint, RunSignature.of(accumulated, MIXTURE))
    assert (written.position.batches, written.position.steps) == (4, 2)


def test_a_resumed_run_ends_where_the_uninterrupted_one_did() -> None:
    stated = configuration(
        budget=budget(epochs=2, batch_size=2), checkpoint=CheckpointPolicy(every_steps=3)
    )
    whole, halved = InMemoryArtifactStore(), InMemoryArtifactStore()

    uninterrupted = list(TorchTrainingRuntime(whole, device="cpu").train(stated, MIXTURE))
    interrupted = TorchTrainingRuntime(halved, device="cpu").train(stated, MIXTURE)
    stopped = next(interrupted)
    # The rest of the run is simply never asked for: that is what an interruption is here.
    resumed = list(
        TorchTrainingRuntime(halved, device="cpu").train(stated, MIXTURE, stopped.checkpoint)
    )

    assert resumed[-1].validation_loss == uninterrupted[-1].validation_loss
    trained = weights_of(whole, uninterrupted[-1].backbone)
    picked_up = weights_of(halved, resumed[-1].backbone)
    for key, value in trained.items():
        assert torch.equal(value, picked_up[key]), key


def test_a_checkpoint_records_where_the_run_stood() -> None:
    stated = configuration(
        budget=budget(epochs=2, batch_size=2), checkpoint=CheckpointPolicy(every_steps=3)
    )
    store = InMemoryArtifactStore()

    outcomes = list(TorchTrainingRuntime(store, device="cpu").train(stated, MIXTURE))

    written = checkpoint_at(store, outcomes[0].checkpoint, RunSignature.of(stated, MIXTURE))
    assert (written.position.epoch, written.position.batches, written.position.steps) == (0, 3, 3)


def test_what_a_run_stored_is_what_it_trained() -> None:
    stated = configuration(budget=budget(epochs=1, batch_size=2))
    store = InMemoryArtifactStore()
    runtime = TorchTrainingRuntime(store, device="cpu")

    outcomes = list(runtime.train(stated, MIXTURE))

    kept = outcomes[-1].backbone
    assert kept is not None
    restored = runtime.restore(kept)
    assert restored.encoder.architecture == stated.architecture
    assert not restored.training


def test_a_precision_this_device_cannot_run_is_refused_before_anything_is_trained() -> None:
    stated = configuration(precision=Precision.FP16)

    with pytest.raises(UnsupportedPrecisionError):
        TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu").train(stated, MIXTURE)


def test_the_device_is_the_machine_s_unless_the_run_is_told_otherwise() -> None:
    assert TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu").device == "cpu"
    assert TorchTrainingRuntime(InMemoryArtifactStore()).device in {"cpu", "mps", "cuda"}


def test_an_epoch_whose_masks_hid_nothing_leaves_the_weights_finite() -> None:
    """A window of one token keeps it visible, so a group of them scores nothing to divide by."""
    single = [TokenWindow.of([Token(channel_id=1, value=0.5, time=0.0, gap=0.0)])] * 4
    nothing_to_hide = TrainingCorpus(
        name="single-token",
        checksum=checksum_of(single),
        training=single,
        validation=CORPUS.validation,
        channels=CHANNEL_NAMES,
    )
    store = InMemoryArtifactStore()

    outcomes = list(
        TorchTrainingRuntime(store, device="cpu").train(
            configuration(budget=budget(epochs=1, batch_size=2)), mixture(nothing_to_hide)
        )
    )

    assert outcomes[-1].training_loss == 0.0
    assert all(
        torch.isfinite(value).all() for value in weights_of(store, outcomes[-1].backbone).values()
    )


def test_a_loss_that_is_not_finite_stops_the_run_before_anything_is_stepped_or_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summed = ReconstructionLoss.summed

    def not_finite(*arguments: Any) -> Any:
        error, scored = summed(*arguments)
        return error * math.nan, scored

    monkeypatch.setattr(ReconstructionLoss, "summed", not_finite)
    store = RecordingStore()

    with pytest.raises(DivergedRunError, match="the loss of batch 0 in epoch 0 is nan"):
        list(
            TorchTrainingRuntime(store, device="cpu").train(
                configuration(checkpoint=CheckpointPolicy(every_steps=1)), MIXTURE
            )
        )
    assert store.kept == []


def test_gradients_that_are_not_finite_stop_the_run_before_their_step_is_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loss stays finite and only the gradient overflows, as half precision without a scaler."""
    summed = ReconstructionLoss.summed

    def overflowing(*arguments: Any) -> Any:
        error, scored = summed(*arguments)
        error.register_hook(lambda gradient: gradient * math.inf)
        return error, scored

    monkeypatch.setattr(ReconstructionLoss, "summed", overflowing)
    store = RecordingStore()

    with pytest.raises(DivergedRunError, match="a gradient is not finite at step 1 of epoch 0"):
        list(
            TorchTrainingRuntime(store, device="cpu").train(
                configuration(checkpoint=CheckpointPolicy(every_steps=1)), MIXTURE
            )
        )
    assert store.kept == []


def test_a_run_is_scored_by_the_reading_its_experiment_states() -> None:
    """The same weights under two readings are two runs: the losses differ and so do the seeds."""
    bounded = configuration(
        budget=budget(epochs=1), loss=ObjectiveLoss(kind=LossKind.HUBER, huber_delta=1.0)
    )
    squared = configuration(budget=budget(epochs=1))

    under_bound = list(
        TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu").train(bounded, MIXTURE)
    )
    under_square = list(
        TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu").train(squared, MIXTURE)
    )

    # The bounded reading never counts a miss for more than the square does, and these windows
    # hold misses past the knee, so it counts strictly less.
    assert under_bound[-1].validation_loss < under_square[-1].validation_loss
    assert RunSignature.of(bounded, MIXTURE) != RunSignature.of(squared, MIXTURE)


def test_an_epoch_reports_the_corpus_it_validated_against_nothing_learnt() -> None:
    outcomes = list(
        TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu").train(
            configuration(budget=budget(epochs=1)), MIXTURE
        )
    )

    scored = outcomes[-1].validation
    assert [entry.corpus for entry in scored] == [CORPUS.name]
    assert scored[0].loss == outcomes[-1].validation_loss
    # The trivial predictor is the channel mean, measured over the same hidden tokens: on windows
    # of drawn standard normals it errs about one, and an untrained model does no better.
    assert scored[0].trivial == pytest.approx(1.0, abs=0.6)
    assert scored[0].relative == pytest.approx(scored[0].loss / scored[0].trivial)
    assert 0 < scored[0].tokens <= sum(len(window.values) for window in CORPUS.validation)


def test_the_backbone_is_the_best_epoch_s_weights_not_the_last_epoch_s() -> None:
    """A run whose validation rises after its first epoch keeps the first epoch's weights."""
    overfitting = configuration(budget=budget(epochs=3, batch_size=2, learning_rate=0.5))
    store = InMemoryArtifactStore()

    outcomes = list(TorchTrainingRuntime(store, device="cpu").train(overfitting, MIXTURE))

    relative = [outcome.relative_validation for outcome in outcomes]
    best = relative.index(min(relative))
    assert best < len(outcomes) - 1, "the test needs a run whose last epoch is not its best"
    kept = outcomes[best].weights
    assert kept is not None
    assert outcomes[-1].backbone == kept
    assert outcomes[-1].weights is None
    assert all(
        outcome.weights is None
        for outcome in outcomes
        if outcome.relative_validation > min(relative[: outcome.epoch + 1])
    )


def test_a_resumed_run_keeps_the_best_epoch_from_before_the_checkpoint() -> None:
    overfitting = configuration(
        budget=budget(epochs=3, batch_size=2, learning_rate=0.5),
        checkpoint=CheckpointPolicy(every_steps=4),
    )
    whole, halved = InMemoryArtifactStore(), InMemoryArtifactStore()

    uninterrupted = list(TorchTrainingRuntime(whole, device="cpu").train(overfitting, MIXTURE))
    interrupted = TorchTrainingRuntime(halved, device="cpu").train(overfitting, MIXTURE)
    first = next(interrupted)
    resumed = list(
        TorchTrainingRuntime(halved, device="cpu").train(overfitting, MIXTURE, first.checkpoint)
    )

    relative = [outcome.relative_validation for outcome in uninterrupted]
    assert relative.index(min(relative)) == 0, "the test needs the best epoch before the checkpoint"
    assert [outcome.relative_validation for outcome in resumed] == relative[1:]
    assert resumed[-1].backbone == uninterrupted[-1].backbone == first.weights
    assert all(outcome.weights is None for outcome in resumed)


def test_a_run_over_two_corpora_scores_each_on_its_own() -> None:
    second = continued(name="second", seed=2, training=4, validation=2)
    mixed = TrainingMixture.of(CORPUS, second)
    stated = configuration(budget=budget(epochs=1, batch_size=2))

    outcomes = list(
        TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu").train(stated, mixed)
    )

    scored = outcomes[-1].validation
    assert [entry.corpus for entry in scored] == ["invented", "second"]
    assert 0 < scored[0].tokens <= sum(len(window.values) for window in CORPUS.validation)
    assert 0 < scored[1].tokens <= sum(len(window.values) for window in second.validation)


def test_the_run_says_where_it_can_be_picked_up_from(caplog: pytest.LogCaptureFixture) -> None:
    stated = configuration(
        budget=budget(epochs=1, batch_size=2), checkpoint=CheckpointPolicy(every_steps=2)
    )
    store = InMemoryArtifactStore()

    with caplog.at_level("INFO", logger="emblema.pretraining"):
        outcomes = list(TorchTrainingRuntime(store, device="cpu").train(stated, MIXTURE))

    checkpoint = outcomes[-1].checkpoint
    assert checkpoint is not None
    said = [record.getMessage() for record in caplog.records]
    assert any(f"checkpoint {checkpoint.key} {checkpoint.checksum}" in message for message in said)
    assert any(
        message.startswith("epoch 1 of 1 done") and "invented validation" in message
        for message in said
    )


def test_the_run_says_how_far_the_epoch_has_come_every_so_many_steps(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stated = configuration(budget=budget(epochs=2, batch_size=2, accumulation_steps=2))
    runtime = TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu", progress_every=1)

    with caplog.at_level("INFO", logger="emblema.pretraining"):
        list(runtime.train(stated, MIXTURE))

    progress = [
        record.getMessage() for record in caplog.records if ", step " in record.getMessage()
    ]
    assert [message[: message.index(":")] for message in progress] == [
        "epoch 1 of 2, step 1 of 2",
        "epoch 1 of 2, step 2 of 2",
        "epoch 2 of 2, step 1 of 2",
        "epoch 2 of 2, step 2 of 2",
    ]
    assert all(
        "over the last 1 steps" in message and "left in the epoch" in message
        for message in progress
    )


def test_a_negative_progress_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="progress_every"):
        TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu", progress_every=-1)


@pytest.mark.parametrize(
    ("seconds", "said"),
    [(-1.0, "0 min"), (0.0, "0 min"), (90.0, "2 min"), (3599.0, "60 min"), (5400.0, "1.5 h")],
)
def test_a_log_says_how_long_is_left_in_the_unit_a_person_reads(seconds: float, said: str) -> None:
    # The only line of a long run a person watches, so its unit switches at the hour rather than
    # counting thousands of minutes.
    assert _duration(seconds) == said
