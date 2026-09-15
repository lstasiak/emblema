"""What the runtime that trains owes beyond the port: the arithmetic it claims to do.

The port's contract covers the shape of a run. Here are the claims only a runtime that really
trains can be held to — that accumulation is the larger batch, that a resumed run is the run it
resumed, and that the weights it stored are the weights it trained.
"""

import pytest

from emblema.pretraining.domain.exceptions import UnsupportedPrecisionError
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.precision import Precision
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import Token, TokenWindow
from tests.support.experiments import budget, checksum_of, configuration, corpus

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.training.torch_training_runtime import (  # noqa: E402
    TorchTrainingRuntime,
)
from emblema.pretraining.adapters.training.trained_model import TrainedModel  # noqa: E402
from emblema.pretraining.adapters.training.training_checkpoint import (  # noqa: E402
    TrainingCheckpoint,
)
from emblema.pretraining.domain.training.run_signature import RunSignature  # noqa: E402

pytestmark = pytest.mark.ml

CORPUS = corpus(training=8, validation=4)


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

    one = list(TorchTrainingRuntime(first, device="cpu").train(stated, CORPUS))
    two = list(TorchTrainingRuntime(second, device="cpu").train(stated, CORPUS))

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

    outcomes = list(TorchTrainingRuntime(store, device="cpu").train(accumulated, CORPUS))

    written = checkpoint_at(store, outcomes[-1].checkpoint, RunSignature.of(accumulated, CORPUS))
    assert (written.position.batches, written.position.steps) == (4, 2)


def test_a_resumed_run_ends_where_the_uninterrupted_one_did() -> None:
    stated = configuration(
        budget=budget(epochs=2, batch_size=2), checkpoint=CheckpointPolicy(every_steps=3)
    )
    whole, halved = InMemoryArtifactStore(), InMemoryArtifactStore()

    uninterrupted = list(TorchTrainingRuntime(whole, device="cpu").train(stated, CORPUS))
    interrupted = TorchTrainingRuntime(halved, device="cpu").train(stated, CORPUS)
    stopped = next(interrupted)
    # The rest of the run is simply never asked for: that is what an interruption is here.
    resumed = list(
        TorchTrainingRuntime(halved, device="cpu").train(stated, CORPUS, stopped.checkpoint)
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

    outcomes = list(TorchTrainingRuntime(store, device="cpu").train(stated, CORPUS))

    written = checkpoint_at(store, outcomes[0].checkpoint, RunSignature.of(stated, CORPUS))
    assert (written.position.epoch, written.position.batches, written.position.steps) == (0, 3, 3)


def test_what_a_run_stored_is_what_it_trained() -> None:
    stated = configuration(budget=budget(epochs=1, batch_size=2))
    store = InMemoryArtifactStore()
    runtime = TorchTrainingRuntime(store, device="cpu")

    outcomes = list(runtime.train(stated, CORPUS))

    kept = outcomes[-1].backbone
    assert kept is not None
    restored = runtime.restore(kept)
    assert restored.encoder.architecture == stated.architecture
    assert not restored.training


def test_a_precision_this_device_cannot_run_is_refused_before_anything_is_trained() -> None:
    stated = configuration(precision=Precision.FP16)

    with pytest.raises(UnsupportedPrecisionError):
        TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu").train(stated, CORPUS)


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
        vocabulary_size=3,
    )
    store = InMemoryArtifactStore()

    outcomes = list(
        TorchTrainingRuntime(store, device="cpu").train(
            configuration(budget=budget(epochs=1, batch_size=2)), nothing_to_hide
        )
    )

    assert outcomes[-1].training_loss == 0.0
    assert all(
        torch.isfinite(value).all() for value in weights_of(store, outcomes[-1].backbone).values()
    )
