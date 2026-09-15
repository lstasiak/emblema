"""A run stopped and picked up is the run that was not stopped.

The claim the training loop is built for: a session that drops at hour two loses the hours, not
the run. It is checked on windows a real corpus produced, over the whole state a run carries —
weights, optimiser, schedule, the generator the masks come from — by comparing an uninterrupted
run with one that was interrupted inside an epoch and resumed from its checkpoint.

Equality here is exact rather than approximate, because on the host the arithmetic is the same
arithmetic in the same order: anything short of bit equality would mean some piece of state was
rebuilt rather than restored, which is exactly the failure this is for. The accelerator legs are
in the verification note, where the same script runs and reports the same table.
"""

import pytest
import torch

from emblema.pretraining.adapters.training.torch_training_runtime import TorchTrainingRuntime
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from tests.support.control_corpus import Control
from tests.support.encoders import SMALL
from tests.support.experiments import budget, configuration

pytestmark = pytest.mark.ml

# Three epochs of eight windows in batches of two, with a checkpoint every fifth step: the run is
# interrupted inside its second epoch, not on a boundary, which is the case a resumed epoch has
# to get right.
CHECKPOINT_EVERY = 5


@pytest.fixture(scope="module")
def corpus(control: Control) -> TrainingCorpus:
    """Windows the control corpus really produced, cut to what a test can train on."""
    return TrainingCorpus(
        name="control-a",
        checksum=control.manifests[0].archived.block.checksum,
        training=control.windows_of(0)[:8],
        validation=control.windows_of(1)[:4],
        vocabulary_size=control.vocabulary_size,
    )


@pytest.fixture(scope="module")
def stated() -> ExperimentConfiguration:
    return configuration(
        architecture=SMALL,
        budget=budget(epochs=3, batch_size=2),
        checkpoint=CheckpointPolicy(every_steps=CHECKPOINT_EVERY),
    )


def weights(store: InMemoryArtifactStore, outcome: EpochOutcome) -> dict[str, torch.Tensor]:
    kept = outcome.backbone
    assert kept is not None
    return TrainedModel.read(store.get(kept)).weights


def test_a_run_resumed_mid_epoch_ends_exactly_where_the_uninterrupted_one_did(
    corpus: TrainingCorpus, stated: ExperimentConfiguration
) -> None:
    whole, dropped = InMemoryArtifactStore(), InMemoryArtifactStore()

    uninterrupted = list(TorchTrainingRuntime(whole, device="cpu").train(stated, corpus))
    epochs = TorchTrainingRuntime(dropped, device="cpu").train(stated, corpus)
    first, second = next(epochs), next(epochs)
    # The third epoch is never asked for: that is the interruption.
    resumed = list(
        TorchTrainingRuntime(dropped, device="cpu").train(stated, corpus, second.checkpoint)
    )

    assert [outcome.epoch for outcome in resumed] == [1, 2]
    assert (first.training_loss, first.validation_loss) == (
        uninterrupted[0].training_loss,
        uninterrupted[0].validation_loss,
    )
    assert resumed[-1].validation_loss == uninterrupted[-1].validation_loss
    trained, picked_up = weights(whole, uninterrupted[-1]), weights(dropped, resumed[-1])
    for key, value in trained.items():
        assert torch.equal(value, picked_up[key]), key


def test_the_weights_of_the_two_runs_are_the_same_artifact(
    corpus: TrainingCorpus, stated: ExperimentConfiguration
) -> None:
    """The store is content-addressed, so equal weights are one reference — the strongest form."""
    whole, dropped = InMemoryArtifactStore(), InMemoryArtifactStore()

    uninterrupted = list(TorchTrainingRuntime(whole, device="cpu").train(stated, corpus))
    epochs = TorchTrainingRuntime(dropped, device="cpu").train(stated, corpus)
    interrupted = [next(epochs), next(epochs)]
    resumed = list(
        TorchTrainingRuntime(dropped, device="cpu").train(
            stated, corpus, interrupted[-1].checkpoint
        )
    )

    assert resumed[-1].backbone == uninterrupted[-1].backbone
