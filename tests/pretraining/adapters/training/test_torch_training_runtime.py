"""What the runtime that trains owes beyond the port: the arithmetic it claims to do.

The port's contract covers the shape of a run. Here are the claims only a runtime that really
trains can be held to — that accumulation is the larger batch, that a resumed run is the run it
resumed, and that the weights it stored are the weights it trained.
"""

import math
import re
from typing import Any

import pytest

from emblema.pretraining.domain.exceptions import (
    DivergedRunError,
    IncompatibleStartError,
    UnsupportedPrecisionError,
)
from emblema.pretraining.domain.masking_strategy import MaskingStrategy
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.corpus_passes import CorpusPasses
from emblema.pretraining.domain.training.objective_loss import LossKind, ObjectiveLoss
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget
from emblema.pretraining.domain.training.precision import Precision
from emblema.pretraining.domain.training.trained_parameters import TrainedParameters
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.retention import Retention
from emblema.shared.kernel.tokens import Token, TokenWindow
from tests.support.experiments import (
    CELLS,
    CHANNEL_NAMES,
    CONTRAST,
    LATENT,
    OBSERVATION,
    SUMMARY,
    budget,
    checksum_of,
    configuration,
    continued,
    corpus,
    irregular,
    mixture,
)

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.objective.cell_reconstruction import (  # noqa: E402
    CellReconstruction,
)
from emblema.pretraining.adapters.objective.contrastive_reconstruction import (  # noqa: E402
    ContrastiveReconstruction,
)
from emblema.pretraining.adapters.objective.latent_prediction import (  # noqa: E402
    LatentPrediction,
)
from emblema.pretraining.adapters.objective.masked_reconstruction import (  # noqa: E402
    MaskedReconstruction,
)
from emblema.pretraining.adapters.objective.objective_score import ObjectiveScore  # noqa: E402
from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape  # noqa: E402
from emblema.pretraining.adapters.objective.pretraining_objective import (  # noqa: E402
    PretrainingObjective,
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
# What a run may train: the value at its hidden tokens, a teacher's state there, the value
# beside the contrast of the window's halves, or the cells of the window with their presence and
# count; what the runtime does is the same under each.
OBJECTIVES: list[dict[str, object]] = [
    {"target": ObjectiveTarget.value()},
    {"target": LATENT},
    {"contrast": CONTRAST},
    {"architecture": CELLS, "observation": OBSERVATION},
]
OBJECTIVE_IDS = ["value", "latent", "contrast", "cells"]


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


@pytest.mark.parametrize("objective", OBJECTIVES, ids=OBJECTIVE_IDS)
def test_the_same_configuration_twice_trains_the_same_weights(objective: dict[str, object]) -> None:
    stated = configuration(budget=budget(epochs=1, batch_size=2), **objective)
    first, second = InMemoryArtifactStore(), InMemoryArtifactStore()

    one = list(TorchTrainingRuntime(first, device="cpu").train(stated, MIXTURE))
    two = list(TorchTrainingRuntime(second, device="cpu").train(stated, MIXTURE))

    assert one[-1].training_loss == two[-1].training_loss
    assert one[-1].backbone == two[-1].backbone


def test_a_run_that_hides_the_tail_trains_and_repeats_itself() -> None:
    forecasting = configuration(
        budget=budget(epochs=1, batch_size=2),
        masking=MaskingStrategy(
            channel_rate=0.15,
            block_rate=0.0,
            block_span=0.5,
            token_rate=0.0,
            horizon_rate=1.0,
            horizon_min_span=0.15,
            horizon_max_span=0.5,
        ),
    )
    first, second = InMemoryArtifactStore(), InMemoryArtifactStore()

    one = list(TorchTrainingRuntime(first, device="cpu").train(forecasting, MIXTURE))
    two = list(TorchTrainingRuntime(second, device="cpu").train(forecasting, MIXTURE))

    assert math.isfinite(one[-1].training_loss)
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


@pytest.mark.parametrize("objective", OBJECTIVES, ids=OBJECTIVE_IDS)
def test_a_resumed_run_ends_where_the_uninterrupted_one_did(objective: dict[str, object]) -> None:
    stated = configuration(
        budget=budget(epochs=2, batch_size=2),
        checkpoint=CheckpointPolicy(every_steps=3),
        **objective,
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
    # The trailing copy of the encoder is state the optimiser does not hold, so it is checked by
    # name: a resume that rebuilt it rather than restored it would still agree on the encoder.
    assert any(key.startswith("teacher.") for key in trained) == stated.target.is_latent
    assert any(key.startswith("projection.") for key in trained) == stated.contrast.is_stated


def test_a_checkpoint_records_where_the_run_stood() -> None:
    stated = configuration(
        budget=budget(epochs=2, batch_size=2), checkpoint=CheckpointPolicy(every_steps=3)
    )
    store = InMemoryArtifactStore()

    outcomes = list(TorchTrainingRuntime(store, device="cpu").train(stated, MIXTURE))

    written = checkpoint_at(store, outcomes[0].checkpoint, RunSignature.of(stated, MIXTURE))
    assert (written.position.epoch, written.position.batches, written.position.steps) == (0, 3, 3)


@pytest.mark.parametrize("objective", OBJECTIVES, ids=OBJECTIVE_IDS)
def test_what_a_run_stored_is_what_it_trained(objective: dict[str, object]) -> None:
    stated = configuration(budget=budget(epochs=1, batch_size=2), **objective)
    store = InMemoryArtifactStore()
    runtime = TorchTrainingRuntime(store, device="cpu")

    outcomes = list(runtime.train(stated, MIXTURE))

    kept = outcomes[-1].backbone
    assert kept is not None
    restored = runtime.restore(kept)
    assert restored.encoder.architecture == stated.architecture
    assert not restored.training
    expected: type[PretrainingObjective] = MaskedReconstruction
    if stated.architecture.reads_cells:
        expected = CellReconstruction
    elif stated.target.is_latent:
        expected = LatentPrediction
    elif stated.contrast.is_stated:
        expected = ContrastiveReconstruction
    assert type(restored) is expected


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
    score = MaskedReconstruction.score

    def not_finite(*arguments: Any) -> ObjectiveScore:
        scored = score(*arguments)
        return ObjectiveScore(
            error=scored.error * math.nan, trivial=scored.trivial, tokens=scored.tokens
        )

    monkeypatch.setattr(MaskedReconstruction, "score", not_finite)
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
    score = MaskedReconstruction.score

    def overflowing(*arguments: Any) -> ObjectiveScore:
        scored = score(*arguments)
        scored.error.register_hook(lambda gradient: gradient * math.inf)
        return scored

    monkeypatch.setattr(MaskedReconstruction, "score", overflowing)
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


def test_a_run_that_predicts_states_is_validated_against_the_visible_mean_state() -> None:
    outcomes = list(
        TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu").train(
            configuration(budget=budget(epochs=1), target=LATENT), MIXTURE
        )
    )

    scored = outcomes[-1].validation
    assert [entry.corpus for entry in scored] == [CORPUS.name]
    assert scored[0].loss == outcomes[-1].validation_loss
    # The targets are normalised states, so predicting their mean costs their spread, which is
    # positive and finite, and what the model costs is read against it.
    assert math.isfinite(scored[0].trivial)
    assert scored[0].trivial > 0
    assert scored[0].relative == pytest.approx(scored[0].loss / scored[0].trivial)
    assert 0 < outcomes[-1].hidden_ratio < 1


def test_a_run_whose_targets_collapse_is_refused_rather_than_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score = LatentPrediction.score

    def collapsed(*arguments: Any) -> ObjectiveScore:
        scored = score(*arguments)
        return ObjectiveScore(
            error=scored.error, trivial=torch.zeros_like(scored.trivial), tokens=scored.tokens
        )

    monkeypatch.setattr(LatentPrediction, "score", collapsed)
    store = RecordingStore()

    with pytest.raises(DivergedRunError, match="have collapsed"):
        list(
            TorchTrainingRuntime(store, device="cpu").train(
                configuration(budget=budget(epochs=1), target=LATENT), MIXTURE
            )
        )
    assert Retention.DURABLE not in store.kept


def test_the_teacher_trails_the_encoder_it_is_trained_beside() -> None:
    """After a run the teacher is neither the encoder it started as nor the one trained."""
    stated = configuration(
        budget=budget(epochs=1, batch_size=2),
        target=ObjectiveTarget.latent(layers=1, momentum=0.5, final_momentum=0.5),
    )
    torch.manual_seed(stated.budget.seed)
    initial = {
        key: value.clone()
        for key, value in ObjectiveShape.of(stated, MIXTURE.vocabulary_size)
        .build()
        .encoder.state_dict()
        .items()
    }
    store = InMemoryArtifactStore()

    outcomes = list(TorchTrainingRuntime(store, device="cpu").train(stated, MIXTURE))

    trained = weights_of(store, outcomes[-1].backbone)
    encoder = {
        k.removeprefix("encoder."): v for k, v in trained.items() if k.startswith("encoder.")
    }
    teacher = {
        k.removeprefix("teacher."): v for k, v in trained.items() if k.startswith("teacher.")
    }
    assert encoder.keys() == teacher.keys() == initial.keys()
    weights = [key for key in encoder if encoder[key].is_floating_point()]
    assert any(not torch.equal(teacher[key], initial[key]) for key in weights)
    assert any(not torch.equal(teacher[key], encoder[key]) for key in weights)


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


def test_the_run_says_what_each_term_of_a_two_term_objective_cost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stated = configuration(budget=budget(epochs=1, batch_size=2), contrast=CONTRAST)

    with caplog.at_level("INFO", logger="emblema.pretraining"):
        list(TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu").train(stated, MIXTURE))

    done = [m for m in (r.getMessage() for r in caplog.records) if m.startswith("epoch 1 of 1")]
    assert len(done) == 1
    assert re.search(r"training [\d.]+ \(reconstruction [\d.]+, contrast [\d.]+\)", done[0])
    assert re.search(
        r"invented validation [\d.]+, [\d.]+ of the trivial predictor's "
        r"\(reconstruction [\d.]+ of [\d.]+, contrast [\d.]+ of [\d.]+\)",
        done[0],
    )


@pytest.mark.parametrize(
    "term", [{"contrast": CONTRAST}, {"summary": SUMMARY}], ids=["contrast", "summary"]
)
def test_a_run_under_a_second_term_is_validated_on_the_hidden_tokens_alone(
    term: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    """The epoch a run keeps is chosen on what a run without the term is chosen on."""
    stated = configuration(budget=budget(epochs=1, batch_size=2), **term)

    with caplog.at_level("INFO", logger="emblema.pretraining"):
        outcomes = list(
            TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu").train(stated, MIXTURE)
        )

    done = [m for m in (r.getMessage() for r in caplog.records) if m.startswith("epoch 1 of 1")]
    said = re.search(
        r"invented validation ([\d.]+), ([\d.]+) of the trivial predictor's "
        r"\(reconstruction ([\d.]+) of ([\d.]+), \w+ [\d.]+ of [\d.]+\)",
        done[0],
    )
    assert said is not None
    validation, relative, reconstruction, trivial = said.groups()
    assert validation == reconstruction
    assert float(relative) == pytest.approx(float(reconstruction) / float(trivial), abs=1e-3)
    scored = outcomes[-1].validation[0]
    assert f"{scored.loss:.5f}" == reconstruction
    assert f"{scored.trivial:.5f}" == trivial


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


def test_a_run_over_corpora_of_both_regimes_learns_and_scores_each() -> None:
    stays = irregular(name="stays", seed=5, training=6, validation=3)
    mixed = TrainingMixture.of(CORPUS, stays)
    stated = configuration(budget=budget(epochs=1, batch_size=2))

    outcomes = list(
        TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu").train(stated, mixed)
    )

    scored = outcomes[-1].validation
    assert math.isfinite(outcomes[-1].training_loss)
    assert [entry.corpus for entry in scored] == ["invented", "stays"]
    assert all(math.isfinite(entry.loss) and entry.tokens > 0 for entry in scored)


def test_a_corpus_read_twice_an_epoch_takes_twice_its_steps(
    caplog: pytest.LogCaptureFixture,
) -> None:
    second = continued(name="second", seed=2, training=4, validation=2)
    mixed = TrainingMixture.of(CORPUS, second)
    weighted = configuration(
        budget=budget(epochs=1, batch_size=2),
        passes=(CorpusPasses(corpus="second", passes=2),),
    )
    runtime = TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu", progress_every=1)

    with caplog.at_level("INFO", logger="emblema.pretraining"):
        list(runtime.train(weighted, mixed))

    progress = [
        record.getMessage() for record in caplog.records if ", step " in record.getMessage()
    ]
    # Four batches of the first corpus and two of the second, read twice: eight steps, not six.
    assert progress[-1].startswith("epoch 1 of 1, step 8 of 8")


TABLE = "encoder.channel_embedding.table.weight"


def pretrained(store: InMemoryArtifactStore) -> ArtifactRef:
    """A backbone trained here over the test corpus, as a stored model a run can continue."""
    outcomes = list(
        TorchTrainingRuntime(store, device="cpu").train(
            configuration(budget=budget(epochs=1, batch_size=2)), MIXTURE
        )
    )
    kept = outcomes[-1].backbone
    assert kept is not None
    return kept


def test_a_run_continuing_a_backbone_starts_from_its_weights_and_draws_the_rest() -> None:
    """Over a wider vocabulary the stored rows are carried and the new rows come from the seed."""
    store = InMemoryArtifactStore()
    parent = pretrained(store)
    wider = mixture(continued(name="second", seed=2))
    stated = configuration(
        name="continued",
        start=parent,
        trained=TrainedParameters.CHANNEL_ROWS,
        budget=budget(epochs=1, batch_size=8, seed=4),
    )

    outcomes = list(TorchTrainingRuntime(store, device="cpu").train(stated, wider))

    child = weights_of(store, outcomes[-1].backbone)
    before = weights_of(store, parent)
    for key, value in before.items():
        if key != TABLE:
            assert torch.equal(child[key], value), key
    # The windows carry channels 1 to 3, whose rows move; the rows the run drew for channels 4
    # and 5 receive no gradient, and the padding row none, so all three stay where they started.
    assert child[TABLE].shape[0] == wider.vocabulary_size + 1
    assert not torch.equal(child[TABLE][1:4], before[TABLE][1:4])
    torch.manual_seed(4)
    drawn = ObjectiveShape.of(stated, wider.vocabulary_size).build().state_dict()[TABLE]
    assert torch.equal(child[TABLE][4:], drawn[4:])
    assert torch.equal(child[TABLE][0], before[TABLE][0])


def test_the_rows_alone_may_be_moved_on_an_encoder_the_seed_drew() -> None:
    stated = configuration(
        name="untrained", trained=TrainedParameters.CHANNEL_ROWS, budget=budget(epochs=1)
    )
    store = InMemoryArtifactStore()

    outcomes = list(TorchTrainingRuntime(store, device="cpu").train(stated, MIXTURE))

    trained = weights_of(store, outcomes[-1].backbone)
    torch.manual_seed(stated.budget.seed)
    drawn = ObjectiveShape.of(stated, MIXTURE.vocabulary_size).build().state_dict()
    assert not torch.equal(trained[TABLE], drawn[TABLE])
    for key, value in drawn.items():
        if key != TABLE:
            assert torch.equal(trained[key], value), key


def test_a_run_continuing_a_backbone_may_move_everything() -> None:
    store = InMemoryArtifactStore()
    parent = pretrained(store)
    stated = configuration(name="continued", start=parent, budget=budget(epochs=1))

    outcomes = list(TorchTrainingRuntime(store, device="cpu").train(stated, MIXTURE))

    child, before = weights_of(store, outcomes[-1].backbone), weights_of(store, parent)
    moved = [key for key, value in before.items() if not torch.equal(child[key], value)]
    assert TABLE in moved
    assert any(key.startswith("encoder.blocks") for key in moved)


def test_a_run_cannot_continue_a_backbone_of_another_shape() -> None:
    store = InMemoryArtifactStore()
    parent = pretrained(store)
    deeper = configuration(name="deeper", start=parent, decoder_layers=2)

    with pytest.raises(IncompatibleStartError, match="another decoder"):
        TorchTrainingRuntime(store, device="cpu").train(deeper, MIXTURE)


def test_a_continued_run_resumes_from_its_checkpoint_as_any_run_does() -> None:
    whole, halved = InMemoryArtifactStore(), InMemoryArtifactStore()
    parents = (pretrained(whole), pretrained(halved))
    assert parents[0] == parents[1]
    stated = configuration(
        name="continued",
        start=parents[0],
        trained=TrainedParameters.CHANNEL_ROWS,
        budget=budget(epochs=2, batch_size=2),
        checkpoint=CheckpointPolicy(every_steps=3),
    )

    uninterrupted = list(TorchTrainingRuntime(whole, device="cpu").train(stated, MIXTURE))
    interrupted = TorchTrainingRuntime(halved, device="cpu").train(stated, MIXTURE)
    stopped = next(interrupted)
    resumed = list(
        TorchTrainingRuntime(halved, device="cpu").train(stated, MIXTURE, stopped.checkpoint)
    )

    trained = weights_of(whole, uninterrupted[-1].backbone)
    picked_up = weights_of(halved, resumed[-1].backbone)
    for key, value in trained.items():
        assert torch.equal(value, picked_up[key]), key


def test_a_run_over_cells_logs_its_three_terms_and_is_validated_on_the_summaries_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stated = configuration(
        budget=budget(epochs=1, batch_size=2), architecture=CELLS, observation=OBSERVATION
    )

    with caplog.at_level("INFO", logger="emblema.pretraining"):
        outcomes = list(
            TorchTrainingRuntime(InMemoryArtifactStore(), device="cpu").train(stated, MIXTURE)
        )

    done = [m for m in (r.getMessage() for r in caplog.records) if m.startswith("epoch 1 of 1")]
    assert len(done) == 1
    assert re.search(
        r"training [\d.]+ \(reconstruction [\d.]+, presence [\d.]+, count [\d.]+\)", done[0]
    )
    said = re.search(
        r"invented validation ([\d.]+), ([\d.]+) of the trivial predictor's "
        r"\(reconstruction ([\d.]+) of ([\d.]+), presence [\d.]+ of [\d.]+, "
        r"count [\d.]+ of [\d.]+\)",
        done[0],
    )
    assert said is not None
    validation, _relative, reconstruction, trivial = said.groups()
    assert validation == reconstruction
    scored = outcomes[-1].validation[0]
    assert f"{scored.loss:.5f}" == reconstruction
    assert f"{scored.trivial:.5f}" == trivial
    assert math.isfinite(outcomes[-1].training_loss)


def test_a_run_over_cells_reads_corpora_of_both_regimes() -> None:
    stays = irregular(name="stays", seed=5, training=6, validation=3)
    mixed = TrainingMixture.of(CORPUS, stays)
    stated = configuration(budget=budget(epochs=1, batch_size=2), architecture=CELLS)
    store = InMemoryArtifactStore()
    runtime = TorchTrainingRuntime(store, device="cpu")

    outcomes = list(runtime.train(stated, mixed))

    scored = outcomes[-1].validation
    assert [entry.corpus for entry in scored] == ["invented", "stays"]
    assert all(math.isfinite(entry.loss) and entry.tokens > 0 for entry in scored)
    kept = outcomes[-1].backbone
    assert kept is not None
    assert isinstance(runtime.restore(kept), CellReconstruction)
