"""What the torch runtime does to the weights under each mode, on a corpus this test publishes.

The definition-of-done test of the ticket is here: under the frozen probe the backbone's weights
do not change their values, and the low-rank mode leaves the layers it wraps as they were. The
encoder a run received is read back through the factory, which keeps every module it hands out.
"""

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates  # noqa: E402
from emblema.evaluation.adapters.blocks.published_corpus_blocks import (  # noqa: E402
    PublishedCorpusBlocks,
)
from emblema.evaluation.adapters.onnx.inference_graph import InferenceGraph  # noqa: E402
from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone  # noqa: E402
from emblema.evaluation.adapters.torch.fitted_candidate import FittedCandidate  # noqa: E402
from emblema.evaluation.adapters.torch.lora_linear import LoraLinear  # noqa: E402
from emblema.evaluation.adapters.torch.ridge_solution import RidgeSolution  # noqa: E402
from emblema.evaluation.adapters.torch.scheduled_training import (  # noqa: E402
    AfterEpoch,
    Forward,
    Loss,
    ScheduledTraining,
)
from emblema.evaluation.adapters.torch.target_link import TargetLink  # noqa: E402
from emblema.evaluation.adapters.torch.torch_adaptation_runtime import (  # noqa: E402
    TorchAdaptationRuntime,
)
from emblema.evaluation.contracts.candidate_kind import CandidateKind  # noqa: E402
from emblema.evaluation.domain.exceptions import (  # noqa: E402
    CandidateNotRetainableError,
    DivergedAdaptationError,
    InvalidTrainingRegimeError,
    LoraTargetNotFoundError,
)
from emblema.evaluation.domain.heads.head_pooling import (  # noqa: E402
    HeadPooling,
    PoolingScheme,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget  # noqa: E402
from emblema.evaluation.domain.labels.label_sample import LabelSample  # noqa: E402
from emblema.evaluation.domain.labels.remaining_life_scheme import (  # noqa: E402
    RemainingLifeScheme,
)
from emblema.evaluation.domain.labels.target_kind import TargetKind  # noqa: E402
from emblema.evaluation.domain.task.downstream_task import DownstreamTask  # noqa: E402
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome  # noqa: E402
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan  # noqa: E402
from emblema.evaluation.domain.transfer.encoder_setting import EncoderSetting  # noqa: E402
from emblema.evaluation.domain.transfer.layer_reading import LayerReading  # noqa: E402
from emblema.evaluation.domain.transfer.training_regime import (  # noqa: E402
    ClassWeight,
    HeadStart,
    StopDivision,
    TrainingRegime,
)
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode  # noqa: E402
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from emblema.shared.kernel.ordering import seeded_rank  # noqa: E402
from tests.evaluation.support import (  # noqa: E402
    LORA,
    OUTCOME,
    OUTCOMES,
    PENALTIES,
    TASK,
    WEIGHTS,
    adaptation_schedule,
    labelled,
    plan,
    task,
)
from tests.support.backbones import PRETRAINED_SEED, SmallBackbones  # noqa: E402
from tests.support.encoders import SMALL  # noqa: E402
from tests.support.published import CHANNELS, publish  # noqa: E402

pytestmark = pytest.mark.ml

# The block holds four windows; three tune and the held-out one validates. Positions are what
# the runtime reads by; the units named here are the task's, not the block's.
SAMPLE = LabelSample(
    task=TASK,
    windows=(labelled("a", 0, 10.0, 5.0), labelled("a", 1, 15.0, 9.0), labelled("b", 3, 10.0, 2.0)),
    budget=LabelBudget.of(3),
    seed=7,
)
VALIDATION = (labelled("c", 2, 10.0, 8.0),)
CEILING = 10.0


class Published(NamedTuple):
    """A runtime over a published corpus, the task defined against it, and its factory."""

    runtime: TorchAdaptationRuntime
    task: DownstreamTask
    backbones: SmallBackbones


@pytest.fixture
def published(tmp_path: Path) -> Published:
    store = InMemoryArtifactStore()
    manifest = publish(store, tmp_path / "scratch").manifest
    backbones = SmallBackbones(vocabulary_size=len(CHANNELS))
    runtime = TorchAdaptationRuntime(
        backbones, PublishedCorpusBlocks(store, tmp_path / "workspace"), device="cpu"
    )
    defined = replace(task(), manifest=manifest, labels=RemainingLifeScheme(CEILING))
    return Published(runtime, defined, backbones)


def adapt(published: Published, stated: AdaptationPlan) -> AdaptationOutcome:
    return published.runtime.adapt(stated, published.task, SAMPLE, VALIDATION, retain=False)


def test_a_run_asked_to_keep_what_it_fitted_stores_both_forms_under_one_manifest(
    tmp_path: Path,
) -> None:
    # What a campaign keeps of its endpoint, and the one thing another context may promote: the
    # state the run fitted, which reads back as a candidate, and the graph derived from it, which
    # answers the validation window as the run did — named together by the manifest the outcome
    # points at, so the campaign's one reference reaches both.
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "scratch")
    blocks = PublishedCorpusBlocks(store, tmp_path / "workspace")
    runtime = TorchAdaptationRuntime(
        SmallBackbones(vocabulary_size=len(CHANNELS)), blocks, device="cpu", store=store
    )
    defined = replace(task(), manifest=published.manifest, labels=RemainingLifeScheme(CEILING))

    outcome = runtime.adapt(plan(), defined, SAMPLE, VALIDATION, retain=True)

    assert outcome.artifact is not None
    kept = KeptCandidates(store).read(outcome.artifact)
    assert kept.kind is CandidateKind.NEURAL
    assert kept.corpus_manifest == published.manifest
    fitted = FittedCandidate.read(store.get(kept.measured.artifact))
    assert (fitted.vocabulary_size, fitted.link) == (
        len(CHANNELS),
        TargetLink(TargetKind.CONTINUOUS, CEILING),
    )
    graph = kept.find(InferenceGraph.FORMAT)
    assert graph is not None
    held = blocks.block_of(blocks.manifest_of(published.manifest)).at([2])
    answered = InferenceGraph.read(store.get(graph.artifact)).predict(
        TokenTensors.from_windows(held)
    )
    assert float(answered[0]) == pytest.approx(outcome.predictions[0].predicted, abs=1e-4)
    assert graph.deviation is not None
    assert graph.deviation <= 1e-4


@pytest.mark.parametrize("mode", [TransferMode.FULL_FINE_TUNING, TransferMode.FROZEN_RIDGE])
def test_a_kept_candidate_of_a_binary_task_answers_probabilities_through_its_graph(
    tmp_path: Path, mode: TransferMode
) -> None:
    # The sigmoid belongs to the graph, not to whoever serves it: the graph answers the same
    # probability the run scored, and the measured form records the link it was read through.
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "scratch")
    blocks = PublishedCorpusBlocks(store, tmp_path / "workspace")
    runtime = TorchAdaptationRuntime(
        SmallBackbones(vocabulary_size=len(CHANNELS)), blocks, device="cpu", store=store
    )
    defined = replace(task(), manifest=published.manifest, labels=OUTCOME, strata=OUTCOMES)
    # Two of each outcome, the fewest the probe's folds can choose a penalty over.
    outcomes = replace(
        SAMPLE,
        windows=(
            labelled("a", 0, 10.0, 1.0),
            labelled("a", 1, 15.0, 1.0),
            labelled("b", 2, 10.0, 0.0),
            labelled("b", 3, 10.0, 0.0),
        ),
        budget=LabelBudget.of(4),
    )

    outcome = runtime.adapt(
        plan(mode), defined, outcomes, (labelled("c", 2, 10.0, 0.0),), retain=True
    )

    assert outcome.artifact is not None
    kept = KeptCandidates(store).read(outcome.artifact)
    fitted = FittedCandidate.read(store.get(kept.measured.artifact))
    assert fitted.link == TargetLink(TargetKind.BINARY, 1.0)
    graph = kept.find(InferenceGraph.FORMAT)
    assert graph is not None
    held = blocks.block_of(blocks.manifest_of(published.manifest)).at([2])
    answered = float(
        InferenceGraph.read(store.get(graph.artifact)).predict(TokenTensors.from_windows(held))[0]
    )
    assert 0.0 < answered < 1.0
    assert answered == pytest.approx(outcome.predictions[0].predicted, abs=1e-4)


def test_a_runtime_with_nowhere_to_keep_a_candidate_refuses_to_keep_one(
    published: Published,
) -> None:
    with pytest.raises(CandidateNotRetainableError):
        published.runtime.adapt(plan(), published.task, SAMPLE, VALIDATION, retain=True)


def pretrained_weights() -> dict[str, Tensor]:
    return (
        SmallBackbones(vocabulary_size=len(CHANNELS))
        .pretrained(WEIGHTS, vocabulary_size=len(CHANNELS), dropout=0.0)
        .state_dict()
    )


def base_weights(encoder: nn.Module) -> dict[str, Tensor]:
    """The encoder's weights with the low-rank wrappers read through, by their original names."""
    return {
        name.replace(".base.", "."): value
        for name, value in encoder.state_dict().items()
        if ".down" not in name and ".up" not in name
    }


def test_under_the_frozen_probe_the_backbone_keeps_every_value(published: Published) -> None:
    adapt(published, plan(TransferMode.FROZEN_PROBE))

    (received,) = published.backbones.built
    after, before = received.state_dict(), pretrained_weights()
    assert after.keys() == before.keys()
    assert all(torch.equal(after[name], before[name]) for name in before)


def test_under_the_low_rank_mode_the_wrapped_layers_keep_every_value(
    published: Published,
) -> None:
    outcome = adapt(published, plan(TransferMode.LORA))

    (received,) = published.backbones.built
    after, before = base_weights(received), pretrained_weights()
    assert after.keys() == before.keys()
    assert all(torch.equal(after[name], before[name]) for name in before)
    assert isinstance(received.get_submodule("blocks.0.attention.qkv"), LoraLinear)
    assert outcome.trainable_parameters < sum(v.numel() for v in before.values())


def test_under_full_fine_tuning_the_backbone_moves(published: Published) -> None:
    adapt(published, plan(TransferMode.FULL_FINE_TUNING))

    (received,) = published.backbones.built
    after, before = received.state_dict(), pretrained_weights()
    assert any(not torch.equal(after[name], before[name]) for name in before)


def test_the_control_arm_never_asks_for_the_pretrained_weights(published: Published) -> None:
    adapt(published, plan(TransferMode.FROM_SCRATCH))

    assert published.backbones.requested == []


def test_answers_come_back_in_cycles_not_in_units_of_the_ceiling(published: Published) -> None:
    outcome = adapt(published, plan(TransferMode.FROZEN_PROBE))

    (answer,) = outcome.predictions
    assert answer.window == VALIDATION[0].window
    assert answer.target == 8.0
    # A head at its random start answers within a few units of zero; the ceiling puts that in
    # cycles, so an answer that stayed in ceiling units would be an order of magnitude smaller.
    assert abs(answer.predicted) < 10 * CEILING


@pytest.mark.parametrize("mode", list(TransferMode))
def test_the_same_plan_over_the_same_sample_repeats_bit_for_bit(
    published: Published, mode: TransferMode
) -> None:
    first = adapt(published, plan(mode))
    again = adapt(published, plan(mode))

    assert first.predictions == again.predictions
    assert first.training_losses == again.training_losses


def test_another_seed_gives_another_run(published: Published) -> None:
    first = adapt(published, plan(TransferMode.FULL_FINE_TUNING))
    other = adapt(published, plan(TransferMode.FULL_FINE_TUNING, seed=2))

    assert first.predictions != other.predictions


def test_a_loss_that_stops_being_finite_ends_the_run(published: Published) -> None:
    runaway = plan(TransferMode.FULL_FINE_TUNING, schedule=adaptation_schedule(learning_rate=1e30))

    with pytest.raises(DivergedAdaptationError):
        adapt(published, runaway)


def test_a_low_rank_update_naming_a_layer_the_backbone_lacks_is_refused(
    published: Published,
) -> None:
    with pytest.raises(LoraTargetNotFoundError):
        adapt(published, plan(TransferMode.LORA, lora=replace(LORA, targets=("ghost",))))


def test_the_pretrained_encoder_the_tests_stand_on_is_the_same_every_time() -> None:
    first, again = pretrained_weights(), pretrained_weights()

    assert PRETRAINED_SEED == 1
    assert all(torch.equal(first[name], again[name]) for name in first)


def test_the_rate_follows_the_schedule_on_every_step_the_probe_included(
    published: Published, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optimiser's rate is the schedule's factor of the peak, stepped once per batch."""
    seen: list[float] = []
    original = torch.optim.lr_scheduler.LambdaLR

    class Recording(original):  # type: ignore[misc, valid-type]
        def step(self, epoch: int | None = None) -> None:
            seen.append(self.optimizer.param_groups[0]["lr"])
            super().step(epoch)

    monkeypatch.setattr(torch.optim.lr_scheduler, "LambdaLR", Recording)
    schedule = adaptation_schedule(
        epochs=3, batch_size=2, learning_rate=1e-2, warmup_fraction=0.4, final_lr_fraction=0.1
    )
    stated = plan(TransferMode.FROZEN_PROBE, schedule=schedule)

    adapt(published, stated)

    # The scheduler steps once as it is built, to set the first rate; every record after that is
    # the rate the optimiser had just stepped under.
    shape = schedule.learning_rate_schedule(len(SAMPLE.windows))
    stepped_under = seen[1:]
    assert len(stepped_under) == shape.total_steps == 6
    assert stepped_under == pytest.approx([1e-2 * shape.factor(step) for step in range(6)])
    assert stepped_under[0] < stepped_under[1] == 1e-2 > stepped_under[-1]


@pytest.mark.parametrize("mode", list(TransferMode))
def test_the_head_starts_where_it_is_told_whatever_the_mode(
    published: Published, mode: TransferMode
) -> None:
    candidate = AdaptedBackbone.under(
        plan(mode), published.backbones, vocabulary_size=len(CHANNELS), starting_at=0.6
    )

    assert candidate.head.linear.bias.item() == pytest.approx(0.6)


def test_a_sample_under_the_floor_of_steps_is_learnt_for_more_epochs(published: Published) -> None:
    stated = plan(schedule=adaptation_schedule(epochs=1, min_steps=5, batch_size=2))

    outcome = adapt(published, stated)

    assert len(outcome.training_losses) == 3


def test_under_the_frozen_probe_a_learnt_pooling_trains_and_the_backbone_still_keeps_every_value(
    published: Published,
) -> None:
    attention = plan(
        TransferMode.FROZEN_PROBE, pooling=HeadPooling(pooling=PoolingScheme.ATTENTION)
    )

    outcome = adapt(published, attention)

    (received,) = published.backbones.built
    before = pretrained_weights()
    assert all(torch.equal(received.state_dict()[name], before[name]) for name in before)
    # The head and the query: the query moves off zero only if the probe ran it in the loop.
    assert outcome.trainable_parameters == SMALL.width + 1 + SMALL.width


def test_a_tail_an_attention_and_static_features_set_apart_each_answer_the_task(
    published: Published,
) -> None:
    tail = plan(
        TransferMode.FULL_FINE_TUNING,
        pooling=HeadPooling(pooling=PoolingScheme.TAIL, tail_share=0.5),
    )
    attention = plan(
        TransferMode.FROM_SCRATCH, pooling=HeadPooling(pooling=PoolingScheme.ATTENTION)
    )

    apart = plan(TransferMode.FROM_SCRATCH, pooling=HeadPooling.mean().tuned("statics", "apart"))

    for stated in (tail, attention, apart):
        outcome = adapt(published, stated)
        assert len(outcome.predictions) == len(VALIDATION)
        assert outcome.plan.parameters()["pooling"] == str(stated.pooling.pooling)
        assert outcome.plan.parameters()["statics"] == str(stated.pooling.statics)


def test_under_the_closed_form_probe_the_backbone_keeps_every_value_and_no_step_is_taken(
    published: Published,
) -> None:
    outcome = adapt(published, plan(TransferMode.FROZEN_RIDGE))

    (received,) = published.backbones.built
    before = pretrained_weights()
    assert all(torch.equal(received.state_dict()[name], before[name]) for name in before)
    assert outcome.training_losses == ()
    assert outcome.optimiser_steps == 0
    assert outcome.trainable_parameters == SMALL.width + 1
    assert len(outcome.predictions) == len(VALIDATION)
    assert outcome.plan.parameters()["ridge_penalties"] == "0.1 1 10"


def test_the_closed_form_probe_answers_what_its_solution_says_over_the_pooled_states(
    published: Published,
) -> None:
    stated = plan(
        TransferMode.FROZEN_RIDGE,
        pooling=HeadPooling(pooling=PoolingScheme.TAIL, tail_share=0.5),
    )

    outcome = adapt(published, stated)

    (received,) = published.backbones.built
    candidate = AdaptedBackbone.under(
        stated, published.backbones, vocabulary_size=len(CHANNELS), starting_at=0.0
    )
    candidate.encoder.load_state_dict(received.state_dict())
    manifest = published.runtime._blocks.manifest_of(published.task.manifest)
    block = published.runtime._blocks.block_of(manifest)
    tuning = block.at([labelled.window.position for labelled in SAMPLE.windows])
    held = block.at([labelled.window.position for labelled in VALIDATION])
    candidate.eval()
    with torch.no_grad():
        states = candidate.embed(TokenTensors.from_windows(list(tuning)))
        targets = torch.tensor([w.target / CEILING for w in SAMPLE.windows])
        RidgeSolution.fitted(states, targets, stated.ridge or PENALTIES).applied_to(candidate.head)
        answered = candidate(TokenTensors.from_windows(list(held))).double() * CEILING
    assert [p.predicted for p in outcome.predictions] == pytest.approx(answered.tolist(), rel=1e-4)


@pytest.mark.parametrize("mode", [TransferMode.FROM_SCRATCH, TransferMode.FULL_FINE_TUNING])
def test_the_encoder_drops_what_the_plan_says_while_it_learns(
    published: Published, mode: TransferMode
) -> None:
    dropped = adapt(published, plan(mode, encoder=EncoderSetting(dropout=0.2)))
    standard = adapt(published, plan(mode))

    received = published.backbones.built[0]
    assert {module.p for module in received.modules() if isinstance(module, nn.Dropout)} == {0.2}
    assert dropped.training_losses != standard.training_losses


def test_a_run_under_a_dropout_repeats_bit_for_bit(published: Published) -> None:
    stated = plan(TransferMode.FULL_FINE_TUNING, encoder=EncoderSetting(dropout=0.2))

    first, again = adapt(published, stated), adapt(published, stated)

    assert first.training_losses == again.training_losses
    assert first.predictions == again.predictions


def test_the_answers_are_read_with_the_dropout_off(published: Published) -> None:
    adapt(published, plan(TransferMode.FULL_FINE_TUNING, encoder=EncoderSetting(dropout=0.5)))

    received = published.backbones.built[0]
    assert not any(module.training for module in received.modules())


def test_the_encoder_reads_the_gridded_readings_where_the_plan_lays_them_on_a_grid(
    published: Published,
) -> None:
    gridded = plan(TransferMode.FULL_FINE_TUNING, encoder=EncoderSetting(grid_resolution=0.5))

    first, again = adapt(published, gridded), adapt(published, gridded)
    raw = adapt(published, plan(TransferMode.FULL_FINE_TUNING))

    assert first.predictions == again.predictions
    assert first.predictions != raw.predictions
    assert first.plan.parameters()["grid_resolution"] == 0.5


def test_a_candidate_fitted_on_gridded_readings_is_kept_without_a_graph(
    tmp_path: Path,
) -> None:
    # A campaign keeps one cell of every candidate, so a gridded one is kept too; its graph
    # would take the raw readings a served model is handed, so only the fitted state is kept,
    # naming the grid it was fitted on, and serving refuses it as a form it cannot run.
    store = InMemoryArtifactStore()
    published = publish(store, tmp_path / "scratch")
    runtime = TorchAdaptationRuntime(
        SmallBackbones(vocabulary_size=len(CHANNELS)),
        PublishedCorpusBlocks(store, tmp_path / "workspace"),
        device="cpu",
        store=store,
    )
    defined = replace(task(), manifest=published.manifest, labels=RemainingLifeScheme(CEILING))
    gridded = plan(TransferMode.FULL_FINE_TUNING, encoder=EncoderSetting(grid_resolution=0.5))

    outcome = runtime.adapt(gridded, defined, SAMPLE, VALIDATION, retain=True)

    assert outcome.artifact is not None
    kept = KeptCandidates(store).read(outcome.artifact)
    assert [form.format for form in kept.representations] == [FittedCandidate.FORMAT]
    fitted = FittedCandidate.read(store.get(kept.measured.artifact))
    assert fitted.parameters["grid_resolution"] == 0.5


def test_a_run_that_stops_learns_from_the_units_left_and_ends_within_its_patience(
    published: Published,
) -> None:
    # Two units: one is held out for the stop, the run learns from the other and is held to the
    # whole sample's count of epochs at most.
    stopping = plan(
        schedule=adaptation_schedule(epochs=6, batch_size=2),
        regime=TrainingRegime(stop_share=0.5, patience=1),
    )

    outcome = adapt(published, stopping)

    assert 1 <= len(outcome.training_losses) <= 6
    assert all(torch.isfinite(torch.tensor(outcome.training_losses)))
    # One window of the held-out unit or two of the other: one step an epoch either way here,
    # but the count runs over the windows learnt from, not the three given.
    assert outcome.stop_windows in (1, 2)
    learnt = len(SAMPLE.windows) - outcome.stop_windows
    assert outcome.optimiser_steps == len(outcome.training_losses) * -(-learnt // 2)
    assert len(outcome.predictions) == len(VALIDATION)


def test_a_stop_needs_a_unit_to_hold_out_and_one_to_learn_from(published: Published) -> None:
    one_unit = replace(SAMPLE, windows=SAMPLE.windows[:2], budget=LabelBudget.of(2))
    stopping = plan(regime=TrainingRegime(stop_share=0.5, patience=1))

    with pytest.raises(InvalidTrainingRegimeError, match="still learn"):
        published.runtime.adapt(stopping, published.task, one_unit, VALIDATION, retain=False)


def test_withheld_channels_change_what_is_learnt_and_repeat_under_the_seed(
    published: Published,
) -> None:
    # Only the first window holds two channels, so a draw changes a step about half the time;
    # eight epochs leave almost no chance that none does.
    schedule = adaptation_schedule(epochs=8, batch_size=2)
    perturbed = plan(schedule=schedule, regime=TrainingRegime(channel_dropout=0.5))

    first = adapt(published, perturbed)
    again = adapt(published, perturbed)
    plain = adapt(published, plan(schedule=schedule))

    assert first.training_losses == again.training_losses
    assert first.training_losses != plain.training_losses


def test_a_class_weight_is_refused_for_a_quantity(published: Published) -> None:
    weighted = plan(regime=TrainingRegime(class_weight=ClassWeight.RATIO))

    with pytest.raises(InvalidTrainingRegimeError, match="no classes"):
        adapt(published, weighted)


def test_a_head_solved_first_starts_the_steps_from_the_solution_over_the_untouched_encoder(
    published: Published, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The probe phase changes the head alone, and the encoder moves only under the steps after
    # it: what the loop receives is read at the moment it is called, before its first step.
    stated = plan(regime=TrainingRegime(head_start=HeadStart.SOLVED), ridge=PENALTIES)
    received: dict[str, Tensor] = {}
    losses = ScheduledTraining.losses

    def spied(
        self: ScheduledTraining,
        model: nn.Module,
        trainable: Iterable[nn.Parameter],
        forward: Forward,
        targets: Tensor,
        *,
        loss: Loss,
        epochs: int | None = None,
        after_epoch: AfterEpoch | None = None,
    ) -> list[float]:
        received.update({k: v.detach().clone() for k, v in model.state_dict().items()})
        return losses(
            self,
            model,
            trainable,
            forward,
            targets,
            loss=loss,
            epochs=epochs,
            after_epoch=after_epoch,
        )

    monkeypatch.setattr(ScheduledTraining, "losses", spied)

    adapt(published, stated)

    (encoder,) = published.backbones.built
    before = pretrained_weights()
    assert all(torch.equal(received[f"encoder.{name}"], before[name]) for name in before)
    candidate = AdaptedBackbone.under(
        stated, published.backbones, vocabulary_size=len(CHANNELS), starting_at=0.0
    )
    candidate.encoder.load_state_dict(before)
    manifest = published.runtime._blocks.manifest_of(published.task.manifest)
    tuning = published.runtime._blocks.block_of(manifest).at(
        [labelled.window.position for labelled in SAMPLE.windows]
    )
    candidate.eval()
    with torch.no_grad():
        states = candidate.embed(TokenTensors.from_windows(list(tuning)))
    targets = torch.tensor([w.target / CEILING for w in SAMPLE.windows])
    RidgeSolution.fitted(states, targets, PENALTIES).applied_to(candidate.head)
    assert torch.allclose(received["head.linear.weight"], candidate.head.linear.weight, atol=1e-5)
    assert any(not torch.equal(encoder.state_dict()[name], before[name]) for name in before)


def test_a_head_solved_first_starts_the_control_arm_too(published: Published) -> None:
    started = plan(
        TransferMode.FROM_SCRATCH,
        regime=TrainingRegime(head_start=HeadStart.SOLVED),
        ridge=PENALTIES,
    )

    outcome = adapt(published, started)

    assert published.backbones.requested == []
    assert outcome.optimiser_steps > 0
    assert outcome.plan.parameters()["ridge_penalties"] == "0.1 1 10"


def test_the_closed_form_probe_reads_an_encoder_at_its_initialisation(
    published: Published,
) -> None:
    outcome = adapt(published, plan(TransferMode.FROZEN_RIDGE, backbone=None))

    assert published.backbones.requested == []
    assert outcome.optimiser_steps == 0
    assert outcome.plan.parameters()["backbone"] == ""
    assert len(outcome.predictions) == len(VALIDATION)


def test_a_patience_in_steps_waits_out_the_warmup_before_it_may_stop(
    published: Published,
) -> None:
    # Every epoch takes as many steps as the next, so a warmup over half the run ends half-way,
    # and a patience of one step cannot end the run before the epoch after that.
    schedule = adaptation_schedule(epochs=6, batch_size=1, warmup_fraction=0.5)
    in_steps = adapt(
        published, plan(schedule=schedule, regime=TrainingRegime(stop_share=0.5, patience_steps=1))
    )

    assert len(in_steps.training_losses) >= 4


OUTCOME_SAMPLE = LabelSample(
    task=TASK,
    windows=(
        labelled("a", 0, 10.0, 1.0),
        labelled("b", 1, 15.0, 1.0),
        labelled("c", 2, 10.0, 0.0),
        labelled("d", 3, 10.0, 0.0),
        labelled("e", 3, 12.0, 0.0),
        labelled("f", 2, 12.0, 0.0),
    ),
    budget=LabelBudget.of(6),
    seed=7,
)


@pytest.mark.parametrize("seed", range(1, 21))
def test_a_stop_divided_by_outcome_holds_out_and_learns_from_each_outcome(seed: int) -> None:
    stopping = plan(
        seed=seed,
        regime=TrainingRegime(stop_share=0.25, patience=1, stop_division=StopDivision.OUTCOMES),
    )

    learnt, stopped = TorchAdaptationRuntime._divided(OUTCOME_SAMPLE, stopping, TargetKind.BINARY)

    assert sorted(labelled.target for labelled in stopped) == [0.0, 1.0]
    assert {labelled.target for labelled in learnt} == {0.0, 1.0}
    assert {w.window.unit for w in learnt}.isdisjoint({w.window.unit for w in stopped})


def test_a_stop_divided_by_units_is_the_division_it_always_was() -> None:
    # The stop registered at every stay ran under this division; a seed draws the same units.
    stopping = plan(seed=3, regime=TrainingRegime(stop_share=0.25, patience=1))

    _, stopped = TorchAdaptationRuntime._divided(OUTCOME_SAMPLE, stopping, TargetKind.BINARY)

    units = sorted({str(w.window.unit) for w in OUTCOME_SAMPLE.windows})
    ranked = sorted(units, key=lambda unit: seeded_rank(3, "stop", unit))
    assert {str(w.window.unit) for w in stopped} == set(ranked[:2])


def test_a_division_by_outcome_is_refused_for_a_quantity() -> None:
    stopping = plan(
        regime=TrainingRegime(stop_share=0.25, patience=1, stop_division=StopDivision.OUTCOMES)
    )

    with pytest.raises(InvalidTrainingRegimeError, match="only an outcome"):
        TorchAdaptationRuntime._divided(SAMPLE, stopping, TargetKind.CONTINUOUS)


@pytest.mark.parametrize("reading", ["0", "mean", "concat"])
def test_under_the_closed_form_probe_another_layer_answers_the_task_and_is_recorded(
    published: Published, reading: str
) -> None:
    stated = plan(TransferMode.FROZEN_RIDGE, encoder=EncoderSetting(layer=LayerReading.of(reading)))

    outcome = adapt(published, stated)

    assert len(outcome.predictions) == len(VALIDATION)
    assert outcome.plan.parameters()["encoder_layer"] == reading
