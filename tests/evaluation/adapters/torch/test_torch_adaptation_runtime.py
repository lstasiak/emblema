"""What the torch runtime does to the weights under each mode, on a corpus this test publishes.

The definition-of-done test of the ticket is here: under the frozen probe the backbone's weights
do not change their values, and the low-rank mode leaves the layers it wraps as they were. The
encoder a run received is read back through the factory, which keeps every module it hands out.
"""

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402

from emblema.evaluation.adapters.blocks.published_corpus_blocks import (  # noqa: E402
    PublishedCorpusBlocks,
)
from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone  # noqa: E402
from emblema.evaluation.adapters.torch.lora_linear import LoraLinear  # noqa: E402
from emblema.evaluation.adapters.torch.torch_adaptation_runtime import (  # noqa: E402
    TorchAdaptationRuntime,
)
from emblema.evaluation.domain.exceptions import (  # noqa: E402
    DivergedAdaptationError,
    LoraTargetNotFoundError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget  # noqa: E402
from emblema.evaluation.domain.labels.label_sample import LabelSample  # noqa: E402
from emblema.evaluation.domain.labels.remaining_life_scheme import (  # noqa: E402
    RemainingLifeScheme,
)
from emblema.evaluation.domain.task.downstream_task import DownstreamTask  # noqa: E402
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome  # noqa: E402
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan  # noqa: E402
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode  # noqa: E402
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore  # noqa: E402
from tests.evaluation.support import (  # noqa: E402
    LORA,
    TASK,
    WEIGHTS,
    adaptation_schedule,
    labelled,
    plan,
    task,
)
from tests.support.backbones import PRETRAINED_SEED, SmallBackbones  # noqa: E402
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
    return published.runtime.adapt(stated, published.task, SAMPLE, VALIDATION)


def pretrained_weights() -> dict[str, Tensor]:
    return SmallBackbones(vocabulary_size=len(CHANNELS)).pretrained(WEIGHTS).state_dict()


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
    candidate = AdaptedBackbone.under(plan(mode), published.backbones, starting_at=0.6)

    assert candidate.head.linear.bias.item() == pytest.approx(0.6)


def test_a_sample_under_the_floor_of_steps_is_learnt_for_more_epochs(published: Published) -> None:
    stated = plan(schedule=adaptation_schedule(epochs=1, min_steps=5, batch_size=2))

    outcome = adapt(published, stated)

    assert len(outcome.training_losses) == 3
