import pytest

torch = pytest.importorskip("torch")

from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone  # noqa: E402
from emblema.evaluation.adapters.torch.lora_linear import LoraLinear  # noqa: E402
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode  # noqa: E402
from tests.evaluation.support import WEIGHTS, plan  # noqa: E402
from tests.support.backbones import SmallBackbones  # noqa: E402
from tests.support.encoders import SMALL  # noqa: E402
from tests.support.token_tensors import VOCABULARY_SIZE, random_batch  # noqa: E402

pytestmark = pytest.mark.ml

HEAD = SMALL.width + 1
ENCODER = SMALL.parameter_count(VOCABULARY_SIZE)
# Two blocks, each with qkv (w -> 3w), projection (w -> w) and two feed-forward linears
# (w -> 4w, 4w -> w), at rank two: rank * (in + out) for each.
LORA_UPDATES = 2 * 2 * ((32 + 96) + (32 + 32) + (32 + 64) + (64 + 32))


def candidate(mode: TransferMode) -> tuple[AdaptedBackbone, SmallBackbones]:
    backbones = SmallBackbones()
    torch.manual_seed(5)
    return AdaptedBackbone.under(plan(mode), backbones), backbones


@pytest.mark.parametrize(
    ("mode", "trainable"),
    [
        (TransferMode.FROM_SCRATCH, ENCODER + HEAD),
        (TransferMode.FROZEN_PROBE, HEAD),
        (TransferMode.LORA, LORA_UPDATES + HEAD),
        (TransferMode.FULL_FINE_TUNING, ENCODER + HEAD),
    ],
)
def test_each_mode_leaves_exactly_its_weights_free(mode: TransferMode, trainable: int) -> None:
    built, _ = candidate(mode)

    assert sum(p.numel() for p in built.trainable_parameters()) == trainable


@pytest.mark.parametrize(
    "mode", [TransferMode.FROZEN_PROBE, TransferMode.LORA, TransferMode.FULL_FINE_TUNING]
)
def test_a_transfer_mode_asks_for_the_plans_weights(mode: TransferMode) -> None:
    _, backbones = candidate(mode)

    assert backbones.requested == [WEIGHTS]


def test_the_control_arm_starts_from_weights_of_its_own() -> None:
    built, backbones = candidate(TransferMode.FROM_SCRATCH)
    pretrained = backbones.pretrained(WEIGHTS)

    assert backbones.requested == [WEIGHTS]
    assert not torch.equal(
        built.encoder.state_dict()["value_projection.weight"],
        pretrained.state_dict()["value_projection.weight"],
    )


def test_the_head_starts_from_the_same_weights_under_every_mode() -> None:
    heads = [candidate(mode)[0].head.linear.weight for mode in TransferMode]

    assert all(torch.equal(head, heads[0]) for head in heads[1:])


def test_the_low_rank_mode_wraps_the_layers_the_plan_names() -> None:
    built, _ = candidate(TransferMode.LORA)

    assert isinstance(built.encoder.get_submodule("blocks.0.attention.qkv"), LoraLinear)


def test_the_candidate_answers_one_number_per_window() -> None:
    built, _ = candidate(TransferMode.FULL_FINE_TUNING)

    answers = built(random_batch(3, 9, seed=2))

    assert answers.shape == (3,)
    assert torch.isfinite(answers).all()
