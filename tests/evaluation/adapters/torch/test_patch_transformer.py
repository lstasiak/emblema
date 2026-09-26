"""The patch model as a module: what it reads, how large it is, and that it can learn."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from emblema.evaluation.adapters.torch.attention_pooling import AttentionPooling  # noqa: E402
from emblema.evaluation.adapters.torch.patch_transformer import PatchTransformer  # noqa: E402
from emblema.evaluation.adapters.torch.scheduled_training import (  # noqa: E402
    ScheduledTraining,
)
from emblema.evaluation.domain.exceptions import InvalidPatchModelSpecError  # noqa: E402
from emblema.evaluation.domain.heads.head_pooling import HeadPooling, PoolingScheme  # noqa: E402
from emblema.evaluation.domain.patching.patch_model_spec import PatchModelSpec  # noqa: E402
from tests.evaluation.support import adaptation_schedule, patch_spec  # noqa: E402

pytestmark = pytest.mark.ml

ENV_EXAMPLE = Path(__file__).parents[4] / "env.example"
# A turbofan window of FD001: twenty-one channels observed over fifty cycles.
TURBOFAN_CHANNELS, TURBOFAN_STEPS = 21, 50


def declared_shape() -> PatchModelSpec:
    """The shape the environment template hands the worker, read out of the template itself."""
    prefix = "# EMBLEMA_WORKER__PATCH__"
    stated = {
        line.removeprefix(prefix).split("=")[0].lower(): line.split("=")[1].strip()
        for line in ENV_EXAMPLE.read_text().splitlines()
        if line.startswith(prefix)
    }
    return PatchModelSpec(
        patch_length=int(stated["patch_length"]),
        stride=int(stated["stride"]),
        width=int(stated["width"]),
        heads=int(stated["heads"]),
        layers=int(stated["layers"]),
        feedforward_width=int(stated["feedforward_width"]),
        dropout=float(stated["dropout"]),
        grid_resolution=float(stated["grid_resolution"]),
    )


def model() -> PatchTransformer:
    torch.manual_seed(1)
    return PatchTransformer(patch_spec(), channels=3, steps=10, starting_at=0.5)


def test_the_declared_shape_stays_within_the_small_tier_on_a_turbofan_window() -> None:
    declared = PatchTransformer(
        declared_shape(), channels=TURBOFAN_CHANNELS, steps=TURBOFAN_STEPS, starting_at=0.0
    )

    assert sum(p.numel() for p in declared.parameters()) <= 2_000_000


def test_one_answer_per_window() -> None:
    values = torch.randn(5, 3, 10)

    assert model()(values, torch.ones_like(values)).shape == (5,)


def test_before_any_step_the_head_answers_where_it_was_told_to_start() -> None:
    zero_head = model()
    torch.nn.init.zeros_(zero_head.head.linear.weight)
    values = torch.randn(2, 3, 10)

    assert zero_head(values, torch.ones_like(values)).tolist() == pytest.approx([0.5, 0.5])


def test_the_mask_is_read_beside_the_values() -> None:
    reader = model().eval()
    values = torch.randn(1, 3, 10)
    carried = torch.ones_like(values)
    carried[0, 1, 4:] = 0.0

    with torch.no_grad():
        assert reader(values, torch.ones_like(values)) != reader(values, carried)


def test_a_row_shorter_than_one_patch_is_refused() -> None:
    with pytest.raises(InvalidPatchModelSpecError, match="shorter than a patch"):
        PatchTransformer(patch_spec(patch_length=8), channels=1, steps=7, starting_at=0.0)


def test_a_small_batch_is_learnt_by_the_loop_every_network_here_learns_by() -> None:
    learner = model()
    values = torch.randn(4, 3, 10)
    observed = torch.ones_like(values)
    targets = torch.tensor([0.1, 0.4, 0.7, 0.9])

    losses = ScheduledTraining(adaptation_schedule(epochs=60, batch_size=4), seed=1).losses(
        learner,
        learner.parameters(),
        lambda indices: learner(values[indices], observed[indices]),
        targets,
    )

    assert losses[-1] < losses[0] / 10


@pytest.mark.parametrize(
    "pooling",
    [
        HeadPooling(pooling=PoolingScheme.TAIL, tail_share=0.5),
        HeadPooling(pooling=PoolingScheme.ATTENTION),
    ],
)
def test_a_turned_pooling_reads_each_channels_patches_and_answers_one_number(
    pooling: HeadPooling,
) -> None:
    turned = PatchTransformer(patch_spec(), channels=3, steps=10, starting_at=0.5, pooling=pooling)
    values = torch.randn(4, 3, 10)

    assert turned(values, torch.ones_like(values)).shape == (4,)
    assert isinstance(turned.pooling, AttentionPooling) == pooling.pooling.learns_weights


def test_a_patch_is_placed_in_the_window_by_the_step_it_starts_at() -> None:
    # Patches of four every two steps over ten steps: starts at 0, 2, 4, 6 and 8, the last one
    # reaching into the padded stride.
    assert model().patch_times.tolist() == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])


def test_the_tail_of_a_row_is_the_last_patches_alone() -> None:
    torch.manual_seed(2)
    whole = PatchTransformer(patch_spec(), channels=1, steps=10, starting_at=0.0)
    torch.manual_seed(2)
    tail = PatchTransformer(
        patch_spec(),
        channels=1,
        steps=10,
        starting_at=0.0,
        pooling=HeadPooling(pooling=PoolingScheme.TAIL, tail_share=0.3),
    )
    values = torch.randn(1, 1, 10)

    with torch.no_grad():
        assert whole(values, torch.ones_like(values)) != tail(values, torch.ones_like(values))
