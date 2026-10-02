"""The parts of the published network an ablation turns towards this project's network."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule  # noqa: E402
from scripts.strats_reference_run import (  # noqa: E402
    ABLATIONS,
    LinearValue,
    MeanWeights,
    ablations_of,
    our_schedule,
    settings,
)

pytestmark = pytest.mark.ml


def test_ours_turns_every_part_and_this_projects_schedule_brings_the_fixed_epochs() -> None:
    assert ablations_of(["ours"]) == frozenset(ABLATIONS)
    assert ablations_of(["our-schedule"]) == {"our-schedule", "fixed-epochs"}
    assert ablations_of([]) == frozenset()


def test_a_part_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ValueError, match="no part is called"):
        ablations_of(["unweighted", "no-attention"])


def test_without_ablations_the_setting_is_its_run_scripts() -> None:
    stated = settings("cpu", 3, Path("out"), logger=None)

    assert (stated.hid_dim, stated.num_layers, stated.num_heads) == (64, 2, 16)
    assert (stated.dropout, stated.attention_dropout) == (0.2, 0.2)
    assert (stated.lr, stated.max_epochs, stated.patience) == (5e-4, 50, 10)


def test_ours_is_this_projects_network_from_nothing_at_every_stay() -> None:
    stated = settings("cpu", 3, Path("out"), logger=None, ablations=ablations_of(["ours"]))

    assert (stated.hid_dim, stated.num_layers, stated.num_heads) == (256, 6, 4)
    assert (stated.dropout, stated.attention_dropout) == (0.0, 0.0)
    assert (stated.lr, stated.max_epochs) == (0.000333, 30)


def test_this_projects_schedule_is_the_one_its_network_trains_under() -> None:
    windows = 2557
    adaptation = AdaptationSchedule(
        epochs=30,
        min_steps=0,
        batch_size=16,
        learning_rate=0.000333,
        weight_decay=0.0,
        warmup_fraction=0.1,
        final_lr_fraction=0.01,
    )

    assert our_schedule(30 * 160) == adaptation.learning_rate_schedule(windows)


def test_equal_weights_pool_the_mean_of_the_observed_triplets() -> None:
    states = torch.randn(2, 5, 3, generator=torch.Generator().manual_seed(1))
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])

    weights = MeanWeights()(states, mask)

    pooled = (states * weights[:, :, None]).sum(dim=1)
    torch.testing.assert_close(pooled[0], states[0, :3].mean(dim=0))
    torch.testing.assert_close(pooled[1], states[1, 0])
    assert torch.isfinite(MeanWeights()(states, torch.zeros(2, 5))).all()


def test_a_value_is_embedded_by_one_linear_map() -> None:
    embedded = LinearValue(8)(torch.randn(2, 5))

    assert embedded.shape == (2, 5, 8)
