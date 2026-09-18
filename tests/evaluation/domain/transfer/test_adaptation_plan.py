import pytest

from emblema.evaluation.domain.exceptions import InvalidAdaptationPlanError
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from tests.evaluation.support import LORA, WEIGHTS, plan


@pytest.mark.parametrize("mode", list(TransferMode))
def test_a_plan_of_every_mode_holds_together(mode: TransferMode) -> None:
    stated = plan(mode)

    assert (stated.backbone is not None) is mode.starts_from_pretrained_weights
    assert (stated.lora is not None) is mode.adds_low_rank_updates


def test_the_control_arm_names_no_weights() -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="starts from no weights and names some"):
        plan(TransferMode.FROM_SCRATCH, backbone=WEIGHTS)


@pytest.mark.parametrize(
    "mode", [TransferMode.FROZEN_PROBE, TransferMode.LORA, TransferMode.FULL_FINE_TUNING]
)
def test_a_transfer_mode_needs_the_pretrained_weights(mode: TransferMode) -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="names none"):
        plan(mode, backbone=None)


def test_low_rank_updates_are_specified_exactly_where_the_mode_adds_them() -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="specifies none"):
        plan(TransferMode.LORA, lora=None)
    with pytest.raises(InvalidAdaptationPlanError, match="specifies some"):
        plan(TransferMode.FULL_FINE_TUNING, lora=LORA)


def test_the_flattened_plan_renders_the_same_columns_for_every_mode() -> None:
    columns = {tuple(plan(mode).parameters()) for mode in TransferMode}

    assert len(columns) == 1


def test_the_flattened_plan_tells_two_plans_apart_by_what_differs() -> None:
    full, lora = (
        plan(TransferMode.FULL_FINE_TUNING).parameters(),
        plan(TransferMode.LORA).parameters(),
    )

    assert {key for key in full if full[key] != lora[key]} == {
        "mode",
        "lora_rank",
        "lora_alpha",
        "lora_targets",
    }
    assert lora["lora_targets"] == "qkv attention.projection feedforward"
    assert plan(TransferMode.FROM_SCRATCH).parameters()["backbone"] == ""


def test_another_seed_is_a_repeat_of_the_same_plan() -> None:
    first, again = plan().parameters(), plan(seed=2).parameters()

    assert {key for key in first if first[key] != again[key]} == {"run_seed"}
    assert again["run_seed"] == 2
