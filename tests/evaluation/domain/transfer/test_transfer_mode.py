import pytest

from emblema.evaluation.domain.heads.head_pooling import HeadPooling, PoolingScheme
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode


@pytest.mark.parametrize(
    ("mode", "takes", "needs", "trains_backbone", "low_rank", "closed_form"),
    [
        (TransferMode.FROM_SCRATCH, False, False, True, False, False),
        (TransferMode.FROZEN_PROBE, True, False, False, False, False),
        (TransferMode.FROZEN_RIDGE, True, False, False, False, True),
        (TransferMode.LORA, True, True, False, True, False),
        (TransferMode.FULL_FINE_TUNING, True, True, True, False, False),
    ],
)
def test_each_mode_says_where_it_may_start_and_what_it_trains(
    mode: TransferMode,
    takes: bool,
    needs: bool,
    trains_backbone: bool,
    low_rank: bool,
    closed_form: bool,
) -> None:
    assert mode.takes_pretrained_weights is takes
    assert mode.needs_pretrained_weights is needs
    assert mode.trains_backbone_weights is trains_backbone
    assert mode.adds_low_rank_updates is low_rank
    assert mode.solves_the_head_in_closed_form is closed_form


def test_the_axis_holds_the_control_arm_and_the_four_transfer_modes() -> None:
    assert [str(mode) for mode in TransferMode] == [
        "from_scratch",
        "frozen_probe",
        "frozen_ridge",
        "lora",
        "full_fine_tuning",
    ]


@pytest.mark.parametrize(
    ("mode", "pooling", "in_the_loop"),
    [
        (TransferMode.FROM_SCRATCH, PoolingScheme.MEAN, True),
        (TransferMode.FULL_FINE_TUNING, PoolingScheme.TAIL, True),
        (TransferMode.LORA, PoolingScheme.MEAN, True),
        (TransferMode.FROZEN_PROBE, PoolingScheme.MEAN, False),
        (TransferMode.FROZEN_PROBE, PoolingScheme.TAIL, False),
        (TransferMode.FROZEN_PROBE, PoolingScheme.ATTENTION, True),
        (TransferMode.FROZEN_RIDGE, PoolingScheme.MEAN, False),
        (TransferMode.FROZEN_RIDGE, PoolingScheme.TAIL, False),
    ],
)
def test_the_encoder_runs_in_the_loop_unless_it_states_every_window_once(
    mode: TransferMode, pooling: PoolingScheme, in_the_loop: bool
) -> None:
    assert mode.encodes_in_the_loop(HeadPooling(pooling=pooling)) is in_the_loop
