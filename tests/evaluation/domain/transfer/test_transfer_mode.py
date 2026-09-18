import pytest

from emblema.evaluation.domain.transfer.transfer_mode import TransferMode


@pytest.mark.parametrize(
    ("mode", "pretrained", "trains_backbone", "low_rank"),
    [
        (TransferMode.FROM_SCRATCH, False, True, False),
        (TransferMode.FROZEN_PROBE, True, False, False),
        (TransferMode.LORA, True, False, True),
        (TransferMode.FULL_FINE_TUNING, True, True, False),
    ],
)
def test_each_mode_says_where_it_starts_and_what_it_trains(
    mode: TransferMode, pretrained: bool, trains_backbone: bool, low_rank: bool
) -> None:
    assert mode.starts_from_pretrained_weights is pretrained
    assert mode.trains_backbone_weights is trains_backbone
    assert mode.adds_low_rank_updates is low_rank


def test_the_axis_holds_the_control_arm_and_the_three_transfer_modes() -> None:
    assert [str(mode) for mode in TransferMode] == [
        "from_scratch",
        "frozen_probe",
        "lora",
        "full_fine_tuning",
    ]
