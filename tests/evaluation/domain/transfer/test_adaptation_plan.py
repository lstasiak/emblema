import pytest

from emblema.evaluation.domain.exceptions import (
    InvalidAdaptationPlanError,
    InvalidEncoderSettingError,
)
from emblema.evaluation.domain.heads.head_pooling import HeadPooling, PoolingScheme
from emblema.evaluation.domain.transfer.encoder_setting import EncoderSetting, ValueEmbedding
from emblema.evaluation.domain.transfer.training_regime import (
    ClassWeight,
    HeadStart,
    TrainingRegime,
)
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from tests.evaluation.support import LORA, PENALTIES, WEIGHTS, plan


@pytest.mark.parametrize("mode", list(TransferMode))
def test_a_plan_of_every_mode_holds_together(mode: TransferMode) -> None:
    stated = plan(mode)

    assert (stated.backbone is not None) is mode.takes_pretrained_weights
    assert (stated.lora is not None) is mode.adds_low_rank_updates
    assert (stated.ridge is not None) is mode.solves_the_head_in_closed_form


def test_the_control_arm_names_no_weights() -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="starts from no weights and names some"):
        plan(TransferMode.FROM_SCRATCH, backbone=WEIGHTS)


@pytest.mark.parametrize("mode", [TransferMode.LORA, TransferMode.FULL_FINE_TUNING])
def test_a_mode_that_adapts_the_weights_needs_pretrained_ones(mode: TransferMode) -> None:
    with pytest.raises(
        InvalidAdaptationPlanError, match="adapts pretrained weights and names none"
    ):
        plan(mode, backbone=None)


@pytest.mark.parametrize("mode", [TransferMode.FROZEN_PROBE, TransferMode.FROZEN_RIDGE])
def test_a_frozen_mode_reads_an_encoder_at_its_initialisation_too(mode: TransferMode) -> None:
    assert plan(mode, backbone=None).backbone is None


def test_low_rank_updates_are_specified_exactly_where_the_mode_adds_them() -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="specifies none"):
        plan(TransferMode.LORA, lora=None)
    with pytest.raises(InvalidAdaptationPlanError, match="specifies some"):
        plan(TransferMode.FULL_FINE_TUNING, lora=LORA)


def test_penalties_are_named_exactly_where_the_head_is_solved_in_closed_form() -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="names no penalties"):
        plan(TransferMode.FROZEN_RIDGE, ridge=None)
    with pytest.raises(InvalidAdaptationPlanError, match="trains its head and names penalties"):
        plan(TransferMode.FROZEN_PROBE, ridge=PENALTIES)


def test_a_closed_form_head_cannot_pool_under_learnt_weights() -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="cannot learn a attention pooling"):
        plan(TransferMode.FROZEN_RIDGE, pooling=HeadPooling(pooling=PoolingScheme.ATTENTION))
    tailed = plan(TransferMode.FROZEN_RIDGE, pooling=HeadPooling(pooling=PoolingScheme.TAIL))
    assert tailed.pooling.pooling is PoolingScheme.TAIL


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
    assert plan(TransferMode.FROZEN_RIDGE).parameters()["ridge_penalties"] == "0.1 1 10"
    assert plan(TransferMode.FROZEN_PROBE).parameters()["ridge_penalties"] == ""


def test_another_seed_is_a_repeat_of_the_same_plan() -> None:
    first, again = plan().parameters(), plan(seed=2).parameters()

    assert {key for key in first if first[key] != again[key]} == {"run_seed"}
    assert again["run_seed"] == 2


@pytest.mark.parametrize(
    ("mode", "pooling"),
    [
        (TransferMode.FROZEN_RIDGE, PoolingScheme.MEAN),
        (TransferMode.FROZEN_PROBE, PoolingScheme.MEAN),
        (TransferMode.FROZEN_PROBE, PoolingScheme.TAIL),
    ],
)
def test_a_dropout_is_refused_where_the_encoder_states_every_window_once(
    mode: TransferMode, pooling: PoolingScheme
) -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="a dropout would change nothing"):
        plan(mode, pooling=HeadPooling(pooling=pooling), encoder=EncoderSetting(dropout=0.2))


@pytest.mark.parametrize(
    ("mode", "pooling"),
    [
        (TransferMode.FROM_SCRATCH, PoolingScheme.MEAN),
        (TransferMode.LORA, PoolingScheme.MEAN),
        (TransferMode.FULL_FINE_TUNING, PoolingScheme.TAIL),
        (TransferMode.FROZEN_PROBE, PoolingScheme.ATTENTION),
    ],
)
def test_a_dropout_is_taken_where_the_encoder_runs_in_the_loop(
    mode: TransferMode, pooling: PoolingScheme
) -> None:
    dropped = plan(mode, pooling=HeadPooling(pooling=pooling), encoder=EncoderSetting(dropout=0.2))

    assert dropped.encodes_in_the_loop
    assert dropped.parameters()["encoder_dropout"] == 0.2


def test_a_grid_is_taken_under_every_mode_and_recorded_with_the_run() -> None:
    for mode in TransferMode:
        gridded = plan(mode, encoder=EncoderSetting(grid_resolution=1.0))
        assert gridded.parameters()["grid_resolution"] == 1.0
    assert plan(TransferMode.FROM_SCRATCH).parameters()["grid_resolution"] == 0.0


def test_where_the_static_features_stand_is_recorded_with_every_run() -> None:
    apart = plan(TransferMode.FROM_SCRATCH, pooling=HeadPooling.mean().tuned("statics", "apart"))

    assert apart.parameters()["statics"] == "apart"
    assert plan(TransferMode.FROM_SCRATCH).parameters()["statics"] == "among"


@pytest.mark.parametrize(
    "encoder",
    [
        EncoderSetting(value_embedding=ValueEmbedding.NONLINEAR),
        EncoderSetting(width=64, heads=16, layers=2, feedforward_width=128),
    ],
)
def test_an_encoder_built_otherwise_than_its_backbone_starts_from_no_weights(
    encoder: EncoderSetting,
) -> None:
    assert plan(TransferMode.FROM_SCRATCH, encoder=encoder).encoder == encoder
    with pytest.raises(InvalidAdaptationPlanError, match="fix its encoder's build"):
        plan(TransferMode.FULL_FINE_TUNING, encoder=encoder)


def test_an_encoders_own_shape_is_stated_whole() -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="states its width, heads"):
        plan(TransferMode.FROM_SCRATCH, encoder=EncoderSetting(width=64, heads=16))
    with pytest.raises(InvalidEncoderSettingError, match="multiple of heads"):
        plan(
            TransferMode.FROM_SCRATCH,
            encoder=EncoderSetting(width=64, heads=5, layers=2, feedforward_width=128),
        )


def test_a_stop_stated_in_part_is_refused_and_whole_it_is_taken_under_every_trained_mode() -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="both the share"):
        plan(regime=TrainingRegime(patience=10))
    for mode in (TransferMode.FROM_SCRATCH, TransferMode.FROZEN_PROBE, TransferMode.LORA):
        stopped = plan(mode, regime=TrainingRegime(stop_share=0.2, patience=10))
        assert stopped.regime.stops
        assert stopped.parameters()["stop_share"] == 0.2


def test_a_closed_form_head_takes_no_regime() -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="takes no step"):
        plan(TransferMode.FROZEN_RIDGE, regime=TrainingRegime(class_weight=ClassWeight.RATIO))


def test_withholding_channels_is_refused_where_the_encoder_states_every_window_once() -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="withholding channels"):
        plan(TransferMode.FROZEN_PROBE, regime=TrainingRegime(channel_dropout=0.2))
    assert plan(TransferMode.FROM_SCRATCH, regime=TrainingRegime(channel_dropout=0.2)).regime


def test_a_head_solved_first_names_penalties_and_pools_under_no_learnt_weights() -> None:
    solved = TrainingRegime(head_start=HeadStart.SOLVED)
    for mode in (TransferMode.FROM_SCRATCH, TransferMode.FULL_FINE_TUNING):
        started = plan(mode, regime=solved, ridge=PENALTIES)
        assert started.solves_a_head
        assert started.parameters()["head_start"] == "solved"
    with pytest.raises(InvalidAdaptationPlanError, match="names no penalties"):
        plan(TransferMode.FULL_FINE_TUNING, regime=solved)
    with pytest.raises(InvalidAdaptationPlanError, match="cannot learn a attention pooling"):
        plan(
            TransferMode.FULL_FINE_TUNING,
            regime=solved,
            ridge=PENALTIES,
            pooling=HeadPooling(pooling=PoolingScheme.ATTENTION),
        )


def test_a_head_the_mode_solves_is_not_solved_first_as_well() -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="no step to start it for"):
        plan(TransferMode.FROZEN_RIDGE, regime=TrainingRegime(head_start=HeadStart.SOLVED))


def test_a_stop_waits_in_steps_as_it_waits_in_epochs() -> None:
    stopped = plan(
        TransferMode.FULL_FINE_TUNING, regime=TrainingRegime(stop_share=0.2, patience_steps=40)
    )
    assert stopped.regime.stops
    assert stopped.parameters()["patience_steps"] == 40
