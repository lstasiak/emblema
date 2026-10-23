from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone  # noqa: E402
from emblema.evaluation.adapters.torch.clipped_values import ClippedValues  # noqa: E402
from emblema.evaluation.adapters.torch.layer_readout import LayerReadout  # noqa: E402
from emblema.evaluation.adapters.torch.lora_linear import LoraLinear  # noqa: E402
from emblema.evaluation.domain.exceptions import InvalidAdaptationPlanError  # noqa: E402
from emblema.evaluation.domain.heads.head_pooling import HeadPooling  # noqa: E402
from emblema.evaluation.domain.transfer.encoder_setting import (  # noqa: E402
    EncoderSetting,
    SummaryReading,
    TimeEncoding,
    ValueEmbedding,
)
from emblema.evaluation.domain.transfer.layer_reading import LayerReading  # noqa: E402
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode  # noqa: E402
from emblema.pretraining.adapters.encoder.learned_time_encoding import (  # noqa: E402
    LearnedTimeEncoding,
)
from emblema.pretraining.adapters.encoder.nonlinear_value_embedding import (  # noqa: E402
    NonlinearValueEmbedding,
)
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from emblema.shared.kernel.artifacts import ArtifactRef  # noqa: E402
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
    return (
        AdaptedBackbone.under(
            plan(mode), backbones, vocabulary_size=VOCABULARY_SIZE, starting_at=0.0
        ),
        backbones,
    )


@pytest.mark.parametrize(
    ("mode", "trainable"),
    [
        (TransferMode.FROM_SCRATCH, ENCODER + HEAD),
        (TransferMode.FROZEN_PROBE, HEAD),
        (TransferMode.FROZEN_RIDGE, HEAD),
        (TransferMode.LORA, LORA_UPDATES + HEAD),
        (TransferMode.FULL_FINE_TUNING, ENCODER + HEAD),
    ],
)
def test_each_mode_leaves_exactly_its_weights_free(mode: TransferMode, trainable: int) -> None:
    built, _ = candidate(mode)

    assert sum(p.numel() for p in built.trainable_parameters()) == trainable


@pytest.mark.parametrize(
    "mode",
    [
        TransferMode.FROZEN_PROBE,
        TransferMode.FROZEN_RIDGE,
        TransferMode.LORA,
        TransferMode.FULL_FINE_TUNING,
    ],
)
def test_a_transfer_mode_asks_for_the_plans_weights(mode: TransferMode) -> None:
    _, backbones = candidate(mode)

    assert backbones.requested == [WEIGHTS]


def test_the_control_arm_starts_from_weights_of_its_own() -> None:
    built, backbones = candidate(TransferMode.FROM_SCRATCH)
    pretrained = backbones.pretrained(WEIGHTS, vocabulary_size=VOCABULARY_SIZE, dropout=0.0)

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


def test_set_apart_the_static_features_double_the_state_the_head_reads() -> None:
    torch.manual_seed(5)
    apart = AdaptedBackbone.under(
        plan(TransferMode.FROM_SCRATCH, pooling=HeadPooling.mean().tuned("statics", "apart")),
        SmallBackbones(),
        vocabulary_size=VOCABULARY_SIZE,
        starting_at=0.0,
    )
    batch = random_batch(3, 9, seed=2)

    assert apart.embed(batch).shape == (3, 2 * SMALL.width)
    assert sum(p.numel() for p in apart.head.parameters()) == 2 * SMALL.width + 1
    assert torch.isfinite(apart(batch)).all()


def test_a_network_from_nothing_is_built_to_its_own_shape_and_value_embedding() -> None:
    torch.manual_seed(5)
    encoder = EncoderSetting(
        value_embedding=ValueEmbedding.NONLINEAR, width=16, heads=4, layers=1, feedforward_width=32
    )
    built = AdaptedBackbone.under(
        plan(TransferMode.FROM_SCRATCH, encoder=encoder),
        SmallBackbones(),
        vocabulary_size=VOCABULARY_SIZE,
        starting_at=0.0,
    )
    batch = random_batch(3, 9, seed=2)

    assert built.embed(batch).shape == (3, 16)
    assert sum(p.numel() for p in built.head.parameters()) == 16 + 1
    assert isinstance(built.encoder, SetEncoder)
    assert isinstance(built.encoder.value_projection, NonlinearValueEmbedding)
    assert len(built.encoder.blocks) == 1
    assert torch.isfinite(built(batch)).all()


@pytest.mark.parametrize("mode", [TransferMode.FROM_SCRATCH, TransferMode.FULL_FINE_TUNING])
def test_a_clip_feeds_any_encoder_bounded_values_and_leaves_the_rest_as_it_was(
    mode: TransferMode,
) -> None:
    torch.manual_seed(5)
    clipped = AdaptedBackbone.under(
        plan(mode, encoder=EncoderSetting(value_clip=5.0)),
        SmallBackbones(),
        vocabulary_size=VOCABULARY_SIZE,
        starting_at=0.0,
    )
    torch.manual_seed(5)
    plain = AdaptedBackbone.under(
        plan(mode), SmallBackbones(), vocabulary_size=VOCABULARY_SIZE, starting_at=0.0
    )
    batch = random_batch(3, 9, seed=2)
    wild = batch.features.clone()
    wild[0, 0, 0] = 88.0
    wild[1, 2, 0] = -31.0
    bounded = wild.clone()
    bounded[0, 0, 0] = 5.0
    bounded[1, 2, 0] = -5.0

    assert isinstance(clipped.encoder, ClippedValues)
    assert [p.shape for p in clipped.trainable_parameters()] == [
        p.shape for p in plain.trainable_parameters()
    ]
    torch.testing.assert_close(
        clipped(replace(batch, features=wild)), plain(replace(batch, features=bounded))
    )
    assert not torch.allclose(
        clipped(replace(batch, features=wild)), plain(replace(batch, features=wild))
    )


GROWN = 6
GROWN_ROWS = GROWN * SMALL.width


def grown(mode: TransferMode) -> AdaptedBackbone:
    torch.manual_seed(5)
    return AdaptedBackbone.under(
        plan(mode), SmallBackbones(), vocabulary_size=VOCABULARY_SIZE + GROWN, starting_at=0.0
    )


@pytest.mark.parametrize(
    ("mode", "trainable"),
    [
        (TransferMode.FROM_SCRATCH, ENCODER + GROWN_ROWS + HEAD),
        (TransferMode.FROZEN_PROBE, GROWN_ROWS + HEAD),
        (TransferMode.LORA, LORA_UPDATES + GROWN_ROWS + HEAD),
        (TransferMode.FULL_FINE_TUNING, ENCODER + GROWN_ROWS + HEAD),
    ],
)
def test_rows_grown_for_the_tasks_new_channels_train_under_every_mode(
    mode: TransferMode, trainable: int
) -> None:
    built = grown(mode)

    assert sum(p.numel() for p in built.trainable_parameters()) == trainable


def test_a_slowed_encoder_learns_at_its_share_and_the_head_and_grown_rows_at_the_rate() -> None:
    built = grown(TransferMode.FULL_FINE_TUNING)

    groups = built.parameter_groups(1e-3, encoder_share=0.1)

    assert [group["lr"] for group in groups] == [1e-3, 1e-4]
    at_rate, slowed = (group["params"] for group in groups)
    assert sum(p.numel() for p in at_rate) == HEAD + GROWN_ROWS
    assert sum(p.numel() for p in slowed) == ENCODER
    assert {id(p) for p in at_rate} | {id(p) for p in slowed} == {
        id(p) for p in built.trainable_parameters()
    }


def test_under_the_whole_rate_the_optimiser_is_handed_one_group_of_everything_trainable() -> None:
    built, _ = candidate(TransferMode.FULL_FINE_TUNING)

    (group,) = built.parameter_groups(1e-3, encoder_share=1.0)

    assert group["lr"] == 1e-3
    assert group["params"] == built.trainable_parameters()


def test_a_grown_candidate_answers_windows_over_the_tasks_new_channels() -> None:
    built = grown(TransferMode.FROZEN_PROBE)
    batch = random_batch(3, 9, seed=2)
    batch.channel_ids[0, :] = VOCABULARY_SIZE + GROWN  # the last new channel, on every token

    answers = built(batch)

    assert answers.shape == (3,)
    assert torch.isfinite(answers).all()


@pytest.mark.parametrize(("reading", "factor"), [("1", 1), ("mean", 1), ("concat", SMALL.layers)])
@pytest.mark.parametrize("backbone", [WEIGHTS, None])
def test_a_probe_reads_the_layer_its_plan_names_under_a_head_as_wide_as_the_reading(
    reading: str, factor: int, backbone: ArtifactRef | None
) -> None:
    torch.manual_seed(5)
    built = AdaptedBackbone.under(
        plan(
            TransferMode.FROZEN_RIDGE,
            backbone=backbone,
            encoder=EncoderSetting(layer=LayerReading.of(reading), value_clip=5.0),
        ),
        SmallBackbones(),
        vocabulary_size=VOCABULARY_SIZE,
        starting_at=0.0,
    )
    batch = random_batch(3, 9, seed=2)

    assert isinstance(built.encoder, ClippedValues)
    assert isinstance(built.encoder.encoder, LayerReadout)
    assert built.embed(batch).shape == (3, SMALL.width * factor)
    assert sum(p.numel() for p in built.trainable_parameters()) == SMALL.width * factor + 1
    assert torch.isfinite(built(batch)).all()


def test_a_probe_at_the_last_layer_holds_the_encoder_as_it_always_did() -> None:
    built, _ = candidate(TransferMode.FROZEN_RIDGE)

    assert isinstance(built.encoder, SetEncoder)


# The statistics of six windows over four columns, each column varying.
SUMMARIES = np.arange(24, dtype=np.float64).reshape(6, 4) ** 1.5
BRANCH = 4 * SMALL.width + SMALL.width


def reading(summaries: SummaryReading) -> AdaptedBackbone:
    torch.manual_seed(5)
    return AdaptedBackbone.under(
        plan(TransferMode.FROM_SCRATCH, encoder=EncoderSetting(summaries=summaries)),
        SmallBackbones(),
        vocabulary_size=VOCABULARY_SIZE,
        starting_at=0.0,
        summary_rows=SUMMARIES,
    )


def test_the_summaries_beside_the_encoder_widen_the_state_the_head_reads() -> None:
    beside = reading(SummaryReading.BESIDE)
    batch = random_batch(3, 9, seed=2)
    rows = torch.from_numpy(SUMMARIES[:3]).to(torch.float32)

    assert beside.embed(batch, rows).shape == (3, 2 * SMALL.width)
    assert sum(p.numel() for p in beside.trainable_parameters()) == (
        ENCODER + BRANCH + 2 * SMALL.width + 1
    )
    assert torch.isfinite(beside(batch, rows)).all()


def test_the_summaries_alone_never_run_the_encoder_and_train_none_of_it() -> None:
    alone = reading(SummaryReading.ONLY)

    def refuse(*_: object) -> None:
        raise AssertionError("the encoder was run")

    alone.encoder.forward = refuse
    rows = torch.from_numpy(SUMMARIES[:3]).to(torch.float32)

    assert alone(random_batch(3, 9, seed=2), rows).shape == (3,)
    assert sum(p.numel() for p in alone.trainable_parameters()) == BRANCH + SMALL.width + 1


def test_a_candidate_reading_the_summaries_refuses_to_answer_without_them() -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="given none"):
        reading(SummaryReading.BESIDE)(random_batch(3, 9, seed=2))


@pytest.mark.parametrize(
    ("summaries", "rows"), [(SummaryReading.BESIDE, None), (SummaryReading.NONE, SUMMARIES)]
)
def test_summaries_are_given_exactly_to_a_plan_that_reads_them(
    summaries: SummaryReading, rows: np.ndarray | None
) -> None:
    with pytest.raises(InvalidAdaptationPlanError, match="summaries"):
        AdaptedBackbone.under(
            plan(TransferMode.FROM_SCRATCH, encoder=EncoderSetting(summaries=summaries)),
            SmallBackbones(),
            vocabulary_size=VOCABULARY_SIZE,
            starting_at=0.0,
            summary_rows=rows,
        )


def test_a_plan_reading_no_summaries_draws_its_head_and_encoder_as_it_always_did() -> None:
    plain, _ = candidate(TransferMode.FROM_SCRATCH)
    torch.manual_seed(5)
    again = AdaptedBackbone.under(
        plan(TransferMode.FROM_SCRATCH, encoder=EncoderSetting(summaries=SummaryReading.NONE)),
        SmallBackbones(),
        vocabulary_size=VOCABULARY_SIZE,
        starting_at=0.0,
    )

    assert plain.summaries is None
    assert all(
        torch.equal(value, again.state_dict()[key]) for key, value in plain.state_dict().items()
    )


def test_a_network_from_nothing_over_cells_reads_nine_features_and_may_learn_its_time() -> None:
    torch.manual_seed(5)
    built = AdaptedBackbone.under(
        plan(
            TransferMode.FROM_SCRATCH,
            encoder=EncoderSetting(bins=4, time_encoding=TimeEncoding.LEARNED),
        ),
        SmallBackbones(),
        vocabulary_size=VOCABULARY_SIZE,
        starting_at=0.0,
        input_features=BinnedTokens.N_FEATURES,
    )
    batch = random_batch(3, 9, seed=2)
    cells = TokenTensors(
        torch.randn(3, 9, BinnedTokens.N_FEATURES),
        batch.channel_ids,
        batch.timestamps,
        batch.timeless,
        batch.padding_mask,
    )

    assert isinstance(built.encoder, SetEncoder)
    assert isinstance(built.encoder.value_projection, nn.Linear)
    assert built.encoder.value_projection.in_features == BinnedTokens.N_FEATURES
    assert isinstance(built.encoder.time_encoding, LearnedTimeEncoding)
    assert built(cells).shape == (3,)
    assert torch.isfinite(built(cells)).all()
