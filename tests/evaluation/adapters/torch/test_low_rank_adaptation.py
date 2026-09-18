import pytest

torch = pytest.importorskip("torch")

from dataclasses import replace  # noqa: E402

from emblema.evaluation.adapters.torch.lora_linear import LoraLinear  # noqa: E402
from emblema.evaluation.adapters.torch.low_rank_adaptation import LowRankAdaptation  # noqa: E402
from emblema.evaluation.domain.exceptions import LoraTargetNotFoundError  # noqa: E402
from tests.evaluation.support import LORA  # noqa: E402
from tests.support.encoders import small_encoder  # noqa: E402
from tests.support.token_tensors import random_batch  # noqa: E402

pytestmark = pytest.mark.ml


def test_every_targeted_linear_of_every_block_is_wrapped_and_nothing_else() -> None:
    encoder = small_encoder()

    wrapped = LowRankAdaptation(LORA).applied_to(encoder)

    assert wrapped == (
        "blocks.0.attention.qkv",
        "blocks.0.attention.projection",
        "blocks.0.feedforward.0",
        "blocks.0.feedforward.2",
        "blocks.1.attention.qkv",
        "blocks.1.attention.projection",
        "blocks.1.feedforward.0",
        "blocks.1.feedforward.2",
    )
    assert all(isinstance(encoder.get_submodule(path), LoraLinear) for path in wrapped)
    assert not isinstance(encoder.value_projection, LoraLinear)


def test_a_narrower_spec_reaches_only_the_layers_it_names() -> None:
    encoder = small_encoder()

    wrapped = LowRankAdaptation(replace(LORA, targets=("qkv",))).applied_to(encoder)

    assert wrapped == ("blocks.0.attention.qkv", "blocks.1.attention.qkv")


def test_targets_that_overlap_each_count_the_layers_they_reach() -> None:
    encoder = small_encoder()

    wrapped = LowRankAdaptation(
        replace(LORA, targets=("attention", "attention.projection"))
    ).applied_to(encoder)

    assert wrapped == (
        "blocks.0.attention.qkv",
        "blocks.0.attention.projection",
        "blocks.1.attention.qkv",
        "blocks.1.attention.projection",
    )


def test_the_encoder_computes_what_it_did_until_the_updates_move() -> None:
    encoder = small_encoder()
    batch = random_batch(2, 7, seed=3)
    before = encoder(*batch.args)

    LowRankAdaptation(LORA).applied_to(encoder)

    assert torch.equal(encoder(*batch.args), before)


def test_a_target_that_reaches_no_linear_layer_is_refused_before_anything_is_wrapped() -> None:
    encoder = small_encoder()

    with pytest.raises(LoraTargetNotFoundError, match=r"under \['ghost'\]") as refused:
        LowRankAdaptation(replace(LORA, targets=("qkv", "ghost"))).applied_to(encoder)

    assert not any(isinstance(module, LoraLinear) for module in encoder.modules())
    assert "blocks.0.attention.qkv" in str(refused.value)
    assert ".base" not in str(refused.value)


def test_a_target_naming_a_module_that_is_not_linear_is_refused() -> None:
    with pytest.raises(LoraTargetNotFoundError):
        LowRankAdaptation(replace(LORA, targets=("attention_norm",))).applied_to(small_encoder())
