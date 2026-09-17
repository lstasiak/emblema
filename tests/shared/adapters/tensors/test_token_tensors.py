import pytest

from emblema.shared.kernel.tokens import Token, TokenWindow

torch = pytest.importorskip("torch")

from emblema.shared.adapters.arrays.token_batch import TokenBatch  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402

pytestmark = pytest.mark.ml

SHORT = TokenWindow.of([Token(1, 0.25, 0.5, 0.5), Token(2, -1.0, 0.75, 0.75)])
LONG = TokenWindow.of(
    [Token(3, 1.5, 0.0, 0.0, timeless=True), Token(1, 0.1, 0.1, 0.1), Token(2, 0.3, 0.4, 0.4)]
)


def test_tensors_carry_the_types_the_graph_declares() -> None:
    tensors = TokenTensors.from_windows([SHORT, LONG])

    assert tensors.features.dtype == torch.float32
    assert tensors.timestamps.dtype == torch.float32
    assert tensors.channel_ids.dtype == torch.int64
    assert tensors.timeless.dtype == torch.bool
    assert tensors.padding_mask.dtype == torch.bool


def test_collating_pads_every_window_to_the_longest_of_them() -> None:
    tensors = TokenTensors.from_windows([SHORT, LONG])

    assert (tensors.batch_size, tensors.token_count) == (2, 3)
    assert tensors.features.shape == (2, 3, 2)
    assert tensors.padding_mask.tolist() == [[False, False, True], [False] * 3]


def test_the_tensors_share_the_memory_of_the_arrays_they_were_laid_out_in() -> None:
    batch = TokenBatch.from_windows([SHORT])
    tensors = TokenTensors.of(batch)

    batch.features[0, 0, 0] = 7.5

    assert tensors.features[0, 0, 0] == 7.5


def test_another_precision_reaches_the_floating_tensors_only() -> None:
    moved = TokenTensors.from_windows([SHORT, LONG]).to(dtype=torch.float16)

    assert moved.features.dtype == torch.float16
    assert moved.timestamps.dtype == torch.float16
    assert moved.channel_ids.dtype == torch.int64
    assert moved.timeless.dtype == torch.bool
    assert moved.padding_mask.dtype == torch.bool


def test_moving_keeps_the_values_the_batch_held() -> None:
    tensors = TokenTensors.from_windows([SHORT, LONG])

    moved = tensors.to("cpu")

    assert torch.equal(moved.channel_ids, tensors.channel_ids)
    assert torch.equal(moved.features, tensors.features)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple-silicon MPS")
def test_widening_on_the_way_off_the_accelerator_keeps_the_values() -> None:
    # A device that has no double precision cannot convert to it, and asking it to move and widen
    # in one call returns zeros rather than refusing (torch 2.14 on MPS).
    tensors = TokenTensors.from_windows([SHORT, LONG])

    widened = tensors.to("mps").to("cpu", torch.float64)

    assert widened.features.dtype == torch.float64
    assert torch.equal(widened.features, tensors.features.to(torch.float64))
    assert torch.equal(widened.timestamps, tensors.timestamps.to(torch.float64))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple-silicon MPS")
def test_every_tensor_reaches_the_accelerator_the_run_uses() -> None:
    # Moving to an accelerator is the case `.to("cpu")` cannot fail on: a field left behind would
    # only surface as a device mismatch inside the model.
    moved = TokenTensors.from_windows([SHORT, LONG]).to("mps")

    assert {tensor.device.type for tensor in moved.args} == {"mps"}


def test_arguments_come_in_the_order_the_model_declares_its_inputs() -> None:
    tensors = TokenTensors.from_windows([SHORT])

    assert tensors.args == (
        tensors.features,
        tensors.channel_ids,
        tensors.timestamps,
        tensors.timeless,
        tensors.padding_mask,
    )
