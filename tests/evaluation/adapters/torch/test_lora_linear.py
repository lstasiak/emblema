import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from emblema.evaluation.adapters.torch.lora_linear import LoraLinear  # noqa: E402
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec  # noqa: E402

pytestmark = pytest.mark.ml

SPEC = LoraSpec(rank=2, alpha=4.0, dropout=0.0, targets=("linear",))


def wrapped(seed: int = 1) -> tuple[nn.Linear, LoraLinear]:
    torch.manual_seed(seed)
    base = nn.Linear(6, 4)
    return base, LoraLinear(base, SPEC)


def test_at_the_start_the_wrapped_layer_computes_what_the_layer_did() -> None:
    base, layer = wrapped()
    inputs = torch.randn(3, 6)

    assert torch.equal(layer(inputs), base(inputs))


def test_only_the_update_receives_a_gradient() -> None:
    base, layer = wrapped()

    layer(torch.randn(3, 6)).sum().backward()

    assert base.weight.grad is None
    assert base.bias.grad is None
    assert layer.down.grad is None or torch.count_nonzero(layer.down.grad) == 0
    assert layer.up.grad is not None
    assert torch.count_nonzero(layer.up.grad) > 0


def test_the_update_has_rank_times_in_plus_out_parameters() -> None:
    _, layer = wrapped()

    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)

    assert trainable == SPEC.rank * (6 + 4)


def test_after_a_step_the_output_moves_while_the_layer_stays() -> None:
    base, layer = wrapped()
    before = {name: value.clone() for name, value in base.state_dict().items()}
    inputs = torch.randn(3, 6)
    optimiser = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)

    layer(inputs).sum().backward()
    optimiser.step()

    assert not torch.equal(layer(inputs), base(inputs))
    assert all(torch.equal(before[name], value) for name, value in base.state_dict().items())


def test_the_update_is_scaled_by_alpha_over_rank() -> None:
    base, layer = wrapped()
    inputs = torch.randn(3, 6)
    with torch.no_grad():
        layer.up.fill_(1.0)

    update = (inputs @ layer.down.T @ layer.up.T) * SPEC.scaling

    torch.testing.assert_close(layer(inputs), base(inputs) + update)
