"""The loop under parameters handed plain or in groups, each group at its own rate."""

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402
from torch.optim.optimizer import ParamsT  # noqa: E402

from emblema.evaluation.adapters.torch.scheduled_training import ScheduledTraining  # noqa: E402
from tests.evaluation.support import adaptation_schedule  # noqa: E402

pytestmark = pytest.mark.ml

RATE = 1e-2


class Toy(nn.Module):
    """An encoder and a head, each one linear map, so their movements can be told apart."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(3)
        self.encoder = nn.Linear(4, 4)
        self.head = nn.Linear(4, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(torch.tanh(self.encoder(x))).squeeze(-1)


def fitted(groups: bool, *, encoder_share: float) -> tuple[Toy, list[float]]:
    torch.manual_seed(11)
    inputs, targets = torch.randn(16, 4), torch.randn(16)
    model = Toy()
    trainable: ParamsT = list(model.parameters())
    if groups:
        trainable = [
            {"params": list(model.head.parameters()), "lr": RATE},
            {"params": list(model.encoder.parameters()), "lr": RATE * encoder_share},
        ]
    losses = ScheduledTraining(adaptation_schedule(learning_rate=RATE), seed=1).losses(
        model,
        trainable,
        lambda indices: model(inputs[list(indices)]),
        targets,
        loss=nn.functional.mse_loss,
    )
    return model, losses


def test_groups_at_the_whole_rate_repeat_the_plain_parameters_bit_for_bit() -> None:
    plain, plain_losses = fitted(groups=False, encoder_share=1.0)
    grouped, grouped_losses = fitted(groups=True, encoder_share=1.0)

    assert plain_losses == grouped_losses
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            plain.state_dict().values(), grouped.state_dict().values(), strict=True
        )
    )


def test_a_group_at_a_share_of_the_rate_moves_less_and_the_other_group_still_moves() -> None:
    start = Toy().state_dict()
    whole, _ = fitted(groups=True, encoder_share=1.0)
    slowed, _ = fitted(groups=True, encoder_share=0.1)

    def moved(model: Toy, part: str) -> float:
        return float((model.state_dict()[f"{part}.weight"] - start[f"{part}.weight"]).norm())

    assert 0.0 < moved(slowed, "encoder") < moved(whole, "encoder")
    assert moved(slowed, "head") > 0.0
