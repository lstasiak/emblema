import pytest

torch = pytest.importorskip("torch")

from emblema.shared.adapters.tensors.masked_mean_pooling import MaskedMeanPooling  # noqa: E402

pytestmark = pytest.mark.ml


def test_the_mean_runs_over_the_observed_tokens_only() -> None:
    states = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    padding_mask = torch.tensor([[False, False, True]])

    assert torch.equal(MaskedMeanPooling()(states, padding_mask), torch.tensor([[2.0, 3.0]]))


def test_a_window_of_nothing_but_padding_pools_to_zeros() -> None:
    states = torch.randn(1, 3, 2)

    pooled = MaskedMeanPooling()(states, torch.ones(1, 3, dtype=torch.bool))

    assert torch.equal(pooled, torch.zeros(1, 2))
