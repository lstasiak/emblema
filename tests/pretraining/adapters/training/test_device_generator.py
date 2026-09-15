import pytest

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.training.device_generator import DeviceGenerator  # noqa: E402
from emblema.pretraining.adapters.training.devices import available_device  # noqa: E402

pytestmark = pytest.mark.ml


def test_the_accelerator_is_preferred_and_the_host_is_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert available_device() == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert available_device() == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert available_device() == "cpu"


def test_the_host_keeps_no_generator_of_its_own_to_save() -> None:
    assert DeviceGenerator("cpu").state() is None


def test_putting_back_nothing_is_not_an_error() -> None:
    DeviceGenerator("cpu").restore(None)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_an_accelerator_gives_its_state_back_as_it_was() -> None:
    generator = DeviceGenerator("mps")
    state = generator.state()
    torch.rand(4, device="mps")

    generator.restore(state)

    assert torch.equal(generator.state(), state)
