import pytest

from emblema.pretraining.domain.exceptions import UnsupportedPrecisionError
from emblema.pretraining.domain.training.precision import Precision

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.training.torch_precision import TorchPrecision  # noqa: E402

pytestmark = pytest.mark.ml


@pytest.mark.parametrize(
    ("device", "precision", "dtype"),
    [
        ("cpu", Precision.FP32, None),
        ("cpu", Precision.BF16, torch.bfloat16),
        ("mps", Precision.FP16, torch.float16),
        ("cuda", Precision.BF16, torch.bfloat16),
    ],
)
def test_a_supported_pair_computes_in_the_type_it_declares(
    device: str, precision: Precision, dtype: object
) -> None:
    assert TorchPrecision(precision, device).dtype == dtype


@pytest.mark.parametrize(
    ("device", "precision"), [("cpu", Precision.FP16), ("mps", Precision.BF16)]
)
def test_a_pair_the_backend_cannot_run_is_refused_rather_than_demoted(
    device: str, precision: Precision
) -> None:
    with pytest.raises(UnsupportedPrecisionError, match=device):
        TorchPrecision(precision, device)


def test_a_device_nothing_is_known_about_is_refused() -> None:
    with pytest.raises(UnsupportedPrecisionError):
        TorchPrecision(Precision.FP32, "meta")


def test_gradients_are_scaled_only_where_half_precision_needs_it() -> None:
    assert TorchPrecision(Precision.FP16, "cuda").scales_gradients
    assert not TorchPrecision(Precision.FP16, "mps").scales_gradients
    assert not TorchPrecision(Precision.BF16, "cuda").scales_gradients


def test_single_precision_enters_no_autocast_and_scales_nothing() -> None:
    single = TorchPrecision(Precision.FP32, "cpu")

    with single.autocast():
        assert not torch.is_autocast_enabled("cpu")
    assert not single.scaler().is_enabled()


def test_a_declared_half_precision_is_what_the_arithmetic_runs_in() -> None:
    with TorchPrecision(Precision.BF16, "cpu").autocast():
        product = torch.ones(2, 2) @ torch.ones(2, 2)

    assert product.dtype == torch.bfloat16
