import pytest

from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from tests.evaluation.adapters.onnx.candidates import Exported, exported


@pytest.fixture
def lora() -> Exported:
    """The richest graph: a grown table, frozen layers and low-rank updates beside them."""
    return exported(TransferMode.LORA)
