import pytest

pytest.importorskip("torch")

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from tests.support.encoders import small_encoder


@pytest.fixture
def encoder() -> SetEncoder:
    return small_encoder()
