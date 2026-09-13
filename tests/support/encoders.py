"""The encoder the tests build: small, seeded, over the test vocabulary."""

import torch

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from tests.support.token_tensors import VOCABULARY_SIZE

SMALL = EncoderArchitecture(width=32, heads=4, layers=2, feedforward_width=64, time_frequencies=8)


def small_encoder(*, seed: int = 1) -> SetEncoder:
    """A small encoder in evaluation mode, its weights drawn under ``seed``."""
    torch.manual_seed(seed)
    return SetEncoder.for_vocabulary(SMALL, VOCABULARY_SIZE).eval()
