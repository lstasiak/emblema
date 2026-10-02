"""The backbones the Evaluation tests adapt: small encoders standing in for pretrained ones.

An implementation of the seam Evaluation gets its encoder through. The "pretrained" encoder is
the small test encoder built under a fixed seed, so it is the same weights every time it is
asked for and different from anything drawn afterwards; a "fresh" one is drawn from torch's
generator as it stands, which is what the runtime seeds before asking.
"""

from dataclasses import replace

import torch
from torch import nn

from emblema.evaluation.domain.transfer.encoder_setting import ValueEmbedding
from emblema.evaluation.domain.transfer.encoder_shape import EncoderShape
from emblema.pretraining.adapters.encoder.nonlinear_value_embedding import (
    NonlinearValueEmbedding,
)
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.shared.kernel.artifacts import ArtifactRef
from tests.support.encoders import SMALL
from tests.support.token_tensors import VOCABULARY_SIZE

PRETRAINED_SEED = 1


class SmallBackbones:
    """Small encoders over the test vocabulary, the pretrained one always the same."""

    def __init__(self, vocabulary_size: int = VOCABULARY_SIZE) -> None:
        self._vocabulary_size = vocabulary_size
        self.requested: list[ArtifactRef] = []
        # Every encoder handed out, so a test can read what a run did to the one it got.
        self.built: list[SetEncoder] = []

    @property
    def width(self) -> int:
        return SMALL.width

    def pretrained(
        self, weights: ArtifactRef, *, vocabulary_size: int, dropout: float
    ) -> nn.Module:
        self.requested.append(weights)
        with torch.random.fork_rng():
            torch.manual_seed(PRETRAINED_SEED)
            encoder = SetEncoder.for_vocabulary(
                SMALL, self._vocabulary_size, dropout=dropout
            ).eval()
        # Grown outside the forked generator, as the production factory grows: the rows for
        # the task's new channels are drawn from the generator the runtime seeded.
        return self._kept(encoder.grown_to(vocabulary_size))

    def fresh(
        self,
        *,
        vocabulary_size: int,
        dropout: float,
        value_embedding: ValueEmbedding = ValueEmbedding.LINEAR,
        shape: EncoderShape | None = None,
    ) -> nn.Module:
        architecture = SMALL
        if shape is not None:
            architecture = replace(
                SMALL,
                width=shape.width,
                heads=shape.heads,
                layers=shape.layers,
                feedforward_width=shape.feedforward_width,
            )
        nonlinear = value_embedding is ValueEmbedding.NONLINEAR
        return self._kept(
            SetEncoder.for_vocabulary(
                architecture,
                max(vocabulary_size, self._vocabulary_size),
                dropout=dropout,
                value_embedding=NonlinearValueEmbedding(architecture.width) if nonlinear else None,
            )
        )

    def _kept(self, encoder: SetEncoder) -> SetEncoder:
        self.built.append(encoder)
        return encoder
