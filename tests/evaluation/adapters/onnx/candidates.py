"""The fitted candidates the export tests stand on: one per transfer mode, over a grown vocabulary.

Apparatus only. The task's corpus names channels past the backbone's vocabulary, so every
pretrained candidate carries grown rows and the graph has to read them; every trainable part is
nudged off its starting value, so that a part which starts at zero — a low-rank update — counts
in the comparison rather than exporting as nothing.
"""

from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import torch

from emblema.evaluation.adapters.onnx.inference_graph import InferenceGraph
from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone
from emblema.evaluation.domain.heads.head_pooling import HeadPooling, PoolingScheme
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.shared.adapters.tensors.token_tensors import TokenTensors
from tests.evaluation.support import plan
from tests.support.backbones import SmallBackbones
from tests.support.token_tensors import VOCABULARY_SIZE

GROWN_CHANNELS = 3
VOCABULARY = VOCABULARY_SIZE + GROWN_CHANNELS
TARGET_SCALE = 125.0


POOLINGS = (
    HeadPooling.mean(),
    HeadPooling(pooling=PoolingScheme.TAIL, tail_share=0.2),
    HeadPooling(pooling=PoolingScheme.ATTENTION),
)


def adapted(
    mode: TransferMode, *, seed: int = 1, pooling: HeadPooling | None = None
) -> AdaptedBackbone:
    """A candidate under ``mode`` as a short run might leave it, in evaluation mode on the host."""
    torch.manual_seed(seed)
    stated = plan(mode) if pooling is None else plan(mode, pooling=pooling)
    candidate = AdaptedBackbone.under(
        stated, SmallBackbones(), vocabulary_size=VOCABULARY, starting_at=0.5
    )
    with torch.no_grad():
        for parameter in candidate.trainable_parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
    return candidate.eval()


@dataclass(frozen=True)
class Exported:
    """A candidate's graph beside the eager candidate it was exported from.

    Every claim about the export is a comparison against the model that produced it.
    """

    candidate: AdaptedBackbone
    graph: InferenceGraph

    def embed_eager(self, batch: TokenTensors) -> np.ndarray[Any, Any]:
        with torch.no_grad():
            return self.candidate.embed(batch).numpy()

    def predict_eager(self, batch: TokenTensors) -> np.ndarray[Any, Any]:
        """What the candidate answers, multiplied back into the task's unit as the graph does."""
        with torch.no_grad():
            return (self.candidate(batch) * TARGET_SCALE).numpy()


@cache
def exported(mode: TransferMode, pooling: HeadPooling | None = None) -> Exported:
    """Export once per mode, pooling and process: it takes seconds, and is deterministic."""
    candidate = adapted(mode, pooling=pooling)
    return Exported(candidate, InferenceGraph.exported(candidate, target_scale=TARGET_SCALE))


def over_grown_channels(batch: TokenTensors) -> TokenTensors:
    """The batch with its first tokens moved onto the channels the task grew."""
    grown = torch.arange(VOCABULARY_SIZE + 1, VOCABULARY + 1)
    channel_ids = batch.channel_ids.clone()
    channel_ids[:, : len(grown)] = grown
    return TokenTensors(
        batch.features, channel_ids, batch.timestamps, batch.timeless, batch.padding_mask
    )
