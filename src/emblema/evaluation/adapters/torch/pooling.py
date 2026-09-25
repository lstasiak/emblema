"""The module a head pools with, built from the pooling a plan names.

A function rather than a class, because it holds no state and is shared by the two networks
that read a pooling — the candidate made out of a backbone and the patch model — each of which
calls the module it gets the same way: states in, one state per window out.
"""

from torch import nn

from emblema.evaluation.adapters.torch.attention_pooling import AttentionPooling
from emblema.evaluation.adapters.torch.mean_pooling import MeanPooling
from emblema.evaluation.adapters.torch.tail_pooling import TailPooling
from emblema.evaluation.domain.heads.head_pooling import HeadPooling, PoolingScheme


def pooling_module(pooling: HeadPooling, *, width: int) -> nn.Module:
    """The module computing ``pooling`` over states of ``width``."""
    match pooling.pooling:
        case PoolingScheme.MEAN:
            return MeanPooling()
        case PoolingScheme.TAIL:
            return TailPooling(pooling.tail_share)
        case PoolingScheme.ATTENTION:
            return AttentionPooling(width)
