from typing import Protocol

from torch import nn

from emblema.shared.kernel.artifacts import ArtifactRef


class BackboneFactory(Protocol):
    """Where this context's torch adapters get an encoder from, without knowing whose it is.

    A seam of the adapter, not a port of the hexagon: no use case sees it, and it speaks torch,
    which a port may not. Evaluation may not import the context that owns the encoder, so the
    process that composes the adapter implements this. A backbone is any module that takes the
    five tensors of a batch in the order ``TokenTensors.args`` gives them and returns one state
    per token, ``[batch, tokens, width]``; it is built on the host and moved by the runtime.

    Every encoder is asked for over the vocabulary of the task's corpus, because a corpus
    published under a vocabulary that continues the backbone's names channels the backbone never
    saw: a pretrained encoder grows rows for them, and the modules holding grown rows say so
    through ``GrownParameters``, so the adapter can train them under every mode.
    """

    @property
    def width(self) -> int:
        """Size of the state a token comes back as, which sizes the head."""
        ...

    def pretrained(self, weights: ArtifactRef, *, vocabulary_size: int) -> nn.Module:
        """The encoder holding the weights stored under ``weights``, in evaluation mode.

        Grown to ``vocabulary_size`` channels where the stored ones cover fewer.

        Raises:
            UnknownBackboneError: If this factory does not serve those weights.
        """
        ...

    def fresh(self, *, vocabulary_size: int) -> nn.Module:
        """An encoder of the same shape with weights drawn anew from torch's generator.

        Over ``vocabulary_size`` channels, or the stored vocabulary where that is larger, so
        that a fresh encoder has the parameters a pretrained one has over the same task.
        """
        ...
