from torch import Tensor, nn

from emblema.pretraining.adapters.encoder.masked_mean_pooling import MaskedMeanPooling
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder


class PooledSetEncoder(nn.Module):
    """The encoder with its pooling on top: one embedding per window, as an inference graph returns.

    Test apparatus. The encoder returns a state per token because its consumers disagree about
    what to do with them; an exported graph has one consumer and returns the pooled embedding, so
    the two are composed here for the export. The adapter that will export a trained backbone
    makes this composition its own.
    """

    def __init__(self, encoder: SetEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = MaskedMeanPooling()
        # An inference graph is traced in evaluation mode; the composition puts itself there so
        # that no caller has to remember, and the exporter has nothing to warn about.
        self.eval()

    @property
    def width(self) -> int:
        return self.encoder.architecture.width

    def forward(
        self,
        features: Tensor,
        channel_ids: Tensor,
        timestamps: Tensor,
        timeless: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        states = self.encoder(features, channel_ids, timestamps, timeless, padding_mask)
        return self.pooling(states, padding_mask)
