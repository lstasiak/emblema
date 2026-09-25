import torch
from torch import Tensor, nn
from torch.nn.functional import pad

from emblema.evaluation.adapters.torch.pooling import pooling_module
from emblema.evaluation.adapters.torch.regression_head import RegressionHead
from emblema.evaluation.domain.heads.head_pooling import HeadPooling
from emblema.evaluation.domain.patching.patch_model_spec import PatchModelSpec


class PatchTransformer(nn.Module):
    """A transformer over patches of each channel's row of the grid, as PatchTST reads a window.

    Every channel is read on its own by the same encoder: its row is padded at the end by one
    stride, cut into overlapping patches, and each patch of values beside the same patch of its
    mask becomes one token. The mask travels with the values because the grid carried readings
    forward, and a token that could not tell a reading from a copy of the last one would learn
    from values the sensor never reported. A token's place in the window is a learnt embedding.

    Unlike PatchTST no window is normalised by its own mean and spread: the level of a reading is
    what a remaining life is read from, and the values are already on the corpus's scale.

    The channels meet only at the head: each channel's tokens are pooled into one state by the
    pooling the plan names — the mean over the patches unless a variant turns it — the states
    are laid side by side, and a linear map reads one number out of them. So the head is as wide
    as the corpus's task has channels, and a model trained on one layout reads no other. A
    patch's place in the window, for a pooling that reads it, is the step its patch starts at
    as a share of the row.
    """

    def __init__(
        self,
        spec: PatchModelSpec,
        *,
        channels: int,
        steps: int,
        starting_at: float,
        pooling: HeadPooling | None = None,
    ) -> None:
        """A model over rows of ``steps`` steps for ``channels`` channels.

        Args:
            spec: How rows are cut into patches and how large the encoder is.
            channels: How many channels a window is read over.
            steps: How many steps each row of the grid has.
            starting_at: What the head answers before any step, in units of the label ceiling.
            pooling: How a channel's patch states become one; the mean unless given.

        Raises:
            InvalidPatchModelSpecError: If a row is shorter than one patch.
        """
        super().__init__()
        self._patch_length = spec.patch_length
        self._stride = spec.stride
        self.embedding = nn.Linear(2 * spec.patch_length, spec.width)
        patches = spec.patches_over(steps)
        self.position = nn.Parameter(torch.randn(patches, spec.width) * 0.02)
        self.pooling: nn.Module = pooling_module(
            HeadPooling.mean() if pooling is None else pooling, width=spec.width
        )
        # Declared as a tensor, since a buffer read back through the module's attribute lookup
        # is typed as either a tensor or a module.
        self.patch_times: Tensor
        self.register_buffer(
            "patch_times", torch.arange(patches, dtype=torch.float32) * spec.stride / steps
        )
        self.dropout = nn.Dropout(spec.dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                spec.width,
                spec.heads,
                spec.feedforward_width,
                spec.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            spec.layers,
            norm=nn.LayerNorm(spec.width),
            # The fast path for nested tensors needs post-norm layers; these are pre-norm.
            enable_nested_tensor=False,
        )
        self.head = RegressionHead(channels * spec.width, starting_at=starting_at)

    def forward(self, values: Tensor, observed: Tensor) -> Tensor:
        """One answer per window out of its grid, ``[batch, channels, steps]`` twice.

        Args:
            values: What the grid holds per channel and step.
            observed: One where a reading was really made, zero where it was carried.
        """
        batch, channels, _ = values.shape
        patches = torch.cat((self._patched(values), self._patched(observed)), dim=-1)
        tokens = self.dropout(self.embedding(patches) + self.position)
        encoded = self.encoder(tokens.reshape(batch * channels, -1, tokens.shape[-1]))
        rows = encoded.shape[0]
        nothing = torch.zeros(rows, encoded.shape[1], dtype=torch.bool, device=encoded.device)
        times = self.patch_times.unsqueeze(0).expand(rows, -1)
        states = self.pooling(encoded, nothing, times, nothing).reshape(batch, -1)
        answer: Tensor = self.head(states)
        return answer

    def _patched(self, rows: Tensor) -> Tensor:
        """Each row padded by one stride of its last step, then cut into patches."""
        padded = pad(rows, (0, self._stride), mode="replicate")
        return padded.unfold(-1, self._patch_length, self._stride)
