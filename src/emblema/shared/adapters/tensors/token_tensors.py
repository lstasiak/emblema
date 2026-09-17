from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Self

import torch
from torch import Tensor

from emblema.shared.adapters.arrays.token_batch import TokenBatch
from emblema.shared.kernel.tokens import TokenWindow


@dataclass(frozen=True)
class TokenTensors:
    """A batch of windows as the tensors a model is called with, in the order it declares them.

    The buffers of ``TokenBatch`` seen through torch, without a copy, and the one path training and
    inference share, so a model is never fed differently than it was trained. The fields and their
    meaning are ``TokenBatch``'s; fixed here is the type each reaches the model as: ``features`` and
    ``timestamps`` floating, in the codec's type, ``channel_ids`` int64, ``timeless`` and
    ``padding_mask`` bool. Device and precision belong to the run, and ``to`` retypes only the
    floating tensors.
    """

    features: Tensor
    channel_ids: Tensor
    timestamps: Tensor
    timeless: Tensor
    padding_mask: Tensor

    @classmethod
    def of(cls, batch: TokenBatch) -> Self:
        """The tensors sharing ``batch``'s memory."""
        return cls(
            features=torch.from_numpy(batch.features),
            channel_ids=torch.from_numpy(batch.channel_ids),
            timestamps=torch.from_numpy(batch.timestamps),
            timeless=torch.from_numpy(batch.timeless),
            padding_mask=torch.from_numpy(batch.padding_mask),
        )

    @classmethod
    def from_windows(cls, windows: Sequence[TokenWindow]) -> Self:
        """The tensors holding ``windows``, each padded to the longest of them.

        The signature is the one a torch data loader collates with, so the classmethod itself is
        the collation step; it is also importable in a worker process, which a closure is not.

        Raises:
            ValueError: If no window is given.
        """
        return cls.of(TokenBatch.from_windows(windows))

    @property
    def batch_size(self) -> int:
        return int(self.padding_mask.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.padding_mask.shape[1])

    @property
    def args(self) -> tuple[Tensor, ...]:
        """The tensors as positional arguments, in the order the model declares its inputs.

        Spelled out rather than read off the fields, because this order is the model's calling
        convention: reordering the fields must be a visible change here, not a silent one.
        """
        return (
            self.features,
            self.channel_ids,
            self.timestamps,
            self.timeless,
            self.padding_mask,
        )

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> Self:
        """The same batch on ``device``, its floating tensors in ``dtype``.

        The move and the change of type are two steps, in that order. Asking an accelerator for
        both at once can give back a tensor that is not the one it holds: on MPS, widening to
        double precision on the way to the host returns zeros with no error, because the device
        has no double precision to convert in (torch 2.14). Widening on the host is exact from
        any device, and costs one pass over a batch.
        """

        def moved(tensor: Tensor) -> Tensor:
            on_device = tensor.to(device=device)
            if dtype is not None and on_device.is_floating_point():
                return on_device.to(dtype=dtype)
            return on_device

        return replace(
            self, **{field.name: moved(getattr(self, field.name)) for field in fields(self)}
        )
