import io
from dataclasses import dataclass
from typing import Any, Self

import torch
from torch import Tensor

from emblema.pretraining.adapters.training.exceptions import UNREADABLE_BYTES
from emblema.pretraining.domain.exceptions import IncompatibleCheckpointError
from emblema.pretraining.domain.training.run_position import RunPosition
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Everything an interrupted run needs to carry on as though it had not been interrupted.

    The weights alone would resume a run that trains differently from the one that stopped:
    momentum, the point the learning rate had reached, the generator the masks are drawn from and
    the global generator dropout draws from all decide the next step as much as the weights do.
    They are held here together with the position, which is what fixes the order the windows come
    in and how much of the epoch is already behind.

    Everything is stored on the CPU. A tensor saved from an accelerator is a tensor that only
    loads where that accelerator exists, and a model kept on MPS cannot be exported at all, so the
    device a run happened on stays out of what the run leaves behind.

    Attributes:
        signature: Which run this belongs to.
        position: How far the run had come.
        model: State of the objective, the encoder inside it included.
        optimiser: State of the optimiser, momentum and step counts included.
        scaler: State of the gradient scaler; empty where the precision needs none.
        masks: State of the generator the masks are drawn from.
        seeds: State of the host's global generator, which initialisation draws from.
        device_seeds: State of the accelerator's own generator, which dropout draws from where
            the run is not on the host; ``None`` on the host.
        best_relative: The lowest mean relative validation an epoch of the run has reached, so
            that a resumed run keeps its best epoch rather than the best since it resumed;
            ``None`` before any epoch was scored.
        best_weights: The weights of that epoch, as written; ``None`` with ``best_relative``.
    """

    signature: RunSignature
    position: RunPosition
    model: dict[str, Any]
    optimiser: dict[str, Any]
    scaler: dict[str, Any]
    masks: Tensor
    seeds: Tensor
    device_seeds: Tensor | None = None
    best_relative: float | None = None
    best_weights: ArtifactRef | None = None

    def to_bytes(self) -> bytes:
        """The checkpoint as the bytes the artifact store keeps, every tensor on the CPU."""
        buffer = io.BytesIO()
        torch.save(
            {
                "signature": self.signature.digest,
                "epoch": self.position.epoch,
                "batches": self.position.batches,
                "steps": self.position.steps,
                "model": _on_cpu(self.model),
                "optimiser": _on_cpu(self.optimiser),
                "scaler": _on_cpu(self.scaler),
                "masks": _on_cpu(self.masks),
                "seeds": _on_cpu(self.seeds),
                "device_seeds": _on_cpu(self.device_seeds),
                "best_relative": self.best_relative,
                "best_weights": (
                    None
                    if self.best_weights is None
                    else [self.best_weights.key, str(self.best_weights.checksum)]
                ),
            },
            buffer,
        )
        return buffer.getvalue()

    @classmethod
    def read(cls, content: bytes, *, signature: RunSignature) -> Self:
        """The checkpoint those bytes hold, refused where it belongs to another run.

        Read with ``weights_only``: a checkpoint is state, and one that asks to run code on the
        way in is not one of ours.

        Raises:
            IncompatibleCheckpointError: If the bytes are not a checkpoint, or are one of a run
                with another configuration or corpus.
        """
        try:
            stored = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
            # Every field is read here, inside the guard: a checkpoint missing one is a checkpoint
            # this cannot read, and saying so is the port's contract — not a stray KeyError.
            read = cls(
                signature=RunSignature(stored["signature"]),
                position=RunPosition(
                    epoch=stored["epoch"], batches=stored["batches"], steps=stored["steps"]
                ),
                model=stored["model"],
                optimiser=stored["optimiser"],
                scaler=stored["scaler"],
                masks=stored["masks"],
                seeds=stored["seeds"],
                device_seeds=stored["device_seeds"],
                best_relative=stored["best_relative"],
                best_weights=(
                    None
                    if stored["best_weights"] is None
                    else ArtifactRef(
                        stored["best_weights"][0], Checksum.parse(stored["best_weights"][1])
                    )
                ),
            )
        except UNREADABLE_BYTES as error:
            raise IncompatibleCheckpointError(f"not a checkpoint this can read: {error}") from error
        if read.signature != signature:
            raise IncompatibleCheckpointError(
                f"checkpoint of run {read.signature} offered to run {signature}"
            )
        return read


def _on_cpu(value: Any) -> Any:
    """``value`` with every tensor in it copied to the host, whatever it is nested in."""
    if isinstance(value, Tensor):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {key: _on_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_on_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_on_cpu(item) for item in value)
    return value
