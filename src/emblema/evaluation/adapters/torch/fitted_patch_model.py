import io
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import torch

from emblema.evaluation.adapters.torch.fitted_candidate import UNREADABLE_BYTES
from emblema.evaluation.adapters.torch.grid_reading import GridReading
from emblema.evaluation.adapters.torch.patch_transformer import PatchTransformer
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError
from emblema.evaluation.domain.heads.head_pooling import HeadPooling, PoolingScheme
from emblema.evaluation.domain.patching.patch_model_spec import PatchModelSpec
from emblema.evaluation.domain.patching.patch_plan import PatchPlan


@dataclass(frozen=True)
class FittedPatchModel:
    """A patch model as one run left it: its weights, and what it takes to answer again.

    The weights alone would not be enough: a window has to be laid on the same grid and read
    over the same channels, the model built in the same shape, and the answer read back in the
    task's unit. So the plan, the reading and the target's scale travel with the weights. A
    kept candidate's manifest names this form ``FORMAT``; it is the measured form and the only
    one.

    Attributes:
        parameters: The plan the model was trained under, flattened to scalars; the shape is
            read back out of them.
        reading: The grid a window is laid on and the channels read off it.
        target_scale: What the targets were divided by.
        weights: State of the whole model, on the host.
    """

    FORMAT: ClassVar[str] = "torch-patch-state"

    parameters: dict[str, str | int | float]
    reading: GridReading
    target_scale: float
    weights: dict[str, Any]

    @classmethod
    def of(
        cls,
        plan: PatchPlan,
        model: PatchTransformer,
        *,
        reading: GridReading,
        target_scale: float,
    ) -> Self:
        """The model as it ended, its weights moved to the host before anything else is done."""
        return cls(
            parameters=plan.parameters(),
            reading=reading,
            target_scale=target_scale,
            weights={key: value.detach().to("cpu") for key, value in model.state_dict().items()},
        )

    def build(self) -> PatchTransformer:
        """The model these weights belong to, in evaluation mode on the host."""
        stated = self.parameters
        spec = PatchModelSpec(
            patch_length=int(stated["patch_length"]),
            stride=int(stated["stride"]),
            width=int(stated["width"]),
            heads=int(stated["heads"]),
            layers=int(stated["layers"]),
            feedforward_width=int(stated["feedforward_width"]),
            dropout=float(stated["dropout"]),
            grid_resolution=float(stated["grid_resolution"]),
        )
        # A model kept before the head had a pooling knob pooled by the mean.
        pooling = HeadPooling(
            pooling=PoolingScheme(str(stated.get("pooling", PoolingScheme.MEAN))),
            tail_share=float(stated.get("tail_share", 1.0)),
        )
        model = PatchTransformer(
            spec,
            channels=len(self.reading.held),
            steps=self.reading.steps,
            starting_at=0.0,
            pooling=pooling,
        )
        model.load_state_dict(self.weights)
        return model.eval()

    def to_bytes(self) -> bytes:
        """The model as the bytes the artifact store keeps."""
        buffer = io.BytesIO()
        torch.save(
            {
                "parameters": self.parameters,
                "grid": [self.reading.steps, self.reading.channels],
                "held": list(self.reading.held),
                "target_scale": self.target_scale,
                "weights": self.weights,
            },
            buffer,
        )
        return buffer.getvalue()

    @classmethod
    def read(cls, content: bytes) -> Self:
        """The model those bytes hold, checked by building it.

        A shape that does not stand up is refused as a value error, which is one of the ways
        bytes turn out not to be ours.

        Raises:
            UnreadableFittedCandidateError: If the bytes are not a patch model of ours.
        """
        try:
            stored = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
            steps, channels = stored["grid"]
            read = cls(
                parameters=stored["parameters"],
                reading=GridReading(
                    steps=int(steps), channels=int(channels), held=tuple(stored["held"])
                ),
                target_scale=float(stored["target_scale"]),
                weights=stored["weights"],
            )
            read.build()
            return read
        except UNREADABLE_BYTES as error:
            raise UnreadableFittedCandidateError(
                f"not a patch model this can read: {error}"
            ) from error
