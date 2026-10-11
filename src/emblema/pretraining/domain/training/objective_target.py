"""What the objective asks the model to predict at a hidden token: its value, or its state.

Two classes because they are one decision: the dictionary of targets a run may state and the
target it states, with the knobs the latent one needs. Neither computes over a tensor — how a
state is read off an encoder belongs to the objective's adapter, which is held to what is stated
here.
"""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Self

from emblema.pretraining.domain.exceptions import InvalidObjectiveTargetError


class TargetKind(StrEnum):
    """What a hidden token's prediction is compared with.

    Attributes:
        VALUE: The normalised value the token was observed at. A dense channel answers it from
            its neighbours, which teaches interpolation.
        LATENT: The state a copy of the encoder, trailing the one being trained, gives the token
            when it sees the whole window. A line through the neighbours' values does not answer
            it; what the encoder makes of the window around the token does.
    """

    VALUE = "value"
    LATENT = "latent"


@dataclass(frozen=True, kw_only=True)
class ObjectiveTarget:
    """The target a run predicts at its hidden tokens, and how the latent one is formed.

    A parameter of the method like the loss: a run that predicts states is a different run over
    the same data, and its losses do not compare with a run that predicts values. It is therefore
    stated by the experiment, reported with the run and digested into its signature.

    Under the latent target the state is read off a trailing copy of the encoder — a teacher
    moved toward the encoder after every optimiser step by an exponential moving average — as the
    mean of its top ``layers`` blocks. The teacher's momentum rises linearly from ``momentum`` at
    the first step to ``final_momentum`` at the last, so it trails closely at first and settles
    as the run ends.

    Invariants: the value target states no layers and no momentum, since a knob nobody reads
    would tell two identical runs apart in the signature; the latent target reads at least one
    block and keeps its momentum in ``(0, 1]``, never falling over the run.

    Attributes:
        kind: Which target.
        layers: How many of the teacher's top blocks are averaged into the target; zero under
            the value target.
        momentum: The teacher's share of itself at the first step; zero under the value target.
        final_momentum: The teacher's share of itself at the last step; one freezes it from
            there, zero under the value target.
    """

    kind: TargetKind
    layers: int = 0
    momentum: float = 0.0
    final_momentum: float = 0.0

    def __post_init__(self) -> None:
        if self.kind is TargetKind.VALUE:
            if (self.layers, self.momentum, self.final_momentum) != (0, 0.0, 0.0):
                raise InvalidObjectiveTargetError(
                    "the value target reads no layers and has no momentum, got "
                    f"layers={self.layers}, momentum={self.momentum}, "
                    f"final_momentum={self.final_momentum}"
                )
            return
        if self.layers < 1:
            raise InvalidObjectiveTargetError(
                f"the latent target reads at least one block, got layers={self.layers}"
            )
        for name, value in (("momentum", self.momentum), ("final_momentum", self.final_momentum)):
            if not isfinite(value) or not 0.0 < value <= 1.0:
                raise InvalidObjectiveTargetError(f"{name} must lie in (0, 1], got {value}")
        if self.final_momentum < self.momentum:
            raise InvalidObjectiveTargetError(
                "the momentum never falls over a run, got "
                f"momentum={self.momentum} above final_momentum={self.final_momentum}"
            )

    @classmethod
    def value(cls) -> Self:
        """The target every run stated before a state could be one: the token's value."""
        return cls(kind=TargetKind.VALUE)

    @classmethod
    def latent(cls, *, layers: int, momentum: float = 0.996, final_momentum: float = 1.0) -> Self:
        """The teacher's state over its top ``layers`` blocks, trailing by this momentum."""
        return cls(
            kind=TargetKind.LATENT, layers=layers, momentum=momentum, final_momentum=final_momentum
        )

    @property
    def is_latent(self) -> bool:
        return self.kind is TargetKind.LATENT

    def momentum_at(self, step: int, steps: int) -> float:
        """The teacher's share of itself after optimiser step ``step`` of a run of ``steps``.

        Linear from ``momentum`` at the first step to ``final_momentum`` at the last, and held
        there past the last: a resumed run asks for the same step and gets the same share.

        Raises:
            InvalidObjectiveTargetError: If the target is not latent, the run has no steps, or
                the step is before the first.
        """
        if not self.is_latent:
            raise InvalidObjectiveTargetError("the value target has no teacher to move")
        if steps < 1:
            raise InvalidObjectiveTargetError(f"a run of {steps} steps has no momentum schedule")
        if step < 0:
            raise InvalidObjectiveTargetError(f"no step of a run is numbered {step}")
        progress = min(step, steps) / steps
        return self.momentum + (self.final_momentum - self.momentum) * progress
