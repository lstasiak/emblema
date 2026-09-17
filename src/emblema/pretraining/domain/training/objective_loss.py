"""What the objective counts a miss as: the reading of an error, and the knee a bounded one has.

Two classes because they are one decision: the reading a run states and the dictionary of
readings it may state. Neither computes over a batch — the arithmetic over tensors belongs to the
objective's adapter, which asks ``of_error`` what a single miss is worth and is held to it.
"""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from emblema.pretraining.domain.exceptions import InvalidObjectiveLossError


class LossKind(StrEnum):
    """How the error of one reconstructed token is turned into a loss.

    Attributes:
        MSE: The squared error. A token that misses by ten weighs a hundred times one that
            misses by one, so a corpus whose values excurse decides its own gradient.
        HUBER: The squared error within the knee and a straight line past it. The loss still
            grows with the error, but the pull of one token stops growing at the knee, so a
            corpus keeps the gradient of the values it repeats.
    """

    MSE = "mse"
    HUBER = "huber"


@dataclass(frozen=True, kw_only=True)
class ObjectiveLoss:
    """The reading a run scores its hidden tokens by, and how far one token may pull.

    A parameter of the method, not of the data: a run over a corpus whose values excurse past a
    hundred standard deviations spends most of a squared error on tokens no held-out unit
    repeats, and which reading it used is what a reported loss means. It is therefore stated by
    the experiment, reported with the run and digested into its signature, like the shape and the
    share; and the tokeniser is left alone, because a task scored on excursions needs the values
    as they were measured.

    Losses of two readings are not comparable, and neither are a bounded loss and a variance: a
    reading is read against the trivial predictor under the same reading, which is one wherever
    nothing was learnt.

    Invariants: the knee is positive and finite under the bounded reading; under the squared one
    it is zero, since a knee nobody reads would tell two identical runs apart in the signature.

    Attributes:
        kind: Which reading.
        huber_delta: Where the bounded reading turns from a square into a line, in the normalised
            units of a token's value; zero under the squared reading, which has no knee.
    """

    kind: LossKind
    huber_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.kind is LossKind.HUBER:
            if not isfinite(self.huber_delta) or self.huber_delta <= 0.0:
                raise InvalidObjectiveLossError(
                    f"huber_delta must be positive and finite, got {self.huber_delta}"
                )
        elif self.huber_delta != 0.0:
            raise InvalidObjectiveLossError(
                f"{self.kind} reads no knee, so huber_delta must be 0, got {self.huber_delta}"
            )

    @property
    def is_bounded(self) -> bool:
        """Whether the pull of one token stops growing past some error."""
        return self.kind is LossKind.HUBER

    def of_error(self, error: float) -> float:
        """What one token that missed by ``error`` contributes.

        The definition the objective's adapter computes over a batch, stated once here for one
        token, so that the tensors are held to it rather than trusted. The bounded reading is
        Huber's: half the square within the knee, and past it the line that continues it with the
        same slope, which is what makes the pull bounded and the loss smooth at the knee.
        """
        if self.kind is LossKind.MSE:
            return error * error
        magnitude = abs(error)
        if magnitude <= self.huber_delta:
            return 0.5 * error * error
        return self.huber_delta * (magnitude - 0.5 * self.huber_delta)
