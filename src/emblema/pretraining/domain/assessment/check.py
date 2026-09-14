from dataclasses import dataclass
from enum import Enum

from emblema.pretraining.domain.mask_kind import MaskKind


class Area(Enum):
    """Where a failed rule sends someone to fix it.

    Attributes:
        IMPLEMENTATION: A correct pipeline cannot fail the rule: the loss that does not fall, a kind
            of mask the strategy draws and the verdict never sees, a model below the measurement
            noise (hidden values reach it, or the loss scores what was not hidden), a model no
            better than the channel mean on every kind at once. Failing it means changing the logic.
        TRAINING: The run cannot be read as it stands: whether a longer one would change the
            verdicts is unknown or known to be yes, the last epoch is off its best, or the model
            memorises.
        MASKING: A kind of mask teaches nothing its trivial baseline did not know. Where the
            baseline already sits at the noise floor that is a fact about the kind on this corpus
            and no training changes it; where it leaves room, a run that has not shown it stopped
            learning may still beat it, and the verdict waits for one that has.
        DIAGNOSTIC: The measurement cannot answer its own question on this corpus.
        EVIDENCE: Too few units or tokens for the verdict to mean anything.
    """

    IMPLEMENTATION = "implementation"
    TRAINING = "training"
    MASKING = "masking"
    DIAGNOSTIC = "diagnostic"
    EVIDENCE = "evidence"


class Rule(Enum):
    """Every question the assessment asks; a check is one rule applied, per kind where it says."""

    LOSS_FALLS = "loss-falls"
    HIDDEN_SHARE = "hidden-share"
    KINDS_PRESENT = "kinds-present"
    COLLAPSED = "collapsed"
    ABOVE_FLOOR = "above-floor"
    CONVERGED = "converged"
    VALIDATION_STABLE = "validation-stable"
    GENERALISATION_GAP = "generalisation-gap"
    TRIVIAL = "trivial"
    ROOM = "room"
    BEATS_MEAN = "beats-mean"
    BEYOND_LINEAR = "beyond-linear"
    EVIDENCE = "evidence"
    SPECTRUM_INFORMATIVE = "spectrum-informative"
    SPECTRUM_FITTED = "spectrum-fitted"

    @property
    def area(self) -> Area:
        return _AREAS[self]


_AREAS = {
    Rule.LOSS_FALLS: Area.IMPLEMENTATION,
    Rule.HIDDEN_SHARE: Area.IMPLEMENTATION,
    Rule.KINDS_PRESENT: Area.IMPLEMENTATION,
    Rule.COLLAPSED: Area.IMPLEMENTATION,
    Rule.ABOVE_FLOOR: Area.IMPLEMENTATION,
    Rule.CONVERGED: Area.TRAINING,
    Rule.VALIDATION_STABLE: Area.TRAINING,
    Rule.GENERALISATION_GAP: Area.TRAINING,
    Rule.TRIVIAL: Area.MASKING,
    Rule.ROOM: Area.MASKING,
    Rule.BEATS_MEAN: Area.MASKING,
    Rule.BEYOND_LINEAR: Area.MASKING,
    Rule.EVIDENCE: Area.EVIDENCE,
    Rule.SPECTRUM_INFORMATIVE: Area.DIAGNOSTIC,
    Rule.SPECTRUM_FITTED: Area.EVIDENCE,
}


class Status(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "n/a"


@dataclass(frozen=True)
class Check:
    """One rule applied to one run.

    Attributes:
        rule: The question asked.
        kind: The kind of mask it was asked about; ``None`` for a question about the whole run.
        status: The outcome.
        measured: What the run shows.
        expected: What the rule asks for.
        reading: What the outcome means and, where it is not a pass, what to do.
    """

    rule: Rule
    kind: MaskKind | None
    status: Status
    measured: str
    expected: str
    reading: str

    @property
    def name(self) -> str:
        """Stable identifier, so that a check can be followed across runs."""
        return self.rule.value if self.kind is None else f"{self.rule.value}-{self.kind.value}"

    @property
    def area(self) -> Area:
        return self.rule.area
