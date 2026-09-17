"""What a saturation curve says about its corpus, and the rules that read it.

Three readings were fixed before any curve was measured, in the order a failure invalidates the
next: a model that memorises at the largest share says nothing about the data beyond that there
is too little of it for the model; a plateau says the corpus has been learnt as far as a model of
this size trained this long can learn it; a loss still falling says the data, not the model, is
the limit. A fourth was added after the first measurement, when the satellite corpus showed a
case the three had not foreseen — a held-out side no run does better on than the channel mean,
which the rules read as overfitting — and comes before them all: nothing learnt is a precondition
of the other three, not a threshold moved. The thresholds are conventions and are changed by
editing them, never to fit a run; ADR-0027 records the addition and what the rules had said.
"""

from dataclasses import dataclass
from enum import Enum

from emblema.pretraining.domain.saturation.saturation_curve import SaturationCurve

# Relative validation loss at the largest share at or past which nothing carried over: the
# channel mean's own error, so the held-out side was not learnt and the curve reads no data.
NOT_LEARNT_FROM = 1.0
# Relative validation loss this many times the relative training loss at the largest share: the
# model fits training windows it will not see again. The same ratio the masked-reconstruction
# assessment warns at, each side read against the trivial predictor's error on that side.
GENERALISATION_GAP = 1.5
# The last step up in data lowered the validation loss by less than this share of it: the corpus
# is learnt at this budget, and a larger share of it teaches nothing a smaller one did not. The
# same tolerance the assessment reads a doubled budget against.
PLATEAU_GAIN = 0.05


class SaturationVerdict(Enum):
    """What the curve shows about the corpus at this model size and budget.

    Attributes:
        DATA_LIMITED: The loss still falls with more data at the largest share: the corpus is
            suited to pretraining and more of it, or more like it, would help.
        SATURATED: The loss stopped falling before the largest share: the corpus is exhausted at
            this size and budget, and a larger model over the same data gains nothing — more
            kinds of data would.
        OVERFITTING: The run over the whole corpus fits its training windows and not the held-out
            ones: too little data for the model, so a smaller model or a larger corpus.
        NOT_LEARNT: The run over the whole corpus does no better than the channel mean on the
            held-out side: nothing carried over, and the curve says nothing about the data — the
            held-out side, or the loss it is scored by, is what to look at first.
    """

    DATA_LIMITED = "data-limited"
    SATURATED = "saturated"
    OVERFITTING = "overfitting"
    NOT_LEARNT = "not learnt"


@dataclass(frozen=True)
class SaturationJudgement:
    """The verdict on one curve, with the two numbers it rests on and what they mean.

    Attributes:
        verdict: What the curve shows.
        gain: Relative fall of the validation loss over the last step up in data.
        generalisation_ratio: Relative validation over relative training loss at the largest
            share.
        reading: The verdict in a sentence, with what to do about it.
    """

    verdict: SaturationVerdict
    gain: float
    generalisation_ratio: float
    reading: str


def judge(curve: SaturationCurve) -> SaturationJudgement:
    """Read the curve by the rules above, in their order of precedence."""
    before, largest = curve.points[-2], curve.largest
    gain = curve.gains()[-1]
    ratio = largest.generalisation_ratio
    step = f"from {before.fraction:.0%} to {largest.fraction:.0%} of the training units"
    if largest.relative_validation_loss >= NOT_LEARNT_FROM:
        return SaturationJudgement(
            SaturationVerdict.NOT_LEARNT,
            gain,
            ratio,
            f"at the whole share validation is {largest.relative_validation_loss:.2f}× the trivial "
            "predictor's: nothing carried over to the held-out side, so the curve reads nothing "
            "about the data — the held-out side, or the loss it is scored by, comes before the "
            "corpus is judged.",
        )
    if ratio > GENERALISATION_GAP:
        return SaturationJudgement(
            SaturationVerdict.OVERFITTING,
            gain,
            ratio,
            f"at the whole share validation is {ratio:.2f}× training, each against its own "
            f"trivial predictor, past {GENERALISATION_GAP:g}×: the model fits windows it will "
            "not see again, so the data is too little for it — a smaller model, or more units.",
        )
    if gain < PLATEAU_GAIN:
        return SaturationJudgement(
            SaturationVerdict.SATURATED,
            gain,
            ratio,
            f"the step {step} lowered validation by {gain:.1%}, under {PLATEAU_GAIN:.0%}: the "
            "corpus is learnt at this size and budget, and a larger model over it gains nothing "
            "— other corpora would.",
        )
    return SaturationJudgement(
        SaturationVerdict.DATA_LIMITED,
        gain,
        ratio,
        f"the step {step} lowered validation by {gain:.1%}: the data, not the model, is the "
        "limit at this budget — the corpus suits pretraining and more of it would help.",
    )
