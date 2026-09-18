from dataclasses import dataclass
from math import isfinite

from emblema.pretraining.domain.exceptions import InvalidTrainingOutcomeError
from emblema.pretraining.domain.training.corpus_validation import CorpusValidation
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class EpochOutcome:
    """What one epoch measured and what it left in the artifact store.

    Three references rather than one, because the artifacts are kept for different reasons and
    for different lengths of time: a checkpoint carries the whole state of the run and exists so
    that a dropped session can be picked up; the weights of an epoch are written when it is the
    best the run has seen by its validation, so that the epoch worth keeping is kept whatever
    comes after it; and the backbone is the weights the run ends with — the best epoch's — which
    appears on the last epoch of a run and nowhere else, which is what lets a caller consuming
    the epochs one by one know when it has them all.

    Invariants: the epoch is not negative; the training loss is finite and not negative; the
    time taken is not negative; the share hidden lies in ``(0, 1]``, since an epoch that hid
    nothing scored nothing; at least one corpus was validated and none of them twice.

    Attributes:
        epoch: Which epoch this was, counted from zero.
        training_loss: The run's reading over the epoch's hidden tokens.
        validation: What each corpus of the held-out side cost, one entry per corpus, beside the
            trivial predictor's loss over the same tokens. A run over one corpus reports one
            entry; the mixture reports one each, because a single number over all of them is the
            largest corpus's number.
        hidden_ratio: Share of the validation windows' observed tokens those masks hid, counted
            rather than taken from what the strategy expects.
        seconds: Wall-clock time the epoch took, training and scoring together.
        checkpoint: The last resumable state written during the epoch; ``None`` where the policy
            asked for none.
        weights: The weights as this epoch left them, written because the epoch is the best the
            run has seen; ``None`` for an epoch that did not improve on an earlier one.
        backbone: The weights the run kept, on its final epoch only: those of the epoch whose
            validation was lowest, which may be an earlier one or one a resumed run inherited.
    """

    epoch: int
    training_loss: float
    validation: tuple[CorpusValidation, ...]
    hidden_ratio: float
    seconds: float
    checkpoint: ArtifactRef | None = None
    weights: ArtifactRef | None = None
    backbone: ArtifactRef | None = None

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise InvalidTrainingOutcomeError(f"epoch must not be negative, got {self.epoch}")
        if not self.validation:
            raise InvalidTrainingOutcomeError("an epoch must have validated a corpus")
        named = [scored.corpus for scored in self.validation]
        if len(set(named)) != len(named):
            raise InvalidTrainingOutcomeError(f"a corpus is validated twice: {sorted(named)}")
        for label, value in (
            ("training_loss", self.training_loss),
            ("seconds", self.seconds),
        ):
            if not isfinite(value) or value < 0.0:
                raise InvalidTrainingOutcomeError(
                    f"{label} must be finite and not negative, got {value}"
                )
        if not 0.0 < self.hidden_ratio <= 1.0:
            raise InvalidTrainingOutcomeError(
                f"hidden_ratio must lie in (0, 1], got {self.hidden_ratio}"
            )

    @property
    def validation_loss(self) -> float:
        """The loss over every hidden validation token, whichever corpus it came from.

        Derived from the corpora rather than reported beside them, so that the number a single
        curve is drawn from and the numbers per corpus cannot disagree. It weighs a corpus by the
        tokens it contributed, which is what a loss over the whole side is.
        """
        tokens = sum(scored.tokens for scored in self.validation)
        return sum(scored.loss * scored.tokens for scored in self.validation) / tokens

    @property
    def relative_validation(self) -> float:
        """The mean over corpora of what each cost against nothing to learn on it.

        What a run that stops on its validation reads, and what keeps the best epoch: a mean over
        corpora rather than over tokens, so that no corpus decides the number by the size of its
        values or the count of its windows.
        """
        return sum(scored.relative for scored in self.validation) / len(self.validation)
