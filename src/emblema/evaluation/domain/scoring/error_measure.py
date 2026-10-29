from enum import StrEnum

from emblema.evaluation.domain.exceptions import MismatchedErrorMeasureError
from emblema.evaluation.domain.labels.target_kind import TargetKind


class ErrorMeasure(StrEnum):
    """What a campaign reads a candidate's answers by, always as an error: lower is better.

    One orientation for every measure, so that a reduction, a practical floor, the rule of one
    standard error and a verdict mean the same whatever the task. A measure that is naturally a
    score enters as its distance from a perfect one; reports state the natural form. Each kind
    of target is read by one measure, registered in the campaign's design so it cannot be chosen
    after the answers are in.

    Attributes:
        RMSE: Root mean squared error over the windows, in the target's unit.
        AUROC_SHORTFALL: One minus the area under the ROC curve: the probability that a positive
            and a negative drawn at random are ordered the wrong way by the answers, a tie
            counting half. Reported as the area itself. Read from how the answers rank and not
            from their values, so a candidate is not credited for calibration here; the squared
            error of its probabilities, the Brier score, is reported beside it.
        PR_AUC_SHORTFALL: One minus the area under the precision-recall curve over the windows,
            every event weighing one, spread evenly over its windows, and every normal window
            weighing one: the precision at each event window as the sweep reaches it, averaged
            by those weights. Reported as the area itself. Read from how
            the scores rank, so a candidate is not credited for where it set its alarm; the
            events found at a fixed share of false alarms and the benchmark's event-wise score
            at the alarm it set are reported beside it.
    """

    RMSE = "rmse"
    AUROC_SHORTFALL = "auroc_shortfall"
    PR_AUC_SHORTFALL = "pr_auc_shortfall"

    @classmethod
    def of(cls, kind: TargetKind) -> "ErrorMeasure":
        """The measure that reads targets of ``kind``: the only one ``accept`` admits for it."""
        match kind:
            case TargetKind.CONTINUOUS:
                return cls.RMSE
            case TargetKind.BINARY:
                return cls.AUROC_SHORTFALL
            case TargetKind.ANOMALY:
                return cls.PR_AUC_SHORTFALL

    def accept(self, kind: TargetKind) -> None:
        """Refuse to read targets of ``kind`` by this measure where it does not apply to them.

        Raises:
            MismatchedErrorMeasureError: If the measure is not the one for that kind: a ranking
                needs two outcomes, a squared error of a probability is a calibration, not the
                task's error, and an area over events needs events.
        """
        if self is not type(self).of(kind):
            raise MismatchedErrorMeasureError(f"a {kind} target is not read by {self}")
