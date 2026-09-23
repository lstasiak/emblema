from enum import StrEnum


class EvaluationProtocol(StrEnum):
    """Which question a task asks, and therefore what a campaign over it is allowed to measure.

    The two are different quantities and never belong on one chart: one says how far a budget of
    labels goes, the other how well anomalies are found without any. Keeping them apart is the
    point of naming them — a campaign that spread a detection task over label budgets would
    report an axis that does not exist for it.

    Attributes:
        LABEL_BUDGET: Supervised learning from a counted number of labelled windows, the axis of
            the label-efficiency curve. The task carries a label scheme and a stratification,
            because a budget is drawn over groups of the target.
        ANOMALY_DETECTION: Detection fitted on normal data and scored on held-out data, with no
            labels spent and so no budget axis. The task carries no label scheme: there is no
            target to read per window.
    """

    LABEL_BUDGET = "label_budget"
    ANOMALY_DETECTION = "anomaly_detection"

    @property
    def spends_labels(self) -> bool:
        return self is EvaluationProtocol.LABEL_BUDGET
