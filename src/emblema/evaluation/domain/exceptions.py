"""Every exception the Evaluation context raises, including those its ports declare."""


class EvaluationError(Exception):
    """Base of everything this context raises."""


class InvalidUnitKeyError(EvaluationError, ValueError):
    """A unit key is blank or carries surrounding whitespace."""


class InvalidTaskWindowError(EvaluationError, ValueError):
    """A window is placed at a negative position or ends at a time that is not finite."""


class InvalidTaskSplitError(EvaluationError, ValueError):
    """The sides of a task's split are empty, overlapping or not named by any corpus."""


class InvalidLabelSchemeError(EvaluationError, ValueError):
    """A label scheme's ceiling is not positive and finite."""


class UnlabelledWindowError(EvaluationError, ValueError):
    """A window ends at or after the moment its unit failed, so it carries no remaining life."""


class EmptyLabelSampleError(EvaluationError, ValueError):
    """A sample without a window has no mean label to start from."""


class InvalidLabelBudgetError(EvaluationError, ValueError):
    """A budget asks for no window, for a fraction of one, or for more than the pool holds."""


class InvalidTargetBinsError(EvaluationError, ValueError):
    """A stratification asks for fewer than one bin, or for more bins than there are windows."""


class InvalidLoraSpecError(EvaluationError, ValueError):
    """A low-rank update has no rank, a scale that is not positive, or names no layer."""


class InvalidAdaptationScheduleError(EvaluationError, ValueError):
    """An adaptation schedule asks for no epoch or window per batch, or for a rate not positive."""


class InvalidAdaptationPlanError(EvaluationError, ValueError):
    """A plan names weights a mode does not start from, or low-rank updates a mode does not add."""


class InvalidWindowPredictionError(EvaluationError, ValueError):
    """A prediction or its target is not a finite number."""


class InvalidUnitErrorError(EvaluationError, ValueError):
    """A unit's error covers no window or is not a finite, non-negative sum."""


class InvalidAdaptationOutcomeError(EvaluationError, ValueError):
    """An outcome predicts nothing or a window twice, or reports other epochs than were planned."""


class DivergedAdaptationError(EvaluationError):
    """A candidate's training loss stopped being finite, before a step was taken on it."""


class FrozenTestSplitClosedError(EvaluationError):
    """The frozen test side was asked for in a run that is not the final one."""


class ProtocolMismatchError(EvaluationError):
    """A task carries a label scheme its protocol has no use for, or lacks one it needs."""


class ForeignLabelSampleError(EvaluationError):
    """A sample of labels was drawn from another task than the one it is used on."""


class LoraTargetNotFoundError(EvaluationError):
    """A low-rank update names a layer the backbone does not have."""


class TaskNotFoundError(EvaluationError):
    """No task is stored under that identity."""


class UnknownBackboneError(EvaluationError):
    """A plan names pretrained weights the runtime was not given and cannot supply."""


class UnknownTaskUnitsError(EvaluationError):
    """A task was defined over units the published corpus does not name."""


class UnknownGroundTruthError(EvaluationError):
    """The ground truth says nothing about a window or a unit the task reads labels for."""


class UnreadableFittedCandidateError(EvaluationError):
    """The bytes an artifact holds are not a fitted candidate this context wrote."""


class CandidateNotRetainableError(EvaluationError):
    """A run was asked to keep the candidate it fitted, in a process with nowhere to keep it."""


class UnreadableTaskCorpusError(EvaluationError):
    """A published corpus is not one this context can read."""


class UnreadableGroundTruthError(EvaluationError):
    """The file the ground truth of a corpus is published in is missing or not what it claims."""


class InvalidComputeBudgetError(EvaluationError, ValueError):
    """A compute budget asks for no epoch, no window per batch, or a negative floor of steps."""


class InvalidCandidateMethodError(EvaluationError, ValueError):
    """A method leaves a parameter unnamed or blank, or names one of them twice."""


class InvalidCampaignCandidateError(EvaluationError, ValueError):
    """A candidate declares a compute budget its kind does not share, or lacks one it must."""


class InvalidCampaignDesignError(EvaluationError, ValueError):
    """A design repeats a coordinate, names a candidate it does not hold, or leaves no control."""


class InvalidCellResultError(EvaluationError, ValueError):
    """A cell's result scores no unit, or reports a time that is not finite and not negative."""


class UnknownCandidateError(EvaluationError):
    """A campaign was asked about a candidate its design does not name."""


class UnknownCampaignCellError(EvaluationError):
    """A result was recorded against a cell the campaign's grid does not hold."""


class CampaignCellAlreadyRecordedError(EvaluationError):
    """A cell of the grid was recorded twice; a repeat is another seed, not another attempt."""


class CampaignClosedError(EvaluationError):
    """A campaign that has already finished was asked to record another cell."""


class IncompleteCampaignError(EvaluationError):
    """A campaign was asked to finish while cells of its grid had not run."""


class CampaignNotCompletedError(EvaluationError):
    """A verdict was asked of a campaign that has not finished."""


class CampaignNotFoundError(EvaluationError):
    """No campaign is stored under that identity."""


class InvalidPairedUnitErrorsError(EvaluationError, ValueError):
    pass


class InvalidBootstrapIntervalError(EvaluationError, ValueError):
    pass


class InvalidPairedUnitBootstrapError(EvaluationError, ValueError):
    pass


class InvalidPairedDifferenceError(EvaluationError, ValueError):
    pass


class InvalidPracticalFloorError(EvaluationError, ValueError):
    pass


class InvalidHolmCorrectionError(EvaluationError, ValueError):
    pass


class InvalidRemainingLifeMetricsError(EvaluationError, ValueError):
    pass


class InvalidComparisonRulesError(EvaluationError, ValueError):
    pass
