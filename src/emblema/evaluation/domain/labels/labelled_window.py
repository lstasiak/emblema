from dataclasses import dataclass

from emblema.evaluation.domain.labels.task_window import TaskWindow


@dataclass(frozen=True, kw_only=True)
class LabelledWindow:
    """A window together with the target a model is asked to predict for it.

    The label is kept beside the window rather than inside it because the same window carries a
    different target under a different scheme, and because the pool a budget is drawn from is
    exactly the windows that have one.

    Attributes:
        window: Which window, and where the block holds it.
        target: What the scheme says the window's answer is.
    """

    window: TaskWindow
    target: float
