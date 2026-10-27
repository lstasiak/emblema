from enum import StrEnum


class CandidateKind(StrEnum):
    """What a candidate is made of, in the one respect other contexts need to know.

    It answers three questions. Serving asks what can run the artifact, since a set of learnt
    weights and a fitted classical model are loaded by different runtimes. This context asks which
    candidates share one compute budget: aligning the budget of a gradient boosting model with
    that of a transformer is not a defined operation, so the classical ones report what they spent
    instead of being held to a common figure. And handing a cell to a queue asks which image can
    run it, since only one of the two carries the machine-learning stack.

    The last two are different questions, and a detector answers them apart: it fits no network
    and shares no compute budget, yet one that reads a frozen encoder's states needs the stack
    that runs the encoder.

    Attributes:
        NEURAL: A network the campaign adapts to the task. Every one in a campaign is held to the
            same compute budget.
        CLASSICAL: A method fitted to the task by its own procedure, serialised as it stands.
        DETECTOR: A detector fitted on windows without a label, scoring a window by its
            distance to the fitted ones; nothing serves it yet.
    """

    NEURAL = "neural"
    CLASSICAL = "classical"
    DETECTOR = "detector"

    @property
    def shares_the_compute_budget(self) -> bool:
        return self is CandidateKind.NEURAL

    @property
    def fits_on_labels(self) -> bool:
        """Whether a candidate of this kind learns from a budget of labels, or from none."""
        return self is not CandidateKind.DETECTOR

    @property
    def needs_the_ml_stack(self) -> bool:
        """Whether running one of these needs the image that carries the training stack."""
        return self is not CandidateKind.CLASSICAL
