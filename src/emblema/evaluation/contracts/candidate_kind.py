from enum import StrEnum


class CandidateKind(StrEnum):
    """What a candidate is made of, in the one respect other contexts need to know.

    It answers three questions. Serving asks what can run the artifact, since a set of learnt
    weights and a fitted classical model are loaded by different runtimes. This context asks which
    candidates share one compute budget: aligning the budget of a gradient boosting model with
    that of a transformer is not a defined operation, so the classical ones report what they spent
    instead of being held to a common figure. And handing a cell to a queue asks which image can
    run it, since only one of the two carries the machine-learning stack.

    The last two coincide today and are asked separately because they are different questions: a
    method that shared no compute budget but still needed the stack would answer them apart.

    Attributes:
        NEURAL: A network the campaign adapts to the task. Every one in a campaign is held to the
            same compute budget.
        CLASSICAL: A method fitted to the task by its own procedure, serialised as it stands.
    """

    NEURAL = "neural"
    CLASSICAL = "classical"

    @property
    def shares_the_compute_budget(self) -> bool:
        return self is CandidateKind.NEURAL

    @property
    def needs_the_ml_stack(self) -> bool:
        """Whether running one of these needs the image that carries the training stack."""
        return self is CandidateKind.NEURAL
