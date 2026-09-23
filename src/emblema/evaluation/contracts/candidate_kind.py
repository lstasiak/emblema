from enum import StrEnum


class CandidateKind(StrEnum):
    """What a candidate is made of, in the one respect other contexts need to know.

    It answers two questions outside this context. Serving asks what can run the artifact, since
    a set of learnt weights and a fitted classical model are loaded by different runtimes. This
    context asks which candidates share one compute budget: aligning the budget of a gradient
    boosting model with that of a transformer is not a defined operation, so the classical ones
    report what they spent instead of being held to a common figure.

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
