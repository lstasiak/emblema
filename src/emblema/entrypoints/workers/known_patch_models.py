from emblema.evaluation.adapters.candidates.patch_model_catalogue import PatchModelCatalogue
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.patching.patch_model_spec import PatchModelSpec
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule


class KnownPatchModels:
    """The network every campaign may compete that is trained from nothing on a grid.

    A register rather than a name read from configuration, for the reason the arms are one: the
    name is what a stored campaign refers to, and one renamed in an environment file would leave
    a finished grid naming a candidate nothing supplies.
    """

    PATCH_TRANSFORMER = CandidateRef("patch_transformer")

    @classmethod
    def refs(cls) -> tuple[CandidateRef, ...]:
        """What the patch models are called, in reporting order."""
        return (cls.PATCH_TRANSFORMER,)

    @classmethod
    def catalogue(cls, spec: PatchModelSpec, schedule: AdaptationSchedule) -> PatchModelCatalogue:
        """What the patch model is under ``spec``, learning under the arms' ``schedule``."""
        return PatchModelCatalogue(cls.PATCH_TRANSFORMER, spec, schedule)
