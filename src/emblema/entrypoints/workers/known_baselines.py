from collections.abc import Sequence

from emblema.evaluation.adapters.candidates.classical_arm import ClassicalArm
from emblema.evaluation.adapters.candidates.classical_baseline_catalogue import (
    ClassicalBaselineCatalogue,
)
from emblema.evaluation.contracts.identifiers import CandidateRef, TaskId
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.classical.gradient_boosting_spec import GradientBoostingSpec


class KnownBaselines:
    """The classical methods every campaign competes, each under the name it is compared by.

    Two of them, differing in the one thing that decides what a baseline can be fitted over: a
    vector as wide as the corpus has channels, and one whose width is the same whatever the
    layout. The second is the only one that can take in another corpus's labels, which is why
    the source tasks arrive with it rather than being a property of both.

    A register rather than a table read from configuration, for the reason the arms are one: the
    names are what a stored campaign refers to, and a baseline renamed in an environment file
    would leave a finished grid naming candidates nothing supplies.
    """

    PER_CHANNEL = CandidateRef("boosted_trees_per_channel")
    ACROSS_CHANNELS = CandidateRef("boosted_trees_across_channels")

    @classmethod
    def over(cls, sources: Sequence[TaskId] = ()) -> tuple[ClassicalArm, ...]:
        """Both baselines, the layout-independent one drawing on ``sources``, in reporting order."""
        return (
            ClassicalArm(ref=cls.PER_CHANNEL, features=FeatureScheme.PER_CHANNEL, sources=()),
            ClassicalArm(
                ref=cls.ACROSS_CHANNELS,
                features=FeatureScheme.CHANNEL_AGGREGATED,
                sources=tuple(sources),
            ),
        )

    @classmethod
    def refs(cls) -> tuple[CandidateRef, ...]:
        """What the baselines are called, in reporting order."""
        return (cls.PER_CHANNEL, cls.ACROSS_CHANNELS)

    @classmethod
    def catalogue(
        cls, boosting: GradientBoostingSpec, sources: Sequence[TaskId] = ()
    ) -> ClassicalBaselineCatalogue:
        """What the two baselines are, with nothing that could fit one."""
        return ClassicalBaselineCatalogue(cls.over(sources), boosting)
