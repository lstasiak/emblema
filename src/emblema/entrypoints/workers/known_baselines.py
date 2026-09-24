from collections.abc import Sequence

from emblema.evaluation.adapters.candidates.classical_arm import ClassicalArm
from emblema.evaluation.adapters.candidates.classical_baseline_catalogue import (
    ClassicalBaselineCatalogue,
)
from emblema.evaluation.contracts.identifiers import CandidateRef, TaskId
from emblema.evaluation.domain.classical.boosted_trees import BoostedTrees
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.classical.gradient_boosting_spec import GradientBoostingSpec
from emblema.evaluation.domain.classical.random_convolutions import RandomConvolutions


class KnownBaselines:
    """The classical methods every campaign competes, each under the name it is compared by.

    Three read a window as a summary and grow trees on it — per channel, as the spectrum per
    channel, and summarised across channels — and one lays it on a grid and convolves it. Only
    the summary across channels has the same width whatever the layout, so it is the only one
    that may take in another corpus's labels, which is why the source tasks arrive with it
    rather than being a property of all four.

    A register rather than a table read from configuration, for the reason the arms are one: the
    names are what a stored campaign refers to, and a baseline renamed in an environment file
    would leave a finished grid naming candidates nothing supplies.
    """

    PER_CHANNEL = CandidateRef("boosted_trees_per_channel")
    SPECTRAL = CandidateRef("boosted_trees_spectral")
    ACROSS_CHANNELS = CandidateRef("boosted_trees_across_channels")
    MINIROCKET = CandidateRef("minirocket")

    @classmethod
    def over(
        cls,
        boosting: GradientBoostingSpec,
        convolutions: RandomConvolutions,
        sources: Sequence[TaskId] = (),
    ) -> tuple[ClassicalArm, ...]:
        """Every baseline, the layout-independent one drawing on ``sources``, in reporting order."""

        def trees(features: FeatureScheme) -> BoostedTrees:
            return BoostedTrees(features=features, boosting=boosting)

        return (
            ClassicalArm(ref=cls.PER_CHANNEL, method=trees(FeatureScheme.PER_CHANNEL), sources=()),
            ClassicalArm(ref=cls.SPECTRAL, method=trees(FeatureScheme.SPECTRAL), sources=()),
            ClassicalArm(
                ref=cls.ACROSS_CHANNELS,
                method=trees(FeatureScheme.CHANNEL_AGGREGATED),
                sources=tuple(sources),
            ),
            ClassicalArm(ref=cls.MINIROCKET, method=convolutions, sources=()),
        )

    @classmethod
    def refs(cls) -> tuple[CandidateRef, ...]:
        """What the baselines are called, in reporting order."""
        return (cls.PER_CHANNEL, cls.SPECTRAL, cls.ACROSS_CHANNELS, cls.MINIROCKET)

    @classmethod
    def catalogue(
        cls,
        boosting: GradientBoostingSpec,
        convolutions: RandomConvolutions,
        sources: Sequence[TaskId] = (),
    ) -> ClassicalBaselineCatalogue:
        """What the baselines are, with nothing that could fit one."""
        return ClassicalBaselineCatalogue(cls.over(boosting, convolutions, sources))
