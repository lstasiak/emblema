from dataclasses import dataclass

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.classical.gradient_boosting_spec import GradientBoostingSpec
from emblema.evaluation.domain.exceptions import InvalidClassicalRecipeError


@dataclass(frozen=True, kw_only=True)
class ClassicalRecipe:
    """How a classical candidate is made for a task: what it reads, how it fits, what else it saw.

    The counterpart of an adaptation plan for a candidate that starts from no weights at all.
    Nothing here names a backbone, a device or a transfer mode, which is the point: a campaign
    made entirely of these runs without the Pretraining context ever being asked for anything.

    The source tasks are the whole of what makes one of these capable of transfer. A candidate
    that names none is fitted on the target task's budget of labels alone; one that names some
    is fitted on those tasks' labels as well and scored on the target, which is the classical
    answer to the question the project's main result asks. Only a scheme whose vector is the
    same width whatever the channel layout may do that, so a recipe that would concatenate a
    channel-bound vector across corpora is refused here rather than crashing on ragged rows.

    Invariants: no source task named twice; sources only under a scheme that spans layouts.

    Attributes:
        features: What a window is turned into before anything is fitted.
        boosting: How many trees are grown and how far each one is trusted.
        seed: Seed of everything the fit draws: which rows and which features each tree sees.
        sources: Other tasks whose labels are fitted alongside the target's; empty for a
            candidate that learns the target alone.
    """

    features: FeatureScheme
    boosting: GradientBoostingSpec
    seed: int
    sources: tuple[TaskId, ...]

    def __post_init__(self) -> None:
        if len(set(self.sources)) != len(self.sources):
            raise InvalidClassicalRecipeError(
                f"a source task is named twice: {sorted(map(str, self.sources))}"
            )
        if self.sources and not self.features.spans_channel_layouts:
            raise InvalidClassicalRecipeError(
                f"{self.features} is as wide as one corpus has channels, so it cannot be fitted "
                f"over the {len(self.sources)} source tasks this recipe names"
            )

    def parameters(self) -> dict[str, str | int | float]:
        """The recipe flattened to scalars, in a fixed order, for whoever reports a fit.

        The sources are rendered as one field rather than one per task, so every recipe renders
        the same columns whatever it transfers from.
        """
        stated: dict[str, str | int | float] = {
            "features": str(self.features),
            "fit_seed": self.seed,
            "sources": " ".join(str(source) for source in self.sources),
        }
        return stated | self.boosting.parameters()
