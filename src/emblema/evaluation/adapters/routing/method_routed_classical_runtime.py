from collections.abc import Sequence

from emblema.evaluation.domain.classical.boosted_trees import BoostedTrees
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.fitting_source import FittingSource
from emblema.evaluation.domain.classical.random_convolutions import RandomConvolutions
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.ports.classical_runtime import ClassicalRuntime


class MethodRoutedClassicalRuntime:
    """Every classical method behind one runtime, each recipe handed to whoever fits its kind.

    Routed by the method's type and never by trying one runtime and catching its refusal, for
    the reason candidates are routed by name: a runtime that refuses a recipe it does own would
    otherwise be indistinguishable from one asked the wrong question. The match is exhaustive,
    so a method added to the union without a runtime here fails the type check rather than a
    campaign.
    """

    def __init__(self, *, trees: ClassicalRuntime, convolutions: ClassicalRuntime) -> None:
        self._trees = trees
        self._convolutions = convolutions

    def fit(
        self,
        recipe: ClassicalRecipe,
        task: DownstreamTask,
        sample: LabelSample,
        sources: Sequence[FittingSource],
        scored: Sequence[LabelledWindow],
        *,
        retain: bool,
    ) -> ScoredOutcome:
        return self._runtime_of(recipe).fit(recipe, task, sample, sources, scored, retain=retain)

    def _runtime_of(self, recipe: ClassicalRecipe) -> ClassicalRuntime:
        match recipe.method:
            case BoostedTrees():
                return self._trees
            case RandomConvolutions():
                return self._convolutions
