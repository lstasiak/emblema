import time
from collections.abc import Sequence

import numpy as np

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.blocks.read_corpus import ReadCorpus
from emblema.evaluation.adapters.minirocket.fitted_convolutions import FittedConvolutions
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.classical.boosted_trees import BoostedTrees
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.fitting_source import FittingSource
from emblema.evaluation.domain.classical.minirocket_spec import MiniRocketSpec
from emblema.evaluation.domain.classical.random_convolutions import RandomConvolutions
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    InvalidScoredOutcomeError,
    UnsupportedClassicalMethodError,
)
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore


class MiniRocketClassicalRuntime:
    """Fits MiniRocket and a ridge map on a window laid on a grid, in this process.

    NumPy and a linear solve, nothing else: no accelerator, no weights to read, which is why it
    runs on the worker without the training stack beside the boosted trees. The seconds an
    outcome reports start once the block is at hand, as they do for every runtime here.

    A recipe that names source tasks never reaches this far — the recipe refuses them for a
    method whose input has a channel axis — so only the target's labels are fitted.
    """

    def __init__(
        self,
        blocks: PublishedCorpusBlocks,
        *,
        store: ArtifactStore | None = None,
    ) -> None:
        """Fit over the corpora ``blocks`` reads, keeping what it is asked to in ``store``."""
        self._blocks = blocks
        self._kept_candidates = None if store is None else KeptCandidates(store)

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
        """Fit the convolutions ``recipe`` names and answer ``scored`` with them.

        Raises:
            UnsupportedClassicalMethodError: If the recipe names a method other than random
                convolutions, or a window of the corpus is too short to convolve.
        """
        task.accept_sample(sample)
        if not scored:
            raise InvalidScoredOutcomeError("there is no window to answer")
        method = self._convolutions_of(recipe)
        corpus = ReadCorpus.every(self._blocks, (task,))[task.manifest]
        steps = self._steps_of(method, corpus)
        started = time.perf_counter()
        scale = task.label_scheme().scale
        fitted = FittedConvolutions.fitted(
            recipe,
            method,
            corpus.channels,
            steps,
            list(corpus.windows([labelled.window for labelled in sample.windows])),
            np.array([labelled.target / scale for labelled in sample.windows]),
            scale,
        )
        predicted = fitted.predict(
            list(corpus.windows([labelled.window for labelled in scored])),
            threads=method.ridge.threads,
        )
        return ScoredOutcome(
            predictions=tuple(
                WindowPrediction(
                    window=labelled.window, target=labelled.target, predicted=float(answer)
                )
                for labelled, answer in zip(scored, predicted.tolist(), strict=True)
            ),
            seconds=time.perf_counter() - started,
            artifact=self._kept(fitted, task) if retain else None,
        )

    @staticmethod
    def _convolutions_of(recipe: ClassicalRecipe) -> RandomConvolutions:
        """The convolutions a recipe names, or a refusal if it names a method fitted otherwise.

        Raises:
            UnsupportedClassicalMethodError: If the recipe is not one of random convolutions.
        """
        match recipe.method:
            case RandomConvolutions():
                return recipe.method
            case BoostedTrees():
                raise UnsupportedClassicalMethodError(
                    f"this runtime fits random convolutions and was handed {recipe.method}"
                )

    @staticmethod
    def _steps_of(method: RandomConvolutions, corpus: ReadCorpus) -> int:
        """The grid's steps over a window: its span in the corpus's time, at the resolution.

        Raises:
            UnsupportedClassicalMethodError: If the grid would be shorter than a kernel.
        """
        steps = method.convolutions.steps_over(corpus.manifest.window_length)
        if steps < MiniRocketSpec.KERNEL_LENGTH:
            raise UnsupportedClassicalMethodError(
                f"a window of {corpus.manifest.corpus} is laid on {steps} steps and a kernel "
                f"spans {MiniRocketSpec.KERNEL_LENGTH}"
            )
        return steps

    def _kept(self, fitted: FittedConvolutions, task: DownstreamTask) -> ArtifactRef:
        """The candidate this fit produced, stored whole under its manifest, for the campaign.

        The convolutions are the measured form and the only one.

        Raises:
            CandidateNotRetainableError: If the runtime was given nowhere to keep it.
        """
        if self._kept_candidates is None:
            raise CandidateNotRetainableError(
                "this runtime was asked to keep what it fitted and was given no store"
            )
        return self._kept_candidates.keep(
            CandidateKind.CLASSICAL,
            corpus_manifest=task.manifest,
            measured=RepresentationBytes(FittedConvolutions.FORMAT, fitted.to_bytes()),
        )
