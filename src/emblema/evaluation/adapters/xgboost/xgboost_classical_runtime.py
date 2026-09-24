import time
from collections.abc import Mapping, Sequence

import numpy as np
import xgboost
from numpy.typing import NDArray

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.blocks.read_corpus import ReadCorpus
from emblema.evaluation.adapters.features.channel_aggregated_features import (
    ChannelAggregatedFeatures,
)
from emblema.evaluation.adapters.features.per_channel_features import PerChannelFeatures
from emblema.evaluation.adapters.features.spectral_features import SpectralFeatures
from emblema.evaluation.adapters.features.window_features import WindowFeatures
from emblema.evaluation.adapters.xgboost.fitted_baseline import FittedBaseline
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.classical.boosted_trees import BoostedTrees
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.classical.fitting_source import FittingSource
from emblema.evaluation.domain.classical.gradient_boosting_spec import GradientBoostingSpec
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


class XgboostClassicalRuntime:
    """Fits gradient-boosted trees over a window's summarised statistics, in this process.

    Nothing here is learnt from anything but labels: there are no pretrained weights to read, no
    accelerator to reach and no checkpoint to resume, which is why this is the runtime a worker
    without a backbone can be built on. The seconds an outcome reports start once the blocks are
    at hand, so the fit that happens to fetch a corpus is not timed against the rest.

    Every task a fit draws on is scaled by its own scheme before the rows are stacked. Remaining
    life in one corpus is counted in cycles and in another in hours, and trees fitted on the two
    raw would learn the units rather than the shape; the answers are multiplied back by the
    target's scale, so what comes out is in the unit the task asks its question in.
    """

    def __init__(
        self,
        blocks: PublishedCorpusBlocks,
        *,
        store: ArtifactStore | None = None,
    ) -> None:
        """Fit over the corpora ``blocks`` reads, keeping what it is asked to in ``store``.

        No thread count here: the recipe carries it, since it is part of what a fit answers.
        A process that never keeps a candidate needs no store.
        """
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
        """Grow the trees ``recipe`` names and answer ``scored`` with them.

        Raises:
            UnsupportedClassicalMethodError: If the recipe names a method other than boosted
                trees.
        """
        task.accept_sample(sample)
        if not scored:
            raise InvalidScoredOutcomeError("there is no window to answer")
        trees = self._trees_of(recipe)
        read = ReadCorpus.every(self._blocks, (task, *(source.task for source in sources)))
        features = self._features(trees.features, read[task.manifest])
        started = time.perf_counter()
        rows, targets = self._fitted_from(features, read, task, sample, sources)
        answered = features.of(read[task.manifest].windows([w.window for w in scored]))
        scale = task.label_scheme().scale
        model = self._grown(trees.boosting, recipe.seed, rows, targets)
        predicted = model.predict(answered) * scale
        return ScoredOutcome(
            predictions=tuple(
                WindowPrediction(
                    window=labelled.window, target=labelled.target, predicted=float(answer)
                )
                for labelled, answer in zip(scored, predicted.tolist(), strict=True)
            ),
            seconds=time.perf_counter() - started,
            artifact=self._kept(recipe, model, features, task, scale) if retain else None,
        )

    @staticmethod
    def _trees_of(recipe: ClassicalRecipe) -> BoostedTrees:
        """The trees a recipe names, or a refusal if it names a method grown some other way.

        Raises:
            UnsupportedClassicalMethodError: If the recipe is not one of boosted trees.
        """
        match recipe.method:
            case BoostedTrees():
                return recipe.method
            case RandomConvolutions():
                raise UnsupportedClassicalMethodError(
                    f"this runtime grows boosted trees and was handed {recipe.method}"
                )

    @staticmethod
    def _features(scheme: FeatureScheme, corpus: ReadCorpus) -> WindowFeatures:
        """How a window is read under ``scheme``, sized to the corpus where that matters."""
        match scheme:
            case FeatureScheme.PER_CHANNEL:
                return PerChannelFeatures(corpus.channels)
            case FeatureScheme.SPECTRAL:
                return SpectralFeatures(corpus.channels)
            case FeatureScheme.CHANNEL_AGGREGATED:
                return ChannelAggregatedFeatures()

    def _fitted_from(
        self,
        features: WindowFeatures,
        read: Mapping[ArtifactRef, ReadCorpus],
        task: DownstreamTask,
        sample: LabelSample,
        sources: Sequence[FittingSource],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Every labelled window the fit may see, as rows and targets on one scale."""
        drawn = [(task, sample), *((source.task, source.sample) for source in sources)]
        rows, targets = [], []
        for fitted, labels in drawn:
            scale = fitted.label_scheme().scale
            rows.append(
                features.of(read[fitted.manifest].windows([w.window for w in labels.windows]))
            )
            targets.append(np.array([w.target / scale for w in labels.windows], dtype=np.float64))
        return np.vstack(rows), np.concatenate(targets)

    @staticmethod
    def _grown(
        boosting: GradientBoostingSpec,
        seed: int,
        rows: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> xgboost.XGBRegressor:
        model = xgboost.XGBRegressor(
            n_estimators=boosting.rounds,
            max_depth=boosting.max_depth,
            learning_rate=boosting.learning_rate,
            subsample=boosting.row_share,
            colsample_bytree=boosting.feature_share,
            min_child_weight=boosting.min_leaf_weight,
            reg_lambda=boosting.l2_penalty,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=boosting.threads,
        )
        model.fit(rows, targets)
        return model

    def _kept(
        self,
        recipe: ClassicalRecipe,
        model: xgboost.XGBRegressor,
        features: WindowFeatures,
        task: DownstreamTask,
        target_scale: float,
    ) -> ArtifactRef:
        """The candidate this fit produced, stored whole under its manifest, for the campaign.

        The trees are the measured form and the only one: the document another context loads
        them from is already free of the fitting stack.

        Raises:
            CandidateNotRetainableError: If the runtime was given nowhere to keep it.
        """
        if self._kept_candidates is None:
            raise CandidateNotRetainableError(
                "this runtime was asked to keep what it fitted and was given no store"
            )
        fitted = FittedBaseline.of(
            recipe, model, feature_names=features.names(), target_scale=target_scale
        )
        return self._kept_candidates.keep(
            CandidateKind.CLASSICAL,
            corpus_manifest=task.manifest,
            measured=RepresentationBytes(FittedBaseline.FORMAT, fitted.to_bytes()),
        )
