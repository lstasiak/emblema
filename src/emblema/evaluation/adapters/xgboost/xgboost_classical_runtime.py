import time
from collections.abc import Sequence

import numpy as np
import xgboost
from numpy.typing import NDArray

from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.features.channel_aggregated_features import (
    ChannelAggregatedFeatures,
)
from emblema.evaluation.adapters.features.per_channel_features import PerChannelFeatures
from emblema.evaluation.adapters.features.window_features import WindowFeatures
from emblema.evaluation.adapters.xgboost.fitted_baseline import FittedBaseline
from emblema.evaluation.domain.classical.classical_outcome import ClassicalOutcome
from emblema.evaluation.domain.classical.classical_recipe import ClassicalRecipe
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.classical.fitting_source import FittingSource
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    InvalidClassicalOutcomeError,
)
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.transfer.window_prediction import WindowPrediction
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
        threads: int,
        store: ArtifactStore | None = None,
    ) -> None:
        """Fit over the corpora ``blocks`` reads.

        Args:
            blocks: Where the published manifests and blocks are read from.
            threads: Threads a fit may use. Part of the result and not a tuning knob: the
                histogram a tree is grown from is summed in an order the split of work decides,
                so two machines agree on an answer only if they agree on this.
            store: Where a candidate this runtime is asked to keep is put; a process that never
                keeps one needs none.

        Raises:
            ValueError: If a fit is given no thread to run on.
        """
        if threads < 1:
            raise ValueError(f"a fit needs at least one thread, got {threads}")
        self._blocks = blocks
        self._threads = threads
        self._store = store

    def fit(
        self,
        recipe: ClassicalRecipe,
        task: DownstreamTask,
        sample: LabelSample,
        sources: Sequence[FittingSource],
        scored: Sequence[LabelledWindow],
        *,
        retain: bool,
    ) -> ClassicalOutcome:
        task.accept_sample(sample)
        if not scored:
            raise InvalidClassicalOutcomeError("there is no window to answer")
        features = self._features(recipe.features, task)
        rows, targets = self._fitted_from(features, task, sample, sources)
        answered = self._rows_of(features, task, [labelled.window for labelled in scored])
        started = time.perf_counter()
        scale = task.label_scheme().scale
        model = self._grown(recipe, rows, targets)
        predicted = model.predict(answered) * scale
        return ClassicalOutcome(
            predictions=tuple(
                WindowPrediction(
                    window=labelled.window, target=labelled.target, predicted=float(answer)
                )
                for labelled, answer in zip(scored, predicted.tolist(), strict=True)
            ),
            seconds=time.perf_counter() - started,
            artifact=self._kept(recipe, model, features, scale) if retain else None,
        )

    def _features(self, scheme: FeatureScheme, task: DownstreamTask) -> WindowFeatures:
        """How a window is read under ``scheme``, sized to the task's corpus where that matters."""
        if scheme is FeatureScheme.CHANNEL_AGGREGATED:
            return ChannelAggregatedFeatures()
        return PerChannelFeatures(len(self._blocks.manifest_of(task.manifest).channels))

    def _fitted_from(
        self,
        features: WindowFeatures,
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
                self._rows_of(features, fitted, [labelled.window for labelled in labels.windows])
            )
            targets.append(
                np.array([labelled.target / scale for labelled in labels.windows], dtype=np.float64)
            )
        return np.vstack(rows), np.concatenate(targets)

    def _rows_of(
        self, features: WindowFeatures, task: DownstreamTask, windows: Sequence[TaskWindow]
    ) -> NDArray[np.float64]:
        """The windows of one task, read out of its block and summarised."""
        published = self._blocks.manifest_of(task.manifest)
        block = self._blocks.block_of(published)
        return features.of(block.at([window.position for window in windows]))

    def _grown(
        self,
        recipe: ClassicalRecipe,
        rows: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> xgboost.XGBRegressor:
        boosting = recipe.boosting
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
            random_state=recipe.seed,
            n_jobs=self._threads,
        )
        model.fit(rows, targets)
        return model

    def _kept(
        self,
        recipe: ClassicalRecipe,
        model: xgboost.XGBRegressor,
        features: WindowFeatures,
        target_scale: float,
    ) -> ArtifactRef:
        """The candidate this fit produced, stored whole, so the campaign can name it.

        Raises:
            CandidateNotRetainableError: If the runtime was given nowhere to keep it.
        """
        if self._store is None:
            raise CandidateNotRetainableError(
                "this runtime was asked to keep what it fitted and was given no store"
            )
        fitted = FittedBaseline.of(
            recipe, model, feature_names=features.names(), target_scale=target_scale
        )
        return self._store.put(fitted.to_bytes())
