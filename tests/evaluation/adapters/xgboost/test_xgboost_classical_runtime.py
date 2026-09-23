"""What the XGBoost runtime does over real published corpora: repeats, scales and spans them.

The transfer claim of this ticket is tested here rather than in the port's contract, because it
needs two corpora of different channel layouts and the contract is about one task.
"""

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple
from uuid import UUID

import numpy as np
import pytest

from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.xgboost.fitted_baseline import FittedBaseline
from emblema.evaluation.adapters.xgboost.xgboost_classical_runtime import XgboostClassicalRuntime
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.classical.classical_outcome import ClassicalOutcome
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.classical.fitting_source import FittingSource
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from tests.evaluation.adapters.xgboost.support import WIDE_CHANNELS, publish_wide
from tests.evaluation.support import TASK, boosting, labelled, recipe, task
from tests.support.published import CHANNELS, publish

AGGREGATED = FeatureScheme.CHANNEL_AGGREGATED
CEILING = 10.0
WIDE_TASK = TaskId(UUID(int=21))
SAMPLE = LabelSample(
    task=TASK,
    windows=(labelled("a", 0, 10.0, 5.0), labelled("a", 1, 15.0, 9.0), labelled("b", 3, 10.0, 2.0)),
    budget=LabelBudget.of(3),
    seed=7,
)
SCORED = (labelled("c", 2, 10.0, 8.0),)
# Drawing rows and columns is where a fit has anything to draw at all; with everything offered
# to every tree there is no randomness for a seed to steer.
DRAWING = boosting(row_share=0.6, feature_share=0.6, rounds=12)


class Published(NamedTuple):
    """A runtime over the published corpus, the task defined against it, and the store."""

    runtime: XgboostClassicalRuntime
    task: DownstreamTask
    store: InMemoryArtifactStore
    workspace: Path


@pytest.fixture
def published(tmp_path: Path) -> Published:
    store = InMemoryArtifactStore()
    manifest = publish(store, tmp_path / "scratch").manifest
    runtime = XgboostClassicalRuntime(
        PublishedCorpusBlocks(store, tmp_path / "workspace"), store=store
    )
    defined = replace(task(), manifest=manifest, labels=RemainingLifeScheme(CEILING))
    return Published(runtime, defined, store, tmp_path)


def fitted(published: Published, scheme: FeatureScheme, **overrides: object) -> ClassicalOutcome:
    return published.runtime.fit(
        recipe(scheme, **overrides), published.task, SAMPLE, (), SCORED, retain=False
    )


@pytest.mark.parametrize("scheme", list(FeatureScheme))
def test_two_fits_of_one_recipe_answer_the_same_on_this_machine(
    published: Published, scheme: FeatureScheme
) -> None:
    once = fitted(published, scheme, boosting=DRAWING)
    again = fitted(published, scheme, boosting=DRAWING)

    assert [p.predicted for p in once.predictions] == [p.predicted for p in again.predictions]


def test_the_seed_is_what_a_fit_draws_its_rows_and_columns_under(published: Published) -> None:
    under_one = fitted(published, AGGREGATED, boosting=DRAWING, seed=1)
    under_two = fitted(published, AGGREGATED, boosting=DRAWING, seed=2)

    assert under_one.predictions[0].predicted != under_two.predictions[0].predicted


def test_an_answer_comes_back_in_the_unit_the_task_asks_its_question_in(
    published: Published,
) -> None:
    # Every label is a remaining life under a ceiling of ten; an answer scaled back wrongly
    # would land near the fractions the fit works in rather than near the labels.
    outcome = fitted(published, AGGREGATED)

    assert 0.0 <= outcome.predictions[0].predicted <= CEILING


def test_a_fit_spans_two_corpora_whose_channel_layouts_have_nothing_in_common(
    published: Published,
) -> None:
    assert len(WIDE_CHANNELS) != len(CHANNELS)
    wide = publish_wide(published.store, published.workspace / "wide", WIDE_TASK, 100.0)

    outcome = published.runtime.fit(
        recipe(AGGREGATED, sources=(WIDE_TASK,)),
        published.task,
        SAMPLE,
        (FittingSource(task=wide.task, sample=wide.sample),),
        SCORED,
        retain=False,
    )

    assert np.isfinite(outcome.predictions[0].predicted)


def test_a_source_task_is_scaled_by_its_own_scheme_before_its_labels_are_pooled(
    published: Published,
) -> None:
    # The wider corpus counts its remaining life in hundreds and the target in tens. Pooled raw,
    # its labels would drag every answer above the target's ceiling.
    wide = publish_wide(published.store, published.workspace / "wide", WIDE_TASK, 100.0)
    at_ceiling = replace(
        wide.sample, windows=tuple(replace(w, target=100.0) for w in wide.sample.windows)
    )

    outcome = published.runtime.fit(
        recipe(AGGREGATED, sources=(WIDE_TASK,)),
        published.task,
        SAMPLE,
        (FittingSource(task=wide.task, sample=at_ceiling),),
        SCORED,
        retain=False,
    )

    assert outcome.predictions[0].predicted <= CEILING


def test_what_a_fit_keeps_answers_the_same_rows_the_fit_itself_did(published: Published) -> None:
    from emblema.evaluation.adapters.features.channel_aggregated_features import (
        ChannelAggregatedFeatures,
    )

    outcome = published.runtime.fit(
        recipe(AGGREGATED), published.task, SAMPLE, (), SCORED, retain=True
    )

    assert outcome.artifact is not None
    kept = FittedBaseline.read(published.store.get(outcome.artifact))
    assert kept.target_scale == CEILING
    assert kept.feature_names == ChannelAggregatedFeatures().names()
    blocks = PublishedCorpusBlocks(published.store, published.workspace / "read")
    manifest = blocks.manifest_of(published.task.manifest)
    rows = ChannelAggregatedFeatures().of(blocks.block_of(manifest).at([2]))
    answered = kept.booster().inplace_predict(rows) * kept.target_scale
    assert float(answered[0]) == pytest.approx(outcome.predictions[0].predicted, rel=1e-6)
