from dataclasses import replace
from pathlib import Path

import pytest

from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.minirocket.minirocket_classical_runtime import (
    MiniRocketClassicalRuntime,
)
from emblema.evaluation.domain.exceptions import UnsupportedClassicalMethodError
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from tests.evaluation.support import TASK, labelled, recipe, task
from tests.support.published import publish


def test_a_recipe_of_trees_is_refused_rather_than_fitted_some_other_way(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    manifest = publish(store, tmp_path / "scratch").manifest
    runtime = MiniRocketClassicalRuntime(PublishedCorpusBlocks(store, tmp_path / "workspace"))
    sample = LabelSample(
        task=TASK, windows=(labelled("a", 0, 10.0, 5.0),), budget=LabelBudget.of(1), seed=1
    )

    with pytest.raises(UnsupportedClassicalMethodError, match="boosted_trees"):
        runtime.fit(
            recipe(),
            replace(task(), manifest=manifest),
            sample,
            (),
            (labelled("c", 2, 10.0, 8.0),),
            retain=False,
        )
