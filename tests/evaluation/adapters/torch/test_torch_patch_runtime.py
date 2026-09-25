"""What the torch runtime does with a patch model, on a corpus this test publishes."""

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch")

from emblema.evaluation.adapters.blocks.published_corpus_blocks import (  # noqa: E402
    PublishedCorpusBlocks,
)
from emblema.evaluation.adapters.torch.fitted_patch_model import FittedPatchModel  # noqa: E402
from emblema.evaluation.adapters.torch.torch_patch_runtime import (  # noqa: E402
    TorchPatchRuntime,
)
from emblema.evaluation.domain.exceptions import (  # noqa: E402
    DivergedAdaptationError,
    InvalidPatchModelSpecError,
    UnreadableFittedCandidateError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget  # noqa: E402
from emblema.evaluation.domain.labels.label_sample import LabelSample  # noqa: E402
from emblema.evaluation.domain.labels.remaining_life_scheme import (  # noqa: E402
    RemainingLifeScheme,
)
from emblema.evaluation.domain.patching.patch_plan import PatchPlan  # noqa: E402
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome  # noqa: E402
from emblema.evaluation.domain.task.downstream_task import DownstreamTask  # noqa: E402
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore  # noqa: E402
from tests.evaluation.support import (  # noqa: E402
    TASK,
    adaptation_schedule,
    labelled,
    patch_plan,
    task,
)
from tests.support.published import CHANNELS, publish  # noqa: E402

pytestmark = pytest.mark.ml

SAMPLE = LabelSample(
    task=TASK,
    windows=(labelled("a", 0, 10.0, 5.0), labelled("a", 1, 15.0, 9.0), labelled("b", 3, 10.0, 2.0)),
    budget=LabelBudget.of(3),
    seed=7,
)
SCORED = (labelled("c", 2, 10.0, 8.0),)
CEILING = 10.0


class Published(NamedTuple):
    """A runtime over a published corpus, the task defined against it, and the store."""

    runtime: TorchPatchRuntime
    task: DownstreamTask
    store: InMemoryArtifactStore
    blocks: PublishedCorpusBlocks


def published_on(tmp_path: Path, device: str) -> Published:
    store = InMemoryArtifactStore()
    manifest = publish(store, tmp_path / "scratch").manifest
    blocks = PublishedCorpusBlocks(store, tmp_path / "workspace")
    runtime = TorchPatchRuntime(blocks, device=device, store=store)
    defined = replace(task(), manifest=manifest, labels=RemainingLifeScheme(CEILING))
    return Published(runtime, defined, store, blocks)


@pytest.fixture
def published(tmp_path: Path) -> Published:
    return published_on(tmp_path, "cpu")


def train(published: Published, plan: PatchPlan, *, retain: bool = False) -> ScoredOutcome:
    return published.runtime.train(plan, published.task, SAMPLE, SCORED, retain=retain)


def test_the_same_plan_over_the_same_sample_repeats_bit_for_bit(published: Published) -> None:
    assert train(published, patch_plan()).predictions == train(published, patch_plan()).predictions


def test_another_seed_gives_another_run(published: Published) -> None:
    first = train(published, patch_plan(seed=1))
    other = train(published, patch_plan(seed=2))

    assert first.predictions != other.predictions


def test_a_loss_that_stops_being_finite_ends_the_run(published: Published) -> None:
    runaway = replace(patch_plan(), schedule=adaptation_schedule(learning_rate=1e30))

    with pytest.raises(DivergedAdaptationError):
        train(published, runaway)


def test_a_window_shorter_than_one_patch_is_refused_before_anything_is_trained(
    published: Published,
) -> None:
    # The test corpus's windows span ten units of time; a patch of eleven steps covers more.
    with pytest.raises(InvalidPatchModelSpecError, match="shorter than a patch"):
        train(published, patch_plan(patch_length=11, stride=4))


def test_the_model_kept_answers_as_the_one_that_was_scored(published: Published) -> None:
    outcome = train(published, patch_plan(), retain=True)

    assert outcome.artifact is not None
    kept = FittedPatchModel.read(published.store.get(outcome.artifact))
    assert kept.target_scale == CEILING
    assert kept.reading.channels == len(CHANNELS)
    assert kept.parameters == patch_plan().parameters()
    manifest = published.blocks.manifest_of(published.task.manifest)
    windows = published.blocks.block_of(manifest).at([w.window.position for w in SCORED])
    values, observed = kept.reading.tensors(windows)
    with torch.no_grad():
        again = (kept.build()(values, observed).to(torch.float64) * CEILING).tolist()
    assert again == [prediction.predicted for prediction in outcome.predictions]


def test_bytes_that_are_not_a_patch_model_are_refused() -> None:
    with pytest.raises(UnreadableFittedCandidateError):
        FittedPatchModel.read(b"not a model")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple-silicon MPS")
def test_on_the_accelerator_the_answers_agree_with_the_processor(tmp_path: Path) -> None:
    on_cpu = train(published_on(tmp_path / "cpu", "cpu"), patch_plan())
    on_mps = train(published_on(tmp_path / "mps", "mps"), patch_plan())

    assert [p.predicted for p in on_mps.predictions] == pytest.approx(
        [p.predicted for p in on_cpu.predictions], abs=1e-3
    )
