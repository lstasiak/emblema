"""One task, one sample, every adapter: what any runtime training a patch model owes the port.

A runtime may learn anything or nothing, so the contract says nothing about how good the answers
are. It holds every adapter to answering exactly the windows it was handed, in the order given,
to refusing a sample of another task's labels or a run with nothing to score, and to keeping what
it trained only where it has somewhere to keep it. The torch adapter runs over the published
test corpus, whose block holds the four positions the sample and the scored side address.
"""

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple
from uuid import UUID

import pytest

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.in_memory.patch_runtime import InMemoryPatchRuntime
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    ForeignLabelSampleError,
    InvalidScoredOutcomeError,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.ports.patch_runtime import PatchRuntime
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from tests.evaluation.support import TASK, labelled, patch_plan, task

SAMPLE = LabelSample(
    task=TASK,
    windows=(labelled("a", 0, 10.0, 5.0), labelled("b", 3, 10.0, 9.0)),
    budget=LabelBudget.of(2),
    seed=7,
)
SCORED = (labelled("c", 2, 10.0, 8.0), labelled("c", 1, 15.0, 3.0))


class Trained(NamedTuple):
    """An adapter, the task it answers for, and the store it keeps what it trains in."""

    runtime: PatchRuntime
    task: DownstreamTask
    store: InMemoryArtifactStore


def runtime(kind: str, tmp_path: Path, *, keeping: bool) -> Trained:
    store = InMemoryArtifactStore()
    kept = store if keeping else None
    if kind == "in memory":
        return Trained(InMemoryPatchRuntime(kept), task(), store)
    pytest.importorskip("torch")
    from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
    from emblema.evaluation.adapters.torch.torch_patch_runtime import TorchPatchRuntime
    from tests.support.published import publish

    manifest = publish(store, tmp_path / "scratch").manifest
    trainer = TorchPatchRuntime(
        PublishedCorpusBlocks(store, tmp_path / "workspace"), device="cpu", store=kept
    )
    return Trained(trainer, replace(task(), manifest=manifest), store)


KINDS = ["in memory", "torch"]


@pytest.fixture(params=KINDS)
def trained(request: pytest.FixtureRequest, tmp_path: Path) -> Trained:
    return runtime(request.param, tmp_path, keeping=False)


def test_every_scored_window_is_answered_in_the_order_given(trained: Trained) -> None:
    outcome = trained.runtime.train(patch_plan(), trained.task, SAMPLE, SCORED, retain=False)

    assert [p.window for p in outcome.predictions] == [w.window for w in SCORED]
    assert [p.target for p in outcome.predictions] == [w.target for w in SCORED]
    assert outcome.artifact is None


def test_a_sample_of_another_task_is_refused(trained: Trained) -> None:
    foreign = LabelSample(
        task=TaskId(UUID(int=9)), windows=SAMPLE.windows, budget=SAMPLE.budget, seed=SAMPLE.seed
    )

    with pytest.raises(ForeignLabelSampleError):
        trained.runtime.train(patch_plan(), trained.task, foreign, SCORED, retain=False)


def test_a_run_with_nothing_to_score_is_refused(trained: Trained) -> None:
    with pytest.raises(InvalidScoredOutcomeError):
        trained.runtime.train(patch_plan(), trained.task, SAMPLE, (), retain=False)


def test_a_runtime_with_nowhere_to_keep_a_model_refuses_to_keep_one(trained: Trained) -> None:
    with pytest.raises(CandidateNotRetainableError):
        trained.runtime.train(patch_plan(), trained.task, SAMPLE, SCORED, retain=True)


@pytest.mark.parametrize("kind", KINDS)
def test_a_model_kept_is_named_by_a_manifest_of_its_form(kind: str, tmp_path: Path) -> None:
    keeping = runtime(kind, tmp_path, keeping=True)

    outcome = keeping.runtime.train(patch_plan(), keeping.task, SAMPLE, SCORED, retain=True)

    assert outcome.artifact is not None
    kept = KeptCandidates(keeping.store).read(outcome.artifact)
    assert kept.kind is CandidateKind.NEURAL
    assert kept.corpus_manifest == keeping.task.manifest
    # Reading back verifies the stored bytes against the checksum the reference carries.
    assert keeping.store.get(kept.measured.artifact)
