"""The synthetic control's transfer leg travels the whole road on a corpus cut to a test's size.

The two control layouts are published the way the command line publishes them, the forecasting
task is defined over the second, its labels are read off the generator, and a candidate learns
from a budget of them and answers every validation window. What this settles before any real run
is that the road exists: nothing here is a reading of transfer.
"""

from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch")

from emblema.catalog.adapters.in_memory.corpus_repository import (  # noqa: E402
    InMemoryCorpusRepository,
)
from emblema.evaluation.adapters.blocks.block_corpus_windows import (  # noqa: E402
    BlockCorpusWindows,
)
from emblema.evaluation.adapters.blocks.published_corpus_blocks import (  # noqa: E402
    PublishedCorpusBlocks,
)
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (  # noqa: E402
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.synthetic.synthetic_ground_truth import (  # noqa: E402
    SyntheticGroundTruth,
)
from emblema.evaluation.adapters.torch.torch_adaptation_runtime import (  # noqa: E402
    TorchAdaptationRuntime,
)
from emblema.evaluation.application.use_cases.define_downstream_task import (  # noqa: E402
    DefineDownstreamTask,
    DefineDownstreamTaskCommand,
)
from emblema.evaluation.application.use_cases.draw_label_budget import DrawLabelBudget  # noqa: E402
from emblema.evaluation.application.use_cases.run_adaptation import (  # noqa: E402
    RunAdaptation,
    RunAdaptationCommand,
)
from emblema.evaluation.contracts.identifiers import TaskId  # noqa: E402
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme  # noqa: E402
from emblema.evaluation.domain.labels.label_budget import LabelBudget  # noqa: E402
from emblema.evaluation.domain.labels.target_bins import TargetBins  # noqa: E402
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit  # noqa: E402
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan  # noqa: E402
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode  # noqa: E402
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore  # noqa: E402
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator  # noqa: E402
from emblema.shared.adapters.synthetic.layouts import (  # noqa: E402
    CONTROL_A,
    CONTROL_B,
    CONTROL_PROCESS,
)
from emblema.shared.adapters.synthetic.sensor_signal import SensorSignal  # noqa: E402
from emblema.shared.kernel.artifacts import ArtifactRef  # noqa: E402
from tests.evaluation.support import LORA, WEIGHTS, adaptation_schedule  # noqa: E402
from tests.support.backbones import SmallBackbones  # noqa: E402
from tests.support.control_corpus import command, process  # noqa: E402
from tests.support.synthetic import miniature  # noqa: E402

pytestmark = pytest.mark.ml

SCHEME = ForecastScheme("s01", 12.0)
# Eight units of each layout: a quarter held out leaves one to freeze and one to validate on.
UNITS = 8


class Leg(NamedTuple):
    """The use case over the published second layout, the task, and what it answers."""

    run: RunAdaptation
    task_id: TaskId
    validation_windows: int
    tuning_windows: int


@pytest.fixture(scope="module")
def leg(tmp_path_factory: pytest.TempPathFactory) -> Leg:
    workspace = tmp_path_factory.mktemp("control")
    store = InMemoryArtifactStore()
    corpora = InMemoryCorpusRepository()
    refs: list[ArtifactRef] = []
    for layout in (CONTROL_A, CONTROL_B):
        root = process(layout, workspace, store, corpora, units=UNITS)
        refs.append(root.services.publish_corpus(command(layout, refs[0] if refs else None)))
    manifest = refs[1]

    blocks = PublishedCorpusBlocks(store, workspace / "blocks")
    corpus = BlockCorpusWindows(blocks)
    tasks = InMemoryDownstreamTaskRepository()
    sides = corpus.describe(manifest)
    held_out = sorted(sides.validation, key=str)
    test = FrozenTestSplit(units=frozenset(held_out[:1]), source="control-b/held-out")
    task_id = DefineDownstreamTask(tasks, corpus, SequentialIdGenerator())(
        DefineDownstreamTaskCommand(
            manifest=manifest,
            units=(sides.training | sides.validation) - test.units,
            test=test,
            labels=SCHEME,
            strata=TargetBins(2),
        )
    )
    truth = SyntheticGroundTruth(
        SensorSignal(CONTROL_PROCESS, miniature(CONTROL_B, units=UNITS)), SCHEME
    )
    # The backbone was pretrained on the first layout alone, so its table covers that layout's
    # channels and grows for the second's, as the real backbone's does.
    pretrained_on = len(blocks.manifest_of(refs[0]).channels)
    runtime = TorchAdaptationRuntime(
        SmallBackbones(vocabulary_size=pretrained_on), blocks, device="cpu"
    )
    task = tasks.get(task_id)
    return Leg(
        RunAdaptation(tasks, corpus, truth, DrawLabelBudget(tasks, corpus, truth), runtime),
        task_id,
        len(corpus.windows_of(manifest, task.validation_units)),
        len(corpus.windows_of(manifest, task.tuning_units)),
    )


def plan_of(mode: TransferMode) -> AdaptationPlan:
    return AdaptationPlan(
        mode=mode,
        backbone=WEIGHTS if mode.starts_from_pretrained_weights else None,
        schedule=adaptation_schedule(epochs=2, batch_size=4, learning_rate=1e-2),
        lora=LORA if mode.adds_low_rank_updates else None,
        seed=1,
    )


@pytest.mark.parametrize("mode", list(TransferMode))
def test_every_mode_learns_the_forecast_from_generated_labels_and_answers_every_window(
    leg: Leg, mode: TransferMode
) -> None:
    outcome = leg.run(
        RunAdaptationCommand(
            task=leg.task_id, plan=plan_of(mode), budget=LabelBudget.everything(), sample_seed=1
        )
    )

    assert len(outcome.predictions) == leg.validation_windows
    assert outcome.labelled_windows == leg.tuning_windows
    assert all(torch.isfinite(torch.tensor(p.predicted)) for p in outcome.predictions)
    targets = {p.target for p in outcome.predictions}
    assert len(targets) > 1, "the exact reading moves from window to window"
    assert all(abs(t) < 5.0 for t in targets), "in the sensor's own units, of unit variance"


def test_the_labels_are_the_generators_readings_and_not_something_the_block_holds(leg: Leg) -> None:
    outcome = leg.run(
        RunAdaptationCommand(
            task=leg.task_id,
            plan=plan_of(TransferMode.FROZEN_PROBE),
            budget=LabelBudget.of(4),
            sample_seed=2,
        )
    )
    truth = SyntheticGroundTruth(
        SensorSignal(CONTROL_PROCESS, miniature(CONTROL_B, units=UNITS)), SCHEME
    )

    windows = [p.window for p in outcome.predictions]
    assert [p.target for p in outcome.predictions] == [truth.truths_of(windows)[w] for w in windows]
    assert all(str(w.unit).startswith("control-b/") for w in windows)
