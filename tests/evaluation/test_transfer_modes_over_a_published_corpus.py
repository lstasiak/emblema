"""Every transfer mode learns the task over a corpus this test really publishes.

The path is the one a report runs: the turbofan sample published through the command line's
use cases, a task defined against the manifest, the labels drawn and the validation side
labelled by the use cases, the windows read out of the block by position, a small encoder in
place of the pretrained backbone. What is held is that each mode trains — its loss over the
labels falls across the epochs — and answers every validation window in cycles; how well is a
question for the curve, on the real backbone, not for a sample of two engines.
"""

from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch")

from emblema.catalog.adapters.in_memory.corpus_repository import (  # noqa: E402
    InMemoryCorpusRepository,
)
from emblema.catalog.contracts.published_corpus_manifest_json import (  # noqa: E402
    PublishedCorpusManifestJson,
)
from emblema.entrypoints.cli.publish_corpus.composition_root import CompositionRoot  # noqa: E402
from emblema.evaluation.adapters.blocks.block_corpus_windows import BlockCorpusWindows  # noqa: E402
from emblema.evaluation.adapters.blocks.published_corpus_blocks import (  # noqa: E402
    PublishedCorpusBlocks,
)
from emblema.evaluation.adapters.in_memory.downstream_task_repository import (  # noqa: E402
    InMemoryDownstreamTaskRepository,
)
from emblema.evaluation.adapters.readers.cmapss_unit_lifetimes import (  # noqa: E402
    CmapssUnitLifetimes,
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
from emblema.evaluation.domain.identifiers import UnitKey  # noqa: E402
from emblema.evaluation.domain.labels.label_budget import LabelBudget  # noqa: E402
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme  # noqa: E402
from emblema.evaluation.domain.labels.target_bins import TargetBins  # noqa: E402
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit  # noqa: E402
from emblema.evaluation.domain.transfer.adaptation_outcome import AdaptationOutcome  # noqa: E402
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan  # noqa: E402
from emblema.evaluation.domain.transfer.adaptation_schedule import (  # noqa: E402
    AdaptationSchedule,
)
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec  # noqa: E402
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode  # noqa: E402
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator  # noqa: E402
from emblema.shared.adapters.storage.local_directory import (  # noqa: E402
    LocalDirectoryArtifactStore,
)
from emblema.shared.kernel.artifacts import ArtifactRef  # noqa: E402
from emblema.shared.kernel.checksums import Checksum  # noqa: E402
from tests.support.backbones import SmallBackbones  # noqa: E402
from tests.support.corpora import CORPUS, SAMPLE, SUBSET, publish_command  # noqa: E402
from tests.support.settings import unreachable_store  # noqa: E402

pytestmark = pytest.mark.ml

ENGINES = (UnitKey("FD001/39"), UnitKey("FD001/91"))
HELD = FrozenTestSplit(units=frozenset({UnitKey("FD001/test/1")}), source="cmapss/test/FD001")
CEILING = 40.0
WEIGHTS = ArtifactRef(key="durable/small", checksum=Checksum.of_bytes(b"small"))
LORA = LoraSpec(
    rank=2, alpha=4.0, dropout=0.0, targets=("qkv", "attention.projection", "feedforward")
)
EPOCHS = 12


class Runs(NamedTuple):
    """The use case over the published sample, and the task it runs for."""

    run: RunAdaptation
    task: TaskId
    validation_windows: int


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Runs:
    root = tmp_path_factory.mktemp("published")
    store = LocalDirectoryArtifactStore(root / "store")
    publishing = CompositionRoot(
        unreachable_store(),
        corpus=CORPUS,
        corpus_root=SAMPLE,
        workspace=root / "blocks",
        subsets=(SUBSET,),
        corpora=InMemoryCorpusRepository(),
        store=store,
    )
    manifest = publishing.services.publish_corpus(publish_command())
    published = PublishedCorpusManifestJson().decode(store.get(manifest))

    # One reader of blocks for both adapters, as the report wires it.
    blocks = PublishedCorpusBlocks(store, root / "workspace")
    corpus = BlockCorpusWindows(blocks)
    tasks = InMemoryDownstreamTaskRepository()
    lifetimes = CmapssUnitLifetimes(SAMPLE)
    task = DefineDownstreamTask(tasks, corpus, SequentialIdGenerator())(
        DefineDownstreamTaskCommand(
            manifest=manifest,
            units=frozenset(ENGINES),
            test=HELD,
            labels=RemainingLifeScheme(CEILING),
            strata=TargetBins(2),
        )
    )
    runtime = TorchAdaptationRuntime(
        SmallBackbones(vocabulary_size=len(published.channels)), blocks, device="cpu"
    )
    validation = corpus.windows_of(manifest, tasks.get(task).validation_units)
    return Runs(
        RunAdaptation(tasks, corpus, lifetimes, DrawLabelBudget(tasks, corpus, lifetimes), runtime),
        task,
        len(validation),
    )


def plan_of(mode: TransferMode) -> AdaptationPlan:
    return AdaptationPlan(
        mode=mode,
        backbone=WEIGHTS if mode.starts_from_pretrained_weights else None,
        schedule=AdaptationSchedule(
            epochs=EPOCHS,
            batch_size=4,
            learning_rate=1e-2,
            weight_decay=0.0,
            warmup_fraction=0.0,
            final_lr_fraction=1.0,
        ),
        lora=LORA if mode.adds_low_rank_updates else None,
        seed=1,
    )


def adapted(runs: Runs, mode: TransferMode) -> AdaptationOutcome:
    return runs.run(
        RunAdaptationCommand(
            task=runs.task, plan=plan_of(mode), budget=LabelBudget.everything(), sample_seed=1
        )
    )


@pytest.mark.parametrize("mode", list(TransferMode))
def test_each_mode_trains_over_the_labels_of_the_tuning_engine(
    runs: Runs, mode: TransferMode
) -> None:
    outcome = adapted(runs, mode)

    assert len(outcome.training_losses) == EPOCHS
    assert outcome.training_losses[-1] < outcome.training_losses[0]


@pytest.mark.parametrize("mode", list(TransferMode))
def test_each_mode_answers_every_window_of_the_validation_engine_in_cycles(
    runs: Runs, mode: TransferMode
) -> None:
    outcome = adapted(runs, mode)

    assert len(outcome.predictions) == runs.validation_windows > 0
    assert len({str(p.window.unit) for p in outcome.predictions}) == 1
    assert all(0.0 <= p.target <= CEILING for p in outcome.predictions)
    # Answers are in cycles like the targets: an untrained head answers near zero and a trained
    # one near the labels, so anything far outside the label range is a scale left unapplied.
    assert all(-CEILING <= p.predicted <= 3 * CEILING for p in outcome.predictions)
    assert outcome.rmse < 3 * CEILING


def test_the_modes_differ_in_how_many_weights_they_free(runs: Runs) -> None:
    trainable = {mode: adapted(runs, mode).trainable_parameters for mode in TransferMode}

    assert trainable[TransferMode.FROZEN_PROBE] < trainable[TransferMode.LORA]
    assert trainable[TransferMode.LORA] < trainable[TransferMode.FULL_FINE_TUNING]
    assert trainable[TransferMode.FULL_FINE_TUNING] == trainable[TransferMode.FROM_SCRATCH]
