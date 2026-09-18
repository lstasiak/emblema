"""One task and one plan, stated once, for every test that needs one rather than a rule.

The overrides are typed ``Any`` because each names a field of the value object it builds and
carries that field's type; the value object refuses anything else on the way in.
"""

from dataclasses import replace
from typing import Any
from uuid import UUID

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.task_split import TaskSplit
from emblema.evaluation.domain.transfer.adaptation_plan import AdaptationPlan
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode
from emblema.evaluation.domain.transfer.window_prediction import WindowPrediction
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum

CORPUS = "turbofans"
MANIFEST = ArtifactRef(key="durable/manifest", checksum=Checksum.of_bytes(b"manifest"))
TASK = TaskId(UUID(int=1))
SCHEME = RemainingLifeScheme(125.0)
STRATA = TargetBins(4)
TEST_SIDE = FrozenTestSplit(units=frozenset({UnitKey("held/1")}), source="turbofans/test")
WEIGHTS = ArtifactRef(key="durable/weights", checksum=Checksum.of_bytes(b"weights"))
# The three kinds of linear layer the encoder's blocks have; the attention's output projection
# is named with its parent, because the time encoding has a projection of its own.
LORA = LoraSpec(
    rank=2, alpha=4.0, dropout=0.0, targets=("qkv", "attention.projection", "feedforward")
)


def units(*names: str) -> frozenset[UnitKey]:
    return frozenset(UnitKey(name) for name in names)


def sides(training: frozenset[UnitKey], validation: frozenset[UnitKey]) -> CorpusSides:
    return CorpusSides(corpus=CORPUS, training=training, validation=validation)


def window(unit: str, position: int, ends_at: float) -> TaskWindow:
    return TaskWindow(unit=UnitKey(unit), position=position, ends_at=ends_at)


def task(
    tuning: frozenset[UnitKey] = units("a", "b"),
    validation: frozenset[UnitKey] = units("c"),
    test: FrozenTestSplit = TEST_SIDE,
) -> DownstreamTask:
    return DownstreamTask(
        task_id=TASK,
        corpus=CORPUS,
        manifest=MANIFEST,
        split=TaskSplit(tuning=tuning, validation=validation, test=test),
        labels=SCHEME,
        strata=STRATA,
    )


def labelled(unit: str, position: int, ends_at: float, target: float) -> LabelledWindow:
    return LabelledWindow(window=window(unit, position, ends_at), target=target)


def prediction(unit: str, position: int, target: float, predicted: float) -> WindowPrediction:
    return WindowPrediction(
        window=window(unit, position, float(position)), target=target, predicted=predicted
    )


def adaptation_schedule(**overrides: Any) -> AdaptationSchedule:
    stated = AdaptationSchedule(epochs=2, batch_size=2, learning_rate=1e-2, weight_decay=0.0)
    return replace(stated, **overrides)


def plan(mode: TransferMode = TransferMode.FULL_FINE_TUNING, **overrides: Any) -> AdaptationPlan:
    """A plan of ``mode`` that holds together; anything named is replaced afterwards."""
    stated = AdaptationPlan(
        mode=mode,
        backbone=WEIGHTS if mode.starts_from_pretrained_weights else None,
        schedule=adaptation_schedule(),
        lora=LORA if mode.adds_low_rank_updates else None,
        seed=1,
    )
    return replace(stated, **overrides)
