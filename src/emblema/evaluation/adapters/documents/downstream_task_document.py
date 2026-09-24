from typing import Any

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.task_split import TaskSplit
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm


class DownstreamTaskDocument:
    """Reads a task to a document and back, whole, for a machine that has no registry to ask.

    The units of every side travel sorted, so one task is always one document, and the frozen
    side travels as the units it names and nothing more: the machine that runs a tuning order
    has to know which units it may not read, not what they hold.
    """

    def encode(self, task: DownstreamTask) -> dict[str, Any]:
        return {
            "task_id": str(task.task_id),
            "corpus": task.corpus,
            "manifest": {
                "key": task.manifest.key,
                "algorithm": str(task.manifest.checksum.algorithm),
                "digest": task.manifest.checksum.digest,
            },
            "protocol": str(task.protocol),
            "tuning": self._units(task.split.tuning),
            "validation": self._units(task.split.validation),
            "test": {"units": self._units(task.split.test.units), "source": task.split.test.source},
            "labels": self._labels(task.labels),
            "strata": None if task.strata is None else task.strata.count,
        }

    def decode(self, document: dict[str, Any]) -> DownstreamTask:
        """The task that document holds.

        Raises:
            KeyError: If the document is missing a part of a task.
            ValueError: If what it holds is not a task that stands up.
        """
        manifest, test, labels = document["manifest"], document["test"], document["labels"]
        return DownstreamTask(
            task_id=TaskId.parse(document["task_id"]),
            corpus=document["corpus"],
            manifest=ArtifactRef(
                manifest["key"], Checksum(HashAlgorithm(manifest["algorithm"]), manifest["digest"])
            ),
            protocol=EvaluationProtocol(document["protocol"]),
            split=TaskSplit(
                tuning=frozenset(UnitKey(unit) for unit in document["tuning"]),
                validation=frozenset(UnitKey(unit) for unit in document["validation"]),
                test=FrozenTestSplit(
                    units=frozenset(UnitKey(unit) for unit in test["units"]),
                    source=test["source"],
                ),
            ),
            labels=None
            if labels is None
            else RemainingLifeScheme(labels["ceiling"])
            if labels["scheme"] == "remaining_life"
            else ForecastScheme(labels["channel"], labels["horizon"]),
            strata=None if document["strata"] is None else TargetBins(document["strata"]),
        )

    @staticmethod
    def _units(units: frozenset[UnitKey]) -> list[str]:
        return sorted(str(unit) for unit in units)

    @staticmethod
    def _labels(labels: RemainingLifeScheme | ForecastScheme | None) -> dict[str, Any] | None:
        match labels:
            case RemainingLifeScheme():
                return {"scheme": "remaining_life", "ceiling": labels.ceiling}
            case ForecastScheme():
                return {"scheme": "forecast", "channel": labels.channel, "horizon": labels.horizon}
            case None:
                return None
