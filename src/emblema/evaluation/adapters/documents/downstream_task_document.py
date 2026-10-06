from typing import Any

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.class_strata import ClassStrata
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.label_scheme import LabelScheme
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.labels.stratification import Stratification
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.evaluation.domain.task.evaluation_protocol import EvaluationProtocol
from emblema.evaluation.domain.task.frozen_test_split import FrozenTestSplit
from emblema.evaluation.domain.task.task_split import TaskSplit
from emblema.evaluation.domain.task.task_windows import TaskWindows
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm


class DownstreamTaskDocument:
    """Reads a task to a document and back, whole, for a machine that has no registry to ask.

    The units of every side travel sorted, so one task is always one document, and the frozen
    side travels as the units it names and nothing more: the machine that runs a tuning order
    has to know which units it may not read, not what they hold. The strata travel as the count
    of a quantity's bins; a task over outcomes is spread over its outcomes, which has nothing to
    count, and its scheme says so. Which windows the task reads travels only when it is not every
    window, so a task written before the choice existed is written as it always was.
    """

    def encode(self, task: DownstreamTask) -> dict[str, Any]:
        encoded: dict[str, Any] = {
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
            "strata": task.strata.count if isinstance(task.strata, TargetBins) else None,
        }
        if task.windows is not TaskWindows.EVERY:
            encoded["windows"] = str(task.windows)
        return encoded

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
            labels=self._scheme(labels),
            strata=self._strata(document["strata"], labels),
            windows=TaskWindows(document.get("windows", TaskWindows.EVERY)),
        )

    @staticmethod
    def _units(units: frozenset[UnitKey]) -> list[str]:
        return sorted(str(unit) for unit in units)

    @staticmethod
    def _labels(labels: LabelScheme | None) -> dict[str, Any] | None:
        match labels:
            case RemainingLifeScheme():
                return {"scheme": "remaining_life", "ceiling": labels.ceiling}
            case ForecastScheme():
                return {"scheme": "forecast", "channel": labels.channel, "horizon": labels.horizon}
            case OutcomeScheme():
                return {"scheme": "outcome", "outcome": labels.outcome}
            case None:
                return None

    @staticmethod
    def _scheme(labels: dict[str, Any] | None) -> LabelScheme | None:
        if labels is None:
            return None
        match labels["scheme"]:
            case "remaining_life":
                return RemainingLifeScheme(labels["ceiling"])
            case "forecast":
                return ForecastScheme(labels["channel"], labels["horizon"])
            case "outcome":
                return OutcomeScheme(labels["outcome"])
            case unknown:
                raise ValueError(f"no label scheme is called {unknown!r}")

    @staticmethod
    def _strata(count: int | None, labels: dict[str, Any] | None) -> Stratification | None:
        if count is not None:
            return TargetBins(count)
        if labels is not None and labels["scheme"] == "outcome":
            return ClassStrata()
        return None
