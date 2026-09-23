from typing import Self
from uuid import UUID

from sqlalchemy import CheckConstraint, Float, Integer, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from emblema.evaluation.adapters.persistence.orm import Base
from emblema.evaluation.adapters.persistence.task_unit_record import (
    TEST,
    TUNING,
    VALIDATION,
    TaskUnitRecord,
)
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

REMAINING_LIFE, FORECAST = "remaining_life", "forecast"


class DownstreamTaskRecord(Base):
    """Row of ``evaluation.downstream_task`` with its units: the persistence model of the task.

    The label scheme is spread over columns rather than kept as a document: which scheme a task
    reads and under what ceiling is asked about whenever results are compared, and a check
    constraint then keeps each scheme to the parameters it has. A protocol that spends no labels
    carries no scheme at all, and the constraints say that too, so the invariant the aggregate
    holds is held by the database as well.
    """

    __tablename__ = "downstream_task"
    __table_args__ = (
        CheckConstraint(
            f"protocol IN ('{EvaluationProtocol.LABEL_BUDGET}', "
            f"'{EvaluationProtocol.ANOMALY_DETECTION}')",
            name="protocol_known",
        ),
        CheckConstraint(
            f"label_scheme IS NULL OR label_scheme IN ('{REMAINING_LIFE}', '{FORECAST}')",
            name="label_scheme_known",
        ),
        CheckConstraint(
            f"(protocol = '{EvaluationProtocol.LABEL_BUDGET}') = (label_scheme IS NOT NULL)",
            name="labels_match_protocol",
        ),
        CheckConstraint("(label_scheme IS NULL) = (strata IS NULL)", name="strata_with_labels"),
        CheckConstraint(
            f"(label_scheme = '{REMAINING_LIFE}') = (label_ceiling IS NOT NULL)",
            name="remaining_life_has_ceiling",
        ),
        CheckConstraint(
            f"(label_scheme = '{FORECAST}') = (label_channel IS NOT NULL)",
            name="forecast_has_channel",
        ),
        CheckConstraint(
            f"(label_scheme = '{FORECAST}') = (label_horizon IS NOT NULL)",
            name="forecast_has_horizon",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    corpus: Mapped[str] = mapped_column(Text)
    manifest_key: Mapped[str] = mapped_column(Text)
    manifest_algorithm: Mapped[str] = mapped_column(Text)
    manifest_digest: Mapped[str] = mapped_column(Text)
    protocol: Mapped[str] = mapped_column(Text)
    test_source: Mapped[str] = mapped_column(Text)
    label_scheme: Mapped[str | None] = mapped_column(Text)
    label_ceiling: Mapped[float | None] = mapped_column(Float)
    label_channel: Mapped[str | None] = mapped_column(Text)
    label_horizon: Mapped[float | None] = mapped_column(Float)
    strata: Mapped[int | None] = mapped_column(Integer)
    units: Mapped[list[TaskUnitRecord]] = relationship(
        cascade="all, delete-orphan", lazy="selectin", order_by=TaskUnitRecord.unit
    )

    @classmethod
    def from_task(cls, task: DownstreamTask) -> Self:
        labels = task.labels
        return cls(
            id=task.task_id.value,
            corpus=task.corpus,
            manifest_key=task.manifest.key,
            manifest_algorithm=str(task.manifest.checksum.algorithm),
            manifest_digest=task.manifest.checksum.digest,
            protocol=str(task.protocol),
            test_source=task.split.test.source,
            label_scheme=cls._scheme_name(labels),
            label_ceiling=labels.ceiling if isinstance(labels, RemainingLifeScheme) else None,
            label_channel=labels.channel if isinstance(labels, ForecastScheme) else None,
            label_horizon=labels.horizon if isinstance(labels, ForecastScheme) else None,
            strata=None if task.strata is None else task.strata.count,
            units=[
                TaskUnitRecord.of(task.task_id, unit, side)
                for side, keys in (
                    (TUNING, task.split.tuning),
                    (VALIDATION, task.split.validation),
                    (TEST, task.split.test.units),
                )
                for unit in sorted(keys, key=str)
            ],
        )

    def to_task(self) -> DownstreamTask:
        sides = {side: self._side(side) for side in (TUNING, VALIDATION, TEST)}
        return DownstreamTask(
            task_id=TaskId(self.id),
            corpus=self.corpus,
            manifest=ArtifactRef(
                self.manifest_key,
                Checksum(HashAlgorithm(self.manifest_algorithm), self.manifest_digest),
            ),
            protocol=EvaluationProtocol(self.protocol),
            split=TaskSplit(
                tuning=sides[TUNING],
                validation=sides[VALIDATION],
                test=FrozenTestSplit(units=sides[TEST], source=self.test_source),
            ),
            labels=self._labels(),
            strata=None if self.strata is None else TargetBins(self.strata),
        )

    def _side(self, side: str) -> frozenset[UnitKey]:
        return frozenset(record.to_unit() for record in self.units if record.side == side)

    def _labels(self) -> RemainingLifeScheme | ForecastScheme | None:
        if self.label_scheme == REMAINING_LIFE and self.label_ceiling is not None:
            return RemainingLifeScheme(self.label_ceiling)
        if (
            self.label_scheme == FORECAST
            and self.label_channel is not None
            and self.label_horizon is not None
        ):
            return ForecastScheme(self.label_channel, self.label_horizon)
        return None

    @staticmethod
    def _scheme_name(labels: RemainingLifeScheme | ForecastScheme | None) -> str | None:
        match labels:
            case RemainingLifeScheme():
                return REMAINING_LIFE
            case ForecastScheme():
                return FORECAST
            case None:
                return None
