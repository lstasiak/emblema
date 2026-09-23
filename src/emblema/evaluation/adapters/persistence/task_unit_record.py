from typing import Self
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from emblema.evaluation.adapters.persistence.orm import Base
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.identifiers import UnitKey

TUNING, VALIDATION, TEST = "tuning", "validation", "test"


class TaskUnitRecord(Base):
    """Row of ``evaluation.task_unit``: one unit of a task, and which side of the split it is on.

    A row per unit rather than three lists in a column, because which side a unit is on is the
    question most often asked of a split — by a report, by whoever checks that the frozen side
    was never trained on — and a document would have to be opened to answer it. The side is
    constrained to the three there are, so a split cannot acquire a fourth by a typo.
    """

    __tablename__ = "task_unit"
    __table_args__ = (
        CheckConstraint(f"side IN ('{TUNING}', '{VALIDATION}', '{TEST}')", name="side_known"),
    )

    task_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("downstream_task.id", ondelete="CASCADE"), primary_key=True
    )
    unit: Mapped[str] = mapped_column(Text, primary_key=True)
    side: Mapped[str] = mapped_column(Text)

    @classmethod
    def of(cls, task_id: TaskId, unit: UnitKey, side: str) -> Self:
        return cls(task_id=task_id.value, unit=str(unit), side=side)

    def to_unit(self) -> UnitKey:
        return UnitKey(self.unit)
