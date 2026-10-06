"""Which of a unit's windows a task reads.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "evaluation"
TABLE = "downstream_task"


def upgrade() -> None:
    # Every task stored before the choice existed read every window of its units.
    op.add_column(
        TABLE,
        sa.Column("windows", sa.Text(), server_default=sa.text("'every'"), nullable=False),
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_downstream_task_windows_known",
        TABLE,
        "windows IN ('every', 'first')",
        schema=SCHEMA,
    )


def downgrade() -> None:
    # A task that reads one window of each unit cannot be kept by the schema before this one, and
    # dropping the choice would leave it reading every window as if it had always been asked so.
    narrowed = (
        op.get_bind()
        .execute(sa.text(f"SELECT count(*) FROM {SCHEMA}.{TABLE} WHERE windows <> 'every'"))
        .scalar_one()
    )
    if narrowed:
        raise RuntimeError(
            f"{narrowed} tasks in {SCHEMA}.{TABLE} read one window of each unit: the schema "
            "before this one reads every window, so downgrading would change what they ask"
        )
    op.drop_constraint("ck_downstream_task_windows_known", TABLE, schema=SCHEMA)
    op.drop_column(TABLE, "windows", schema=SCHEMA)
