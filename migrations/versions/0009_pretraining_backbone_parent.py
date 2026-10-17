"""The backbone a run continued the weights of.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "pretraining"
TABLE = "backbone"
KEY = "fk_backbone_parent_id"


def upgrade() -> None:
    # Every backbone registered so far drew its weights from its seed, so the column is empty
    # for them and stays so.
    op.add_column(TABLE, sa.Column("parent_id", sa.Uuid(), nullable=True), schema=SCHEMA)
    op.create_foreign_key(
        KEY, TABLE, TABLE, ["parent_id"], ["id"], source_schema=SCHEMA, referent_schema=SCHEMA
    )


def downgrade() -> None:
    # A backbone continued from another's weights cannot be kept by the schema before this one
    # without losing where it came from, which is what the row is for.
    continued = (
        op.get_bind()
        .execute(sa.text(f"SELECT count(*) FROM {SCHEMA}.{TABLE} WHERE parent_id IS NOT NULL"))
        .scalar_one()
    )
    if continued:
        raise RuntimeError(
            f"{continued} backbones in {SCHEMA}.{TABLE} name the backbone they continued: the "
            "schema before this one holds no parent, so downgrading would lose it"
        )
    op.drop_constraint(KEY, TABLE, schema=SCHEMA, type_="foreignkey")
    op.drop_column(TABLE, "parent_id", schema=SCHEMA)
