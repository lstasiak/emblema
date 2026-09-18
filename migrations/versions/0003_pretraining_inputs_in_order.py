"""A backbone's inputs in the order their vocabulary was chained.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "pretraining"
TABLE = "pretraining_input"
KEY = "pk_pretraining_input"


def upgrade() -> None:
    # Every backbone registered so far read one corpus, which takes the first position; the
    # default exists for those rows alone and is dropped once they have it.
    op.add_column(
        TABLE,
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        schema=SCHEMA,
    )
    op.alter_column(TABLE, "position", server_default=None, schema=SCHEMA)
    op.drop_constraint(KEY, TABLE, schema=SCHEMA, type_="primary")
    op.create_primary_key(KEY, TABLE, ["backbone_id", "position"], schema=SCHEMA)


def downgrade() -> None:
    # A backbone over several corpora cannot be kept by the schema before this one, and a
    # migration that dropped its inputs to fit would lose what the backbone was trained on.
    beyond_first = (
        op.get_bind()
        .execute(sa.text(f"SELECT count(*) FROM {SCHEMA}.{TABLE} WHERE position > 0"))
        .scalar_one()
    )
    if beyond_first:
        raise RuntimeError(
            f"{beyond_first} input rows beyond position 0 in {SCHEMA}.{TABLE}: the schema "
            "before this one holds one input per backbone, so downgrading would drop them"
        )
    op.drop_constraint(KEY, TABLE, schema=SCHEMA, type_="primary")
    op.create_primary_key(KEY, TABLE, ["backbone_id"], schema=SCHEMA)
    op.drop_column(TABLE, "position", schema=SCHEMA)
