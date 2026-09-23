from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

from emblema.shared.adapters.persistence.naming import NAMING_CONVENTION


class Base(DeclarativeBase):
    """Declarative base of the Evaluation context's persistence model: its schema and names.

    Every record class of this context derives from it, so every table lands in the
    ``evaluation`` schema; the constraint names follow the convention every context shares.
    """

    metadata = MetaData(schema="evaluation", naming_convention=NAMING_CONVENTION)
