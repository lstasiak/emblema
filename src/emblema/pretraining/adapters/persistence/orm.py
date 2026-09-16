from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

from emblema.shared.adapters.persistence.naming import NAMING_CONVENTION


class Base(DeclarativeBase):
    """Declarative base of Pretraining's persistence model: its schema and constraint names.

    Every record class of Pretraining derives from it, so every table lands in the
    ``pretraining`` schema; the constraint names follow the convention every context shares.
    """

    metadata = MetaData(schema="pretraining", naming_convention=NAMING_CONVENTION)
