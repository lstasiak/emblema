"""Identifiers of the Data Catalog aggregates and entities."""

from dataclasses import dataclass

from emblema.shared.kernel.identifiers import EntityId


@dataclass(frozen=True)
class CorpusId(EntityId):
    pass


@dataclass(frozen=True)
class CorpusVersionId(EntityId):
    pass
