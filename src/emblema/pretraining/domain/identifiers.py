"""Identifiers private to Pretraining."""

from dataclasses import dataclass

from emblema.shared.kernel.identifiers import EntityId


@dataclass(frozen=True)
class BackboneId(EntityId):
    """Identity of a backbone.

    Private to the context until another one refers to a backbone by identity; Evaluation sees
    candidates through its anti-corruption layer, by an opaque reference and a checksum, and the
    day it needs the type itself this moves to the published language.
    """
