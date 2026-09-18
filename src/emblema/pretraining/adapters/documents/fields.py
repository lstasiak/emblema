from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm

type Document = dict[str, object]


class Fields:
    """A decoded JSON object read one typed field at a time, refusing what is missing or mistyped.

    Every reader raises ``ValueError`` with the key in the message, so a codec catches one class
    and reports which field a malformed document broke on rather than a bare ``KeyError``.
    """

    def __init__(self, document: Mapping[str, object]) -> None:
        self._document = document

    def text(self, key: str) -> str:
        value = self._required(key)
        if not isinstance(value, str):
            raise ValueError(f"{key!r} must be text, got {type(value).__name__}")
        return value

    def optional_text(self, key: str) -> str | None:
        value = self._document.get(key)
        return None if value is None else self.text(key)

    def integer(self, key: str) -> int:
        value = self._required(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key!r} must be an integer, got {type(value).__name__}")
        return value

    def number(self, key: str) -> float:
        value = self._required(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{key!r} must be a number, got {type(value).__name__}")
        return float(value)

    def mapping(self, key: str) -> Mapping[str, object]:
        value = self._required(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"{key!r} must be an object, got {type(value).__name__}")
        return value

    def fields(self, key: str) -> Self:
        return type(self)(self.mapping(key))

    def optional_fields(self, key: str) -> Self | None:
        return None if self._document.get(key) is None else self.fields(key)

    def each(self, key: str) -> list[Self]:
        value = self._required(key)
        if not isinstance(value, list):
            raise ValueError(f"{key!r} must be a list, got {type(value).__name__}")
        return [type(self)(self._object(key, item)) for item in value]

    def checksum(self, key: str) -> Checksum:
        fields = self.fields(key)
        return Checksum(HashAlgorithm(fields.text("algorithm")), fields.text("digest"))

    def ref(self, key: str) -> ArtifactRef:
        fields = self.fields(key)
        return ArtifactRef(fields.text("key"), fields.checksum("checksum"))

    def optional_ref(self, key: str) -> ArtifactRef | None:
        return None if self._document.get(key) is None else self.ref(key)

    def refs(self, key: str) -> list[ArtifactRef]:
        return [ArtifactRef(item.text("key"), item.checksum("checksum")) for item in self.each(key)]

    @staticmethod
    def of_checksum(checksum: Checksum) -> Document:
        return {"algorithm": str(checksum.algorithm), "digest": checksum.digest}

    @classmethod
    def of_ref(cls, ref: ArtifactRef) -> Document:
        return {"key": ref.key, "checksum": cls.of_checksum(ref.checksum)}

    @classmethod
    def of_optional_ref(cls, ref: ArtifactRef | None) -> Document | None:
        return None if ref is None else cls.of_ref(ref)

    def _required(self, key: str) -> object:
        if key not in self._document:
            raise ValueError(f"{key!r} is missing")
        return self._document[key]

    @staticmethod
    def _object(key: str, item: object) -> Mapping[str, object]:
        if not isinstance(item, Mapping):
            raise ValueError(f"every item of {key!r} must be an object")
        return item
