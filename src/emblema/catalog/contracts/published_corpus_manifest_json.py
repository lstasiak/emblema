from __future__ import annotations

import json
from typing import Final

from emblema.catalog.contracts.exceptions import MalformedManifestError
from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.published_channel_statistics import PublishedChannelStatistics
from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm

type _Document = dict[str, object]


class PublishedCorpusManifestJson:
    """The published manifest as JSON, the form it is stored in.

    Compact and key-ordered, so the same manifest is always the same bytes and two publications
    of one corpus store one description. Part of the published language because other contexts
    read the manifest from the artifact store, with nothing but the standard library.
    """

    FORMAT: Final = "emblema.tokenisation-manifest"
    # Bumped when the meaning of a field changes; a new field is one an older reader ignores.
    VERSION: Final = 1

    def encode(self, manifest: PublishedCorpusManifest) -> bytes:
        document: _Document = {
            "format": self.FORMAT,
            "version": self.VERSION,
            "corpus": manifest.corpus,
            "corpus_version": str(manifest.corpus_version.value),
            "corpus_checksum": self._checksum_document(manifest.corpus_checksum),
            "block": {
                "key": manifest.block.key,
                "checksum": self._checksum_document(manifest.block.checksum),
            },
            "window": {"length": manifest.window_length, "stride": manifest.window_stride},
            "channels": [self._channel_document(channel) for channel in manifest.channels],
            "units": list(manifest.units),
            "empty_units": list(manifest.empty_units),
            "split": {
                "seed": manifest.split_seed,
                "training": list(manifest.training_units),
                "validation": list(manifest.validation_units),
            },
            "window_count": manifest.window_count,
            "token_count": manifest.token_count,
        }
        return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def decode(self, content: bytes) -> PublishedCorpusManifest:
        """The manifest those bytes describe.

        Raises:
            MalformedManifestError: If the bytes are not a manifest of a version this reads, a
                field is missing or of the wrong type, or the manifest breaks its own rules.
        """
        try:
            document = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MalformedManifestError(f"manifest is not JSON: {error}") from error
        if not isinstance(document, dict):
            raise MalformedManifestError("manifest is not an object")
        if document.get("format") != self.FORMAT or document.get("version") != self.VERSION:
            raise MalformedManifestError(
                f"manifest is a {document.get('format')!r} of version {document.get('version')!r}"
            )
        try:
            return self._manifest(_Fields(document))
        except ValueError as error:
            raise MalformedManifestError(f"manifest is not well formed: {error}") from error

    def _manifest(self, fields: _Fields) -> PublishedCorpusManifest:
        block = fields.fields("block")
        window = fields.fields("window")
        split = fields.fields("split")
        return PublishedCorpusManifest(
            corpus=fields.text("corpus"),
            corpus_version=CorpusVersionId.parse(fields.text("corpus_version")),
            corpus_checksum=self._checksum(fields.fields("corpus_checksum")),
            block=ArtifactRef(block.text("key"), self._checksum(block.fields("checksum"))),
            window_length=window.number("length"),
            window_stride=window.number("stride"),
            channels=tuple(self._channel(channel) for channel in fields.each("channels")),
            units=fields.texts("units"),
            empty_units=fields.texts("empty_units"),
            training_units=split.texts("training"),
            validation_units=split.texts("validation"),
            split_seed=split.integer("seed"),
            window_count=fields.integer("window_count"),
            token_count=fields.integer("token_count"),
        )

    @staticmethod
    def _channel_document(channel: PublishedChannel) -> _Document:
        statistics = channel.statistics
        return {
            "channel_id": channel.channel_id,
            "corpus": channel.corpus,
            "channel": channel.channel,
            "timeless": channel.timeless,
            "unit": channel.unit,
            "statistics": None
            if statistics is None
            else {"count": statistics.count, "mean": statistics.mean, "std": statistics.std},
        }

    @staticmethod
    def _channel(fields: _Fields) -> PublishedChannel:
        statistics = fields.optional_fields("statistics")
        return PublishedChannel(
            channel_id=fields.integer("channel_id"),
            corpus=fields.text("corpus"),
            channel=fields.text("channel"),
            timeless=fields.flag("timeless"),
            unit=fields.optional_text("unit"),
            statistics=None
            if statistics is None
            else PublishedChannelStatistics(
                statistics.integer("count"), statistics.number("mean"), statistics.number("std")
            ),
        )

    @staticmethod
    def _checksum_document(checksum: Checksum) -> _Document:
        return {"algorithm": str(checksum.algorithm), "digest": checksum.digest}

    @staticmethod
    def _checksum(fields: _Fields) -> Checksum:
        return Checksum(HashAlgorithm(fields.text("algorithm")), fields.text("digest"))


class _Fields:
    """A JSON object read one field at a time; a failure names the field and what it had to be."""

    def __init__(self, document: _Document) -> None:
        self._document = document

    def text(self, key: str) -> str:
        value = self._required(key)
        if not isinstance(value, str):
            raise MalformedManifestError(f"field {key!r} must be text, got {value!r}")
        return value

    def optional_text(self, key: str) -> str | None:
        value = self._document.get(key)
        if value is not None and not isinstance(value, str):
            raise MalformedManifestError(f"field {key!r} must be text or null, got {value!r}")
        return value

    def integer(self, key: str) -> int:
        value = self._required(key)
        # A JSON ``true`` decodes as a Python ``bool``, which is an ``int``; a count is never one.
        if isinstance(value, bool) or not isinstance(value, int):
            raise MalformedManifestError(f"field {key!r} must be an integer, got {value!r}")
        return value

    def number(self, key: str) -> float:
        value = self._required(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise MalformedManifestError(f"field {key!r} must be a number, got {value!r}")
        return float(value)

    def flag(self, key: str) -> bool:
        value = self._required(key)
        if not isinstance(value, bool):
            raise MalformedManifestError(f"field {key!r} must be true or false, got {value!r}")
        return value

    def fields(self, key: str) -> _Fields:
        value = self._required(key)
        if not isinstance(value, dict):
            raise MalformedManifestError(f"field {key!r} must be an object, got {value!r}")
        return _Fields(value)

    def optional_fields(self, key: str) -> _Fields | None:
        return None if self._document.get(key) is None else self.fields(key)

    def each(self, key: str) -> list[_Fields]:
        value = self._required(key)
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise MalformedManifestError(f"field {key!r} must be a list of objects")
        return [_Fields(item) for item in value]

    def texts(self, key: str) -> tuple[str, ...]:
        value = self._required(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise MalformedManifestError(f"field {key!r} must be a list of text")
        return tuple(value)

    def _required(self, key: str) -> object:
        if key not in self._document:
            raise MalformedManifestError(f"field {key!r} is missing")
        return self._document[key]
