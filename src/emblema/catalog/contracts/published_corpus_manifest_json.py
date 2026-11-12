from __future__ import annotations

import json
from typing import Final

from emblema.catalog.contracts.exceptions import MalformedManifestError
from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.published_channel_statistics import (
    NORMALISATION_QUANTILE,
    NORMALISATION_ROBUST,
    NORMALISATION_STANDARD,
    ROBUST_BOUND_UNSTATED,
    PublishedChannelStatistics,
)
from emblema.catalog.contracts.published_corpus_manifest import (
    SHORT_UNITS_DROPPED,
    PublishedCorpusManifest,
)
from emblema.catalog.contracts.published_value_bounds import PublishedValueBounds
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
    # The version of a manifest whose values were normalised otherwise than by the standard score:
    # a reader that knew only that score would normalise a served value wrongly without noticing,
    # so it must refuse such a manifest rather than ignore the field that says so.
    VERSION_OTHER_NORMALISATION: Final = 2
    # The version of a manifest whose robust score is held to another bound than the one every
    # earlier robust manifest was held to: a reader that knew only that bound would clip a
    # served value wrongly without noticing, so it must refuse such a manifest too.
    VERSION_ROBUST_BOUND: Final = 3

    def encode(self, manifest: PublishedCorpusManifest) -> bytes:
        document: _Document = {
            "format": self.FORMAT,
            "version": self._version_of(manifest),
            "corpus": manifest.corpus,
            "corpus_version": str(manifest.corpus_version.value),
            "corpus_checksum": self._checksum_document(manifest.corpus_checksum),
            "block": {
                "key": manifest.block.key,
                "checksum": self._checksum_document(manifest.block.checksum),
            },
            "window": self._window_document(manifest),
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
        # Stated only where the statistics were adopted, so every manifest that fitted its own
        # keeps its bytes, and so its checksum.
        if manifest.statistics_from is not None:
            document["statistics_from"] = {
                "key": manifest.statistics_from.key,
                "checksum": self._checksum_document(manifest.statistics_from.checksum),
            }
        # Stated only where the values were cleaned, for the same reason.
        if manifest.cleaning:
            document["cleaning"] = [self._bounds_document(each) for each in manifest.cleaning]
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
        version = document.get("version")
        if document.get("format") != self.FORMAT or version not in (
            self.VERSION,
            self.VERSION_OTHER_NORMALISATION,
            self.VERSION_ROBUST_BOUND,
        ):
            raise MalformedManifestError(
                f"manifest is a {document.get('format')!r} of version {version!r}"
            )
        try:
            manifest = self._manifest(_Fields(document))
        except ValueError as error:
            raise MalformedManifestError(f"manifest is not well formed: {error}") from error
        # Each version states exactly what the encoder would write for it, so a manifest read
        # and written again keeps its bytes, and so its checksum.
        if version != self._version_of(manifest):
            other = self._other_normalisations(manifest)
            raise MalformedManifestError(
                f"version {self.VERSION_OTHER_NORMALISATION} is written exactly where a channel is "
                f"normalised otherwise than by the standard score and version "
                f"{self.VERSION_ROBUST_BOUND} where a robust score is held to another bound than "
                f"{ROBUST_BOUND_UNSTATED:g}; version {version} states {other or 'none'}"
                + (" and a bound" if self._bounds_stated(manifest) else "")
            )
        return manifest

    @classmethod
    def _version_of(cls, manifest: PublishedCorpusManifest) -> int:
        if cls._bounds_stated(manifest):
            return cls.VERSION_ROBUST_BOUND
        if cls._other_normalisations(manifest):
            return cls.VERSION_OTHER_NORMALISATION
        return cls.VERSION

    @staticmethod
    def _bounds_stated(manifest: PublishedCorpusManifest) -> bool:
        return any(
            channel.statistics is not None
            and channel.statistics.bound is not None
            and channel.statistics.bound != ROBUST_BOUND_UNSTATED
            for channel in manifest.channels
        )

    @staticmethod
    def _other_normalisations(manifest: PublishedCorpusManifest) -> list[str]:
        return sorted(
            {
                channel.statistics.normalisation
                for channel in manifest.channels
                if channel.statistics is not None
            }
            - {NORMALISATION_STANDARD}
        )

    def _manifest(self, fields: _Fields) -> PublishedCorpusManifest:
        block = fields.fields("block")
        window = fields.fields("window")
        split = fields.fields("split")
        adopted = fields.optional_fields("statistics_from")
        return PublishedCorpusManifest(
            corpus=fields.text("corpus"),
            corpus_version=CorpusVersionId.parse(fields.text("corpus_version")),
            corpus_checksum=self._checksum(fields.fields("corpus_checksum")),
            block=ArtifactRef(block.text("key"), self._checksum(block.fields("checksum"))),
            window_length=window.number("length"),
            window_stride=window.number("stride"),
            window_short_units=self._short_units(window),
            channels=tuple(self._channel(channel) for channel in fields.each("channels")),
            units=fields.texts("units"),
            empty_units=fields.texts("empty_units"),
            training_units=split.texts("training"),
            validation_units=split.texts("validation"),
            split_seed=split.optional_integer("seed"),
            window_count=fields.integer("window_count"),
            token_count=fields.integer("token_count"),
            statistics_from=None
            if adopted is None
            else ArtifactRef(adopted.text("key"), self._checksum(adopted.fields("checksum"))),
            cleaning=tuple(self._bounds(each) for each in fields.optional_each("cleaning")),
        )

    @staticmethod
    def _window_document(manifest: PublishedCorpusManifest) -> _Document:
        """The window, its policy for short units stated only where it is not the usual one.

        Every manifest written before the policy existed keeps its bytes, and so its checksum.
        """
        document: _Document = {
            "length": manifest.window_length,
            "stride": manifest.window_stride,
        }
        if manifest.window_short_units != SHORT_UNITS_DROPPED:
            document["short_units"] = manifest.window_short_units
        return document

    @staticmethod
    def _short_units(window: _Fields) -> str:
        stated = window.optional_text("short_units")
        return SHORT_UNITS_DROPPED if stated is None else stated

    @classmethod
    def _channel_document(cls, channel: PublishedChannel) -> _Document:
        statistics = channel.statistics
        return {
            "channel_id": channel.channel_id,
            "corpus": channel.corpus,
            "channel": channel.channel,
            "timeless": channel.timeless,
            "unit": channel.unit,
            "statistics": None if statistics is None else cls._statistics_document(statistics),
        }

    @staticmethod
    def _statistics_document(statistics: PublishedChannelStatistics) -> _Document:
        """The statistics, their normalisation and its facts stated only where it is not the usual.

        Every manifest written before the other normalisations existed keeps its bytes.
        """
        document: _Document = {
            "count": statistics.count,
            "mean": statistics.mean,
            "std": statistics.std,
        }
        if statistics.normalisation != NORMALISATION_STANDARD:
            document["normalisation"] = statistics.normalisation
        if statistics.median is not None:
            document["median"] = statistics.median
        if statistics.mad is not None:
            document["mad"] = statistics.mad
        if statistics.quantiles:
            document["quantiles"] = list(statistics.quantiles)
        if statistics.bound is not None and statistics.bound != ROBUST_BOUND_UNSTATED:
            document["bound"] = statistics.bound
        return document

    @classmethod
    def _channel(cls, fields: _Fields) -> PublishedChannel:
        statistics = fields.optional_fields("statistics")
        return PublishedChannel(
            channel_id=fields.integer("channel_id"),
            corpus=fields.text("corpus"),
            channel=fields.text("channel"),
            timeless=fields.flag("timeless"),
            unit=fields.optional_text("unit"),
            statistics=None if statistics is None else cls._statistics(statistics),
        )

    @staticmethod
    def _statistics(fields: _Fields) -> PublishedChannelStatistics:
        named = fields.optional_text("normalisation")
        # The standard score is never written, so a field naming it is not the encoder's.
        facts = {NORMALISATION_ROBUST: {"median", "mad"}, NORMALISATION_QUANTILE: {"quantiles"}}
        if named is not None and named not in facts:
            raise MalformedManifestError(
                f"field 'normalisation' must be one of {sorted(facts)} where stated, got {named!r}"
            )
        normalisation = NORMALISATION_STANDARD if named is None else named
        expected = facts.get(normalisation, set())
        stated = {key for key in ("median", "mad", "quantiles") if fields.has(key)}
        if stated != expected:
            raise MalformedManifestError(
                f"statistics normalised by {normalisation!r} state {sorted(expected)}, "
                f"not {sorted(stated)}"
            )
        robust = normalisation == NORMALISATION_ROBUST
        if fields.has("bound") and not robust:
            raise MalformedManifestError(
                f"a bound belongs to a robust normalisation, not a {normalisation!r} one"
            )
        if fields.has("bound") and fields.number("bound") == ROBUST_BOUND_UNSTATED:
            # Written again it would be left out: other bytes, other checksum.
            raise MalformedManifestError(
                f"a bound of {ROBUST_BOUND_UNSTATED:g} is the one a robust score is held to "
                "unless stated, and is never written"
            )
        return PublishedChannelStatistics(
            fields.integer("count"),
            fields.number("mean"),
            fields.number("std"),
            normalisation=normalisation,
            median=fields.number("median") if fields.has("median") else None,
            mad=fields.number("mad") if fields.has("mad") else None,
            quantiles=fields.numbers("quantiles") if fields.has("quantiles") else (),
            bound=(fields.number("bound") if fields.has("bound") else ROBUST_BOUND_UNSTATED)
            if robust
            else None,
        )

    @staticmethod
    def _bounds_document(bounds: PublishedValueBounds) -> _Document:
        return {
            "channel": bounds.channel,
            "removed_below": bounds.removed_below,
            "clipped_below": bounds.clipped_below,
            "clipped_above": bounds.clipped_above,
            "removed_above": bounds.removed_above,
            "removed": bounds.removed,
            "clipped": bounds.clipped,
        }

    @staticmethod
    def _bounds(fields: _Fields) -> PublishedValueBounds:
        return PublishedValueBounds(
            channel=fields.text("channel"),
            removed_below=fields.number("removed_below"),
            clipped_below=fields.number("clipped_below"),
            clipped_above=fields.number("clipped_above"),
            removed_above=fields.number("removed_above"),
            removed=fields.integer("removed"),
            clipped=fields.integer("clipped"),
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

    def optional_integer(self, key: str) -> int | None:
        value = self._document.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise MalformedManifestError(f"field {key!r} must be an integer or null, got {value!r}")
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

    def has(self, key: str) -> bool:
        return key in self._document

    def numbers(self, key: str) -> tuple[float, ...]:
        value = self._required(key)
        if not isinstance(value, list) or not all(
            isinstance(item, int | float) and not isinstance(item, bool) for item in value
        ):
            raise MalformedManifestError(f"field {key!r} must be a list of numbers")
        return tuple(float(item) for item in value)

    def optional_each(self, key: str) -> list[_Fields]:
        return [] if key not in self._document else self.each(key)

    def texts(self, key: str) -> tuple[str, ...]:
        value = self._required(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise MalformedManifestError(f"field {key!r} must be a list of text")
        return tuple(value)

    def _required(self, key: str) -> object:
        if key not in self._document:
            raise MalformedManifestError(f"field {key!r} is missing")
        return self._document[key]
