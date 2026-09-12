"""The manifest of a tokenised corpus as JSON, in both directions.

A manifest is read by people as often as by programs — it is what a run consults to decide
whether an artifact suits it, and what a reader of the repository consults to find out what a
published corpus is — so it is text, and the same manifest is always the same text: keys are
ordered and nothing here depends on the order a dictionary was built in. Two runs of one
configuration must produce one artifact, and that holds for the description as much as for the
block it describes.
"""

import json
from typing import Any, Final

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.domain.channel_statistics import ChannelStatistics
from emblema.catalog.domain.channel_vocabulary import ChannelVocabulary, VocabularyEntry
from emblema.catalog.domain.exceptions import MalformedManifestError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation_manifest import TokenisationManifest
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.unit_split import UnitSplit
from emblema.catalog.domain.window_spec import WindowSpec
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm

FORMAT_NAME: Final = "emblema.tokenisation-manifest"
VERSION: Final = 1


def encode_manifest(manifest: TokenisationManifest) -> bytes:
    """The manifest as the bytes that are stored: compact, key-ordered, UTF-8."""
    document = {
        "format": FORMAT_NAME,
        "version": VERSION,
        "corpus": manifest.corpus,
        "corpus_version": str(manifest.corpus_version.value),
        "corpus_checksum": _checksum(manifest.corpus_checksum),
        "block": {"key": manifest.block.key, "checksum": _checksum(manifest.block.checksum)},
        "window": {"length": manifest.window.length, "stride": manifest.window.stride},
        "vocabulary": [_entry(entry) for entry in manifest.scheme.vocabulary.entries],
        "statistics": [_statistics(item) for item in manifest.scheme.statistics],
        "units": [str(key) for key in manifest.units],
        "empty_units": [str(key) for key in manifest.empty_units],
        "split": {
            "seed": manifest.split_seed,
            "training": sorted(str(key) for key in manifest.split.training),
            "validation": sorted(str(key) for key in manifest.split.validation),
        },
        "window_count": manifest.window_count,
        "token_count": manifest.token_count,
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode_manifest(content: bytes) -> TokenisationManifest:
    """The manifest those bytes describe.

    Raises:
        MalformedManifestError: If the bytes are not a manifest of a version this reads, or a
            value in one breaks the rules of the object it belongs to.
    """
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MalformedManifestError(f"manifest is not JSON: {error}") from error
    if not isinstance(document, dict):
        raise MalformedManifestError("manifest is not an object")
    if document.get("format") != FORMAT_NAME or document.get("version") != VERSION:
        raise MalformedManifestError(
            f"manifest is a {document.get('format')!r} of version {document.get('version')!r}"
        )
    try:
        return _manifest(document)
    except (KeyError, TypeError, ValueError) as error:
        raise MalformedManifestError(f"manifest is not well formed: {error}") from error


def _manifest(document: dict[str, Any]) -> TokenisationManifest:
    vocabulary = ChannelVocabulary(
        tuple(
            VocabularyEntry(
                channel_id=entry["channel_id"],
                corpus=entry["corpus"],
                channel=entry["channel"],
                timeless=entry["timeless"],
                unit=entry["unit"],
            )
            for entry in document["vocabulary"]
        )
    )
    statistics = tuple(
        None
        if item is None
        else ChannelStatistics(count=item["count"], mean=item["mean"], std=item["std"])
        for item in document["statistics"]
    )
    split = document["split"]
    return TokenisationManifest(
        corpus=document["corpus"],
        corpus_version=CorpusVersionId.parse(document["corpus_version"]),
        corpus_checksum=_parsed_checksum(document["corpus_checksum"]),
        block=ArtifactRef(
            key=document["block"]["key"], checksum=_parsed_checksum(document["block"]["checksum"])
        ),
        window=WindowSpec(document["window"]["length"], document["window"]["stride"]),
        scheme=TokenisationScheme(vocabulary=vocabulary, statistics=statistics),
        units=tuple(UnitKey(key) for key in document["units"]),
        split=UnitSplit(
            training=frozenset(UnitKey(key) for key in split["training"]),
            validation=frozenset(UnitKey(key) for key in split["validation"]),
        ),
        split_seed=split["seed"],
        window_count=document["window_count"],
        token_count=document["token_count"],
        empty_units=tuple(UnitKey(key) for key in document["empty_units"]),
    )


def _checksum(checksum: Checksum) -> dict[str, str]:
    return {"algorithm": str(checksum.algorithm), "digest": checksum.digest}


def _parsed_checksum(document: dict[str, str]) -> Checksum:
    return Checksum(HashAlgorithm(document["algorithm"]), document["digest"])


def _entry(entry: VocabularyEntry) -> dict[str, Any]:
    return {
        "channel_id": entry.channel_id,
        "corpus": entry.corpus,
        "channel": entry.channel,
        "timeless": entry.timeless,
        "unit": entry.unit,
    }


def _statistics(statistics: ChannelStatistics | None) -> dict[str, Any] | None:
    if statistics is None:
        return None
    return {"count": statistics.count, "mean": statistics.mean, "std": statistics.std}
