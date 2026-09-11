"""Stateful property test: no sequence of operations breaks the invariants of a corpus."""

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.domain.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.corpus import Corpus
from emblema.catalog.domain.corpus_content import CorpusContent
from emblema.catalog.domain.corpus_version import CorpusVersion
from emblema.catalog.domain.exceptions import (
    CorpusVersionFrozenError,
    CorpusVersionNotValidatedError,
    SameDataAlreadyFrozenError,
)
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime
from tests.catalog.domain.support import LICENCE, empty_corpus, instant

# Small vocabularies make collisions (duplicate data, repeated operations on one version) likely.
schemas = st.lists(st.sampled_from(["a", "b", "c"]), min_size=1, max_size=3, unique=True).map(
    lambda names: ChannelSchema(frozenset(Channel(name) for name in names))
)
contents = st.builds(
    CorpusContent,
    checksum=st.sampled_from([b"x", b"y", b"z"]).map(Checksum.of_bytes),
    unit_count=st.integers(min_value=1, max_value=3),
    observation_count=st.integers(min_value=1, max_value=3),
)


def fingerprint(version: CorpusVersion) -> tuple[Checksum, ChannelSchema, SamplingRegime]:
    assert version.content is not None
    return version.content.checksum, version.channel_schema, version.sampling_regime


class CorpusLifecycle(RuleBasedStateMachine):
    versions = Bundle("versions")

    def __init__(self) -> None:
        super().__init__()
        self.corpus: Corpus = empty_corpus()
        self.ids = SequentialIdGenerator()
        self.ticks = 0
        self.frozen_snapshots: dict[CorpusVersionId, CorpusVersion] = {}

    @rule(target=versions, schema=schemas, regime=st.sampled_from(SamplingRegime))
    def add_version(self, schema: ChannelSchema, regime: SamplingRegime) -> CorpusVersionId:
        version_id = self.ids.generate(CorpusVersionId)
        self.corpus = self.corpus.add_version(version_id, schema, regime, LICENCE)
        return version_id

    @rule(version_id=versions, content=contents)
    def record_content(self, version_id: CorpusVersionId, content: CorpusContent) -> None:
        if self.corpus.get_version(version_id).is_frozen:
            with pytest.raises(CorpusVersionFrozenError):
                self.corpus.record_content(version_id, content)
        else:
            self.corpus = self.corpus.record_content(version_id, content)

    @rule(version_id=versions)
    def freeze(self, version_id: CorpusVersionId) -> None:
        version = self.corpus.get_version(version_id)
        self.ticks += 1
        at = instant(self.ticks)
        if version.is_frozen:
            with pytest.raises(CorpusVersionFrozenError):
                self.corpus.freeze_version(version_id, at)
        elif version.content is None:
            with pytest.raises(CorpusVersionNotValidatedError):
                self.corpus.freeze_version(version_id, at)
        elif fingerprint(version) in {fingerprint(v) for v in self.frozen_snapshots.values()}:
            with pytest.raises(SameDataAlreadyFrozenError):
                self.corpus.freeze_version(version_id, at)
        else:
            self.corpus = self.corpus.freeze_version(version_id, at)
            self.frozen_snapshots[version_id] = self.corpus.get_version(version_id)

    @invariant()
    def version_numbers_are_one_to_n(self) -> None:
        numbers = [version.number for version in self.corpus.versions]
        assert numbers == list(range(1, len(numbers) + 1))

    @invariant()
    def frozen_versions_never_change(self) -> None:
        for version_id, snapshot in self.frozen_snapshots.items():
            assert self.corpus.get_version(version_id) == snapshot

    @invariant()
    def frozen_versions_have_content_and_a_freeze_instant(self) -> None:
        for version in self.corpus.frozen_versions:
            assert version.content is not None
            assert version.frozen_at is not None

    @invariant()
    def frozen_versions_describe_distinct_data(self) -> None:
        fingerprints = [fingerprint(version) for version in self.corpus.frozen_versions]
        assert len(set(fingerprints)) == len(fingerprints)


TestCorpusLifecycle = CorpusLifecycle.TestCase
TestCorpusLifecycle.settings = settings(max_examples=100, stateful_step_count=30, deadline=None)
