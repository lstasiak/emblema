import pytest

from emblema.catalog.application.assemblers.corpus_version_ref_assembler import (
    CorpusVersionRefAssembler,
)
from emblema.catalog.contracts.corpus_version_ref import ChannelSpec, CorpusVersionRef
from emblema.catalog.domain.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.corpus_version import CorpusVersion
from emblema.catalog.domain.exceptions import CorpusVersionNotFrozenError
from emblema.shared.kernel.sampling import SamplingRegime
from tests.catalog.domain.support import AT, LICENCE, SCHEMA, content, version_id

ASSEMBLER = CorpusVersionRefAssembler()


def frozen_version(schema: ChannelSchema = SCHEMA) -> CorpusVersion:
    return CorpusVersion(version_id(), 1, schema, SamplingRegime.REGULAR, LICENCE, content(), AT)


def test_reference_pins_identity_checksum_channels_and_regime() -> None:
    assert ASSEMBLER.assemble(frozen_version()) == CorpusVersionRef(
        version_id=version_id(),
        checksum=content().checksum,
        channels=(ChannelSpec("pressure", "Pa"), ChannelSpec("temperature", "K")),
        sampling_regime=SamplingRegime.REGULAR,
    )


def test_channels_are_published_in_name_order() -> None:
    schema = ChannelSchema(frozenset({Channel("zeta"), Channel("alpha", "V"), Channel("mid")}))

    assert ASSEMBLER.assemble(frozen_version(schema)).channels == (
        ChannelSpec("alpha", "V"),
        ChannelSpec("mid"),
        ChannelSpec("zeta"),
    )


def test_a_static_channel_is_published_as_timeless() -> None:
    schema = ChannelSchema(frozenset({Channel("age", "years", timeless=True), Channel("hr")}))

    assert ASSEMBLER.assemble(frozen_version(schema)).channels == (
        ChannelSpec("age", "years", timeless=True),
        ChannelSpec("hr"),
    )


def test_a_draft_is_not_published() -> None:
    draft = CorpusVersion(version_id(), 1, SCHEMA, SamplingRegime.REGULAR, LICENCE, content())

    with pytest.raises(CorpusVersionNotFrozenError):
        ASSEMBLER.assemble(draft)
