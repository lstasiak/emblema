from uuid import UUID

import pytest

from emblema.catalog.contracts.corpus_version_ref import ChannelSpec, CorpusVersionRef
from emblema.catalog.contracts.exceptions import (
    InvalidChannelSpecError,
    InvalidCorpusVersionRefError,
)
from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime

VERSION_ID = CorpusVersionId(UUID(int=101))
CHECKSUM = Checksum.of_bytes(b"records")
PRESSURE = ChannelSpec("pressure", "Pa")
TEMPERATURE = ChannelSpec("temperature", "K")


def ref(channels: tuple[ChannelSpec, ...]) -> CorpusVersionRef:
    return CorpusVersionRef(VERSION_ID, CHECKSUM, channels, SamplingRegime.REGULAR)


def test_reference_keeps_channels_in_name_order() -> None:
    assert ref((PRESSURE, TEMPERATURE)).channels == (PRESSURE, TEMPERATURE)


def test_channels_out_of_name_order_are_rejected() -> None:
    with pytest.raises(InvalidCorpusVersionRefError, match="sorted"):
        ref((TEMPERATURE, PRESSURE))


def test_duplicate_channel_names_are_rejected() -> None:
    with pytest.raises(InvalidCorpusVersionRefError, match="duplicate"):
        ref((PRESSURE, ChannelSpec("pressure", "bar")))


def test_reference_needs_at_least_one_channel() -> None:
    with pytest.raises(InvalidCorpusVersionRefError, match="at least one"):
        ref(())


@pytest.mark.parametrize("name", ["", " pressure", "pressure "])
def test_channel_name_must_be_non_blank_without_padding(name: str) -> None:
    with pytest.raises(InvalidChannelSpecError):
        ChannelSpec(name)
