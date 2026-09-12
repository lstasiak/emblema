from emblema.catalog.contracts.corpus_version_ref import ChannelSpec, CorpusVersionRef
from emblema.catalog.domain.registry.corpus_version import CorpusVersion


class CorpusVersionRefAssembler:
    """Builds the Catalog's published references from its domain model.

    This is the Catalog's side of its published language, not an anti-corruption layer: it
    translates the context's own model outwards. Only frozen versions leave the context, because
    a draft has no checksum to pin and may still change. Channels come out in name order, the
    canonical order of a reference.
    """

    def assemble(self, version: CorpusVersion) -> CorpusVersionRef:
        """Reference of one frozen version.

        Raises:
            CorpusVersionNotFrozenError: If the version is still a draft.
        """
        content = version.frozen_content()
        return CorpusVersionRef(
            version_id=version.id,
            checksum=content.checksum,
            channels=tuple(
                ChannelSpec(channel.name, channel.unit, channel.timeless)
                for channel in version.channel_schema
            ),
            sampling_regime=version.sampling_regime,
        )
