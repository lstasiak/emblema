"""The names a published vocabulary's channels carry, as a training corpus states them.

Shared by the report scripts that build a training corpus straight from a tokenisation manifest
rather than through the published reader, so that all of them name channels as the reader does.
"""

from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary
from emblema.pretraining.adapters.blocks.block_training_corpus_reader import (
    BlockTrainingCorpusReader,
)


def channel_names(vocabulary: ChannelVocabulary) -> tuple[str, ...]:
    """The vocabulary's channels in identifier order, named as the reader names them."""
    return tuple(
        BlockTrainingCorpusReader.channel_name(entry.corpus, entry.channel)
        for entry in vocabulary.entries
    )
