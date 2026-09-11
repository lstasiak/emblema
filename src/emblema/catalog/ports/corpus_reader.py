from typing import Protocol

from emblema.catalog.domain.corpus_description import CorpusDescription


class CorpusReader(Protocol):
    """Reads one corpus from its source and describes what it holds, keeping none of it.

    An adapter is bound to the location and format of its corpus, the way an artifact store is
    bound to its bucket; the port carries no location, so the hexagon knows nothing about files.
    ``describe`` validates the data, checksums it and counts its units and observations in one
    pass. A description is a function of the bytes: the same data described twice gives equal
    descriptions, and changed data gives a different checksum.
    """

    def describe(self) -> CorpusDescription:
        """Validate the corpus and describe it.

        Raises:
            CorpusDataNotFoundError: If the source holds no data where the adapter expects it.
            MalformedCorpusDataError: If the data is present but violates its own format.
        """
        ...
