from emblema.catalog.domain.corpus_description import CorpusDescription


class InMemoryCorpusReader:
    """Reader over a description held in memory: hands back what it was given.

    Replacing the description stands in for data that changed between two registrations, so the
    detection of modified data can be exercised without any file.
    """

    def __init__(self, description: CorpusDescription) -> None:
        self._description = description

    def describe(self) -> CorpusDescription:
        return self._description

    def replace_data(self, description: CorpusDescription) -> None:
        self._description = description
