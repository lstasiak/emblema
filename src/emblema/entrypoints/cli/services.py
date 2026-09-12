from dataclasses import dataclass

from emblema.catalog.application.use_cases.publish_corpus import PublishCorpus
from emblema.catalog.application.use_cases.register_corpus import RegisterCorpus
from emblema.catalog.application.use_cases.register_corpus_version import RegisterCorpusVersion
from emblema.catalog.application.use_cases.tokenise_corpus_version import TokeniseCorpusVersion


@dataclass(frozen=True)
class Services:
    """The use cases this process can run, each already holding its dependencies."""

    register_corpus: RegisterCorpus
    register_corpus_version: RegisterCorpusVersion
    tokenise_corpus_version: TokeniseCorpusVersion
    publish_corpus: PublishCorpus
