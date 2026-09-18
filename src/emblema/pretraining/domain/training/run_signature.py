from dataclasses import dataclass
from typing import Self

from emblema.pretraining.domain.exceptions import InvalidRunSignatureError
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_corpus_shape import TrainingCorpusShape
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from emblema.pretraining.domain.training.training_mixture_shape import TrainingMixtureShape
from emblema.shared.kernel.checksums import Checksum


@dataclass(frozen=True)
class RunSignature:
    """What says which run a checkpoint belongs to: the configuration and the data, as a digest.

    A checkpoint holds the state of one particular run, and putting it back into a different one
    resumes nothing — it starts a run that reports a configuration it did not train under. So the
    identity is computed here, once, from everything the configuration states and from each
    corpus read, in order: the checksum of the data, because the same configuration over other
    windows is another run, and the shape beside it, because a corpus read in part is not the
    corpus its checksum names. A run over one corpus digests as it did before runs read several,
    so what the registry holds for those runs still names them.

    Invariants: the digest is a hexadecimal checksum.

    Attributes:
        digest: Hexadecimal digest of what the run is.
    """

    digest: str

    def __post_init__(self) -> None:
        if not self.digest:
            raise InvalidRunSignatureError("a signature must have a digest")

    @classmethod
    def of(cls, configuration: ExperimentConfiguration, mixture: TrainingMixture) -> Self:
        """The signature of a run of ``configuration`` over ``mixture``."""
        return cls.of_shape(configuration, mixture.shape)

    @classmethod
    def of_shape(cls, configuration: ExperimentConfiguration, shape: TrainingMixtureShape) -> Self:
        """The signature of a run of ``configuration`` over data of that shape.

        The same digest ``of`` computes, from the description alone: a run made elsewhere is
        checked against what was ordered without the windows being here.
        """
        stated = "\n".join(f"{key}={value}" for key, value in configuration.parameters().items())
        read = "\n".join(cls._read(corpus) for corpus in shape.corpora)
        return cls(Checksum.of_bytes(f"{stated}\n{read}".encode()).digest)

    @staticmethod
    def _read(corpus: TrainingCorpusShape) -> str:
        return (
            f"corpus={corpus.name}\nchecksum={corpus.checksum}\n"
            f"vocabulary={corpus.vocabulary_size}\ntraining={corpus.training_windows}\n"
            f"validation={corpus.validation_windows}"
        )

    def __str__(self) -> str:
        return self.digest
