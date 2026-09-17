from pathlib import Path

import pytest

from emblema.pretraining.adapters.experiments.experiment_file import ExperimentFile
from emblema.pretraining.domain.exceptions import InvalidRunSignatureError
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.pretraining.domain.training.training_corpus_shape import TrainingCorpusShape
from emblema.shared.kernel.checksums import Checksum
from tests.support.experiments import budget, configuration, corpus

# The control experiment at tier S over the control corpus as it was published (window 32,
# stride 12, a quarter held out), as the registry and the result documents sign it. The digest
# moves only when the configuration gains a parameter or renders one anew, and then on purpose:
# the first backbone registered against the remote bucket signed as 21c07d44…, before the share
# of the corpus became a parameter every run states and the warmup a fraction of an epoch.
CONTROL_A = TrainingCorpusShape(
    name="control-a",
    checksum=Checksum.parse(
        "sha256:39c03b8c03302b4dfcef510dd23f94e0b2f972e451c3b20c638d7bae7be562fb"
    ),
    training_windows=4732,
    validation_windows=1534,
    vocabulary_size=9,
)
CONTROL_A_S_SIGNATURE = "96fa0aca4d082a6e15f5fc48eee6997410bfd795e9ef0fccc5b4b735c31fb4c9"


def test_the_same_run_signs_the_same_and_another_configuration_does_not() -> None:
    stated, windows = configuration(), corpus()

    assert RunSignature.of(stated, windows) == RunSignature.of(stated, windows)
    assert RunSignature.of(configuration(budget=budget(seed=2)), windows) != RunSignature.of(
        stated, windows
    )


def test_the_same_configuration_over_another_corpus_is_another_run() -> None:
    stated = configuration()

    assert RunSignature.of(stated, corpus(name="other")) != RunSignature.of(stated, corpus())
    assert RunSignature.of(stated, corpus(training=6)) != RunSignature.of(stated, corpus())


def test_two_corpora_of_one_shape_holding_other_data_are_other_runs() -> None:
    """The shape of a corpus is not what it is: the checksum of the data decides."""
    stated = configuration()
    read, other = corpus(seed=1), corpus(seed=7)

    assert (other.name, len(other.training)) == (read.name, len(read.training))
    assert other.checksum != read.checksum
    assert RunSignature.of(stated, other) != RunSignature.of(stated, read)


def test_a_configuration_built_from_integers_signs_as_the_same_run() -> None:
    windows = corpus()

    assert RunSignature.of(configuration(dropout=0), windows) == RunSignature.of(
        configuration(dropout=0.0), windows
    )


def test_the_signature_of_a_registered_run_is_the_digest_the_registry_holds() -> None:
    stated = ExperimentFile.load(Path("experiments/control-a-s.toml")).configuration()

    assert RunSignature.of_shape(stated, CONTROL_A).digest == CONTROL_A_S_SIGNATURE


def test_a_signature_without_a_digest_is_refused() -> None:
    with pytest.raises(InvalidRunSignatureError):
        RunSignature("")
