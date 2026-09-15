import pytest

from emblema.pretraining.domain.exceptions import InvalidRunSignatureError
from emblema.pretraining.domain.training.run_signature import RunSignature
from tests.support.experiments import budget, configuration, corpus


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


def test_a_signature_without_a_digest_is_refused() -> None:
    with pytest.raises(InvalidRunSignatureError):
        RunSignature("")
