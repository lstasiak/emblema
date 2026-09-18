import pytest

from emblema.pretraining.domain.exceptions import (
    InvalidPretrainingOrderError,
    PretrainingOrderRejectedError,
)
from emblema.pretraining.domain.handoff.pretraining_order import PretrainingOrder
from tests.support.experiments import corpus, mixture
from tests.support.handoff import COMMIT, MIXTURE, OTHER_COMMIT, backbone, order


def test_the_order_of_a_backbone_is_its_run_as_registered() -> None:
    ordered = backbone()

    placed = PretrainingOrder.of(ordered)

    assert placed == order(
        backbone=ordered.id,
        configuration=ordered.configuration,
        manifests=(ordered.inputs[0].manifest,),
        run=ordered.run,
        git_commit=ordered.git_commit,
        signature=ordered.signature,
    )


def test_the_corpus_ordered_passes_and_another_one_stops_the_run() -> None:
    placed = order()

    placed.require_read(MIXTURE)
    with pytest.raises(PretrainingOrderRejectedError, match="sign run"):
        placed.require_read(mixture(corpus(training=6)))


def test_the_code_ordered_passes_and_other_code_stops_the_run() -> None:
    placed = order()

    placed.require_commit(COMMIT)
    with pytest.raises(PretrainingOrderRejectedError) as refused:
        placed.require_commit(OTHER_COMMIT)

    assert f"runs commit {OTHER_COMMIT}" in str(refused.value)
    assert f"order is for commit {COMMIT}" in str(refused.value)


@pytest.mark.parametrize("field", ["run", "git_commit"])
def test_a_blank_name_is_refused(field: str) -> None:
    with pytest.raises(InvalidPretrainingOrderError, match=field):
        order(**{field: " "})


def test_an_order_names_at_least_one_manifest_and_none_twice() -> None:
    with pytest.raises(InvalidPretrainingOrderError, match="at least one manifest"):
        order(manifests=())
    with pytest.raises(InvalidPretrainingOrderError, match="no manifest twice"):
        order(manifests=(order().manifests[0], order().manifests[0]))
