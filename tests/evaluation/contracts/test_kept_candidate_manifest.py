"""The manifest of a kept candidate: what holds it together, and what it refuses."""

from dataclasses import replace
from typing import Any

import pytest

from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.contracts.exceptions import (
    InvalidKeptCandidateManifestError,
    InvalidKeptRepresentationError,
)
from emblema.evaluation.contracts.kept_candidate_manifest import KeptCandidateManifest
from emblema.evaluation.contracts.kept_representation import KeptRepresentation
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum


def ref(name: str) -> ArtifactRef:
    return ArtifactRef(key=f"durable/{name}", checksum=Checksum.of_bytes(name.encode()))


STATE = KeptRepresentation(format="torch-state", artifact=ref("state"), deviation=None)
GRAPH = KeptRepresentation(format="onnx", artifact=ref("graph"), deviation=1.5e-5)


def manifest(**overrides: Any) -> KeptCandidateManifest:
    stated = KeptCandidateManifest(
        kind=CandidateKind.NEURAL,
        corpus_manifest=ref("corpus"),
        representations=(STATE, GRAPH),
        measured_as=STATE.format,
    )
    return replace(stated, **overrides)


def test_the_measured_form_is_the_one_named_and_a_form_is_found_by_its_format() -> None:
    kept = manifest()

    assert kept.measured == STATE
    assert kept.find("onnx") == GRAPH
    assert kept.find("npz") is None


def test_a_candidate_stored_in_no_form_is_refused() -> None:
    with pytest.raises(InvalidKeptCandidateManifestError, match="some form"):
        manifest(representations=())


def test_a_format_listed_twice_is_refused() -> None:
    twice = replace(GRAPH, artifact=ref("other graph"))

    with pytest.raises(InvalidKeptCandidateManifestError, match="twice"):
        manifest(representations=(STATE, GRAPH, twice))


def test_a_measured_form_that_is_not_among_the_forms_is_refused() -> None:
    with pytest.raises(InvalidKeptCandidateManifestError, match="not among"):
        manifest(measured_as="npz")


def test_the_measured_form_reports_no_deviation_and_every_derived_form_reports_one() -> None:
    with pytest.raises(InvalidKeptCandidateManifestError, match="reports a deviation"):
        manifest(representations=(replace(STATE, deviation=0.0), GRAPH))
    with pytest.raises(InvalidKeptCandidateManifestError, match="reports no deviation"):
        manifest(representations=(STATE, replace(GRAPH, deviation=None)))


@pytest.mark.parametrize("named", ["", " onnx", "onnx "])
def test_a_form_without_a_readable_format_is_refused(named: str) -> None:
    with pytest.raises(InvalidKeptRepresentationError):
        replace(GRAPH, format=named)


@pytest.mark.parametrize("deviation", [-1e-9, float("inf"), float("nan")])
def test_a_deviation_that_is_negative_or_not_a_number_is_refused(deviation: float) -> None:
    with pytest.raises(InvalidKeptRepresentationError):
        replace(GRAPH, deviation=deviation)
