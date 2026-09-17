"""The report runs on one machine and is pasted into a note, so its arithmetic is checked here.

What it times is machine-dependent and untestable; what it *says* is not. The verdict about
publishing twice is the line this report exists to produce — it is how the duplicated-manifest
finding was caught — so a report that could only ever print the reassuring answer would be worse
than no report. Everything below runs on the miniature sample, which every machine has.
"""

from pathlib import Path
from typing import NamedTuple

import pytest

from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.domain.tokenisation.split_policy import SeededSplit
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from scripts.published_corpus_report import (
    Published,
    corpus_root,
    default_window,
    measured_counts,
    publish_once,
    read_back,
    render,
    writing,
)
from tests.support.corpora import CORPUS, SAMPLE, SAMPLE_WINDOW, SUBSET, publish_command


class Run(NamedTuple):
    """Two publications of the sample into one registry, and the workspace they were made in."""

    first: Published
    second: Published
    workspace: Path


def publish_into(registry: InMemoryCorpusRepository, workspace: Path) -> Published:
    command = publish_command(window=SAMPLE_WINDOW, split=SeededSplit(0.5, 1))
    return publish_once(SAMPLE, workspace, (SUBSET,), command, registry)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    workspace = tmp_path_factory.mktemp("report")
    registry = InMemoryCorpusRepository()
    return Run(
        publish_into(registry, workspace / "first"),
        publish_into(registry, workspace / "second"),
        workspace,
    )


def test_the_window_reported_is_the_one_the_budget_marks_default() -> None:
    # Pinned: every published artifact and every note that quotes one was cut with this window,
    # so moving the default has to be a deliberate change here rather than a silent one there.
    assert default_window(CORPUS) == WindowSpec(50.0, 5.0)


def test_the_counts_reported_are_the_ones_the_data_spike_measured() -> None:
    counts = measured_counts(CORPUS)

    assert counts is not None
    assert counts["units"] > 0
    assert counts["observations"] > 0


def test_a_corpus_that_was_never_fetched_stops_the_report() -> None:
    with pytest.raises(SystemExit, match="needs the real corpus"):
        corpus_root("never-downloaded")


def test_the_sizes_reported_are_those_of_the_artifacts(run: Run) -> None:
    assert run.first.block_bytes > 0
    assert run.first.manifest_bytes > 0
    assert run.first.manifest.window_count > 0
    assert run.first.seconds > 0.0


def test_two_publications_into_one_registry_report_one_block_and_one_manifest(run: Run) -> None:
    report = rendered(run.first, run.second, run.workspace)

    assert "gives one block and one manifest." in report
    assert "two manifests" not in report


def test_a_second_manifest_for_one_block_is_reported_as_the_finding_it_is(run: Run) -> None:
    # Two registries stand for two processes that persist nothing between them: one lot of data,
    # described twice. The report has to name that rather than print the reassuring sentence.
    forgetful = publish_into(InMemoryCorpusRepository(), run.workspace / "forgetful")

    report = rendered(run.first, forgetful, run.workspace)

    assert "**two manifests**" in report
    assert "did not find the first run's registration" in report


def test_reading_back_weighs_the_windows_against_holding_them(run: Run) -> None:
    reading = read_back(run.first)

    assert reading.windows == run.first.manifest.window_count
    assert reading.per_window_ms > 0.0
    assert reading.per_window_bytes > 0.0


def test_writing_reports_the_check_apart_from_the_rest_of_the_write(run: Run) -> None:
    # The check is what a reader pays too, so it is the yardstick that tells a slower machine
    # from a slower code path; folded into the whole write it would say nothing.
    writes = writing(run.first, run.workspace / "timing")

    assert writes.windows == run.first.manifest.window_count
    assert writes.add_ms > 0.0
    assert writes.check_ms > 0.0


def rendered(first: Published, second: Published, workspace: Path) -> str:
    return render(
        CORPUS, first, second, read_back(first), writing(first, workspace / "timing"), SAMPLE_WINDOW
    )
