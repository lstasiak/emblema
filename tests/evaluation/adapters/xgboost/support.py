"""A second published corpus, laid out unlike the first, for the fits that are meant to span them.

The transfer claim this ticket makes is that a reading summarised across channels can be fitted
over corpora whose channel layouts have nothing in common, so a test of it needs two layouts
that really differ: five channels here against the three of the shared test corpus.
"""

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.published_channel_statistics import PublishedChannelStatistics
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.shared.adapters.windows.window_block_writer import WindowBlockWriter
from emblema.shared.kernel.tokens import Token, TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore
from tests.evaluation.support import labelled, task
from tests.support.published import manifest_of

WIDE_CORPUS = "wide-test-corpus"
WIDE_CHANNELS = tuple(
    PublishedChannel(
        channel_id=channel,
        corpus=WIDE_CORPUS,
        channel=f"s{channel}",
        statistics=PublishedChannelStatistics(4, 0.0, 1.0),
    )
    for channel in range(1, 6)
)


def wide_window(offset: float) -> TokenWindow:
    """One window over all five channels, each read twice, shifted by ``offset``."""
    return TokenWindow.of(
        [
            Token(channel, offset + channel * step, time, time)
            for channel in range(1, 6)
            for step, time in ((0.0, 0.25), (0.5, 0.75))
        ]
    )


class WideCorpus(NamedTuple):
    """A task over the wider corpus, and the labels it contributes to a fit."""

    task: DownstreamTask
    sample: LabelSample


def publish_wide(
    store: ArtifactStore, workspace: Path, task_id: TaskId, ceiling: float
) -> WideCorpus:
    """Write a four-window block of five channels into ``store`` and define a task over it."""
    path = workspace / "wide.block"
    with WindowBlockWriter(path, scratch=workspace) as writer:
        for position in range(4):
            writer.add(wide_window(float(position)), unit=0, start=0.0, end=10.0)
    block = store.put_file(path)
    manifest = store.put(
        PublishedCorpusManifestJson().encode(
            manifest_of(
                block,
                corpus=WIDE_CORPUS,
                channels=WIDE_CHANNELS,
                units=("w1", "w2"),
                empty_units=(),
                training_units=("w1",),
                validation_units=("w2",),
                window_count=4,
                token_count=40,
            )
        )
    )
    defined = replace(
        task(),
        task_id=task_id,
        corpus=WIDE_CORPUS,
        manifest=manifest,
        labels=RemainingLifeScheme(ceiling),
    )
    windows = tuple(labelled("w1", position, 10.0, 1.0) for position in range(3))
    drawn = LabelSample(task=task_id, windows=windows, budget=LabelBudget.of(3), seed=3)
    return WideCorpus(defined, drawn)
