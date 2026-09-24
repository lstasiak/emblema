import argparse
import logging
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout

from emblema.config.settings import Settings
from emblema.entrypoints.cli.campaign_run.campaign_run_invocation import CampaignRunInvocation
from emblema.entrypoints.cli.campaign_run.composition_root import CompositionRoot
from emblema.entrypoints.cli.campaign_run.services import Services
from emblema.entrypoints.source_revision import SourceRevision
from emblema.evaluation.application.use_cases.fulfil_campaign_order import (
    FulfilCampaignOrderCommand,
)
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum


class CampaignRunCli:
    """Command line of a run of an order: the cells of a campaign, where no queue reaches.

    On the machine that runs them — a notebook that installed the package at the commit the
    order names, or this one, natively, for its accelerator::

        uv run python -m emblema.entrypoints.cli.campaign_run --order <key> <checksum>

    which prints the reference of the result, to be accepted where the campaign is kept. Every
    cell answered is also reported at once and logged with its reference, so a session that is
    ended early is picked up with ``--resume <key> <checksum>`` from the last one logged and
    runs only the cells that are left.

    It needs the store and the worker's settings of the order's pool; it needs no database and
    no broker, which is the whole of why it exists. Standard output is the reference and nothing
    else; the log of every report goes to standard error.
    """

    def __init__(self, revision: SourceRevision | None = None) -> None:
        self._revision = revision

    def parse(self, argv: Sequence[str] | None = None) -> CampaignRunInvocation:
        """What the arguments ask for, at the revision of the code this machine runs.

        Raises:
            RuntimeError: If the revision cannot be read.
        """
        arguments = self._parser().parse_args(argv)
        revision = SourceRevision() if self._revision is None else self._revision
        return CampaignRunInvocation(
            command=FulfilCampaignOrderCommand(
                order=self._ref(arguments.order),
                git_commit=revision.current(),
                resume=None if arguments.resume is None else self._ref(arguments.resume),
            )
        )

    def run(self, argv: Sequence[str] | None = None) -> None:  # pragma: no cover - environment
        invocation = self.parse(argv)
        settings = Settings()
        logging.basicConfig(
            stream=sys.stderr,
            level=settings.log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        root = CompositionRoot.from_environment(invocation.command.order, settings)
        with redirect_stdout(sys.stderr):
            printed = self.execute(invocation, root.services)
        print(printed)

    @staticmethod
    def execute(invocation: CampaignRunInvocation, services: Services) -> str:
        """Run the order and return what the process prints for it: the result's reference."""
        reported = services.fulfil_campaign_order(invocation.command)
        return f"{reported.key} {reported.checksum}"

    @staticmethod
    def _ref(pair: Sequence[str]) -> ArtifactRef:
        key, checksum = pair
        return ArtifactRef(key, Checksum.parse(checksum))

    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Run the cells of a campaign's order where no queue reaches."
        )
        parser.add_argument(
            "--order",
            nargs=2,
            metavar=("KEY", "CHECKSUM"),
            required=True,
            help="the order the campaign command line placed",
        )
        parser.add_argument(
            "--resume",
            nargs=2,
            metavar=("KEY", "CHECKSUM"),
            help="the last result an interrupted run of the same order logged",
        )
        return parser
