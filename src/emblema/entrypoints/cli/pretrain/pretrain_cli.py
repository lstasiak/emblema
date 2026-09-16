import argparse
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path

from emblema.config.settings import Settings
from emblema.entrypoints.cli.pretrain.composition_root import CompositionRoot
from emblema.entrypoints.cli.pretrain.pretrain_invocation import PretrainInvocation
from emblema.entrypoints.cli.pretrain.services import Services
from emblema.entrypoints.cli.pretrain.source_revision import SourceRevision
from emblema.pretraining.adapters.experiments.experiment_file import ExperimentFile
from emblema.pretraining.application.use_cases.accept_pretraining_result import (
    AcceptPretrainingResultCommand,
)
from emblema.pretraining.application.use_cases.fulfil_pretraining_order import (
    FulfilPretrainingOrderCommand,
)
from emblema.pretraining.application.use_cases.order_pretraining import OrderPretrainingCommand
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum

ORDER, RUN, ACCEPT = "order", "run", "accept"
NO_REGISTRY = (
    "this process has no registry: ordering and accepting need EMBLEMA_DATABASE__* to be set"
)

Command = OrderPretrainingCommand | FulfilPretrainingOrderCommand | AcceptPretrainingResultCommand


class PretrainCli:
    """Command line of the pretraining process: order a run, fulfil it somewhere, accept it back.

    Three invocations make one backbone. On the machine that keeps the registry::

        uv run python -m emblema.entrypoints.cli.pretrain order
            --experiment experiments/control-a-s.toml --corpus <manifest key> <checksum>
            --run first

    which prints the backbone's identifier and the order's reference. On whichever machine
    trains — this one, or a notebook that installed the package at the commit the order names::

        uv run python -m emblema.entrypoints.cli.pretrain run --order <key> <checksum>

    which prints the result's reference. Back on the first machine::

        uv run python -m emblema.entrypoints.cli.pretrain accept
            --result <key> <checksum> --track <tracking URI>

    which records the run and registers the weights. The commit the order and the result carry
    is read from the installed package or the working tree; an order is placed only from a
    committed tree, because it names what another machine installs. ``--commit`` states the
    revision where it cannot be read, or where the tree is not what the run is made of.

    Standard output is the references a person copies into the next invocation, and nothing
    else: whatever a library prints while a use case runs goes to standard error.
    """

    def __init__(self, revision: SourceRevision | None = None) -> None:
        self._revision = SourceRevision() if revision is None else revision

    def parse(self, argv: Sequence[str] | None = None) -> PretrainInvocation:
        arguments = self._parser().parse_args(argv)
        return PretrainInvocation(
            command=self._command(arguments),
            workspace=arguments.workspace,
            device=getattr(arguments, "device", None),
            tracking_uri=getattr(arguments, "track", None),
            num_workers=getattr(arguments, "num_workers", 0),
        )

    def run(self, argv: Sequence[str] | None = None) -> None:  # pragma: no cover - environment
        invocation = self.parse(argv)
        command = invocation.command
        root = CompositionRoot(
            Settings(),
            workspace=invocation.workspace,
            device=invocation.device,
            tracking_uri=invocation.tracking_uri,
            result=(
                command.result if isinstance(command, AcceptPretrainingResultCommand) else None
            ),
            num_workers=invocation.num_workers,
        )
        with redirect_stdout(sys.stderr):
            printed = self.execute(command, root.services)
        print(printed)

    def execute(self, command: Command, services: Services) -> str:
        """Run ``command`` on ``services`` and return what the process prints for it.

        Raises:
            SystemExit: If the command needs the registry and this process has none.
        """
        match command:
            case OrderPretrainingCommand():
                if services.order_pretraining is None:
                    raise SystemExit(NO_REGISTRY)
                placed = services.order_pretraining(command)
                return f"{placed.backbone}\n{placed.order.key}\n{placed.order.checksum}"
            case FulfilPretrainingOrderCommand():
                reported = services.fulfil_pretraining_order(command)
                return f"{reported.key}\n{reported.checksum}"
            case _:
                if services.accept_pretraining_result is None:
                    raise SystemExit(NO_REGISTRY)
                return str(services.accept_pretraining_result(command))

    def _command(self, arguments: argparse.Namespace) -> Command:
        match arguments.what:
            case "order":
                experiment = ExperimentFile.load(arguments.experiment)
                return OrderPretrainingCommand(
                    configuration=experiment.configuration(),
                    corpus=experiment.corpus,
                    manifest=self._ref(arguments.corpus),
                    run=arguments.run,
                    git_commit=self._commit(arguments.commit, committed=True),
                )
            case "run":
                return FulfilPretrainingOrderCommand(
                    order=self._ref(arguments.order),
                    git_commit=self._commit(arguments.commit),
                    resume_from=(
                        None if arguments.resume_from is None else self._ref(arguments.resume_from)
                    ),
                )
            case _:
                return AcceptPretrainingResultCommand(result=self._ref(arguments.result))

    def _commit(self, stated: str | None, *, committed: bool = False) -> str:
        if stated is not None:
            return stated
        try:
            return self._revision.committed() if committed else self._revision.current()
        except RuntimeError as error:
            raise SystemExit(str(error)) from None

    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Order a pretraining run, fulfil an order, or accept a result."
        )
        what = parser.add_subparsers(dest="what", required=True)

        order = what.add_parser(ORDER, help="register a backbone and place the order for it")
        order.add_argument("--experiment", type=Path, required=True, help="experiment file")
        self._reference(
            order, "--corpus", "manifest of the published corpus to train on", required=True
        )
        order.add_argument("--run", required=True, help="name of the run within the experiment")
        self._common(order, commit=True)

        run = what.add_parser(RUN, help="train an order on this machine and report the result")
        self._reference(run, "--order", "the order to fulfil", required=True)
        run.add_argument("--device", help="where to train; the machine's accelerator by default")
        self._reference(run, "--resume-from", "checkpoint of an interrupted run of this order")
        run.add_argument("--num-workers", type=int, default=0, help="batch collating processes")
        run.add_argument(
            "--track", help="MLflow tracking URI; the run is kept in memory unless given"
        )
        self._common(run, commit=True)

        accept = what.add_parser(ACCEPT, help="record a result and register its weights")
        self._reference(accept, "--result", "the result to accept", required=True)
        # Accepting is the one chance to record the run: the backbone refuses a second delivery,
        # so a replay into a tracker that dies with the process would be the curve lost for good.
        accept.add_argument(
            "--track", required=True, help="MLflow tracking URI the replayed run is recorded at"
        )
        self._common(accept)
        return parser

    @staticmethod
    def _common(parser: argparse.ArgumentParser, *, commit: bool = False) -> None:
        parser.add_argument(
            "--workspace",
            type=Path,
            default=Path("data/artifacts"),
            help="where blocks are fetched to and mapped from",
        )
        if commit:
            parser.add_argument(
                "--commit",
                help="revision of the code; read from the package or the tree unless given",
            )

    @staticmethod
    def _reference(
        parser: argparse.ArgumentParser, name: str, help: str, *, required: bool = False
    ) -> None:
        parser.add_argument(
            name, nargs=2, metavar=("KEY", "CHECKSUM"), required=required, help=help
        )

    @staticmethod
    def _ref(pair: Sequence[str]) -> ArtifactRef:
        key, checksum = pair
        return ArtifactRef(key, Checksum.parse(checksum))
