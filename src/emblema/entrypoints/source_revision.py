import json
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname


class SourceRevision:
    """The revision of the code this process runs, read from evidence rather than typed in.

    Two places know it. A package installed from a repository at a revision records that
    revision in its distribution metadata, which is how a notebook that installed
    ``git+…@<sha>`` knows what it runs without a checkout. A package installed editable from a
    working tree records the tree instead, and the tree's own ``git rev-parse HEAD`` says where
    it stands; a tree with uncommitted changes is marked as such, because a run made on it is
    not a run of that commit and an acceptance comparing commits must see the difference.
    """

    DIRTY: str = "-dirty"

    def __init__(self, distribution_name: str = "emblema", tree: Path | None = None) -> None:
        """Read the revision of ``distribution_name``, or of the tree at ``tree``.

        Args:
            distribution_name: The installed package whose metadata may record a revision.
            tree: The working tree to ask where the package records no revision; the tree the
                package was installed editable from unless given, and the current directory
                where it records none either.
        """
        self._distribution = distribution_name
        self._tree = tree

    def current(self) -> str:
        """The revision, from the installed package or the working tree, marked when dirty.

        Raises:
            RuntimeError: If neither says: the package was not installed from a repository and
                there is no working tree to ask.
        """
        recorded = self._recorded()
        installed = self._installed_at(recorded)
        if installed is not None:
            return installed
        checked_out = self._checked_out(self._tree_of(recorded))
        if checked_out is not None:
            return checked_out
        raise RuntimeError(
            f"the revision of {self._distribution!r} is unknown: it was not installed from a "
            f"repository and no git tree answers here; state it with --commit"
        )

    def committed(self) -> str:
        """The revision, refused where the working tree has changes no commit holds.

        An order names the revision another machine installs, and a marked revision is one
        nobody can install: the tree is committed first, or the revision is stated.

        Raises:
            RuntimeError: If the tree is dirty, or the revision is unknown.
        """
        revision = self.current()
        if revision.endswith(self.DIRTY):
            raise RuntimeError(
                "the working tree has uncommitted changes: commit them, or state the revision "
                "the order is for with --commit"
            )
        return revision

    def _recorded(self) -> dict[str, object]:
        try:
            recorded = distribution(self._distribution).read_text("direct_url.json")
        except PackageNotFoundError:
            return {}
        parsed = json.loads(recorded) if recorded else {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _installed_at(recorded: dict[str, object]) -> str | None:
        vcs = recorded.get("vcs_info")
        commit = vcs.get("commit_id") if isinstance(vcs, dict) else None
        return str(commit) if commit else None

    def _tree_of(self, recorded: dict[str, object]) -> Path | None:
        if self._tree is not None:
            return self._tree
        location, url = recorded.get("dir_info"), recorded.get("url")
        if isinstance(location, dict) and location.get("editable") and isinstance(url, str):
            parsed = urlparse(url)
            if parsed.scheme == "file":
                return Path(url2pathname(parsed.path))
        return None

    def _checked_out(self, tree: Path | None) -> str | None:
        head = self._git(tree, "rev-parse", "HEAD")
        if head is None:
            return None
        changes = self._git(tree, "status", "--porcelain")
        return head + (self.DIRTY if changes else "")

    @staticmethod
    def _git(tree: Path | None, *arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments], cwd=tree, capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None
