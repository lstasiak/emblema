from collections.abc import Iterable, Iterator
from pathlib import Path

from emblema.catalog.adapters.readers.subsets import chosen_subsets
from emblema.catalog.domain.exceptions import CorpusDataNotFoundError, UnknownUnitError
from emblema.catalog.domain.identifiers import UnitKey


class UnitFiles:
    """The files of a corpus that keeps one unit per file, under a directory per part.

    The clinical challenges ship a stay per file in a directory per set. The files are visited in
    canonical order — parts as declared, files by name within one — and a unit is keyed by its
    part and file stem. Units named in ``excluded`` are left out as if they were not there: a
    downstream task's frozen side, cut from the corpus by name.
    """

    def __init__(
        self,
        root: Path,
        subsets: Iterable[str],
        known: tuple[str, ...],
        extension: str,
        *,
        what: str,
        part: str,
        excluded: Iterable[str] = (),
    ) -> None:
        """Bind to the selected parts under ``root``.

        Args:
            root: Directory holding one directory per part.
            subsets: The parts to read, each one of ``known``.
            known: Every part the corpus has, in canonical order.
            extension: What a unit's file ends in, with its dot.
            what: What one unit of this corpus is called, for messages.
            part: What the publisher calls one part of this corpus, for messages.
            excluded: Unit names (file stems) left out of the corpus.

        Raises:
            ValueError: If no part is named, a name is not one of ``known``, or an excluded name
                is blank or carries surrounding whitespace.
        """
        self._root = root
        self._subsets = chosen_subsets(subsets, known, part)
        self._extension = extension
        self._what = what
        self._excluded = frozenset(excluded)
        for name in self._excluded:
            if not name or name != name.strip():
                raise ValueError(f"an excluded {what} must be named without surrounding whitespace")

    @property
    def subsets(self) -> tuple[str, ...]:
        return self._subsets

    def paths(self) -> Iterator[Path]:
        """Every file read, in canonical order.

        Raises:
            CorpusDataNotFoundError: If a selected part is missing or holds no file, or an
                excluded name matches no file of the selected parts — an exclusion that silently
                missed would let a unit meant to stay out be read.
        """
        excluded_seen: set[str] = set()
        for subset in self._subsets:
            folder = self._folder(subset)
            files = sorted(folder.glob(f"*{self._extension}"), key=lambda file: file.name)
            if not files:
                raise CorpusDataNotFoundError(f"{folder} holds no {self._what}")
            for file in files:
                if file.stem in self._excluded:
                    excluded_seen.add(file.stem)
                else:
                    yield file
        missing = sorted(self._excluded - excluded_seen)
        if missing:
            raise CorpusDataNotFoundError(
                f"excluded {self._what}s that no selected set holds: {missing}"
            )

    def key_of(self, path: Path) -> UnitKey:
        return UnitKey.within(path.parent.name, path.stem)

    def locate(self, unit: UnitKey) -> Path:
        """The file of ``unit``.

        Raises:
            UnknownUnitError: If the key names no file of a selected part — an excluded unit, a
                nested path, a step upwards — however the file system would resolve it.
            CorpusDataNotFoundError: If the part is selected but its directory is missing.
        """
        subset = unit.part
        if subset is None or subset not in self._subsets or unit.name in self._excluded:
            raise UnknownUnitError(f"{unit} is not a {self._what} of a selected set")
        folder = self._folder(subset)
        path = folder / f"{unit.name}{self._extension}"
        if path.parent != folder or not path.is_file():
            raise UnknownUnitError(f"{folder} has no {self._what} {unit.name}")
        return path

    def _folder(self, subset: str) -> Path:
        folder = self._root / subset
        if not folder.is_dir():
            raise CorpusDataNotFoundError(f"{folder} is missing")
        return folder
