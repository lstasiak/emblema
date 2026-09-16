"""Choosing which parts of a corpus a reader covers, shared by the adapters that read files.

Every downloaded corpus arrives split into parts its publisher named — C-MAPSS subsets, SKAB
folders, SMD machine groups, ESA missions — and a reader is bound to a selection of them. What a
part is called differs; that a selection must be non-empty, drawn from the parts the corpus has,
and ordered the way a checksum covers them does not.
"""

from collections.abc import Iterable


def chosen_subsets(subsets: Iterable[str], known: tuple[str, ...], part: str) -> tuple[str, ...]:
    """The selected parts in the order of ``known``, whatever order they were named in.

    Raises:
        ValueError: If no part is named or a name is not one of ``known``; ``part`` says what one
            part of this corpus is called.
    """
    chosen = set(subsets)
    unknown = sorted(chosen - set(known))
    if unknown:
        raise ValueError(f"unknown {part}s {unknown}; expected some of {known}")
    if not chosen:
        raise ValueError(f"at least one {part} is needed")
    return tuple(name for name in known if name in chosen)
